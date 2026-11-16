#!/usr/bin/env python3
"""Offline protocol/provenance audit for archived industrial virtual-sensing runs.

This verifies *recorded* split indices and metadata. It cannot establish how
a model was trained without the genuine execution artifacts or raw data.
Only inspect NPZ files from a trusted source (legacy object arrays may pickle).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_splits(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required execution artifact is missing: {path}")
    with np.load(path, allow_pickle=True) as saved:
        required = ("train_idx", "val_idx", "test_idx")
        if not all(k in saved.files for k in required):
            raise ValueError(f"{path} must include {', '.join(required)}")
        splits = {k: np.asarray(saved[k]).reshape(-1) for k in required}
    sets = {k: set(splits[k].tolist()) for k in splits}
    disjoint = all(
        sets[a].isdisjoint(sets[b])
        for a, b in (("train_idx", "val_idx"), ("train_idx", "test_idx"), ("val_idx", "test_idx"))
    )
    meta = {
        "sha256": sha256(path),
        "sizes_observed_target": {k: int(len(v)) for k, v in splits.items()},
        "pairwise_disjoint": disjoint,
        "unique_counts": {k: len(sets[k]) for k in sets},
        "unique_within_fold": all(len(sets[k]) == len(splits[k]) for k in splits),
    }
    return splits, meta


def discover(run: Path) -> Path:
    if run.is_file() and run.suffix == ".npz":
        return run
    for rel in ("splits_indices.npz", "shared/splits_indices.npz"):
        candidate = run / rel
        if candidate.is_file():
            return candidate
    candidates = sorted(run.rglob("splits_indices.npz")) if run.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple split artifacts under {run}; pass the specific run directory.")
    raise FileNotFoundError(f"No splits_indices.npz under {run}")


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        k: {
            "same_indices_in_same_order": bool(np.array_equal(reference[k], other[k])),
            "same_index_set": set(reference[k].tolist()) == set(other[k].tolist()),
        }
        for k in ("train_idx", "val_idx", "test_idx")
    }


def report_one(name: str, path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = discover(path)
    indices, metadata = load_splits(source)
    run = source.parent
    metadata["source"] = str(source)
    metadata["run"] = name
    metadata["coverage_report"] = read_json_if_present(run / "coverage_report.json")
    metadata["run_manifest"] = read_json_if_present(run / "run_manifest.json")
    metadata["dataset_metadata"] = read_json_if_present(run / "dataset_metadata.json")
    checkpoint = run / "best_model.pth"
    metadata["checkpoint_sha256"] = sha256(checkpoint) if checkpoint.is_file() else None
    config_candidates = [run / "controlled_test_results.json", run / "complete_pipeline_results.json"]
    metadata["results_file"] = next((str(p) for p in config_candidates if p.is_file()), None)
    return indices, metadata


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1", required=True, type=Path, help="Phase 1 output directory or its split NPZ")
    p.add_argument("--phase2", nargs="*", default=[], type=Path, help="Controlled-run directories, e.g. MCAR MAR MNAR")
    p.add_argument("--baseline", type=Path, help="Tabular shared-protocol directory")
    p.add_argument("--output", type=Path, default=Path("protocol_audit.json"))
    args = p.parse_args()

    ref, ref_report = report_one("phase1", args.phase1)
    output: dict[str, Any] = {
        "reference": ref_report,
        "comparisons": [],
        "interpretation": (
            "Exact index agreement is necessary but not sufficient for independent "
            "reproduction. This utility does not compare model weights, training histories, "
            "feature/configuration hashes, raw dataset versions or execution commands."
        ),
    }
    failures = [] if ref_report["pairwise_disjoint"] and ref_report["unique_within_fold"] else ["phase1 overlaps/duplicates"]
    entries = [(f"phase2_{i}", path) for i, path in enumerate(args.phase2, 1)]
    if args.baseline:
        entries.append(("baseline", args.baseline))
    for name, path in entries:
        index, metadata = report_one(name, path)
        agreement = check(ref, index)
        entry = {"metadata": metadata, "comparison_to_phase1": agreement}
        output["comparisons"].append(entry)
        if not metadata["pairwise_disjoint"] or not metadata["unique_within_fold"]:
            failures.append(f"{name} split overlaps or duplicates")
        if not all(x["same_indices_in_same_order"] for x in agreement.values()):
            failures.append(f"{name} differs from Phase 1 split indices")

    neural_hashes = [
        entry["metadata"]["checkpoint_sha256"]
        for entry in output["comparisons"]
        if entry["metadata"]["run"].startswith("phase2_")
    ]
    output["phase2_checkpoint_hashes"] = neural_hashes
    output["phase2_checkpoints_identical_as_files"] = (
        len(neural_hashes) >= 2 and all(hash_value is not None for hash_value in neural_hashes)
        and len(set(neural_hashes)) == 1
    ) if neural_hashes else None
    output["phase2_checkpoint_hash_status"] = (
        "NOT_AVAILABLE" if not neural_hashes or any(x is None for x in neural_hashes)
        else ("IDENTICAL_FILE_HASH" if len(set(neural_hashes)) == 1 else "DIFFERENT_FILE_HASH")
    )
    output["status"] = "FAIL" if failures else ("PASS_INDICES_ONLY" if entries else "INCOMPLETE_NO_COMPARATORS")
    output["issues"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{output['status']}: {args.output}")
    if output["phase2_checkpoint_hash_status"] == "DIFFERENT_FILE_HASH":
        print("NOTE: phase2 checkpoints have different file hashes; model-state equivalence is not established.")
    if failures:
        for issue in failures:
            print(" -", issue)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
