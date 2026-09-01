"""Aggregate baseline runs, export LaTeX macros, and build summary figures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from baselines.config import (
    COMPARISON_METHOD_LABELS,
    COMPARISON_METHODS,
    DEFAULT_CDAE_RESULTS_ROOT,
    PROJECT_ROOT,
    SUPPORTED_MECHANISMS,
    SUPPORTED_MODELS,
)
from baselines.plots import save_comparison_bars


def _find_cdae_run_dir(base: Path, mechanism: str) -> Optional[Path]:
    """Locate the latest controlled_cdae_v2_* folder for a mechanism."""
    candidates = [
        base / "phase2" / f"{mechanism}_80pct",
        base / f"{mechanism}_80pct",
        base / mechanism,
    ]
    for mech_dir in candidates:
        if not mech_dir.is_dir():
            continue
        subdirs = sorted(mech_dir.glob("controlled_cdae_v2_*"))
        if subdirs:
            return subdirs[-1]
    return None


def _load_cdae_metrics(cdae_root: Path, variant: str, mechanism: str) -> Dict[str, float]:
    """Load masked-index metrics for a CDAE variant (lambda=1 or lambda=0)."""
    if variant == "cdae_l1":
        search_base = cdae_root
    elif variant == "cdae_l0":
        search_base = cdae_root / "ablation_lambda0"
    else:
        raise ValueError(f"Unknown CDAE variant: {variant}")

    run_dir = _find_cdae_run_dir(search_base, mechanism)
    if run_dir is None:
        raise FileNotFoundError(
            f"No CDAE run for {variant}/{mechanism} under {search_base}"
        )
    result_path = run_dir / "controlled_test_results.json"
    with open(result_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics payload in {result_path}")
    return {
        "MAE": float(metrics["MAE"]),
        "RMSE": float(metrics["RMSE"]),
        "R2": float(metrics["R2"]),
        "MAE_CI_low": float(metrics["MAE_CI_low"]),
        "MAE_CI_high": float(metrics["MAE_CI_high"]),
        "RMSE_CI_low": float(metrics["RMSE_CI_low"]),
        "RMSE_CI_high": float(metrics["RMSE_CI_high"]),
        "n_samples": float(metrics.get("n_samples", 0)),
    }


def _load_run_metrics(output_root: Path, model: str, mechanism: str) -> Dict[str, float]:
    result_path = output_root / model / mechanism / "baseline_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing baseline result: {result_path}")
    with open(result_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics payload in {result_path}")
    return {
        "MAE": float(metrics["MAE"]),
        "RMSE": float(metrics["RMSE"]),
        "R2": float(metrics["R2"]),
        "MAE_CI_low": float(metrics["MAE_CI_low"]),
        "MAE_CI_high": float(metrics["MAE_CI_high"]),
        "RMSE_CI_low": float(metrics["RMSE_CI_low"]),
        "RMSE_CI_high": float(metrics["RMSE_CI_high"]),
        "n_samples": float(metrics.get("n_samples", 0)),
    }


def build_summary(output_root: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collect metrics for all model x mechanism combinations."""
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in SUPPORTED_MODELS:
        summary[model] = {}
        for mechanism in SUPPORTED_MECHANISMS:
            summary[model][mechanism] = _load_run_metrics(output_root, model, mechanism)
    return summary


def build_comparison_summary(
    output_root: Path,
    cdae_results_root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Merge tabular baselines with CDAE Phase~2 and ablation metrics."""
    summary = build_summary(output_root)
    for variant in ("cdae_l1", "cdae_l0"):
        summary[variant] = {}
        for mechanism in SUPPORTED_MECHANISMS:
            summary[variant][mechanism] = _load_cdae_metrics(
                cdae_results_root,
                variant,
                mechanism,
            )
    return summary


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def export_latex_macros(summary: Dict[str, Dict[str, Dict[str, float]]], output_path: Path) -> None:
    """Write macros_results-compatible baseline macros."""
    lines: List[str] = [
        "% Auto-generated baseline macros (leakage-safe protocol)",
        f"% Source: {output_path.parent.as_posix()}",
        "",
    ]
    macro_map: List[Tuple[str, str, str]] = [
        ("MAE", "MAE"),
        ("MAECIlow", "MAE_CI_low"),
        ("MAECIhigh", "MAE_CI_high"),
        ("RMSE", "RMSE"),
        ("RMSECIlow", "RMSE_CI_low"),
        ("RMSECIhigh", "RMSE_CI_high"),
        ("Rtwo", "R2"),
    ]
    for model in SUPPORTED_MODELS:
        for mechanism in SUPPORTED_MECHANISMS:
            mech_key = mechanism.lower()
            model_key = model
            metrics = summary[model][mechanism]
            for suffix, field in macro_map:
                macro_name = f"\\{model_key}{mech_key.capitalize()}{suffix}"
                if suffix == "Rtwo":
                    value = _fmt(metrics[field], 3)
                else:
                    value = _fmt(metrics[field], 2 if "CI" in suffix else 3)
                lines.append(f"\\newcommand{{{macro_name}}}{{{value}}}")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def consolidate_baselines(
    output_root: Path,
    copy_figures_to: Path | None = None,
    cdae_results_root: Path | None = None,
) -> Dict[str, object]:
    """Build consolidated JSON, macros, and comparison figures."""
    cdae_root = cdae_results_root or DEFAULT_CDAE_RESULTS_ROOT
    baseline_summary = build_summary(output_root)
    comparison_summary = build_comparison_summary(output_root, cdae_root)
    consolidated_dir = output_root / "consolidated"
    fig_dir = consolidated_dir / "figures"
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    with open(consolidated_dir / "baseline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(baseline_summary, fh, indent=2, ensure_ascii=True)
    with open(consolidated_dir / "comparison_summary.json", "w", encoding="utf-8") as fh:
        json.dump(comparison_summary, fh, indent=2, ensure_ascii=True)

    export_latex_macros(baseline_summary, consolidated_dir / "macros_baselines.tex")

    methods: Sequence[str] = list(COMPARISON_METHODS)
    mechanisms: Sequence[str] = list(SUPPORTED_MECHANISMS)
    save_comparison_bars(
        comparison_summary,
        metric="MAE",
        methods=methods,
        mechanisms=mechanisms,
        output_path=fig_dir / "comparison_mae_by_mechanism.png",
        title="Method comparison: MAE on masked indices (r=0.8)",
        method_labels=COMPARISON_METHOD_LABELS,
        ci_metric=True,
    )
    save_comparison_bars(
        comparison_summary,
        metric="R2",
        methods=methods,
        mechanisms=mechanisms,
        output_path=fig_dir / "comparison_r2_by_mechanism.png",
        title="Method comparison: R2 on masked indices (r=0.8)",
        method_labels=COMPARISON_METHOD_LABELS,
        ci_metric=False,
    )

    if copy_figures_to is not None:
        copy_figures_to.mkdir(parents=True, exist_ok=True)
        for src in fig_dir.glob("comparison_*.png"):
            shutil.copy2(src, copy_figures_to / src.name)

    macros_dest = PROJECT_ROOT / "texto-tex" / "macros_baselines_generated.tex"
    shutil.copy2(consolidated_dir / "macros_baselines.tex", macros_dest)

    print(f"[OK] Consolidated summary: {consolidated_dir / 'baseline_summary.json'}")
    print(f"[OK] Comparison summary: {consolidated_dir / 'comparison_summary.json'}")
    print(f"[OK] LaTeX macros: {consolidated_dir / 'macros_baselines.tex'}")
    print(f"[OK] Figures: {fig_dir}")
    return {
        "baseline_summary": baseline_summary,
        "comparison_summary": comparison_summary,
        "consolidated_dir": str(consolidated_dir),
    }
