#!/usr/bin/env python3
"""
run_baselines.py

Organized leakage-safe baseline matrix:
  - KNN-Imputer (multivariate imputation)
  - Ridge regression (linear soft sensor)
  - HistGradientBoosting (strong tabular soft sensor)

All runs share train-only preprocessing artifacts under:
  testes/baselines_leakage_safe/primary_target_v1/shared/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baselines.config import (
    DEFAULT_CDAE_RESULTS_ROOT,
    DEFAULT_DATA,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    SUPPORTED_MECHANISMS,
    SUPPORTED_MODELS,
    BaselineProtocolConfig,
)
from baselines.consolidate import consolidate_baselines
from baselines.protocol import prepare_protocol_artifacts
from baselines.run_single import run_single_baseline

PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TEX_FIGS = PROJECT_ROOT / "texto-tex" / "figs"


def build_config_from_args(args: argparse.Namespace) -> BaselineProtocolConfig:
    """Map CLI args to protocol config."""
    splits = Path(args.cdae_splits_file) if args.cdae_splits_file else Path("")
    mechanism = getattr(args, "mechanism", "MCAR")
    model = getattr(args, "model", "ridge")
    return BaselineProtocolConfig(
        data_path=Path(args.data),
        feature_config_path=Path(args.feature_config),
        output_root=Path(args.output),
        cdae_splits_file=splits,
        mechanism=str(mechanism),
        model=str(model),
        missing_rate=float(args.missing_rate),
        k_neighbors=int(args.k_neighbors),
        seed=int(args.seed),
        bootstrap_n=int(args.bootstrap_n),
        aux_col=str(args.aux_col or ""),
        robust_target=bool(args.robust_target),
        blocked_by_index=bool(args.blocked_by_index),
        time_test_size=float(args.time_test_size),
        time_val_size=float(args.time_val_size),
        min_target_coverage=float(args.min_target_coverage),
        min_target_count=int(args.min_target_count),
        auto_skip_until_coverage=bool(args.auto_skip_until_coverage),
        max_skip_head_frac=float(args.max_skip_head_frac),
        strict_validation=bool(args.strict_validation),
    )


def cmd_prepare(cfg: BaselineProtocolConfig) -> None:
    """Create shared protocol artifacts once."""
    artifacts = prepare_protocol_artifacts(cfg)
    print("[OK] Shared protocol artifacts prepared.")
    print(f"     Output root: {cfg.output_root}")
    print(f"     Train/val/test: {artifacts['metadata']['train_samples']}/"
          f"{artifacts['metadata']['val_samples']}/"
          f"{artifacts['metadata']['test_samples']}")


def cmd_run_single(cfg: BaselineProtocolConfig) -> None:
    """Run one model and mechanism."""
    run_single_baseline(cfg)


def cmd_run_all(cfg: BaselineProtocolConfig, models: list[str], mechanisms: list[str]) -> None:
    """Run full model x mechanism matrix."""
    artifacts = prepare_protocol_artifacts(cfg)
    matrix_summary = {}
    for model in models:
        for mechanism in mechanisms:
            cfg.model = model
            cfg.mechanism = mechanism
            result = run_single_baseline(cfg, artifacts=artifacts)
            matrix_summary[f"{model}_{mechanism}"] = result["metrics"]
    matrix_path = cfg.output_root / "run_matrix_summary.json"
    with open(matrix_path, "w", encoding="utf-8") as fh:
        json.dump(matrix_summary, fh, indent=2, ensure_ascii=True)
    print(f"[OK] Matrix summary: {matrix_path}")


def cmd_consolidate(cfg: BaselineProtocolConfig, copy_figs: bool, cdae_results_root: Path) -> None:
    """Aggregate results and export publication assets."""
    dest = DEFAULT_TEX_FIGS if copy_figs else None
    consolidate_baselines(
        cfg.output_root,
        copy_figures_to=dest,
        cdae_results_root=cdae_results_root,
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    parser = argparse.ArgumentParser(
        description="Leakage-safe baseline comparison runner (KNN, Ridge, HGB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", default=str(DEFAULT_DATA))
        p.add_argument("--feature_config", default=str(DEFAULT_FEATURE_CONFIG))
        p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
        p.add_argument("--cdae_splits_file", default="")
        p.add_argument("--missing_rate", type=float, default=0.8)
        p.add_argument("--k_neighbors", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--bootstrap_n", type=int, default=1000)
        p.add_argument("--aux_col", default="")
        p.add_argument("--robust_target", action="store_true")
        p.add_argument("--blocked_by_index", action="store_true", default=True)
        p.add_argument("--time_test_size", type=float, default=0.05)
        p.add_argument("--time_val_size", type=float, default=0.1)
        p.add_argument("--min_target_coverage", type=float, default=0.2)
        p.add_argument("--min_target_count", type=int, default=5000)
        p.add_argument("--auto_skip_until_coverage", action="store_true", default=True)
        p.add_argument("--max_skip_head_frac", type=float, default=0.95)
        p.add_argument("--strict_validation", action="store_true")

    def add_cdae_root(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--cdae_results_root",
            default=str(DEFAULT_CDAE_RESULTS_ROOT),
            help="Root with phase2/ and ablation_lambda0/ CDAE runs for comparison figures",
        )

    p_prepare = sub.add_parser("prepare", help="Create shared protocol artifacts")
    add_common(p_prepare)

    p_single = sub.add_parser("single", help="Run one model x mechanism")
    add_common(p_single)
    add_cdae_root(p_single)
    p_single.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    p_single.add_argument("--mechanism", choices=SUPPORTED_MECHANISMS, required=True)

    p_all = sub.add_parser("all", help="Run full baseline matrix")
    add_common(p_all)
    add_cdae_root(p_all)
    p_all.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=SUPPORTED_MODELS,
    )
    p_all.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(SUPPORTED_MECHANISMS),
        choices=SUPPORTED_MECHANISMS,
    )

    p_all.add_argument(
        "--copy_figures_to_tex",
        action="store_true",
        help="Copy consolidated figures to texto-tex/figs",
    )

    p_cons = sub.add_parser("consolidate", help="Aggregate results and export figures/macros")
    add_common(p_cons)
    add_cdae_root(p_cons)
    p_cons.add_argument(
        "--copy_figures_to_tex",
        action="store_true",
        help="Copy consolidated figures to texto-tex/figs",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)

    if args.command == "prepare":
        cmd_prepare(cfg)
    elif args.command == "single":
        prepare_protocol_artifacts(cfg)
        cmd_run_single(cfg)
    elif args.command == "all":
        cmd_run_all(cfg, list(args.models), list(args.mechanisms))
        consolidate_baselines(
            cfg.output_root,
            copy_figures_to=DEFAULT_TEX_FIGS if args.copy_figures_to_tex else None,
            cdae_results_root=Path(args.cdae_results_root),
        )
    elif args.command == "consolidate":
        cmd_consolidate(cfg, bool(args.copy_figures_to_tex), Path(args.cdae_results_root))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
