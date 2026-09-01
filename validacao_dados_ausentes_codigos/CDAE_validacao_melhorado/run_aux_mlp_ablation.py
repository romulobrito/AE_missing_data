#!/usr/bin/env python3
"""
run_aux_mlp_ablation.py

Capacity-matched auxiliary-only MLP ablation under the leakage-safe protocol.

Subcommands:
  prepare  - copy shared protocol artifacts + freeze MCAR/MAR/MNAR masks
  train    - train aux_mlp once
  eval     - evaluate frozen checkpoint on frozen masks
  export   - write LaTeX macros
  all      - prepare + train + eval + export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ablations.config import (
    DEFAULT_DATA,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROTOCOL_SHARED,
    SUPPORTED_MECHANISMS,
    AblationConfig,
)
from ablations.evaluate import evaluate_aux_mlp
from ablations.export_latex import export_latex_artifacts
from ablations.prepare import create_and_save_masks, prepare_shared_protocol
from ablations.train import train_aux_mlp


def build_config(args: argparse.Namespace) -> AblationConfig:
    return AblationConfig(
        data_path=Path(args.data),
        feature_config_path=Path(args.feature_config),
        protocol_shared_source=Path(args.protocol_shared),
        output_root=Path(args.output),
        missing_rate=float(args.missing_rate),
        seed=int(args.seed),
        bootstrap_n=int(args.bootstrap_n),
        aux_col=str(args.aux_col or ""),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        patience=int(args.patience),
        dropout=float(args.dropout),
        latent_dim=int(args.latent_dim),
        device=str(args.device),
    )


def cmd_prepare(cfg: AblationConfig) -> None:
    artifacts = prepare_shared_protocol(cfg)
    audit = create_and_save_masks(cfg, artifacts)
    print("[OK] Shared protocol copied and masks frozen.")
    for mech, info in audit.items():
        print(f"     {mech}: n_masked={info['n_masked']} / {info['n_eligible']}")


def cmd_train(cfg: AblationConfig) -> None:
    train_aux_mlp(cfg)


def cmd_eval(cfg: AblationConfig, mechanisms: list[str]) -> None:
    evaluate_aux_mlp(cfg, mechanisms=mechanisms)


def cmd_export(cfg: AblationConfig, copy_to_tex: bool) -> None:
    export_latex_artifacts(cfg, copy_to_tex=copy_to_tex)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auxiliary-only MLP ablation (capacity-matched to TCDR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", default=str(DEFAULT_DATA))
        p.add_argument("--feature_config", default=str(DEFAULT_FEATURE_CONFIG))
        p.add_argument("--protocol_shared", default=str(DEFAULT_PROTOCOL_SHARED))
        p.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
        p.add_argument("--missing_rate", type=float, default=0.8)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--bootstrap_n", type=int, default=1000)
        p.add_argument("--aux_col", default="")
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--learning_rate", type=float, default=1e-3)
        p.add_argument("--patience", type=int, default=20)
        p.add_argument("--dropout", type=float, default=0.2)
        p.add_argument("--latent_dim", type=int, default=16)
        p.add_argument("--device", default="cpu")

    p_prepare = sub.add_parser("prepare", help="Copy shared protocol and freeze masks")
    add_common(p_prepare)

    p_train = sub.add_parser("train", help="Train aux_mlp once")
    add_common(p_train)

    p_eval = sub.add_parser("eval", help="Evaluate frozen checkpoint on frozen masks")
    add_common(p_eval)
    p_eval.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(SUPPORTED_MECHANISMS),
        choices=SUPPORTED_MECHANISMS,
    )

    p_export = sub.add_parser("export", help="Export LaTeX macros")
    add_common(p_export)
    p_export.add_argument("--copy_to_tex", action="store_true", default=True)
    p_export.add_argument("--no_copy_to_tex", action="store_true")

    p_all = sub.add_parser("all", help="prepare + train + eval + export")
    add_common(p_all)
    p_all.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(SUPPORTED_MECHANISMS),
        choices=SUPPORTED_MECHANISMS,
    )
    p_all.add_argument("--copy_to_tex", action="store_true", default=True)
    p_all.add_argument("--no_copy_to_tex", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config(args)

    if args.command == "prepare":
        cmd_prepare(cfg)
    elif args.command == "train":
        cmd_train(cfg)
    elif args.command == "eval":
        cmd_eval(cfg, list(args.mechanisms))
    elif args.command == "export":
        cmd_export(cfg, copy_to_tex=not bool(getattr(args, "no_copy_to_tex", False)))
    elif args.command == "all":
        cmd_prepare(cfg)
        cmd_train(cfg)
        cmd_eval(cfg, list(args.mechanisms))
        cmd_export(cfg, copy_to_tex=not bool(args.no_copy_to_tex))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
