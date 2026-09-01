"""Export aux_mlp metrics to LaTeX macros and a comparison table snippet."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

from ablations.config import AblationConfig, PROJECT_ROOT, SUPPORTED_MECHANISMS


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _load_eval_metrics(cfg: AblationConfig) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for mechanism in SUPPORTED_MECHANISMS:
        path = cfg.eval_dir(mechanism) / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        metrics = payload["metrics"]
        out[mechanism] = {
            "MAE": float(metrics["MAE"]),
            "RMSE": float(metrics["RMSE"]),
            "R2": float(metrics["R2"]),
            "MAE_CI_low": float(metrics["MAE_CI_low"]),
            "MAE_CI_high": float(metrics["MAE_CI_high"]),
            "RMSE_CI_low": float(metrics["RMSE_CI_low"]),
            "RMSE_CI_high": float(metrics["RMSE_CI_high"]),
            "n_samples": float(metrics["n_samples"]),
        }
    return out


def export_latex_artifacts(
    cfg: AblationConfig,
    copy_to_tex: bool = True,
) -> Dict[str, Path]:
    """Write macros and table snippet under consolidated/."""
    metrics = _load_eval_metrics(cfg)
    model_config_path = cfg.model_dir() / "model_config.json"
    param_count = 0
    if model_config_path.exists():
        with open(model_config_path, "r", encoding="utf-8") as fh:
            param_count = int(json.load(fh).get("parameter_count", 0))

    cons = cfg.consolidated_dir()
    cons.mkdir(parents=True, exist_ok=True)

    with open(cons / "aux_mlp_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"metrics": metrics, "parameter_count": param_count},
            fh,
            indent=2,
            ensure_ascii=True,
        )

    lines: List[str] = [
        "% AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.",
        f"% Source: {cons.as_posix()}",
        "% Variant: aux_mlp (capacity-matched auxiliary-only MLP)",
        "",
        f"\\newcommand{{\\auxMlpParamCount}}{{{param_count}}}",
        "",
    ]
    for mechanism in SUPPORTED_MECHANISMS:
        key = mechanism.lower().capitalize()
        row = metrics[mechanism]
        prefix = f"auxMlp{key}"
        lines.append(f"\\newcommand{{\\{prefix}MAE}}{{{_fmt(row['MAE'], 3)}}}")
        lines.append(f"\\newcommand{{\\{prefix}MAECIlow}}{{{_fmt(row['MAE_CI_low'], 2)}}}")
        lines.append(f"\\newcommand{{\\{prefix}MAECIhigh}}{{{_fmt(row['MAE_CI_high'], 2)}}}")
        lines.append(f"\\newcommand{{\\{prefix}RMSE}}{{{_fmt(row['RMSE'], 3)}}}")
        lines.append(f"\\newcommand{{\\{prefix}RMSECIlow}}{{{_fmt(row['RMSE_CI_low'], 2)}}}")
        lines.append(f"\\newcommand{{\\{prefix}RMSECIhigh}}{{{_fmt(row['RMSE_CI_high'], 2)}}}")
        lines.append(f"\\newcommand{{\\{prefix}Rtwo}}{{{_fmt(row['R2'], 3)}}}")
        lines.append("")

    macros_path = cons / "macros_aux_mlp.tex"
    macros_path.write_text("\n".join(lines), encoding="utf-8")

    table_lines = [
        "% AUTO-GENERATED snippet for appendix table tab:aux_mlp_ablation",
        "\\multirow{3}{*}{Auxiliary-only MLP} & MCAR & "
        "\\auxMlpMcarMAE\\ [\\auxMlpMcarMAECIlow, \\auxMlpMcarMAECIhigh] & "
        "\\auxMlpMcarRMSE\\ [\\auxMlpMcarRMSECIlow, \\auxMlpMcarRMSECIhigh] & "
        "\\auxMlpMcarRtwo \\\\",
        " & MAR & "
        "\\auxMlpMarMAE\\ [\\auxMlpMarMAECIlow, \\auxMlpMarMAECIhigh] & "
        "\\auxMlpMarRMSE\\ [\\auxMlpMarRMSECIlow, \\auxMlpMarRMSECIhigh] & "
        "\\auxMlpMarRtwo \\\\",
        " & MNAR & "
        "\\auxMlpMnarMAE\\ [\\auxMlpMnarMAECIlow, \\auxMlpMnarMAECIhigh] & "
        "\\auxMlpMnarRMSE\\ [\\auxMlpMnarRMSECIlow, \\auxMlpMnarRMSECIhigh] & "
        "\\auxMlpMnarRtwo \\\\",
    ]
    table_path = cons / "table_aux_mlp_comparison.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    paths = {"macros": macros_path, "table": table_path, "summary": cons / "aux_mlp_summary.json"}
    if copy_to_tex:
        tex_dir = PROJECT_ROOT / "texto-tex"
        dest = tex_dir / "macros_aux_mlp.tex"
        shutil.copy2(macros_path, dest)
        paths["macros_tex_copy"] = dest
        print(f"[OK] Copied macros to {dest}")

    print(f"[OK] LaTeX macros: {macros_path}")
    print(f"[OK] Table snippet: {table_path}")
    return paths
