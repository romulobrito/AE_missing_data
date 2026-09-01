"""Default configuration for leakage-safe baseline experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_DATA = PROJECT_ROOT / "sulfatos_dados_concatenados_formatados_2021.h5"
DEFAULT_FEATURE_CONFIG = (
    SCRIPT_DIR / "outputs" / "cdae_optimal_20251226_171939" / "feature_selection_config.json"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "testes" / "baselines_leakage_safe" / "primary_target_v1"
DEFAULT_CDAE_RESULTS_ROOT = (
    SCRIPT_DIR / "testes_blocked_20260211_092413"
)

SUPPORTED_MODELS: Tuple[str, ...] = ("knn", "ridge", "hgb")
SUPPORTED_MECHANISMS: Tuple[str, ...] = ("MCAR", "MAR", "MNAR")

# Plot order for unified method comparison figures (tabular + CDAE variants).
COMPARISON_METHODS: Tuple[str, ...] = (
    "knn",
    "ridge",
    "hgb",
    "cdae_l1",
    "cdae_l0",
)
COMPARISON_METHOD_LABELS: Dict[str, str] = {
    "knn": "KNN",
    "ridge": "Ridge",
    "hgb": "HGB",
    "cdae_l1": "CDAE (lambda=1)",
    "cdae_l0": "CDAE (lambda=0)",
}

RIDGE_ALPHA_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
HGB_PARAMS: Dict[str, object] = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "max_iter": 500,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "early_stopping": True,
    "n_iter_no_change": 20,
}


@dataclass
class BaselineProtocolConfig:
    """Shared protocol settings aligned with CDAE Phase 1/2."""

    data_path: Path = DEFAULT_DATA
    feature_config_path: Path = DEFAULT_FEATURE_CONFIG
    output_root: Path = DEFAULT_OUTPUT_ROOT
    cdae_splits_file: Path = Path("")
    mechanism: str = "MCAR"
    model: str = "ridge"
    missing_rate: float = 0.8
    k_neighbors: int = 5
    seed: int = 42
    bootstrap_n: int = 1000
    aux_col: str = ""
    robust_target: bool = False
    blocked_by_index: bool = True
    time_test_size: float = 0.05
    time_val_size: float = 0.1
    min_target_coverage: float = 0.2
    min_target_count: int = 5000
    auto_skip_until_coverage: bool = True
    max_skip_head_frac: float = 0.95
    time_skip_head_frac: float = 0.0
    strict_validation: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def shared_dir(self) -> Path:
        return self.output_root / "shared"

    def run_dir(self, model: str, mechanism: str) -> Path:
        return self.output_root / model / mechanism
