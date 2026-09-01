"""Configuration for auxiliary-only MLP ablation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_DATA = PROJECT_ROOT / "sulfatos_dados_concatenados_formatados_2021.h5"
DEFAULT_FEATURE_CONFIG = (
    SCRIPT_DIR / "outputs" / "cdae_optimal_20251226_171939" / "feature_selection_config.json"
)
DEFAULT_PROTOCOL_SHARED = (
    SCRIPT_DIR / "testes" / "baselines_leakage_safe" / "primary_target_v1" / "shared"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "testes" / "ablation_aux_mlp" / "primary_target_v1"
)

SUPPORTED_MECHANISMS: Tuple[str, ...] = ("MCAR", "MAR", "MNAR")


@dataclass
class AblationConfig:
    """Settings for the capacity-matched auxiliary-only MLP ablation."""

    data_path: Path = DEFAULT_DATA
    feature_config_path: Path = DEFAULT_FEATURE_CONFIG
    protocol_shared_source: Path = DEFAULT_PROTOCOL_SHARED
    output_root: Path = DEFAULT_OUTPUT_ROOT
    missing_rate: float = 0.8
    seed: int = 42
    bootstrap_n: int = 1000
    aux_col: str = ""
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    patience: int = 20
    dropout: float = 0.2
    latent_dim: int = 16
    device: str = "cpu"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def shared_dir(self) -> Path:
        return self.output_root / "shared"

    def model_dir(self) -> Path:
        return self.output_root / "model"

    def masks_dir(self) -> Path:
        return self.output_root / "masks"

    def eval_dir(self, mechanism: str) -> Path:
        return self.output_root / "eval" / mechanism

    def consolidated_dir(self) -> Path:
        return self.output_root / "consolidated"
