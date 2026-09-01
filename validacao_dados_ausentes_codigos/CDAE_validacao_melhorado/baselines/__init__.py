"""Leakage-safe baseline comparison package (KNN, Ridge, HistGradientBoosting)."""

from baselines.config import BaselineProtocolConfig
from baselines.protocol import prepare_protocol_artifacts
from baselines.run_single import run_single_baseline

__all__ = [
    "BaselineProtocolConfig",
    "prepare_protocol_artifacts",
    "run_single_baseline",
]
