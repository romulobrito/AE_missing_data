"""Data loading, splits, and train-only preprocessing for baseline runs."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from teste_controlado_cdae_v2 import (
    blocked_split_indices,
    blocked_split_indices_with_skip,
    compute_target_coverage,
    temporal_split_indices,
    temporal_split_indices_with_skip,
)
from teste_controlado_dados_faltantes_improved import load_feature_config, set_seed

from baselines.config import BaselineProtocolConfig


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load HDF5 or CSV industrial dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if data_path.suffix in {".h5", ".hdf5"}:
        return pd.read_hdf(str(data_path))
    return pd.read_csv(data_path)


def create_splits(
    df: pd.DataFrame,
    target_var: str,
    cfg: BaselineProtocolConfig,
) -> Tuple[List, List, List, Dict[str, object]]:
    """Create chronological splits using the same logic as CDAE."""
    skip_used = 0.0
    if cfg.blocked_by_index:
        if cfg.auto_skip_until_coverage or cfg.time_skip_head_frac > 0.0:
            train_idx, val_idx, test_idx, skip_used = blocked_split_indices_with_skip(
                df,
                cfg.time_test_size,
                cfg.time_val_size,
                target_var,
                cfg.min_target_coverage,
                cfg.min_target_count,
                skip_head_frac=cfg.time_skip_head_frac,
                auto_skip_until_coverage=bool(cfg.auto_skip_until_coverage),
                max_skip_head_frac=cfg.max_skip_head_frac,
            )
        else:
            train_idx, val_idx, test_idx = blocked_split_indices(
                df, cfg.time_test_size, cfg.time_val_size
            )
    else:
        from sklearn.model_selection import train_test_split

        all_indices = df.index.to_numpy()
        train_idx, temp_idx = train_test_split(
            all_indices, test_size=0.3, random_state=cfg.seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=cfg.seed
        )

    cov_train, cnt_train, tot_train = compute_target_coverage(
        df, np.array(train_idx), target_var
    )
    cov_val, cnt_val, tot_val = compute_target_coverage(
        df, np.array(val_idx), target_var
    )
    cov_test, cnt_test, tot_test = compute_target_coverage(
        df, np.array(test_idx), target_var
    )

    coverage_report: Dict[str, object] = {
        "train": {"coverage": cov_train, "present": cnt_train, "total": tot_train},
        "val": {"coverage": cov_val, "present": cnt_val, "total": tot_val},
        "test": {"coverage": cov_test, "present": cnt_test, "total": tot_test},
        "thresholds": {
            "min_target_coverage": float(cfg.min_target_coverage),
            "min_target_count": int(cfg.min_target_count),
        },
        "split_mode": "blocked_by_index" if cfg.blocked_by_index else "random",
        "skip_head_frac_used": float(skip_used),
    }
    return train_idx, val_idx, test_idx, coverage_report


def validate_splits_against_reference(
    train_idx: List,
    val_idx: List,
    test_idx: List,
    splits_file: Path,
    strict: bool,
) -> bool:
    """Optional check against persisted CDAE split indices."""
    splits_text = str(splits_file).strip()
    if not splits_text:
        return True
    splits_path = Path(splits_text)
    if not splits_path.is_file():
        return True
    ref = np.load(str(splits_path), allow_pickle=True)
    ok = (
        np.array_equal(np.sort(train_idx), np.sort(ref["train_idx"]))
        and np.array_equal(np.sort(val_idx), np.sort(ref["val_idx"]))
        and np.array_equal(np.sort(test_idx), np.sort(ref["test_idx"]))
    )
    if not ok and strict:
        raise ValueError("Split indices do not match CDAE reference file.")
    return ok


def prepare_protocol_artifacts(cfg: BaselineProtocolConfig) -> Dict[str, object]:
    """
    Build shared leakage-safe artifacts once:
    splits, train-only imputation means, scalers, and feature metadata.
    """
    set_seed(cfg.seed)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    shared_dir = cfg.shared_dir()
    shared_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.output_root / "protocol_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, ensure_ascii=True)

    df = load_dataset(cfg.data_path)
    target_var, features = load_feature_config(str(cfg.feature_config_path))
    train_idx, val_idx, test_idx, coverage_report = create_splits(df, target_var, cfg)

    if str(cfg.cdae_splits_file).strip():
        validate_splits_against_reference(
            train_idx, val_idx, test_idx, cfg.cdae_splits_file, cfg.strict_validation
        )

    df_with_target = df[df[target_var].notna()].copy()
    train_idx_model = [i for i in train_idx if i in df_with_target.index]
    val_idx_model = [i for i in val_idx if i in df_with_target.index]
    test_idx_model = [i for i in test_idx if i in df_with_target.index]

    if (
        len(train_idx_model) == 0
        or len(val_idx_model) == 0
        or len(test_idx_model) == 0
    ):
        from sklearn.model_selection import train_test_split

        available_idx = list(df_with_target.index)
        train_idx_model, temp_idx_model = train_test_split(
            available_idx, test_size=0.3, random_state=cfg.seed
        )
        val_idx_model, test_idx_model = train_test_split(
            temp_idx_model, test_size=0.5, random_state=cfg.seed
        )

    x_all = df_with_target[features].apply(pd.to_numeric, errors="coerce")
    y_all = df_with_target[[target_var]]
    train_feature_means = x_all.loc[train_idx_model].mean()
    if train_feature_means.isna().any():
        bad = train_feature_means[train_feature_means.isna()].index.tolist()
        raise ValueError(f"No training data to compute mean for: {bad}")

    def impute_train_means(part: pd.DataFrame) -> pd.DataFrame:
        return part.fillna(train_feature_means)

    x_train_raw = impute_train_means(x_all.loc[train_idx_model]).values.astype(np.float32)
    x_val_raw = impute_train_means(x_all.loc[val_idx_model]).values.astype(np.float32)
    x_test_raw = impute_train_means(x_all.loc[test_idx_model]).values.astype(np.float32)
    y_train_raw = y_all.loc[train_idx_model].values.astype(np.float32).reshape(-1)
    y_val_raw = y_all.loc[val_idx_model].values.astype(np.float32).reshape(-1)
    y_test_raw = y_all.loc[test_idx_model].values.astype(np.float32).reshape(-1)

    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train_raw).astype(np.float32)
    x_val_scaled = scaler_x.transform(x_val_raw).astype(np.float32)
    x_test_scaled = scaler_x.transform(x_test_raw).astype(np.float32)

    scaler_y = RobustScaler() if cfg.robust_target else StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).astype(np.float32)

    np.savez(
        shared_dir / "splits_indices.npz",
        train_pos=np.array(
            [df_with_target.index.get_loc(i) for i in train_idx_model], dtype=np.int64
        ),
        val_pos=np.array(
            [df_with_target.index.get_loc(i) for i in val_idx_model], dtype=np.int64
        ),
        test_pos=np.array(
            [df_with_target.index.get_loc(i) for i in test_idx_model], dtype=np.int64
        ),
        train_idx=np.array(train_idx_model),
        val_idx=np.array(val_idx_model),
        test_idx=np.array(test_idx_model),
    )
    with open(shared_dir / "coverage_report.json", "w", encoding="utf-8") as fh:
        json.dump(coverage_report, fh, indent=2, ensure_ascii=True)
    train_feature_means.to_json(shared_dir / "train_feature_means.json")
    joblib.dump(scaler_x, shared_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, shared_dir / "scaler_y.joblib")
    shutil.copy2(cfg.feature_config_path, shared_dir / "feature_selection_config.json")

    metadata = {
        "target_variable": target_var,
        "features": features,
        "dataset_shape": list(df.shape),
        "train_samples": len(train_idx_model),
        "val_samples": len(val_idx_model),
        "test_samples": len(test_idx_model),
    }
    with open(shared_dir / "dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=True)

    return {
        "df_with_target": df_with_target,
        "target_var": target_var,
        "features": features,
        "train_idx_model": train_idx_model,
        "val_idx_model": val_idx_model,
        "test_idx_model": test_idx_model,
        "train_feature_means": train_feature_means,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "x_train_scaled": x_train_scaled,
        "x_val_scaled": x_val_scaled,
        "x_test_scaled": x_test_scaled,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "y_train_scaled": y_train_scaled.reshape(-1),
        "y_val_scaled": y_val_scaled.reshape(-1),
        "y_test_scaled": y_test_scaled.reshape(-1),
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "coverage_report": coverage_report,
        "metadata": metadata,
    }


def load_protocol_artifacts(cfg: BaselineProtocolConfig) -> Dict[str, object]:
    """Reload shared artifacts if they already exist."""
    shared_dir = cfg.shared_dir()
    required = [
        shared_dir / "splits_indices.npz",
        shared_dir / "train_feature_means.json",
        shared_dir / "scaler_X.joblib",
        shared_dir / "scaler_y.joblib",
        shared_dir / "dataset_metadata.json",
    ]
    if not all(path.exists() for path in required):
        return prepare_protocol_artifacts(cfg)

    df = load_dataset(cfg.data_path)
    with open(shared_dir / "dataset_metadata.json", "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    target_var = str(metadata["target_variable"])
    features = [str(item) for item in metadata["features"]]
    df_with_target = df[df[target_var].notna()].copy()

    splits = np.load(shared_dir / "splits_indices.npz", allow_pickle=True)
    if "train_pos" in splits.files:
        train_idx_model = df_with_target.index[splits["train_pos"]].tolist()
        val_idx_model = df_with_target.index[splits["val_pos"]].tolist()
        test_idx_model = df_with_target.index[splits["test_pos"]].tolist()
    else:
        train_idx_model = splits["train_idx"].tolist()
        val_idx_model = splits["val_idx"].tolist()
        test_idx_model = splits["test_idx"].tolist()

    train_feature_means = pd.read_json(shared_dir / "train_feature_means.json", typ="series")
    scaler_x = joblib.load(shared_dir / "scaler_X.joblib")
    scaler_y = joblib.load(shared_dir / "scaler_y.joblib")

    x_all = df_with_target[features].apply(pd.to_numeric, errors="coerce")
    y_all = df_with_target[[target_var]]

    def impute_train_means(part: pd.DataFrame) -> pd.DataFrame:
        return part.fillna(train_feature_means)

    x_train_raw = impute_train_means(x_all.loc[train_idx_model]).values.astype(np.float32)
    x_val_raw = impute_train_means(x_all.loc[val_idx_model]).values.astype(np.float32)
    x_test_raw = impute_train_means(x_all.loc[test_idx_model]).values.astype(np.float32)
    y_train_raw = y_all.loc[train_idx_model].values.astype(np.float32).reshape(-1)
    y_val_raw = y_all.loc[val_idx_model].values.astype(np.float32).reshape(-1)
    y_test_raw = y_all.loc[test_idx_model].values.astype(np.float32).reshape(-1)

    x_train_scaled = scaler_x.transform(x_train_raw).astype(np.float32)
    x_val_scaled = scaler_x.transform(x_val_raw).astype(np.float32)
    x_test_scaled = scaler_x.transform(x_test_raw).astype(np.float32)
    y_train_scaled = scaler_y.transform(y_train_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    with open(shared_dir / "coverage_report.json", "r", encoding="utf-8") as fh:
        coverage_report = json.load(fh)

    return {
        "df_with_target": df_with_target,
        "target_var": target_var,
        "features": features,
        "train_idx_model": train_idx_model,
        "val_idx_model": val_idx_model,
        "test_idx_model": test_idx_model,
        "train_feature_means": train_feature_means,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "x_train_scaled": x_train_scaled,
        "x_val_scaled": x_val_scaled,
        "x_test_scaled": x_test_scaled,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "coverage_report": coverage_report,
        "metadata": metadata,
    }
