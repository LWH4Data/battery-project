"""Confirmed Day 2 baseline: Batch 1 grouped splits and one-feature OLS.

Importing this module does not split data, fit a model, or write artifacts.
``prepare_splits(root)`` exposes the split before learning; ``run(root)`` fits
only development/CV samples. Hold-out and Batch 2/3 are never predicted here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE = "delta_q_log10var"
TARGET = "cycle_life"
GROUP_KEY = "policy_readable"
SEED = 42
N_FOLDS = 5
EXPERIMENT_ID = "v01_baseline"
SOURCE_PATH = Path("outputs/day1/early_feature_candidates.csv")
SPLIT_PATH = Path("artifacts/splits/batch1_group_holdout_cv.json")
ASSIGNMENTS_PATH = Path("artifacts/splits/split_assignments.csv")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _settings() -> dict[str, Any]:
    return {
        "batch": "batch1",
        "expected_cell_count": 46,
        "expected_group_count": 23,
        "group_key": GROUP_KEY,
        "group_key_normalization": "none; original policy_readable string",
        "group_key_evidence": "All 23 Batch 1 original policy strings match the 23 numeric (C1, SOC_percent, C2) groups; no whitespace variants.",
        "holdout": {"method": "GroupShuffleSplit", "n_splits": 1,
                    "test_size": 0.2, "random_state": SEED},
        "cv": {"method": "GroupKFold", "n_splits": N_FOLDS,
               "shuffle": True, "random_state": SEED,
               "scope": "development partition only"},
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
    }


def _load_batch1(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    path = root / SOURCE_PATH
    # Only the confirmed columns enter this module; other batches are filtered
    # before any feature/label validation or summary is computed.
    columns = ["batch_id", "cell_id", GROUP_KEY, FEATURE, TARGET]
    contents = path.read_bytes()
    table = pd.read_csv(BytesIO(contents), usecols=columns, dtype={"batch_id": str, GROUP_KEY: str})
    data = table.loc[table.batch_id == "batch1", columns].copy()
    if len(data) != 46 or data.cell_id.isna().any() or data.cell_id.duplicated().any():
        raise ValueError("Batch 1 must contain all 46 unique original cells; no filtering is allowed.")
    ids = data.cell_id.to_numpy(dtype=float)
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Batch 1 cell_id must be finite integers.")
    data["cell_id"] = ids.astype(int)
    if set(data.cell_id) != set(range(46)):
        raise ValueError("Batch 1 cell IDs differ from the original 0..45 contract.")
    if data[GROUP_KEY].isna().any() or (data[GROUP_KEY].str.len() == 0).any():
        raise ValueError("A Batch 1 protocol group key is missing; no replacement is allowed.")
    if data[GROUP_KEY].nunique() != 23:
        raise ValueError("Expected the previously verified 23 original Batch 1 protocol groups.")
    for column in (FEATURE, TARGET):
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite Batch 1 {column}; stop without imputation or cell exclusion.")
        data[column] = values
    if (data[TARGET] <= 0).any():
        raise ValueError("Original cycle_life must be positive for the confirmed MAPE metric.")
    data = data.sort_values("cell_id").reset_index(drop=True)
    source = {"relative_path": SOURCE_PATH.as_posix(), "sha256": hashlib.sha256(contents).hexdigest()}
    return data, source


def _new_split(data: pd.DataFrame, source: dict[str, str]) -> dict[str, Any]:
    settings = _settings()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    dev_positions, holdout_positions = next(splitter.split(data[[FEATURE]], groups=data[GROUP_KEY]))
    dev = data.iloc[dev_positions]
    holdout = data.iloc[holdout_positions]
    folds = []
    cv = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_positions, valid_positions) in enumerate(
            cv.split(dev[[FEATURE]], groups=dev[GROUP_KEY]), start=1):
        train, valid = dev.iloc[train_positions], dev.iloc[valid_positions]
        folds.append({
            "fold": fold,
            "train_cell_ids": sorted(train.cell_id.astype(int).tolist()),
            "validation_cell_ids": sorted(valid.cell_id.astype(int).tolist()),
            "train_groups": sorted(train[GROUP_KEY].unique().tolist()),
            "validation_groups": sorted(valid[GROUP_KEY].unique().tolist()),
        })
    fold_by_id = {cell_id: item["fold"] for item in folds for cell_id in item["validation_cell_ids"]}
    dev_ids = set(dev.cell_id.astype(int))
    identity = json.dumps({"source": source, "settings": settings}, sort_keys=True).encode()
    return {
        "schema_version": 1,
        "split_id": hashlib.sha256(identity).hexdigest(),
        "created_at_utc": _utc_now(),
        "source": source,
        "settings": settings,
        "development_cell_ids": sorted(dev_ids),
        "holdout_cell_ids": sorted(holdout.cell_id.astype(int).tolist()),
        "development_groups": sorted(dev[GROUP_KEY].unique().tolist()),
        "holdout_groups": sorted(holdout[GROUP_KEY].unique().tolist()),
        "cells": [{"cell_id": int(row.cell_id), GROUP_KEY: getattr(row, GROUP_KEY),
                   "partition": "development" if row.cell_id in dev_ids else "holdout",
                   "cv_validation_fold": fold_by_id.get(row.cell_id)}
                  for row in data.itertuples(index=False)],
        "folds": folds,
        "holdout_status": "reserved; no prediction or evaluation",
        "note": "test_size=0.2 refers to protocol groups, not an exact 20% cell count; group count rounds upward.",
    }


def _validate_split(data: pd.DataFrame, manifest: dict[str, Any], source: dict[str, str]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("source") != source or manifest.get("settings") != _settings():
        raise ValueError("Stored split source hash/settings differ. The existing split will not be overwritten.")
    expected_identity = json.dumps({"source": source, "settings": _settings()}, sort_keys=True).encode()
    if manifest.get("split_id") != hashlib.sha256(expected_identity).hexdigest():
        raise ValueError("Stored split identity is inconsistent; refusing to overwrite it.")
    all_ids = set(data.cell_id.astype(int))
    dev_list, holdout_list = manifest["development_cell_ids"], manifest["holdout_cell_ids"]
    dev_ids, holdout_ids = set(dev_list), set(holdout_list)
    if len(dev_ids) != len(dev_list) or len(holdout_ids) != len(holdout_list):
        raise ValueError("Duplicate cells in the saved split.")
    assert dev_ids.isdisjoint(holdout_ids), "Development/hold-out cell leakage"
    assert dev_ids | holdout_ids == all_ids, "The split must preserve all 46 Batch 1 cells"
    by_id = data.set_index("cell_id")
    dev_groups = set(by_id.loc[sorted(dev_ids), GROUP_KEY])
    holdout_groups = set(by_id.loc[sorted(holdout_ids), GROUP_KEY])
    assert dev_groups.isdisjoint(holdout_groups), "Development/hold-out protocol leakage"
    assert len(holdout_groups) == math.ceil(0.2 * data[GROUP_KEY].nunique()), "Wrong hold-out group count"
    assert manifest["development_groups"] == sorted(dev_groups)
    assert manifest["holdout_groups"] == sorted(holdout_groups)
    folds = manifest["folds"]
    assert len(folds) == N_FOLDS and [f["fold"] for f in folds] == list(range(1, N_FOLDS + 1))
    validation_counts = {cell_id: 0 for cell_id in dev_ids}
    expected_cells = {}
    for item in folds:
        train_list, valid_list = item["train_cell_ids"], item["validation_cell_ids"]
        train_ids, valid_ids = set(train_list), set(valid_list)
        assert len(train_ids) == len(train_list) and len(valid_ids) == len(valid_list)
        assert train_ids and valid_ids, "Every fold must have training and validation cells"
        assert train_ids.isdisjoint(valid_ids), "CV cell leakage"
        assert train_ids | valid_ids == dev_ids, "CV must use development cells only"
        assert not (train_ids | valid_ids) & holdout_ids, "Hold-out entered CV"
        train_groups = set(by_id.loc[sorted(train_ids), GROUP_KEY])
        valid_groups = set(by_id.loc[sorted(valid_ids), GROUP_KEY])
        assert train_groups.isdisjoint(valid_groups), "CV protocol leakage"
        assert item["train_groups"] == sorted(train_groups)
        assert item["validation_groups"] == sorted(valid_groups)
        for cell_id in valid_ids:
            validation_counts[cell_id] += 1
            expected_cells[cell_id] = item["fold"]
    assert set(validation_counts.values()) == {1}, "Every development cell needs exactly one OOF prediction"
    saved_cells = manifest["cells"]
    assert len(saved_cells) == len(all_ids) and len({c["cell_id"] for c in saved_cells}) == len(all_ids)
    for cell in saved_cells:
        cell_id = cell["cell_id"]
        assert cell_id in all_ids and cell[GROUP_KEY] == by_id.loc[cell_id, GROUP_KEY]
        assert cell["partition"] == ("development" if cell_id in dev_ids else "holdout")
        assert cell["cv_validation_fold"] == expected_cells.get(cell_id)


def _assignments(manifest: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(manifest["cells"]).sort_values("cell_id").reset_index(drop=True)
    frame.insert(0, "batch_id", "batch1")
    frame["cv_validation_fold"] = frame["cv_validation_fold"].astype("Int64")
    return frame


def _exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures a concurrently created saved split is never overwritten.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _get_split(root: Path, data: pd.DataFrame, source: dict[str, str]) -> tuple[dict[str, Any], bool]:
    path, assignment_path = root / SPLIT_PATH, root / ASSIGNMENTS_PATH
    reused = path.exists()
    if reused:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        if assignment_path.exists():
            raise ValueError("An assignment CSV exists without its split manifest; refusing to overwrite it.")
        manifest = _new_split(data, source)
        _validate_split(data, manifest, source)
        try:
            _exclusive_text(path, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        except FileExistsError:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            reused = True
    _validate_split(data, manifest, source)
    expected = _assignments(manifest)
    if not assignment_path.exists():
        try:
            _exclusive_text(assignment_path, expected.to_csv(index=False))
        except FileExistsError:
            pass
    actual = pd.read_csv(assignment_path, dtype={"batch_id": str, GROUP_KEY: str})
    actual["cv_validation_fold"] = actual["cv_validation_fold"].astype("Int64")
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved assignment CSV differs from the validated manifest; it will not be overwritten.") from error
    return manifest, reused


def _split_tables(data: pd.DataFrame, manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_id = data.set_index("cell_id")
    counts = []
    for partition in ("development", "holdout"):
        ids = manifest[f"{partition}_cell_ids"]
        groups = by_id.loc[ids, GROUP_KEY].nunique()
        counts.append({"partition": partition, "n_cells": len(ids), "n_protocol_groups": groups,
                       "cell_fraction": len(ids) / len(data),
                       "group_fraction": groups / data[GROUP_KEY].nunique(),
                       "usage": "5-fold CV and development fit" if partition == "development"
                       else "reserved; no prediction/evaluation"})
    folds = pd.DataFrame([
        {"fold": f["fold"], "n_train_cells": len(f["train_cell_ids"]),
         "n_validation_cells": len(f["validation_cell_ids"]),
         "n_train_groups": len(f["train_groups"]),
         "n_validation_groups": len(f["validation_groups"])} for f in manifest["folds"]
    ])
    return pd.DataFrame(counts), folds


def prepare_splits(root: Path | str) -> dict[str, Any]:
    """Create/reuse and validate grouped splits, without fitting any estimator.

    Returns ``summary`` (dict), ``split_summary``, ``assignments`` and ``fold_summary``
    (DataFrames), and ``split_manifest`` (dict). Both split artifacts are saved.
    """
    root = Path(root).resolve()
    data, source = _load_batch1(root)
    manifest, reused = _get_split(root, data, source)
    split_summary, folds = _split_tables(data, manifest)
    return {
        "summary": {"status": "splits prepared; no estimator fitted",
                    "split_id": manifest["split_id"], "split_reused": reused,
                    "batch1_cell_count": len(data), "group_count": data[GROUP_KEY].nunique(),
                    "source_csv_sha256": source["sha256"],
                    "holdout_status": "reserved; no prediction/evaluation"},
        "split_summary": split_summary,
        "assignments": _assignments(manifest),
        "fold_summary": folds,
        "split_manifest": manifest,
    }


def _pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])


def _coefficient_record(model: Pipeline, stage: str, n_cells: int) -> dict[str, Any]:
    scaler, regressor = model.named_steps["scaler"], model.named_steps["regressor"]
    scaled_coefficient = float(regressor.coef_[0])
    original_coefficient = scaled_coefficient / float(scaler.scale_[0])
    original_intercept = float(regressor.intercept_) - original_coefficient * float(scaler.mean_[0])
    return {"stage": stage, "n_fit_cells": n_cells, "feature": FEATURE,
            "scaler_mean": float(scaler.mean_[0]), "scaler_scale": float(scaler.scale_[0]),
            "coefficient_standardized": scaled_coefficient,
            "intercept_standardized": float(regressor.intercept_),
            "coefficient_original_input_units": original_coefficient,
            "intercept_original_input_units": original_intercept,
            "equation": f"predicted_cycle_life = {original_intercept:.12g} + ({original_coefficient:.12g}) * {FEATURE}",
            "target_unit": "cycles", "prediction_clipping": False}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run(root: Path | str) -> dict[str, Any]:
    """Run grouped development CV and save a development-only OLS baseline.

    ``summary`` is a JSON-compatible dict. ``split_summary``, ``assignments``,
    ``fold_metrics``, ``coefficients`` and ``cv_predictions`` are displayable DataFrames.
    No hold-out, Batch 2 or Batch 3 prediction/evaluation is implemented.
    """
    root = Path(root).resolve()
    started_at = _utc_now()
    data, source = _load_batch1(root)
    manifest, reused = _get_split(root, data, source)
    module_path = Path(__file__).resolve()
    environment = _environment()
    output_dir = root / "artifacts/experiments" / EXPERIMENT_ID
    config_path = root / "configs/experiments" / f"{EXPERIMENT_ID}.json"
    holdout_ids = set(manifest["holdout_cell_ids"])
    dev_ids = set(manifest["development_cell_ids"])
    by_id = data.set_index("cell_id", drop=False)
    status = {"holdout": "not_predicted_or_evaluated", "batch2": "not_predicted_or_evaluated",
              "batch3": "not_predicted_or_evaluated", "model_selection": "not_performed",
              "hyperparameter_tuning": "not_performed"}
    config = {
        "experiment_id": EXPERIMENT_ID, "started_at_utc": started_at,
        "input_csv": source, "module_path": str(module_path), "module_sha256": _sha256(module_path),
        "environment": environment, "split_id": manifest["split_id"],
        "split_manifest": SPLIT_PATH.as_posix(), "split_settings": _settings(),
        "features": [FEATURE],
        "feature_definition": "log10 of population variance (ddof=0) of original Qdlin(cycle 100)-Qdlin(cycle 10) over the verified common 1000-point Vdlin grid; Ah^2 numerical values, without epsilon.",
        "target": {"column": TARGET, "definition": "Original final cycle_life", "unit": "cycles",
                   "transformation": "none", "correction": "none"},
        "observation_scope": "first 100 cycles; the only feature uses actual cycles 10 and 100",
        "model": {"class": "sklearn.linear_model.LinearRegression", "parameters": LinearRegression().get_params()},
        "preprocessing": {"class": "sklearn.preprocessing.StandardScaler",
                          "parameters": StandardScaler().get_params(),
                          "fit_scope": "each CV training fold; final baseline on development cells only",
                          "imputation": "none", "sampling": "none", "cell_exclusion": "none"},
        "target_transform": "none", "prediction_clipping": False,
        "primary_metric": {"name": "MAPE_percent", "aggregation": "arithmetic mean of the five validation-fold MAPEs",
                           "uncertainty_summary": "sample standard deviation across five folds (ddof=1), not a confidence interval"},
        "auxiliary_metrics": ["mean validation-fold MAE_cycles", "mean validation-fold RMSE_cycles", "pooled development OOF MAPE_percent"],
        "evaluation_status": status,
        "saved_model_scope": "all development cells only; baseline, not a finally selected model",
    }
    # All source/split validations precede estimator fitting. Configuration is
    # saved before learning so any interrupted run still states its design.
    _write_json(config_path, config)
    fold_metrics, predictions, coefficients = [], [], []
    oof_seen = {cell_id: 0 for cell_id in dev_ids}
    for item in manifest["folds"]:
        fold = item["fold"]
        train = by_id.loc[item["train_cell_ids"]]
        valid = by_id.loc[item["validation_cell_ids"]]
        assert set(train.cell_id).isdisjoint(valid.cell_id)
        assert set(train[GROUP_KEY]).isdisjoint(valid[GROUP_KEY])
        assert not (set(train.cell_id) | set(valid.cell_id)) & holdout_ids
        model = _pipeline()
        model.fit(train[[FEATURE]], train[TARGET])
        # Explicitly verify the preprocessing statistics come from this fold's
        # training cells, never its validation cells or the reserved hold-out.
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, train[[FEATURE]].mean().to_numpy())
        np.testing.assert_allclose(model.named_steps["scaler"].var_, train[[FEATURE]].var(ddof=0).to_numpy())
        predicted = model.predict(valid[[FEATURE]])
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite CV predictions; stop rather than alter or clip them.")
        observed = valid[TARGET].to_numpy()
        fold_metrics.append({"fold": fold, "n_train_cells": len(train), "n_validation_cells": len(valid),
                             "n_train_groups": train[GROUP_KEY].nunique(), "n_validation_groups": valid[GROUP_KEY].nunique(),
                             "mape_percent": float(100 * mean_absolute_percentage_error(observed, predicted)),
                             "mae_cycles": float(mean_absolute_error(observed, predicted)),
                             "rmse_cycles": float(root_mean_squared_error(observed, predicted))})
        coefficients.append(_coefficient_record(model, f"cv_fold_{fold}", len(train)))
        for cell_id, policy, actual, estimate in zip(valid.cell_id, valid[GROUP_KEY], observed, predicted):
            cell_id = int(cell_id)
            assert cell_id in dev_ids and cell_id not in holdout_ids
            oof_seen[cell_id] += 1
            predictions.append({"batch_id": "batch1", "cell_id": cell_id, GROUP_KEY: policy,
                                "cv_validation_fold": fold, "y_true_cycle_life": float(actual),
                                "y_pred_cycle_life": float(estimate),
                                "absolute_error_cycles": float(abs(estimate - actual)),
                                "absolute_percentage_error_percent": float(100 * abs(estimate - actual) / actual)})
    assert set(oof_seen.values()) == {1}, "Development OOF prediction count must be exactly one"
    metrics_frame = pd.DataFrame(fold_metrics)
    prediction_frame = pd.DataFrame(predictions).sort_values("cell_id").reset_index(drop=True)
    assert not prediction_frame.cell_id.duplicated().any()
    assert set(prediction_frame.cell_id) == dev_ids
    assert set(prediction_frame.cell_id).isdisjoint(holdout_ids), "Hold-out was predicted"
    development = by_id.loc[sorted(dev_ids)]
    development_model = _pipeline()
    development_model.fit(development[[FEATURE]], development[TARGET])
    np.testing.assert_allclose(development_model.named_steps["scaler"].mean_, development[[FEATURE]].mean().to_numpy())
    coefficients.append(_coefficient_record(development_model, "development_baseline", len(development)))
    coefficient_frame = pd.DataFrame(coefficients)
    summary = {
        "experiment_id": EXPERIMENT_ID, "status": "development CV baseline completed; hold-out/test evaluation pending",
        "split_id": manifest["split_id"], "split_reused": reused,
        "batch1_cell_count": len(data), "development_cell_count": len(dev_ids), "holdout_cell_count": len(holdout_ids),
        "development_group_count": development[GROUP_KEY].nunique(),
        "holdout_group_count": len(manifest["holdout_groups"]),
        "primary": {"mape_percent_mean": float(metrics_frame.mape_percent.mean()),
                    "mape_percent_sample_sd": float(metrics_frame.mape_percent.std(ddof=1)),
                    "fold_count": N_FOLDS, "aggregation": "unweighted arithmetic mean of fold validation MAPEs",
                    "sd_interpretation": "sample SD across folds, not a confidence interval"},
        "auxiliary": {"mae_cycles_fold_mean": float(metrics_frame.mae_cycles.mean()),
                      "mae_cycles_fold_sample_sd": float(metrics_frame.mae_cycles.std(ddof=1)),
                      "rmse_cycles_fold_mean": float(metrics_frame.rmse_cycles.mean()),
                      "rmse_cycles_fold_sample_sd": float(metrics_frame.rmse_cycles.std(ddof=1)),
                      "pooled_oof_mape_percent": float(100 * mean_absolute_percentage_error(
                          prediction_frame.y_true_cycle_life, prediction_frame.y_pred_cycle_life)),
                      "pooled_oof_interpretation": "cell-weighted pooled OOF result; auxiliary, not the reported fold-mean Train metric"},
        "evaluation_status": status,
        "saved_model_scope": "development cells only; not a finally selected model",
        "development_equation_original_input_units": coefficients[-1],
        "feature_columns": [FEATURE], "target_column": TARGET,
        "preprocessing_or_exclusion_applied": "StandardScaler only, fitted within each training partition; no imputation, target transformation, cell exclusion or prediction clipping",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "cv_metrics.csv", metrics_frame.to_csv(index=False))
    _atomic_text(output_dir / "cv_predictions.csv", prediction_frame.to_csv(index=False))
    _write_json(output_dir / "metrics.json", summary)
    handle, temporary_name = tempfile.mkstemp(dir=output_dir, prefix=".model.", suffix=".joblib")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        joblib.dump(development_model, temporary)
        os.replace(temporary, output_dir / "model.joblib")
    finally:
        temporary.unlink(missing_ok=True)
    paths = {"cv_metrics": output_dir / "cv_metrics.csv", "cv_predictions": output_dir / "cv_predictions.csv",
             "metrics": output_dir / "metrics.json", "model": output_dir / "model.joblib",
             "configuration": config_path, "split_manifest": root / SPLIT_PATH,
             "split_assignments": root / ASSIGNMENTS_PATH}
    run_manifest = {
        "experiment_id": EXPERIMENT_ID, "started_at_utc": started_at, "completed_at_utc": _utc_now(),
        "input_csv": source, "module_path": str(module_path), "module_sha256": config["module_sha256"],
        "environment": environment, "split_id": manifest["split_id"], "split_reused": reused,
        "group_key": GROUP_KEY, "cv_shuffle": True, "random_state": SEED,
        "evaluation_status": status, "saved_model_fit_cell_ids": sorted(dev_ids),
        "saved_model_is_finally_selected": False, "feature_columns": [FEATURE], "target_column": TARGET,
        "target_transform": "none", "prediction_clipping": False,
        "all_batch1_cells_preserved": len(dev_ids | holdout_ids) == 46,
        "oof_prediction_count_per_development_cell": 1,
        "holdout_cell_ids_reserved": sorted(holdout_ids),
        "artifacts": {name: {"relative_path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
                      for name, path in paths.items()},
    }
    _write_json(output_dir / "run-manifest.json", run_manifest)
    split_summary, _ = _split_tables(data, manifest)
    return {"summary": summary, "split_summary": split_summary, "assignments": _assignments(manifest),
            "fold_metrics": metrics_frame, "coefficients": coefficient_frame, "cv_predictions": prediction_frame,
            "config": config, "run_manifest": run_manifest,
            "artifact_paths": {**{name: str(path) for name, path in paths.items()},
                               "run_manifest": str(output_dir / "run-manifest.json")}}
