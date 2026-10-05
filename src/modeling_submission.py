"""Approved Day 2 selection, hold-out validation and fixed Batch 2 test.

Import is side-effect free. ``run(root)`` executes the approved experiment once.
An existing experiment is never overwritten. ``run(root, reproduce=True)``
replays the locked winner without searching or selecting again, and writes into
a separate reproduction directory. A replay is not a new independent test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import modeling_baseline as baseline


FEATURE = baseline.FEATURE
TARGET = baseline.TARGET
GROUP_KEY = baseline.GROUP_KEY
EXPERIMENT_ID = "v02_submission"
SOURCE_SHA256 = "dc4e7979037b974d68e88eebc66232b2221095154e747b8f2b15d32a17ddfe21"
SPLIT_ID = "5ac6e1b0a3010116290b6d6a5675dc54a9ecc17867404b891c2d9989369b9fa1"
EXPECTED_UNKNOWN_IDS = [22, 23, 35, 36, 37, 38, 39, 40]
ALPHAS = [float(value) for value in np.logspace(-4, 4, 17)]
L1_RATIOS = [0.1, 0.5, 0.9]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def confirmed_config() -> dict[str, Any]:
    """Return only the design approved by the user; no silent overrides."""
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "decision_status": "user_approved_final_submission_design",
        "source_csv": baseline.SOURCE_PATH.as_posix(),
        "source_csv_sha256": SOURCE_SHA256,
        "split_manifest": baseline.SPLIT_PATH.as_posix(),
        "split_id": SPLIT_ID,
        "random_state": 42,
        "features": [FEATURE],
        "target": TARGET,
        "target_transform": "none",
        "observation_scope": "first 100 cycles; actual cycles 10 and 100 for this feature",
        "feature_definition": "log10(var(Qdlin_100-Qdlin_10, ddof=0)) on the verified original common 1000-point Vdlin grid; numeric Ah^2 variance; no epsilon",
        "preprocessing": "StandardScaler fitted on each training partition only",
        "imputation": "none",
        "cell_exclusion": "none",
        "prediction_clipping": False,
        "candidate_models": ["LinearRegression", "Ridge", "ElasticNet"],
        "alpha_values": ALPHAS,
        "elastic_net_l1_ratio_values": L1_RATIOS,
        "candidate_count": 69,
        "grid_search": {
            "class": "sklearn.model_selection.GridSearchCV",
            "n_jobs": 1,
            "error_score": "raise",
            "refit": False,
            "selection_metric": "lowest unweighted arithmetic mean of five validation-fold MAPE values",
            "exact_tie_break": "fixed order: LinearRegression, Ridge then ElasticNet; within family ascending alpha then ascending l1_ratio; exact floats only, no tolerance",
            "cv": "explicit saved five-fold group indices on the 35 development cells",
            "optuna_used": False,
        },
        "fixed_solver_settings": {
            "Ridge": {"solver": "svd", "fit_intercept": True},
            "ElasticNet": {"selection": "cyclic", "max_iter": 100000, "tol": 1e-8,
                           "random_state": 42, "fit_intercept": True},
            "convergence_warnings": "raise and stop; no performance-dependent solver adjustments",
        },
        "fit_and_evaluation_order": [
            "69-setting group CV using development 35 cells only",
            "lock the winner and training design on disk before hold-out/test prediction",
            "fit winner on development 35 cells and evaluate hold-out 11 cells once",
            "refit the same winner parameters on all Batch 1 46 cells",
            "predict all Batch 2 47 cells once and evaluate the 39 known positive targets",
        ],
        "primary_metric": "MAPE_percent",
        "auxiliary_metrics": ["MAE_cycles", "RMSE_cycles", "pooled_OOF_MAPE_percent"],
        "paper_target_mape_percent": 9.1,
        "gaps_percent_points": {
            "Train-Valid": "Valid MAPE minus CV mean MAPE",
            "Valid-Test": "Batch 2 MAPE minus Valid MAPE",
            "Target-Test": "Batch 2 MAPE minus 9.1",
        },
        "batch2_missing_target_policy": "preserve all 47 rows and predictions; compute errors and metrics only for 39 known targets; preserve eight unknown targets without imputation",
        "batch3": "not evaluated; optional assignment omitted",
        "reproduction_policy": "fixed-winner replay only; no search or selection; writes a separate reproduction and is not an independent final test",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    baseline._atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _git_record(root: Path) -> dict[str, Any]:
    def read(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=root, text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"head_revision": read("rev-parse", "HEAD"),
            "working_tree_status": read("status", "--porcelain"),
            "note": "Run-time module/config hashes identify uncommitted changes independently of the prior HEAD."}


def _saved_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    data, source = baseline._load_batch1(root)
    if source["sha256"] != SOURCE_SHA256:
        raise ValueError("Confirmed source CSV hash changed; no new split or silent source change is allowed.")
    split_path = root / baseline.SPLIT_PATH
    assignments_path = root / baseline.ASSIGNMENTS_PATH
    if not split_path.exists() or not assignments_path.exists():
        raise FileNotFoundError("Both saved v01 split artifacts are required; v02 never creates a split.")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    baseline._validate_split(data, manifest, source)
    if manifest["split_id"] != SPLIT_ID:
        raise ValueError("This experiment must reuse the explicitly approved split ID.")
    actual = pd.read_csv(assignments_path, dtype={"batch_id": str, GROUP_KEY: str})
    actual["cv_validation_fold"] = actual["cv_validation_fold"].astype("Int64")
    pd.testing.assert_frame_equal(actual, baseline._assignments(manifest), check_dtype=False)
    if len(manifest["development_cell_ids"]) != 35 or len(manifest["holdout_cell_ids"]) != 11:
        raise ValueError("The approved development/hold-out sizes must remain 35/11.")
    return data, manifest, source


def _batch2_after_selection(root: Path) -> pd.DataFrame:
    contents = (root / baseline.SOURCE_PATH).read_bytes()
    if hashlib.sha256(contents).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source changed while the run was executing.")
    columns = ["batch_id", "cell_id", GROUP_KEY, FEATURE, TARGET]
    table = pd.read_csv(BytesIO(contents), usecols=columns,
                        dtype={"batch_id": str, GROUP_KEY: str})
    data = table.loc[table.batch_id == "batch2", columns].copy()
    ids = data.cell_id.to_numpy(dtype=float)
    if (len(data) != 47 or not np.isfinite(ids).all()
            or not np.equal(ids, np.floor(ids)).all()):
        raise ValueError("Batch 2 must retain all 47 unique original integer cell IDs.")
    data["cell_id"] = ids.astype(int)
    if data.cell_id.duplicated().any() or set(data.cell_id) != set(range(47)):
        raise ValueError("Batch 2 original cell IDs must be exactly 0..46.")
    if data[GROUP_KEY].isna().any() or (data[GROUP_KEY].str.len() == 0).any():
        raise ValueError("Batch 2 contains a missing protocol key.")
    feature = data[FEATURE].to_numpy(dtype=float)
    targets = data[TARGET].to_numpy(dtype=float)
    if not np.isfinite(feature).all():
        raise ValueError("Batch 2 feature is nonfinite; no imputation or exclusion is allowed.")
    missing = np.isnan(targets)
    if not np.isfinite(targets[~missing]).all() or (targets[~missing] <= 0).any():
        raise ValueError("A nonmissing Batch 2 target is nonfinite/nonpositive; stop without exclusion.")
    if sorted(data.loc[missing, "cell_id"].tolist()) != EXPECTED_UNKNOWN_IDS:
        raise ValueError("The eight recorded unknown Batch 2 targets have changed.")
    return data.sort_values("cell_id").reset_index(drop=True)


def _estimator(spec: dict[str, Any]) -> Any:
    family = spec["model_class"]
    if family == "LinearRegression":
        return LinearRegression()
    if family == "Ridge":
        return Ridge(alpha=spec["alpha"], solver="svd", fit_intercept=True)
    if family == "ElasticNet":
        return ElasticNet(alpha=spec["alpha"], l1_ratio=spec["l1_ratio"],
                         selection="cyclic", max_iter=100000, tol=1e-8,
                         random_state=42, fit_intercept=True)
    raise ValueError(f"Unapproved estimator {family!r}.")


def _candidate_specs() -> list[dict[str, Any]]:
    return [{"model_class": "LinearRegression", "alpha": None, "l1_ratio": None}] + [
        {"model_class": "Ridge", "alpha": alpha, "l1_ratio": None} for alpha in ALPHAS
    ] + [
        {"model_class": "ElasticNet", "alpha": alpha, "l1_ratio": ratio}
        for alpha in ALPHAS for ratio in L1_RATIOS
    ]


def _pipeline(spec: dict[str, Any]) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("regressor", _estimator(spec))])


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite predictions; stop without clipping.")
    if not len(observed) or not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Evaluation requires finite positive known targets.")
    return {"mape_percent": float(100 * mean_absolute_percentage_error(observed, predicted)),
            "mae_cycles": float(mean_absolute_error(observed, predicted)),
            "rmse_cycles": float(root_mean_squared_error(observed, predicted))}


def _scoring(estimator: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    scores = _metrics(target.to_numpy(dtype=float), estimator.predict(features))
    return {"mape": -scores["mape_percent"] / 100,
            "mae": -scores["mae_cycles"], "rmse": -scores["rmse_cycles"]}


def _cv_indices(development: pd.DataFrame, split: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = {int(cell_id): index for index, cell_id in enumerate(development.cell_id)}
    return [(np.array([positions[cell] for cell in fold["train_cell_ids"]], dtype=int),
             np.array([positions[cell] for cell in fold["validation_cell_ids"]], dtype=int))
            for fold in split["folds"]]


def _assert_scaler(model: Pipeline, training: pd.DataFrame) -> None:
    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, training[[FEATURE]].mean().to_numpy())
    np.testing.assert_allclose(scaler.var_, training[[FEATURE]].var(ddof=0).to_numpy())


def _solver_record(model: Pipeline, stage: str) -> dict[str, Any]:
    regressor = model.named_steps["regressor"]
    return {"stage": stage, "model_class": type(regressor).__name__,
            "convergence_warning_count": 0,
            "n_iter": int(regressor.n_iter_) if isinstance(regressor, ElasticNet) else None,
            "dual_gap": float(regressor.dual_gap_) if isinstance(regressor, ElasticNet) else None}


def _predictions(data: pd.DataFrame, predicted: np.ndarray, stage: str,
                 fold: int | None = None) -> pd.DataFrame:
    if len(predicted) != len(data) or not np.isfinite(predicted).all():
        raise ValueError("Prediction count/finite check failed.")
    result = data[["batch_id", "cell_id", GROUP_KEY, FEATURE]].copy()
    result["evaluation_stage"] = stage
    result["cv_validation_fold"] = fold
    result["y_true_cycle_life"] = data[TARGET].to_numpy(dtype=float)
    result["y_pred_cycle_life"] = predicted
    result["target_available"] = np.isfinite(result.y_true_cycle_life)
    result["evaluated"] = result.target_available
    # Unknown targets remain NaN in all error columns and stay in the CSV.
    result["signed_error_cycles"] = result.y_pred_cycle_life - result.y_true_cycle_life
    result["absolute_error_cycles"] = result.signed_error_cycles.abs()
    result["absolute_percentage_error_percent"] = 100 * result.absolute_error_cycles / result.y_true_cycle_life
    return result


def _cv_winner(development: pd.DataFrame, split: dict[str, Any], spec: dict[str, Any]
               ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    by_id = development.set_index("cell_id", drop=False)
    rows, predictions, solver = [], [], []
    for fold in split["folds"]:
        train = by_id.loc[fold["train_cell_ids"]]
        valid = by_id.loc[fold["validation_cell_ids"]]
        model = _pipeline(spec).fit(train[[FEATURE]], train[TARGET])
        _assert_scaler(model, train)
        predicted = model.predict(valid[[FEATURE]])
        rows.append({"fold": fold["fold"], "n_train_cells": len(train),
                     "n_validation_cells": len(valid), "n_train_groups": train[GROUP_KEY].nunique(),
                     "n_validation_groups": valid[GROUP_KEY].nunique(),
                     **_metrics(valid[TARGET].to_numpy(), predicted)})
        predictions.append(_predictions(valid, predicted, "development_oof", fold["fold"]))
        solver.append(_solver_record(model, f"winner_cv_fold_{fold['fold']}"))
    oof = pd.concat(predictions, ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    if len(oof) != 35 or oof.cell_id.duplicated().any() or set(oof.cell_id) != set(development.cell_id):
        raise ValueError("Each of the 35 development cells must have exactly one OOF prediction.")
    return pd.DataFrame(rows), oof, solver


def _candidate_search(development: pd.DataFrame, split: dict[str, Any]
                      ) -> tuple[pd.DataFrame, dict[str, Any], int]:
    specs = _candidate_specs()
    search = GridSearchCV(
        _pipeline(specs[0]), [{"regressor": [_estimator(spec)]} for spec in specs],
        scoring=_scoring, cv=_cv_indices(development, split),
        n_jobs=1, refit=False, error_score="raise", return_train_score=False,
    )
    search.fit(development[[FEATURE]], development[TARGET])
    results = search.cv_results_
    means = -np.asarray(results["mean_test_mape"], dtype=float) * 100
    if len(means) != 69 or not np.isfinite(means).all():
        raise ValueError("All 69 candidate CV means must be finite.")
    # np.argmin returns the first exact minimum in the approved deterministic
    # candidate order; no tolerance is used to alter the winning score.
    winner_index = int(np.argmin(means))
    rows = []
    for index, spec in enumerate(specs):
        fold_scores = [-float(results[f"split{fold}_test_mape"][index]) * 100
                       for fold in range(5)]
        rows.append({"candidate_id": index + 1, **spec,
                     "mean_mape_percent": float(means[index]),
                     "mape_percent_sample_sd": float(np.std(fold_scores, ddof=1)),
                     "mean_mae_cycles": -float(results["mean_test_mae"][index]),
                     "mean_rmse_cycles": -float(results["mean_test_rmse"][index]),
                     "selected": index == winner_index,
                     **{f"fold_{fold+1}_mape_percent": value for fold, value in enumerate(fold_scores)}})
    return pd.DataFrame(rows), specs[winner_index], winner_index + 1


def load_results(root: Path | str) -> dict[str, Any]:
    """Read saved submission artifacts without fitting/predicting/searching."""
    directory = Path(root).resolve() / "artifacts/experiments" / EXPERIMENT_ID
    summary = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {"summary": summary,
                            "run_manifest": json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))}
    for name in ("candidate_scores", "cv_metrics", "cv_predictions", "holdout_predictions", "batch2_predictions", "performance"):
        result[name] = pd.read_csv(directory / f"{name}.csv")
    result["artifact_directory"] = str(directory)
    return result


def verify_only(root: Path | str, *, require_models: bool = True) -> dict[str, Any]:
    """Validate source/split/lock/artifact hashes without fit or predict calls.

    Public GitHub exports exclude joblib binaries. ``require_models=False``
    permits only these two absent model files and explicitly reports that their
    hashes were not verified; every present file is still hash checked.
    """
    root = Path(root).resolve()
    data, split, source = _saved_inputs(root)
    directory = root / "artifacts/experiments" / EXPERIMENT_ID
    manifest = json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "selection-lock.json").read_text(encoding="utf-8"))
    missing_models: list[str] = []
    verified_count = 0
    for name, record in manifest["artifacts"].items():
        path = root / record["relative_path"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Unexpected artifact path for {name!r}.")
        if not path.exists() and not require_models and name in ("development_model.joblib", "final_model.joblib"):
            missing_models.append(name)
            continue
        if baseline._sha256(path) != record["sha256"]:
            raise ValueError(f"Saved artifact hash mismatch: {name}.")
        verified_count += 1
    checked = {
        "module_sha256": Path(__file__).resolve(),
        "baseline_module_sha256": Path(baseline.__file__).resolve(),
        "split_manifest_sha256": root / baseline.SPLIT_PATH,
        "split_assignments_sha256": root / baseline.ASSIGNMENTS_PATH,
        "config_sha256": root / "configs/experiments" / f"{EXPERIMENT_ID}.json",
        "uv_lock_sha256": root / "uv.lock",
    }
    for key, path in checked.items():
        if baseline._sha256(path) != manifest["provenance"][key]:
            raise ValueError(f"Run provenance hash mismatch: {key}.")
    if manifest["provenance"]["source_csv"] != source or manifest["split_id"] != split["split_id"]:
        raise ValueError("Source/split differs from run provenance.")
    config = json.loads(checked["config_sha256"].read_text(encoding="utf-8"))
    if config != confirmed_config() or lock["winner"] != manifest["winner"]:
        raise ValueError("Confirmed configuration or locked winner differs from final manifest.")
    winner_id = lock["winner_candidate_id"]
    specs = _candidate_specs()
    if not isinstance(winner_id, int) or not 1 <= winner_id <= len(specs) or lock["winner"] != specs[winner_id - 1]:
        raise ValueError("Locked winner must be the recorded approved candidate.")
    candidates = pd.read_csv(directory / "candidate_scores.csv")
    selected_ids = candidates.loc[candidates.selected, "candidate_id"].astype(int).tolist()
    if len(candidates) != 69 or selected_ids != [winner_id]:
        raise ValueError("Candidate selection CSV does not match the locked winner.")
    if lock["holdout_predictions_performed"] or lock["batch2_predictions_performed"]:
        raise ValueError("Selection lock was not saved before evaluation.")
    if lock["provenance"] != manifest["provenance"]:
        raise ValueError("Lock/final provenance differs.")
    sequence = [manifest[key] for key in ("started_at_utc", "selection_locked_at_utc",
                "holdout_prediction_started_at_utc", "final_fit_completed_at_utc",
                "batch2_prediction_started_at_utc", "completed_at_utc")]
    times = [datetime.fromisoformat(stamp) for stamp in sequence]
    if times != sorted(times):
        raise ValueError("Recorded selection/validation/refit/test order is inconsistent.")
    return {"status": "pass", "experiment_id": EXPERIMENT_ID, "read_only": True,
            "fit_calls": 0, "predict_calls": 0, "source_csv_sha256": source["sha256"],
            "split_id": split["split_id"], "n_batch1_cells": len(data),
            "artifact_count_verified": verified_count,
            "missing_model_files_not_verified": missing_models,
            "selection_preceded_holdout_and_test": True}


def run(root: Path | str, *, reproduce: bool = False) -> dict[str, Any]:
    """Run once, or explicitly replay the already locked winner without search.

    No v01 source, split, config or artifact is written. Existing v02 results
    cannot be overwritten. Replays are clearly labelled and saved separately.
    """
    root = Path(root).resolve()
    experiment_dir = root / "artifacts/experiments" / EXPERIMENT_ID
    started = _utc_now()
    if experiment_dir.exists() and not reproduce:
        raise FileExistsError("v02 already exists. Use load_results(root); --reproduce is fixed-winner replay only, never retuning.")
    if reproduce and not (experiment_dir / "run-manifest.json").exists():
        raise FileNotFoundError("A completed original v02 run is required for fixed-winner reproduction.")
    data, split, source = _saved_inputs(root)
    config = confirmed_config()
    config_path = root / "configs/experiments" / f"{EXPERIMENT_ID}.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Saved v02 configuration differs from the approved fixed design.")
    elif reproduce:
        raise FileNotFoundError("The original v02 config is required to reproduce.")
    else:
        _write_json(config_path, config)
    module_path = Path(__file__).resolve()
    provenance = {"source_csv": source,
                  "module_sha256": baseline._sha256(module_path),
                  "baseline_module_sha256": baseline._sha256(Path(baseline.__file__).resolve()),
                  "split_manifest_sha256": baseline._sha256(root / baseline.SPLIT_PATH),
                  "split_assignments_sha256": baseline._sha256(root / baseline.ASSIGNMENTS_PATH),
                  "config_sha256": baseline._sha256(config_path),
                  "uv_lock_sha256": baseline._sha256(root / "uv.lock"),
                  "environment": baseline._environment(), "git": _git_record(root)}
    if reproduce:
        # Verify public audit artifacts before accepting their frozen winner.
        # The original exported joblib binaries may be absent in a fresh clone;
        # they are not loaded or needed to reproduce a fit from the locked spec.
        reproduction_source_verification = verify_only(root, require_models=False)
        original_lock = json.loads((experiment_dir / "selection-lock.json").read_text(encoding="utf-8"))
        for key in ("module_sha256", "baseline_module_sha256", "split_manifest_sha256",
                    "split_assignments_sha256", "config_sha256", "uv_lock_sha256"):
            if original_lock["provenance"][key] != provenance[key]:
                raise ValueError(f"Cannot replay changed {key}; no automatic re-selection is permitted.")
        if original_lock["provenance"]["environment"]["packages"] != provenance["environment"]["packages"]:
            raise ValueError("Package versions differ from the locked original environment.")
        if original_lock["provenance"]["environment"]["python"] != provenance["environment"]["python"]:
            raise ValueError("Python version differs from the locked original environment.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = experiment_dir / "reproductions" / stamp
        winner_spec = original_lock["winner"]
        winner_id = original_lock["winner_candidate_id"]
        candidates = pd.read_csv(experiment_dir / "candidate_scores.csv")
        search_status = "not rerun; frozen original candidate results copied for provenance"
    else:
        output_dir = experiment_dir
        search_status = "69 candidates x 5 saved development folds; GridSearchCV completed"
    output_dir.mkdir(parents=True, exist_ok=False)
    dev = data.set_index("cell_id", drop=False).loc[split["development_cell_ids"]]
    holdout = data.set_index("cell_id", drop=False).loc[split["holdout_cell_ids"]]
    _write_json(output_dir / "run-start.json", {"started_at_utc": started, "configuration": config,
                "provenance": provenance, "mode": "fixed_winner_reproduction" if reproduce else "original_final_evaluation",
                "holdout_status": "not predicted", "batch2_status": "not loaded for evaluation or predicted"})
    # A warning indicates solver failure; never change tolerances after seeing
    # scores and never continue into Hold-out/Test with questionable fits.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if not reproduce:
            candidates, winner_spec, winner_id = _candidate_search(dev, split)
        baseline._atomic_text(output_dir / "candidate_scores.csv", candidates.to_csv(index=False))
        cv_metrics, cv_predictions, solver_records = _cv_winner(dev, split, winner_spec)
        selected_mean = float(cv_metrics.mape_percent.mean())
        original_mean = float(candidates.loc[candidates.candidate_id == winner_id, "mean_mape_percent"].iloc[0])
        np.testing.assert_allclose(selected_mean, original_mean, rtol=1e-12, atol=1e-12)
        development_model = _pipeline(winner_spec).fit(dev[[FEATURE]], dev[TARGET])
        _assert_scaler(development_model, dev)
        solver_records.append(_solver_record(development_model, "development_fit_35"))
        joblib.dump(development_model, output_dir / "development_model.joblib")
        baseline._atomic_text(output_dir / "cv_metrics.csv", cv_metrics.to_csv(index=False))
        baseline._atomic_text(output_dir / "cv_predictions.csv", cv_predictions.to_csv(index=False))
        selection_locked_at = _utc_now()
        lock = {"schema_version": 1, "experiment_id": EXPERIMENT_ID,
                "selection_locked_at_utc": selection_locked_at,
                "mode": "fixed_winner_reproduction" if reproduce else "original_selection",
                "winner": winner_spec, "winner_candidate_id": winner_id,
                "winner_estimator_parameters": _estimator(winner_spec).get_params(),
                "selection_cv_mape_percent": selected_mean, "search_status": search_status,
                "split_id": split["split_id"], "provenance": provenance,
                "holdout_predictions_performed": False, "batch2_predictions_performed": False,
                "holdout_in_selection": False, "batch2_in_selection": False,
                "final_fit_cell_ids": data.cell_id.astype(int).tolist(),
                "development_fit_cell_ids": dev.cell_id.astype(int).tolist(),
                "note": "Winner/config frozen before any Hold-out or Batch 2 predictions; later test performance never changes this record."}
        if reproduce:
            lock["original_selection_lock_sha256"] = baseline._sha256(experiment_dir / "selection-lock.json")
            lock["original_artifact_verification"] = reproduction_source_verification
        _write_json(output_dir / "selection-lock.json", lock)
        holdout_started_at = _utc_now()
        holdout_pred = development_model.predict(holdout[[FEATURE]])
        holdout_scores = _metrics(holdout[TARGET].to_numpy(), holdout_pred)
        holdout_predictions = _predictions(holdout, holdout_pred, "holdout")
        baseline._atomic_text(output_dir / "holdout_predictions.csv", holdout_predictions.to_csv(index=False))
        final_model = _pipeline(winner_spec).fit(data[[FEATURE]], data[TARGET])
        _assert_scaler(final_model, data)
        solver_records.append(_solver_record(final_model, "final_batch1_fit_46"))
        joblib.dump(final_model, output_dir / "final_model.joblib")
        final_fit_completed_at = _utc_now()
        # Batch 2 values/targets are not used for candidate search or winner fit.
        batch2 = _batch2_after_selection(root)
        batch2_started_at = _utc_now()
        batch2_pred = final_model.predict(batch2[[FEATURE]])
        batch2_predictions = _predictions(batch2, batch2_pred, "batch2_final_test")
        known = batch2[TARGET].notna().to_numpy()
        test_scores = _metrics(batch2.loc[known, TARGET].to_numpy(), batch2_pred[known])
        baseline._atomic_text(output_dir / "batch2_predictions.csv", batch2_predictions.to_csv(index=False))
    gaps = {"train_valid": holdout_scores["mape_percent"] - selected_mean,
            "valid_test": test_scores["mape_percent"] - holdout_scores["mape_percent"],
            "target_test": test_scores["mape_percent"] - 9.1}
    performance = pd.DataFrame([
        {"index": "Train (Batch 1 CV)", "mape_percent": selected_mean, "unit": "%",
         "note": "35 development cells; five-fold unweighted validation mean; selection CV estimate"},
        {"index": "Valid (Batch 1 Hold-out)", **holdout_scores, "unit": "%",
         "note": "11 cells; winner fitted on development 35 cells"},
        {"index": "Test (Batch 2)", **test_scores, "unit": "%",
         "note": "39 known targets of 47 cells; same settings refitted on all Batch 1 46 cells"},
        {"index": "Gap (Train-Valid)", "mape_percent": gaps["train_valid"], "unit": "%p",
         "note": "Valid-CV; positive suggests poorer validation, not proof of overfitting"},
        {"index": "Gap (Valid-Test)", "mape_percent": gaps["valid_test"], "unit": "%p",
         "note": "Test-Valid; different batches AND fit sample counts (46 versus 35)"},
        {"index": "Gap (Target-Test)", "mape_percent": gaps["target_test"], "unit": "%p",
         "note": "Batch 2 Test-9.1; assignment comparison, not identical paper protocol"},
    ])
    performance.loc[0, "mae_cycles"] = float(cv_metrics.mae_cycles.mean())
    performance.loc[0, "rmse_cycles"] = float(cv_metrics.rmse_cycles.mean())
    baseline._atomic_text(output_dir / "performance.csv", performance.to_csv(index=False))
    summary = {"experiment_id": EXPERIMENT_ID, "status": "required Day 2 model selection and evaluation completed",
               "mode": "fixed_winner_reproduction" if reproduce else "original_final_evaluation",
               "winner": winner_spec, "winner_candidate_id": winner_id,
               "search_status": search_status, "candidate_count": 69, "cv_fit_count_for_search": 0 if reproduce else 345,
               "features": [FEATURE], "target": TARGET, "target_transform": "none",
               "split_id": split["split_id"],
               "train_cv": {"n_cells": 35, "n_protocol_groups": 18, "n_folds": 5,
                            "mape_percent_mean": selected_mean,
                            "mape_percent_sample_sd": float(cv_metrics.mape_percent.std(ddof=1)),
                            "mae_cycles_fold_mean": float(cv_metrics.mae_cycles.mean()),
                            "rmse_cycles_fold_mean": float(cv_metrics.rmse_cycles.mean()),
                            "pooled_oof": _metrics(cv_predictions.y_true_cycle_life.to_numpy(), cv_predictions.y_pred_cycle_life.to_numpy()),
                            "aggregation": "unweighted arithmetic validation-fold mean",
                            "limitations": "Used for candidate selection; not nested/unbiased post-selection CV. Fold sample SD is not a confidence interval."},
               "holdout": {"n_cells": 11, "n_protocol_groups": 5, "model_fit_n_cells": 35, **holdout_scores},
               "batch2": {"n_original_cells": 47, "n_predicted_cells": 47, "n_evaluated_cells": 39,
                          "n_unknown_targets": 8, "unknown_target_cell_ids": EXPECTED_UNKNOWN_IDS,
                          "model_fit_n_cells": 46, **test_scores},
               "gaps_percent_points": gaps, "gap_definitions": config["gaps_percent_points"],
               "paper_target_mape_percent": 9.1,
               "development_model_equation": baseline._coefficient_record(development_model, "development_fit", 35),
               "final_model_equation": baseline._coefficient_record(final_model, "all_batch1_final_fit", 46),
               "evaluation_status": {"model_selection": "completed", "grid_search": "not rerun; fixed winner" if reproduce else "completed",
                                     "holdout": "completed", "batch2": "completed", "batch3": "not performed; optional"},
               "honest_evaluation": {"holdout_or_test_used_for_selection": False,
                                     "test_driven_retuning": False, "test_prediction_calls_this_run": 1,
                                     "batch2_prior_eda": True,
                                     "note": "Assignment required Batch 2 EDA in Day 1; this is not a wholly unobserved dataset. No Batch 2 modeling feedback used for tuning. Replay is not a new independent test."},
               "solver_records": solver_records,
               "prediction_quality": {
                   stage: {"nonpositive_prediction_count": int((frame.y_pred_cycle_life <= 0).sum()),
                           "nonpositive_prediction_cell_ids": frame.loc[frame.y_pred_cycle_life <= 0, "cell_id"].astype(int).tolist(),
                           "predictions_clipped": False}
                   for stage, frame in (("development_oof", cv_predictions),
                                        ("holdout", holdout_predictions), ("batch2", batch2_predictions))},
               "preprocessing": "fold/partition train-only StandardScaler; no target transform, feature additions, imputation, exclusion or prediction clipping"}
    _write_json(output_dir / "metrics.json", summary)
    paths = {path.name: {"relative_path": path.relative_to(root).as_posix(), "sha256": baseline._sha256(path)}
             for path in output_dir.iterdir() if path.is_file()}
    run_manifest = {"schema_version": 1, "experiment_id": EXPERIMENT_ID,
                    "started_at_utc": started, "selection_locked_at_utc": selection_locked_at,
                    "holdout_prediction_started_at_utc": holdout_started_at,
                    "final_fit_completed_at_utc": final_fit_completed_at,
                    "batch2_prediction_started_at_utc": batch2_started_at,
                    "completed_at_utc": _utc_now(), "provenance": provenance,
                    "module_relative_path": "src/modeling_submission.py", "split_id": split["split_id"],
                    "config": config, "winner": winner_spec,
                    "development_model_fit_cell_ids": dev.cell_id.astype(int).tolist(),
                    "holdout_evaluation_cell_ids": holdout.cell_id.astype(int).tolist(),
                    "final_model_fit_cell_ids": data.cell_id.astype(int).tolist(),
                    "batch2_predicted_cell_ids": batch2.cell_id.astype(int).tolist(),
                    "batch2_evaluated_cell_ids": batch2.loc[known, "cell_id"].astype(int).tolist(),
                    "batch2_unknown_target_cell_ids": EXPECTED_UNKNOWN_IDS,
                    "all_original_batch1_and_batch2_cells_preserved": True,
                    "evaluation_status": summary["evaluation_status"], "honest_evaluation": summary["honest_evaluation"],
                    "artifacts": paths}
    _write_json(output_dir / "run-manifest.json", run_manifest)
    return {"summary": summary, "candidate_scores": candidates, "cv_metrics": cv_metrics,
            "cv_predictions": cv_predictions, "holdout_predictions": holdout_predictions,
            "batch2_predictions": batch2_predictions, "performance": performance,
            "run_manifest": run_manifest, "config": config, "artifact_directory": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reproduce", action="store_true",
                        help="Replay the locked winner in a new directory; no new search or selection.")
    args = parser.parse_args()
    result = run(args.root, reproduce=args.reproduce)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
