"""DAY 1 Q1/Q4: 원본 셀을 보존하는 수명·충전 조건 탐색 분석.

원본 label/셀을 수정하거나 학습하지 않는다. 전류 상세값은 실제 cycle 10만 읽는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CycleNotFoundError, RawBatch


COLORS = {"batch1": "#27647B", "batch2": "#D58037", "batch3": "#6B7A51"}
LABELS = {"batch1": "Batch 1", "batch2": "Batch 2", "batch3": "Batch 3"}
FILENAMES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
POLICY_PATTERN = re.compile(
    r"^(?P<C1>\d+(?:\.\d+)?)C\((?P<SOCswitch>\d+(?:\.\d+)?)%\)-"
    r"(?P<C2>\d+(?:\.\d+)?)C(?P<suffix>-newstructure|\(SLOWCYCLE)?$"
)


def parse_policy(text: str) -> dict:
    """관찰된 2단 표기만 파싱하며 suffix와 원문을 보존한다."""
    match = POLICY_PATTERN.fullmatch(text)
    if match is None:
        return {"C1": np.nan, "SOCswitch": np.nan, "C2": np.nan,
                "policy_suffix": None, "policy_parse_status": "unsupported_format"}
    values = {name: float(match.group(name)) for name in ("C1", "SOCswitch", "C2")}
    suffix = match.group("suffix") or ""
    return {**values, "policy_suffix": suffix,
            "policy_parse_status": "parsed_with_suffix" if suffix else "parsed_standard"}


def correlate(frame: pd.DataFrame, x: str, y: str) -> dict:
    """각 상관계수의 실제 유한값 쌍 분모를 별도로 기록한다."""
    pairs = frame[[x, y]].to_numpy(dtype=float)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    record = {"x": x, "y": y, "n_pairs": len(pairs),
              "pearson_r": np.nan, "pearson_p": np.nan,
              "spearman_rho": np.nan, "spearman_p": np.nan}
    if len(pairs) >= 3 and np.unique(pairs[:, 0]).size > 1 and np.unique(pairs[:, 1]).size > 1:
        pearson = stats.pearsonr(pairs[:, 0], pairs[:, 1])
        spearman = stats.spearmanr(pairs[:, 0], pairs[:, 1])
        record.update(pearson_r=float(pearson.statistic), pearson_p=float(pearson.pvalue),
                      spearman_rho=float(spearman.statistic), spearman_p=float(spearman.pvalue))
    return record


def life_summary(group: pd.DataFrame) -> dict:
    life = group.loc[np.isfinite(group.cycle_life), "cycle_life"]
    n = len(life)
    q1, q3 = life.quantile([0.25, 0.75]) if n else (np.nan, np.nan)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    short, middle, long = (life < 500).sum(), life.between(500, 1000).sum(), (life > 1000).sum()
    return {
        "n_cells": len(group), "n_valid_life": n, "n_missing_or_nonfinite_life": len(group) - n,
        "min": life.min(), "max": life.max(), "mean": life.mean(), "std": life.std(),
        "median": life.median(), "q25": q1, "q75": q3,
        "short_lt500_count": int(short), "short_lt500_pct": 100 * short / n if n else np.nan,
        "middle_500_to1000_count": int(middle), "middle_500_to1000_pct": 100 * middle / n if n else np.nan,
        "long_gt1000_count": int(long), "long_gt1000_pct": 100 * long / n if n else np.nan,
        "histogram_range": [150, 2300], "below_histogram": int((life < 150).sum()),
        "above_histogram": int((life > 2300).sum()),
        "iqr_lower_fence": lower, "iqr_upper_fence": upper,
        "iqr_candidate_cell_ids": group.loc[np.isfinite(group.cycle_life) &
                                             ((group.cycle_life < lower) | (group.cycle_life > upper)),
                                             "cell_id"].tolist(),
    }


def analyze_distribution_protocol(project_root: str | Path):
    """139개 원본 셀과 batch별 통계/상관 결과를 반환한다."""
    root = Path(project_root)
    tables, patterns = [], {}
    for batch_id, filename in FILENAMES.items():
        with RawBatch(root / "data/raw" / batch_id / filename, batch_id) as raw:
            cells = raw.cell_table()
            features = []
            for cell_id in range(raw.cell_count):
                summary = raw.summary_for_cell(cell_id)
                early = summary.loc[summary.cycle.between(10, 100), ["cycle", "QDischarge"]]
                finite = early.loc[np.isfinite(early.to_numpy(dtype=float)).all(axis=1)]
                slope = np.nan
                if len(finite) >= 2 and finite.cycle.nunique() >= 2:
                    slope = float(stats.linregress(finite.cycle, finite.QDischarge).slope)
                row = {"cell_id": cell_id, "qd_slope_cycle10_100": slope,
                       "qd_slope_pair_count": len(finite), "qd_window_row_count": len(early),
                       "qd_window_nonfinite_count": len(early) - len(finite),
                       "qd_window_zero_count": int((early.QDischarge == 0).sum())}
                try:
                    time = raw.time_frame(cell_id, 10)
                except CycleNotFoundError:
                    time = pd.DataFrame(columns=["t", "I"])
                    status = "missing_cycle"
                else:
                    status = "empty_cycle" if time.empty else "available"
                current = time.I.to_numpy(dtype=float)
                positive = current[np.isfinite(current) & (current > 0)]
                row.update(current_cycle10_status=status,
                           current_cycle10_raw_count=len(current),
                           current_cycle10_nonfinite_count=int((~np.isfinite(current)).sum()),
                           positive_current_count=len(positive),
                           positive_current_mean=float(positive.mean()) if len(positive) else np.nan,
                           positive_current_max=float(positive.max()) if len(positive) else np.nan,
                           positive_current_std=float(positive.std(ddof=1)) if len(positive) > 1 else np.nan,
                           positive_current_median=float(np.median(positive)) if len(positive) else np.nan)
                features.append(row)
                if cell_id == 0:
                    patterns[batch_id] = time[["t", "I"]].copy()
            cells = cells.merge(pd.DataFrame(features), on="cell_id", validate="one_to_one")
            parsed = pd.DataFrame([parse_policy(text) for text in cells.policy_readable])
            cells = pd.concat([cells.reset_index(drop=True), parsed], axis=1)
            cells["life_label_status"] = np.where(np.isfinite(cells.cycle_life), "finite", "missing_or_nonfinite")
            tables.append(cells)
    cells = pd.concat(tables, ignore_index=True)
    assert len(cells) == 139 and not cells.duplicated(["batch_id", "cell_id"]).any()

    q1 = {batch: life_summary(group) for batch, group in cells.groupby("batch_id", sort=False)}
    for batch, group in cells.groupby("batch_id", sort=False):
        life = group.loc[np.isfinite(group.cycle_life), "cycle_life"]
        q1[batch]["below_batch1_min_count"] = int((life < q1["batch1"]["min"]).sum())
        q1[batch]["above_batch1_max_count"] = int((life > q1["batch1"]["max"]).sum())
    short = cells.loc[np.isfinite(cells.cycle_life) & (cells.cycle_life < 500),
                      ["batch_id", "cell_id", "cycle_life", "policy_readable", "C1", "SOCswitch", "C2"]]
    protocols = []
    for (batch, policy), group in cells.groupby(["batch_id", "policy_readable"], sort=False):
        life = group.loc[np.isfinite(group.cycle_life), "cycle_life"]
        protocols.append({"batch_id": batch, "policy_readable": policy,
                          "n_cells": len(group), "n_valid_life": len(life), "n_missing_life": len(group) - len(life),
                          "mean_life": life.mean(), "median_life": life.median(),
                          "std_life": life.std(), "min_life": life.min(), "max_life": life.max()})
    correlations = []
    for batch, group in [("all", cells), *cells.groupby("batch_id", sort=False)]:
        for outcome in ("cycle_life", "qd_slope_cycle10_100"):
            for signal in ("C1", "SOCswitch", "C2", "positive_current_mean", "positive_current_max"):
                correlations.append({"scope": batch, **correlate(group, signal, outcome)})
    result = {
        "analysis": "DAY1 Q1 and Q4 exploratory analysis", "n_original_cells": len(cells),
        "source_files": {batch: str(root / "data/raw" / batch / filename) for batch, filename in FILENAMES.items()},
        "source_snapshot_reference": str(root / "configs/data-snapshot.json"),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": sys.version, **{package: importlib.metadata.version(package) for package in ("numpy", "pandas", "matplotlib", "scipy", "h5py")}},
        "definitions": {
            "life": "Unmodified original cycle_life. Missing values remain missing.",
            "life_ratios": "Denominator: finite cycle_life labels within each batch; short <500; long >1000.",
            "histogram": "Common equal-width bins in 150-2300; outside-range counts are reported separately.",
            "protocol": "Parse exact two-stage numeric notation with known suffixes only; retain the complete raw string.",
            "positive_current": "Cycle 10, finite samples with I>0; sample mean/max/std. Not time-weighted, not converted to C-rate.",
            "slope": "OLS slope of finite (actual cycle,QDischarge) pairs in cycles 10-100. Raw zeros are retained.",
            "correlations": "Finite cell-level pairs. Pearson and Spearman; pooled and within batch; p-values exploratory, no multiple-testing decision.",
            "model_context": "User confirmed first 100 cycles -> original final cycle_life regression; no model is fitted here.",
            "iqr_candidates": "Descriptive Tukey fences per batch only, not an exclusion rule.",
        },
        "q1": q1, "short_cells": short.sort_values(["batch_id", "cycle_life"]).to_dict("records"),
        "missing_label_cells": cells.loc[cells.life_label_status != "finite", ["batch_id", "cell_id", "policy_readable"]].to_dict("records"),
        "protocol_parse_counts": cells.groupby(["batch_id", "policy_parse_status"]).size().rename("count").reset_index().to_dict("records"),
        "protocol_parse_failures": cells.loc[cells.policy_parse_status == "unsupported_format", ["batch_id", "cell_id", "policy_readable"]].to_dict("records"),
        "protocol_summary": protocols, "correlations": correlations,
        "cell_records": cells.drop(columns=["barcode_raw", "channel_id_raw"]).to_dict("records"),
        "limitations": [
            "All 139 raw cells remain; missing life labels are not imputed. Correlations use finite pairs, not all cells.",
            "This raw cohort and raw life values differ from the paper's merged/cleaned cohort. Batch 2 file identity also differs.",
            "Short life is not proof of a defect. Continuation, special protocols, or data quality are candidates for review, not verified causes.",
            "Protocol means often have small n; numeric protocol components, current patterns, and batch differ jointly.",
            "Pooled correlations may mix batch effects; associations do not establish charging-current causation.",
            "I>0 is a simple exploratory definition. Sampling-frequency differences and tiny positive values can affect the sample mean.",
            "Early capacity may increase; its signed slope is not automatically a degradation rate.",
            "No raw values/cells are changed, no protocol suffixes are collapsed, and no training/test optimization occurs.",
        ],
    }
    return cells, result, patterns


def _style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.18)


def create_figures(cells, result, patterns, destination: str | Path):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    filenames = []

    def save(figure, name):
        figure.savefig(destination / name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        filenames.append(name)

    with plt.rc_context({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True, layout="constrained")
        for axis, (batch, group) in zip(axes, cells.groupby("batch_id", sort=False)):
            life = group.loc[np.isfinite(group.cycle_life), "cycle_life"]
            summary = result["q1"][batch]
            axis.axvspan(150, 500, color="#B86B61", alpha=0.10)
            axis.axvspan(1000, 2300, color="#527886", alpha=0.06)
            axis.hist(life, bins=np.linspace(150, 2300, 22), color=COLORS[batch], edgecolor="white")
            axis.axvline(life.median(), color="#263238", linewidth=1.2, linestyle="--")
            axis.set(title=f"{LABELS[batch]}: {len(life)}/{len(group)} labels", xlabel="Cycle life", xlim=(150, 2300))
            axis.text(0.97, 0.94, f"Median: {life.median():.1f}\nMissing: {len(group)-len(life)}\nOutside range: {summary['below_histogram']+summary['above_histogram']}",
                      transform=axis.transAxes, ha="right", va="top")
            axis.set_xticks([150, 500, 1000, 1500, 2000, 2300])
            axis.tick_params(axis="x", rotation=35)
            _style(axis)
        axes[0].set_ylabel("Cells")
        save(fig, "q1_life_distribution.png")

        fig, axis = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
        positions = np.arange(3)
        keys = ["short_lt500_pct", "middle_500_to1000_pct", "long_gt1000_pct"]
        for offset, batch in enumerate(COLORS):
            values = [result["q1"][batch][key] for key in keys]
            bars = axis.bar(positions + (offset-1)*0.24, values, width=0.24, color=COLORS[batch],
                            label=f"{LABELS[batch]} (n={result['q1'][batch]['n_valid_life']})")
            axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
        axis.set(xticks=positions, xticklabels=["Short: <500", "Middle: 500-1000", "Long: >1000"],
                 ylabel="Share of finite life labels (%)", ylim=(0, 105), title="Short-life cells are concentrated in Batch 2")
        axis.legend(loc="upper left")
        _style(axis)
        save(fig, "q1_life_groups.png")

        protocol_table = pd.DataFrame(result["protocol_summary"])
        for batch in COLORS:
            table = protocol_table.loc[protocol_table.batch_id == batch].sort_values("mean_life", na_position="last")
            fig, axis = plt.subplots(figsize=(8.5, max(4.3, len(table)*0.32+1.9)), layout="constrained")
            y = np.arange(len(table))
            finite = np.isfinite(table.mean_life.to_numpy(dtype=float))
            valid = table.loc[finite]
            axis.errorbar(valid.mean_life, y[finite],
                          xerr=np.stack([valid.mean_life-valid.min_life, valid.max_life-valid.mean_life]),
                          fmt="o", color=COLORS[batch], ecolor=COLORS[batch], alpha=0.85, capsize=2)
            for position in y[~finite]:
                axis.text(0.03, position, "No finite life label", transform=axis.get_yaxis_transform(), va="center", color="#777777")
            axis.set_yticks(y, [f"{row.policy_readable}  [{row.n_valid_life}/{row.n_cells}]" for row in table.itertuples()])
            axis.set(xlabel="Cycle life: mean and observed min-max", ylabel="Raw protocol [valid/all cells]",
                     title=f"{LABELS[batch]}: cycle life by raw protocol", xlim=(0, 2300))
            axis.set_ylim(-0.75, len(table)-0.25)
            _style(axis)
            save(fig, f"q4_protocol_means_{batch}.png")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharey=True, layout="constrained")
        for axis, (batch, group) in zip(axes, cells.groupby("batch_id", sort=False)):
            pairs = group.loc[np.isfinite(group.C1) & np.isfinite(group.cycle_life)]
            correlation = correlate(group, "C1", "cycle_life")
            axis.scatter(pairs.C1, pairs.cycle_life, color=COLORS[batch], s=38, alpha=0.75, edgecolor="white", linewidth=0.4)
            axis.set(title=f"{LABELS[batch]}\nn={correlation['n_pairs']}, Spearman rho={correlation['spearman_rho']:.2f}",
                     xlabel="Parsed first-stage C-rate (C)")
            _style(axis)
        axes[0].set_ylabel("Raw cycle life")
        save(fig, "q4_c1_life_relationship.png")

        fig, axes = plt.subplots(2, 3, figsize=(12, 8.1), layout="constrained")
        for column, (batch, group) in enumerate(cells.groupby("batch_id", sort=False)):
            for row, outcome in enumerate(("cycle_life", "qd_slope_cycle10_100")):
                axis = axes[row, column]
                mask = np.isfinite(group.positive_current_mean) & np.isfinite(group[outcome])
                axis.scatter(group.loc[mask, "positive_current_mean"], group.loc[mask, outcome],
                             color=COLORS[batch], s=34, alpha=0.75, edgecolor="white", linewidth=0.4)
                correlation = correlate(group, "positive_current_mean", outcome)
                axis.set(title=f"{LABELS[batch]}\nn={correlation['n_pairs']}, Spearman rho={correlation['spearman_rho']:.2f}",
                         xlabel="Mean positive I, cycle 10\n(raw units)")
                if row:
                    axis.axhline(0, color="#8A8A8A", linestyle="--", linewidth=0.8)
                    axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                _style(axis)
        axes[0, 0].set_ylabel("Raw cycle life")
        axes[1, 0].set_ylabel("QDischarge slope, cycles 10-100\n(raw capacity / cycle)")
        save(fig, "q4_current_life_slope.png")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, layout="constrained")
        for axis, batch in zip(axes, COLORS):
            time = patterns[batch]
            axis.plot(time.t, time.I, color=COLORS[batch], linewidth=1)
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set(title=f"{LABELS[batch]}: cell 0, cycle 10", xlabel="t (raw units)")
            _style(axis)
        axes[0].set_ylabel("I (raw units)")
        fig.suptitle("Illustrative raw current patterns; one cell per batch")
        save(fig, "q4_current_patterns_cell0.png")
    return filenames


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def run(project_root: str | Path, output_dir: str | Path | None = None):
    root = Path(project_root)
    destination = Path(output_dir) if output_dir is not None else root / "outputs/day1"
    destination.mkdir(parents=True, exist_ok=True)
    cells, result, patterns = analyze_distribution_protocol(root)
    result["figure_files"] = create_figures(cells, result, patterns, destination / "figures")
    output = destination / "q1_q4_results.json"
    output.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_ready({"q1": result["q1"], "protocol_parse_counts": result["protocol_parse_counts"],
                                "selected_correlations": [record for record in result["correlations"] if record["x"] in ("C1", "positive_current_mean")],
                                "figure_files": result["figure_files"]}),
                     ensure_ascii=False, indent=2, allow_nan=False))
    return json_ready(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.project_root, args.output_dir)
