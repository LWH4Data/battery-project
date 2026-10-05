"""DAY 1 Q2: 원본 방전용량 곡선과 탐색적 열화 가속 / knee 후보.

모든 셀과 원본 행을 유지한다. Batch 1 cycle 1의 상세 측정 empty marker가
확인된 0값만 수치 추세 계산에서 측정값으로 취급하지 않는다. 모든 방법은
EDA용이며, 전체 곡선이나 knee 결과를 초기 수명 예측 입력으로 사용하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from src.data import RawBatch

BATCHES = ("batch1", "batch2", "batch3")
COLORS = {"batch1": "#27647B", "batch2": "#D58037", "batch3": "#6B7A51"}


def _distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0}
    return {"count": int(len(a)), "min": float(a.min()), "q25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "q75": float(np.quantile(a, .75)), "max": float(a.max())}


def _slope(x, y):
    if len(x) < 10 or len(np.unique(x)) < 10:
        return None
    return float(theilslopes(y, x)[0])


def _hinge_fit(x, y, window):
    """중앙 rolling median 뒤 연속 2직선의 SSE를 최소화하는 break 탐색.

    최대 200개 관측 위치를 등간격으로 사용한다. Break 후보는 관측 기간의
    15~85%, 양쪽 최소 50사이클이다. 강제 가속 제약 없이 적합한다.
    """
    smooth = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    idx = np.unique(np.linspace(0, len(x) - 1, min(200, len(x))).astype(int))
    xs, ys = x[idx], smooth[idx]
    origin, span = float(x[0]), float(x[-1] - x[0])
    if span < 150:
        return None
    z = (xs - origin) / span
    line_matrix = np.column_stack((np.ones(len(z)), z))
    line_coef = np.linalg.lstsq(line_matrix, ys, rcond=None)[0]
    line_sse = float(np.square(ys - line_matrix @ line_coef).sum())
    candidates = np.linspace(max(x[0] + 50, x[0] + .15 * span),
                             min(x[-1] - 50, x[0] + .85 * span), 81)
    best = None
    for knee in candidates:
        zk = (knee - origin) / span
        matrix = np.column_stack((np.ones(len(z)), z, np.maximum(z - zk, 0)))
        coef = np.linalg.lstsq(matrix, ys, rcond=None)[0]
        sse = float(np.square(ys - matrix @ coef).sum())
        if best is None or sse < best[0]:
            best = (sse, float(knee), coef)
    sse, knee, coef = best
    return {"cycle": knee, "left_slope_Ah_per_cycle": float(coef[1] / span),
            "right_slope_Ah_per_cycle": float((coef[1] + coef[2]) / span),
            "relative_sse_reduction": float(1 - sse / line_sse) if line_sse > 0 else 0.,
            "coefficients": coef.tolist(), "origin": origin, "span": span,
            "smoothing_window": window}


def _candidate(x, y):
    a, b = _hinge_fit(x, y, 21), _hinge_fit(x, y, 51)
    if a is None or b is None:
        return {"status": "not_detected", "reason": "insufficient_observed_span", "fits": [a, b]}
    sensitivity = float(abs(a["cycle"] - b["cycle"]) / (x[-1] - x[0]))
    reasons = []
    for fit in (a, b):
        if not (fit["right_slope_Ah_per_cycle"] < 0 and
                fit["right_slope_Ah_per_cycle"] < fit["left_slope_Ah_per_cycle"]):
            reasons.append("not_accelerating_in_both_smoothings")
        if fit["relative_sse_reduction"] < .20:
            reasons.append("less_than_20pct_sse_improvement")
    if sensitivity > .10:
        reasons.append("break_moves_over_10pct_observed_span")
    return {"status": "exploratory_candidate" if not reasons else "not_detected",
            "reason": ";".join(sorted(set(reasons))) if reasons else None,
            "smoothing_sensitivity_fraction": sensitivity, "fits": [a, b]}


def _fit_curve(x, fit):
    z = (x - fit["origin"]) / fit["span"]
    zk = (fit["cycle"] - fit["origin"]) / fit["span"]
    c = fit["coefficients"]
    return c[0] + c[1] * z + c[2] * np.maximum(z - zk, 0)


def run(project_root=None):
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    out = root / "outputs/day1"
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables, cells, records = {}, {}, []
    report = {"question": "Q2: discharge capacity decline, acceleration and exploratory knee",
              "all_cells_retained": True, "raw_data_modified": False,
              "model_features": "No full-curve, late-window or knee result is a prediction input.",
              "methods": {
                  "raw_curves": "All original summary QDischarge rows against original summary.cycle; no outlier deletion or capacity-based truncation.",
                  "marker_handling": "Batch1 cycle1 QDischarge=0 is marked on raw plots; only verified empty time/QV detail rows are outside the measurement trend calculation domain.",
                  "slopes": "Theil-Sen median pairwise slope from cycle10-100 and final100 observed cycles, finite measurement rows; original spikes retained.",
                  "knee": "Exploratory continuous two-line hinge least squares on 21-/51-point centred rolling-median curves, at most200 equally-spaced observation positions;81 break candidates in15%-85% of observed span, at least50 cycles on each side.",
                  "knee_screen": "Exploratory candidate requires negative right slope, right slope more negative than left in both smoothings, >=20% SSE improvement over one line in both, break agreement within10% observed span. These are descriptive sensitivity screens, not significance tests or physical EOL definitions.",
                  "quality_flags": "Qd>1.3Ah, Tmin<0C or Tmax>100C are diagnostic flags only; no row/cell is excluded based on these thresholds."},
              "limitations": ["Observed final cycle is not cycle_life and can include post-target observations.",
                              "Original missing cycle_life is encoded as null in JSON, not imputed; those cells remain in curve and knee analyses.",
                              "Rolling-median smoothing is an EDA transformation and does not replace stored raw values.",
                              "Piecewise fits can approximate gradual curvature, activation, experimental changes or noise rather than a physical knee.",
                              "Late slopes and knees depend on observation length; missing late deterioration or truncation may prevent detection.",
                              "Theil-Sen robust slope does not establish the cause of anomalous measurements."],
              "batches": {}}
    for batch_id in BATCHES:
        paths = list((root / "data/raw" / batch_id).glob("*.mat"))
        if len(paths) != 1:
            raise ValueError(f"Expected one MAT file in {batch_id}; got {len(paths)}")
        with RawBatch(paths[0], batch_id) as raw:
            table, cell_table = raw.summary_table(), raw.cell_table()
            empty_marker_cells = []
            if batch_id == "batch1":
                for cell_id in range(raw.cell_count):
                    first = table[(table.cell_id == cell_id) & (table.cycle == 1)]
                    if len(first) == 1 and first.QDischarge.item() == 0:
                        if raw.time_frame(cell_id, 1).empty and raw.voltage_frame(cell_id, 1).empty:
                            empty_marker_cells.append(cell_id)
            tables[batch_id], cells[batch_id] = table, cell_table
            batch_records = []
            for row in cell_table.itertuples(index=False):
                sub = table[table.cell_id == row.cell_id].sort_values("cycle")
                marker = ((sub.cycle == 1) & (sub.QDischarge == 0)) if row.cell_id in empty_marker_cells else np.zeros(len(sub), dtype=bool)
                finite = np.isfinite(sub.cycle) & np.isfinite(sub.QDischarge)
                valid = sub[finite & ~marker]
                x, y = valid.cycle.to_numpy(), valid.QDischarge.to_numpy()
                early = (x >= 10) & (x <= 100)
                late = x >= x[-1] - 99
                early_slope, late_slope = _slope(x[early], y[early]), _slope(x[late], y[late])
                record = {"batch_id": batch_id, "cell_id": int(row.cell_id),
                          "cycle_life": float(row.cycle_life) if np.isfinite(row.cycle_life) else None,
                          "original_cycle_life_missing": bool(not np.isfinite(row.cycle_life)),
                          "observed_first_cycle": float(sub.cycle.min()),
                          "observed_last_cycle": float(sub.cycle.max()), "observed_cycle_rows": len(sub),
                          "last_cycle_minus_target": float(sub.cycle.max() - row.cycle_life) if np.isfinite(row.cycle_life) else None,
                          "empty_marker_rows_not_measurements": int(np.sum(marker)),
                          "nonfinite_measurement_rows": int(np.sum(~finite)),
                          "early_slope_Ah_per_cycle": early_slope, "late_slope_Ah_per_cycle": late_slope,
                          "late_more_negative_than_early": bool(late_slope < early_slope) if early_slope is not None and late_slope is not None else None,
                          "knee": _candidate(x, y),
                          "diagnostic_Qd_gt_1_3_rows": int((sub.QDischarge > 1.3).sum()),
                          "diagnostic_temperature_rows": int(((sub.Tmin < 0) | (sub.Tmax > 100)).sum())}
                records.append(record)
                batch_records.append(record)
            high = table[table.QDischarge > 1.3]
            thermal = table[(table.Tmin < 0) | (table.Tmax > 100)]
            n = len(batch_records)
            candidates = [r for r in batch_records if r["knee"]["status"] == "exploratory_candidate"]
            report["batches"][batch_id] = {
                "cell_count": n, "summary_row_count": len(table), "zero_capacity_rows": int((table.QDischarge == 0).sum()),
                "verified_empty_marker_cell_ids": empty_marker_cells, "QD_range_Ah": [float(table.QDischarge.min()), float(table.QDischarge.max())],
                "early_slope_Ah_per_1000_cycles": _distribution([r["early_slope_Ah_per_cycle"] * 1000 for r in batch_records if r["early_slope_Ah_per_cycle"] is not None]),
                "late_slope_Ah_per_1000_cycles": _distribution([r["late_slope_Ah_per_cycle"] * 1000 for r in batch_records if r["late_slope_Ah_per_cycle"] is not None]),
                "late_more_negative_count": sum(r["late_more_negative_than_early"] is True for r in batch_records),
                "late_more_negative_pct": 100 * sum(r["late_more_negative_than_early"] is True for r in batch_records) / n,
                "exploratory_knee_candidate_count": len(candidates), "not_detected_count": n - len(candidates),
                "candidate_cycle_distribution": _distribution([r["knee"]["fits"][0]["cycle"] for r in candidates]),
                "last_cycle_minus_target": _distribution([r["last_cycle_minus_target"] for r in batch_records]),
                "original_cycle_life_missing_cell_ids": [r["cell_id"] for r in batch_records if r["original_cycle_life_missing"]],
                "observed_end_before_target_count": sum(r["last_cycle_minus_target"] is not None and r["last_cycle_minus_target"] < 0 for r in batch_records),
                "observed_end_after_target_count": sum(r["last_cycle_minus_target"] is not None and r["last_cycle_minus_target"] > 0 for r in batch_records),
                "QD_gt_1_3_flags": high[["cell_id", "cycle", "QDischarge"]].to_dict("records"),
                "temperature_flag_rows": len(thermal), "temperature_flag_cells": sorted(int(v) for v in thermal.cell_id.unique()),
                "IR_zero_rows": int((table.IR == 0).sum()),
                "Tmin_range_C": [float(table.Tmin.min()), float(table.Tmin.max())],
                "Tmax_range_C": [float(table.Tmax.min()), float(table.Tmax.max())]}
    report["cell_results"] = records
    with plt.rc_context({"font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), constrained_layout=True)
        for col, batch_id in enumerate(BATCHES):
            t = tables[batch_id]
            for _, sub in t.groupby("cell_id"):
                for ax in axes[:, col]:
                    ax.plot(sub.cycle, sub.QDischarge, color=COLORS[batch_id], lw=.65, alpha=.28)
            high = t[t.QDischarge > 1.3]
            axes[0, col].scatter(high.cycle, high.QDischarge, color="#9C3D35", s=15, zorder=4, label="Qd > 1.3 Ah (flag)")
            zeros = t[t.QDischarge == 0]
            if len(zeros):
                axes[0, col].scatter(zeros.cycle, zeros.QDischarge, c="#9C3D35", marker="x", s=25, label="Empty-detail marker")
            axes[0, col].set_title(f"{batch_id.capitalize()}: all {cells[batch_id].shape[0]} cells")
            axes[0, col].axhline(1.3, color="#8B8B8B", linestyle=":", linewidth=.7, alpha=.6)
            axes[0, col].set_ylim(-.08, 3.05)
            axes[1, col].set_ylim(.75, 1.15)
            axes[1, col].set_title("Zoom: 0.75–1.15 Ah")
            for ax in axes[:, col]:
                ax.set_xlabel("Original cycle number")
                ax.set_ylabel("Discharge capacity (Ah)")
                ax.grid(alpha=.15)
            if len(high) or len(zeros):
                axes[0, col].legend(fontsize=9)
        fig.suptitle("Q2 | Full discharge trajectories and measurement-quality flags")
        fig.savefig(figure_dir / "q2_degradation_curves.png", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        for ax, key, title in zip(axes, ("early_slope_Ah_per_cycle", "late_slope_Ah_per_cycle"),
                                  ("Early: original cycles 10–100", "Late: final 100 observed cycles")):
            vals = [[r[key] * 1000 for r in records if r["batch_id"] == b and r[key] is not None] for b in BATCHES]
            boxes = ax.boxplot(vals, tick_labels=[b.capitalize() for b in BATCHES], patch_artist=True, showfliers=True)
            for patch, b in zip(boxes["boxes"], BATCHES):
                patch.set_facecolor(COLORS[b]); patch.set_alpha(.65)
            ax.axhline(0, c="#555555", ls="--", lw=1)
            ax.set_title(title)
            ax.set_ylabel("Theil–Sen capacity slope (Ah / 1,000 cycles)")
            ax.grid(axis="y", alpha=.15)
        fig.suptitle("Q2 | Robust early vs late slopes: no capacity outlier deletion")
        fig.savefig(figure_dir / "q2_early_late_slopes.png", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
        example_ids = {}
        for ax, b in zip(axes, BATCHES):
            viable = [r for r in records if r["batch_id"] == b and r["knee"]["status"] == "exploratory_candidate"]
            if not viable:
                viable = [r for r in records if r["batch_id"] == b]
            viable.sort(key=lambda r: r["observed_last_cycle"])
            r = viable[len(viable) // 2]
            example_ids[b] = r["cell_id"]
            sub = tables[b][tables[b].cell_id == r["cell_id"]]
            if r["empty_marker_rows_not_measurements"]:
                sub = sub[~((sub.cycle == 1) & (sub.QDischarge == 0))]
            ax.plot(sub.cycle, sub.QDischarge, color=COLORS[b], lw=.85, alpha=.6, label="Original Qd")
            for fit, style in zip(r["knee"]["fits"], ("-", "--")):
                if fit is not None:
                    x = sub.cycle.to_numpy()
                    ax.plot(x, _fit_curve(x, fit), style, c="#393939", lw=1.4, label=f"Median{fit['smoothing_window']} hinge")
                    ax.axvline(fit["cycle"], c="#393939", ls=style, lw=.9, alpha=.65)
            ax.set_title(f"{b.capitalize()} / cell {r['cell_id']}\n{r['knee']['status'].replace('_', ' ')}")
            ax.set_xlabel("Original cycle number"); ax.set_ylabel("Discharge capacity (Ah)")
            ax.grid(alpha=.15); ax.legend(fontsize=9)
        fig.suptitle("Q2 | Piecewise change points: illustrative candidates, not confirmed physical knees")
        fig.savefig(figure_dir / "q2_knee_examples.png", dpi=300)
        plt.close(fig)
        report["illustrative_example_cell_ids"] = example_ids
        report["knee_example_plot_note"] = "Verified empty-detail cycle1 markers are shown in the full-curve figure, not treated as measurements in knee example views."
        report["example_selection"] = "Median-observed-end-cycle cell among exploratory candidates per batch; fallback to median-observed-end-cycle cell if none. Missing-target cells remain eligible. Not a random or representative cohort claim."
    (out / "q2_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else None)
