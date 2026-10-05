"""원본 MAT 배터리 데이터를 읽고 측정 단위별로 연결하는 코드.

저장 형식과 독립적으로 원본 필드와 배열 길이를 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd


SUMMARY_FIELDS = ("cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin", "chargetime")
TIME_FIELDS = ("t", "I", "V", "T", "Qc", "Qd")
CURVE_FIELDS = ("Qdlin", "Tdlin", "discharge_dQdV")


def numeric_vector(dataset: h5py.Dataset) -> np.ndarray:
    """MATLAB empty marker를 실제 측정값으로 해석하지 않는다."""
    if dataset.attrs.get("MATLAB_empty", False):
        return np.empty(0, dtype=np.float64)
    values = np.asarray(dataset[()])
    if values.dtype.kind not in "biuf":
        raise TypeError(f"Expected numeric data, got dtype={values.dtype}")
    return values.ravel(order="F")


def matlab_class(dataset: h5py.Dataset) -> str:
    value = dataset.attrs.get("MATLAB_class", b"")
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def character_value(dataset: h5py.Dataset) -> str:
    if matlab_class(dataset) != "char":
        raise TypeError("Expected MATLAB char")
    if dataset.attrs.get("MATLAB_empty", False):
        return ""
    return np.asarray(dataset[()], dtype="<u2").ravel(order="F").tobytes().decode("utf-16-le")


def encoded_string_value(dataset: h5py.Dataset) -> str:
    """MATLAB string 객체의 저장값을 JSON으로 보존한다. ID로 재해석하지 않는다."""
    return json.dumps(np.asarray(dataset[()]).tolist(), separators=(",", ":"))


@dataclass
class RawCycle:
    cell_id: int
    cycle_index: int
    cycle: float
    timeseries: dict[str, np.ndarray]
    voltage: dict[str, np.ndarray]


class CycleNotFoundError(KeyError):
    """요청한 실제 사이클 번호가 해당 셀에 없을 때 발생한다."""


class RawBatch:
    """파일을 열어 둔 범위 안에서만 원본 객체를 접근한다."""

    def __init__(self, path: str | Path, batch_id: str):
        self.path = Path(path)
        self.batch_id = batch_id

    def __enter__(self):
        self.file = h5py.File(self.path, "r")
        batch = self.file["batch"]
        self.references = {key: value[()].ravel(order="F") for key, value in batch.items()}
        if len({len(refs) for refs in self.references.values()}) != 1:
            self.file.close()
            raise ValueError("Different cell counts in original batch fields")
        self.cell_count = len(self.references["summary"])
        return self

    def __exit__(self, *args):
        self.file.close()

    def field(self, name: str, cell_id: int):
        return self.file[self.references[name][cell_id]]

    def cell_table(self) -> pd.DataFrame:
        records = []
        for cell_id in range(self.cell_count):
            life = numeric_vector(self.field("cycle_life", cell_id))
            if life.size != 1:
                raise ValueError(f"cell_id={cell_id}: cycle_life is not scalar")
            records.append({
                "batch_id": self.batch_id,
                "cell_id": cell_id,
                "cycle_life": life.item(),
                "policy": character_value(self.field("policy", cell_id)),
                "policy_readable": character_value(self.field("policy_readable", cell_id)),
                "barcode_raw": encoded_string_value(self.field("barcode", cell_id)),
                "channel_id_raw": encoded_string_value(self.field("channel_id", cell_id)),
            })
        return pd.DataFrame(records)

    def summary_for_cell(self, cell_id: int) -> pd.DataFrame:
        source = self.field("summary", cell_id)
        if set(source) != set(SUMMARY_FIELDS):
            raise ValueError(f"Unexpected summary fields for cell_id={cell_id}: {list(source)}")
        values = {name: numeric_vector(source[name]) for name in SUMMARY_FIELDS}
        if len({len(value) for value in values.values()}) != 1:
            raise ValueError(f"Summary lengths do not match for cell_id={cell_id}")
        count = len(values["cycle"])
        result = pd.DataFrame(values)
        result.insert(0, "cycle_index", np.arange(count, dtype=np.int32))
        result.insert(0, "cell_id", np.full(count, cell_id, dtype=np.int16))
        result.insert(0, "batch_id", self.batch_id)
        return result

    def summary_table(self) -> pd.DataFrame:
        return pd.concat([self.summary_for_cell(i) for i in range(self.cell_count)], ignore_index=True)


    def _cycle_location(self, cell_id: int, cycle_number: int | float):
        """저장 순서가 아닌 원본 summary.cycle의 정확한 번호를 찾는다."""
        if isinstance(cell_id, (bool, np.bool_)) or not isinstance(cell_id, (int, np.integer)):
            raise TypeError("cell_id must be an integer")
        if not 0 <= cell_id < self.cell_count:
            raise IndexError(f"cell_id={cell_id}: expected 0 <= cell_id < {self.cell_count}")
        if isinstance(cycle_number, (bool, np.bool_)) or not isinstance(
            cycle_number, (int, float, np.integer, np.floating)
        ):
            raise TypeError("cycle_number must be a numeric cycle number")
        if not np.isfinite(cycle_number):
            raise ValueError("cycle_number must be finite")
        cycles = numeric_vector(self.field("summary", cell_id)["cycle"])
        positions = np.flatnonzero(cycles == cycle_number)
        if not len(positions):
            raise CycleNotFoundError(f"cell_id={cell_id}: actual cycle={cycle_number} is missing")
        if len(positions) != 1:
            raise ValueError(f"cell_id={cell_id}: actual cycle={cycle_number} occurs more than once")
        position = int(positions[0])
        return position, float(cycles[position]), len(cycles)

    def _cycle_vectors(self, cell_id: int, position: int, cycle_count: int, fields):
        """요청한 필드의 해당 참조만 읽는다. 다른 사이클 측정값은 읽지 않는다."""
        source = self.field("cycles", cell_id)
        values = {}
        for name in fields:
            references = source[name][()].ravel(order="F")
            if len(references) != cycle_count:
                raise ValueError(f"Summary/cycles counts differ: cell_id={cell_id}, field={name}")
            values[name] = numeric_vector(self.file[references[position]])
        return values

    def _measurement_frame(self, cell_id: int, position: int, cycle: float, values):
        count = len(next(iter(values.values())))
        frame = pd.DataFrame({
            "batch_id": np.full(count, self.batch_id, dtype=object),
            "cell_id": np.full(count, cell_id, dtype=np.int64),
            "cycle_index": np.full(count, position, dtype=np.int32),
            "cycle": np.full(count, cycle, dtype=np.float64),
            "point_index": np.arange(count, dtype=np.int32),
            **values,
        })
        # 빈 측정값도 실제 셀·사이클 위치를 잃지 않도록 보관한다.
        frame.attrs.update(batch_id=self.batch_id, cell_id=int(cell_id), cycle_index=position, cycle=cycle)
        return frame

    def time_frame(self, cell_id: int, cycle_number: int | float) -> pd.DataFrame:
        """한 셀의 실제 사이클에서 t/I/V/T/Qc/Qd 원본 값만 읽는다.

        없는 번호와 중복 번호는 오류로 드러내며 근접 사이클로 대체하지 않는다.
        모든 시간 배열이 비어 있으면 원본 위치를 attrs에 가진 빈 표를 반환한다.
        """
        position, cycle, cycle_count = self._cycle_location(cell_id, cycle_number)
        values = self._cycle_vectors(cell_id, position, cycle_count, TIME_FIELDS)
        if len({len(vector) for vector in values.values()}) != 1:
            raise ValueError(f"Time-series lengths differ: cell_id={cell_id}, cycle_index={position}")
        return self._measurement_frame(cell_id, position, cycle, values)

    def voltage_frame(self, cell_id: int, cycle_number: int | float) -> pd.DataFrame:
        """한 셀의 실제 사이클에서 원본 Vdlin과 Qdlin만 읽는다.

        Qdlin이 비어 있으면 관측 전압-용량 쌍이 없는 빈 표를 반환한다.
        비어 있지 않은 곡선은 원본 전압 축과 길이가 같아야 한다.
        보간, 정렬, 차이 계산은 수행하지 않는다.
        """
        position, cycle, cycle_count = self._cycle_location(cell_id, cycle_number)
        qd = self._cycle_vectors(cell_id, position, cycle_count, ("Qdlin",))["Qdlin"]
        grid = numeric_vector(self.field("Vdlin", cell_id))
        if len(qd) and len(qd) != len(grid):
            raise ValueError(f"Voltage grid/curve lengths differ: cell_id={cell_id}, cycle_index={position}")
        values = {"Vdlin": grid if len(qd) else grid[:0], "Qdlin": qd}
        return self._measurement_frame(cell_id, position, cycle, values)

    def qv_table(self, cycles=(10, 100)) -> tuple[pd.DataFrame, pd.DataFrame]:
        """모든 셀에서 요청한 실제 사이클의 Q(V)와 가용 상태만 준비한다.

        반환값은 (곡선 표, 상태 표)다. 상태는 available / missing_cycle /
        empty_curve이며 없는 번호나 빈 곡선의 셀도 상태 표에 남는다.
        중복 사이클·길이 불일치는 오류를 내며 ΔQ나 피처를 계산하지 않는다.
        """
        requested = tuple(cycles)
        if not requested:
            raise ValueError("At least one cycle number must be requested")
        if len(set(requested)) != len(requested):
            raise ValueError("Requested cycle numbers must be unique")
        frames, records = [], []
        for cell_id in range(self.cell_count):
            for cycle_number in requested:
                try:
                    frame = self.voltage_frame(cell_id, cycle_number)
                except CycleNotFoundError:
                    records.append({
                        "batch_id": self.batch_id,
                        "cell_id": cell_id,
                        "cycle": float(cycle_number),
                        "cycle_index": None,
                        "point_count": 0,
                        "status": "missing_cycle",
                    })
                    continue
                frames.append(frame)
                records.append({
                    "batch_id": self.batch_id,
                    "cell_id": cell_id,
                    "cycle": frame.attrs["cycle"],
                    "cycle_index": frame.attrs["cycle_index"],
                    "point_count": len(frame),
                    "status": "empty_curve" if frame.empty else "available",
                })
        columns = ["batch_id", "cell_id", "cycle_index", "cycle", "point_index", "Vdlin", "Qdlin"]
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        availability = pd.DataFrame(records, columns=[
            "batch_id", "cell_id", "cycle", "cycle_index", "point_count", "status"
        ])
        availability["cycle_index"] = availability["cycle_index"].astype("Int64")
        return table, availability

    def iter_cycles(self, cell_id: int):
        summary = self.summary_for_cell(cell_id)
        source = self.field("cycles", cell_id)
        if set(source) != set(TIME_FIELDS + CURVE_FIELDS):
            raise ValueError(f"Unexpected cycles fields for cell_id={cell_id}: {list(source)}")
        references = {name: source[name][()].ravel(order="F") for name in source}
        if any(len(refs) != len(summary) for refs in references.values()):
            raise ValueError(f"Summary/cycles counts differ for cell_id={cell_id}")
        grid = numeric_vector(self.field("Vdlin", cell_id))

        for position, cycle in enumerate(summary["cycle"].to_numpy()):
            vectors = {name: numeric_vector(self.file[refs[position]]) for name, refs in references.items()}
            time = {name: vectors[name] for name in TIME_FIELDS}
            curve = {name: vectors[name] for name in CURVE_FIELDS}
            if len({len(value) for value in time.values()}) != 1:
                raise ValueError(f"Time-series lengths differ: cell_id={cell_id}, cycle_index={position}")
            if len({len(value) for value in curve.values()}) != 1:
                raise ValueError(f"Curve lengths differ: cell_id={cell_id}, cycle_index={position}")
            count = len(curve["Qdlin"])
            if count not in (0, len(grid)):
                raise ValueError(f"Voltage grid/curve lengths differ: cell_id={cell_id}, cycle_index={position}")
            voltage = {"Vdlin": grid if count else grid[:0], **curve}
            yield RawCycle(cell_id, position, float(cycle), time, voltage)


def measurement_columns(raw_cycle: RawCycle, table: str) -> dict[str, np.ndarray]:
    """같은 좌표의 측정값만 같은 행에 두며 서로 다른 축을 섞지 않는다."""
    if table not in {"timeseries", "voltage"}:
        raise ValueError(f"Unknown measurement table: {table}")
    values = raw_cycle.timeseries if table == "timeseries" else raw_cycle.voltage
    count = len(next(iter(values.values())))
    return {
        "cell_id": np.full(count, raw_cycle.cell_id, dtype=np.int16),
        "cycle_index": np.full(count, raw_cycle.cycle_index, dtype=np.int32),
        "cycle": np.full(count, raw_cycle.cycle, dtype=np.float64),
        "point_index": np.arange(count, dtype=np.int32),
        **values,
    }


def iter_detail_frames(path: str | Path, batch_id: str):
    """모든 셀·사이클을 시간별/전압별 DataFrame으로 순서대로 펼친다."""
    with RawBatch(path, batch_id) as raw:
        for cell_id in range(raw.cell_count):
            for cycle in raw.iter_cycles(cell_id):
                frames = []
                for table in ("timeseries", "voltage"):
                    frame = pd.DataFrame(measurement_columns(cycle, table))
                    frame.insert(0, "batch_id", batch_id)
                    frames.append(frame)
                yield frames[0], frames[1]


class RunningStatistics:
    """모든 값을 순회하며 유한값의 통계와 비유한값 개수를 누적한다."""

    def __init__(self):
        self.total = self.nan = self.positive_inf = self.negative_inf = self.count = 0
        self.mean = self.m2 = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: np.ndarray):
        self.total += values.size
        self.nan += int(np.isnan(values).sum())
        self.positive_inf += int(np.isposinf(values).sum())
        self.negative_inf += int(np.isneginf(values).sum())
        finite = values[np.isfinite(values)]
        count = finite.size
        if not count:
            return
        mean = float(finite.mean())
        m2 = float(np.square(finite - mean).sum())
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))

    def record(self):
        return {
            "total_count": self.total,
            "finite_count": self.count,
            "nan_count": self.nan,
            "positive_inf_count": self.positive_inf,
            "negative_inf_count": self.negative_inf,
            "mean": self.mean if self.count else np.nan,
            "std": np.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else np.nan,
            "min": self.minimum if self.count else np.nan,
            "max": self.maximum if self.count else np.nan,
        }


def profile_detail_values(path: str | Path, batch_id: str, progress: bool = True) -> pd.DataFrame:
    """샘플링 없이 모든 상세 측정값을 읽어 기본 데이터 품질을 확인한다.

    수치 임계값에 따른 이상치 판정·제외·보간은 하지 않는다.
    """
    fields = {"timeseries": TIME_FIELDS, "voltage": ("Vdlin",) + CURVE_FIELDS}
    accumulators = {table: {name: RunningStatistics() for name in names} for table, names in fields.items()}
    counts = {"cycles": 0, "timeseries_rows": 0, "voltage_rows": 0, "empty_timeseries_cycles": 0, "empty_voltage_cycles": 0}
    with RawBatch(path, batch_id) as raw:
        for cell_id in range(raw.cell_count):
            for cycle in raw.iter_cycles(cell_id):
                n_time = len(cycle.timeseries["t"])
                n_voltage = len(cycle.voltage["Vdlin"])
                counts["cycles"] += 1
                counts["timeseries_rows"] += n_time
                counts["voltage_rows"] += n_voltage
                counts["empty_timeseries_cycles"] += int(n_time == 0)
                counts["empty_voltage_cycles"] += int(n_voltage == 0)
                for table, values in [("timeseries", cycle.timeseries), ("voltage", cycle.voltage)]:
                    for name, vector in values.items():
                        accumulators[table][name].update(vector)
            if progress and ((cell_id + 1) % 10 == 0 or cell_id + 1 == raw.cell_count):
                print(f"{batch_id}: 전체 상세값 검사 {cell_id + 1}/{raw.cell_count} 셀", flush=True)

    result = pd.DataFrame([
        {"batch_id": batch_id, "table": table, "variable": name, **statistics.record()}
        for table, fields in accumulators.items()
        for name, statistics in fields.items()
    ])
    result.attrs["coverage"] = counts
    result.attrs["statistics_basis"] = "Finite values only; all non-finite values counted separately; raw values unchanged."
    return result
