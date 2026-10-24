"""Rank candidate distance-3 patches from a saved IQM calibration snapshot.

Assumption: the patch-enumeration task will later provide a JSON file containing
a list of objects with ``name``, 17 ``qubits``, and 24 ``cz_pairs``.  This ranks
calibration-derived proxies only; validate the hypothesis with the patch scan.

Example:
    python verification/score_patch.py calib/SNAPSHOT.json results/patches.json --output patch_ranking.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Patch:
    """One candidate d=3 surface-code patch provided by the patch scan."""

    name: str
    qubits: tuple[str, ...]
    cz_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PatchScore:
    """Transparent calibration measurements and their derived ranks."""

    name: str
    cz_error_sum: float
    readout_error_sum: float
    single_qubit_error_sum: float
    minimum_t1: float
    minimum_t2: float
    cz_rank: int = 0
    readout_rank: int = 0
    single_qubit_rank: int = 0
    t1_rank: int = 0
    t2_rank: int = 0
    combined_z_score: float = 0.0
    combined_rank: int = 0


def _pair_key(first: str, second: str) -> str:
    """Return the ordered pair-key format used by the IQM calibration snapshot."""
    return str([first, second])


def _validate_patch(patch: Patch) -> None:
    if len(patch.qubits) != 17 or len(set(patch.qubits)) != 17:
        raise ValueError(f"{patch.name}: expected exactly 17 unique qubits")
    if len(patch.cz_pairs) != 24:
        raise ValueError(f"{patch.name}: expected exactly 24 CZ pairs")
    if len({frozenset(pair) for pair in patch.cz_pairs}) != 24:
        raise ValueError(f"{patch.name}: CZ pairs must be unique, ignoring direction")
    patch_qubits = set(patch.qubits)
    invalid_pairs = [pair for pair in patch.cz_pairs if len(pair) != 2 or not set(pair) <= patch_qubits]
    if invalid_pairs:
        raise ValueError(f"{patch.name}: pair references a qubit outside the patch: {invalid_pairs}")


def _error(fidelity: float) -> float:
    return 1.0 - float(fidelity)


def score_patch(patch: Patch, calibration: Mapping[str, Mapping[str, float]]) -> PatchScore:
    """Compute requested component measurements for one patch."""
    _validate_patch(patch)
    required_metrics = ("CZ", "readout", "1Q", "t1", "t2")
    absent_metrics = [metric for metric in required_metrics if metric not in calibration]
    if absent_metrics:
        raise KeyError(f"Snapshot lacks calibration metrics: {', '.join(absent_metrics)}")

    cz, readout, one_qubit, t1, t2 = (
        calibration["CZ"],
        calibration["readout"],
        calibration["1Q"],
        calibration["t1"],
        calibration["t2"],
    )
    absent_pairs = [pair for pair in patch.cz_pairs if _pair_key(*pair) not in cz]
    absent_qubits = {
        metric: [qubit for qubit in patch.qubits if qubit not in values]
        for metric, values in (("readout", readout), ("1Q", one_qubit), ("t1", t1), ("t2", t2))
    }
    absent_qubits = {metric: values for metric, values in absent_qubits.items() if values}
    if absent_pairs or absent_qubits:
        raise KeyError(
            f"{patch.name}: incomplete snapshot; missing CZ pairs={absent_pairs}; "
            f"missing qubits={absent_qubits}"
        )

    return PatchScore(
        name=patch.name,
        cz_error_sum=sum(_error(cz[_pair_key(*pair)]) for pair in patch.cz_pairs),
        readout_error_sum=sum(_error(readout[qubit]) for qubit in patch.qubits),
        single_qubit_error_sum=sum(_error(one_qubit[qubit]) for qubit in patch.qubits),
        minimum_t1=min(float(t1[qubit]) for qubit in patch.qubits),
        minimum_t2=min(float(t2[qubit]) for qubit in patch.qubits),
    )


def _ranks(values: Sequence[float], *, higher_is_better: bool) -> list[int]:
    """Return stable one-based ranks; ties follow the input patch order."""
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=higher_is_better)
    result = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        result[index] = rank
    return result


def _z_scores(values: Sequence[float]) -> list[float]:
    spread = pstdev(values)
    if spread == 0:
        return [0.0] * len(values)
    average = mean(values)
    return [(value - average) / spread for value in values]


def rank_patches(
    patches: Iterable[Patch], calibration: Mapping[str, Mapping[str, float]]
) -> list[PatchScore]:
    """Rank all patches by components and an equal-weight, direction-aware z-score.

    Lower error sums improve the score; higher minimum T1/T2 improve it.  Equal
    weighting is intentional: it is a clear baseline, not a calibrated model.
    """
    raw_scores = [score_patch(patch, calibration) for patch in patches]
    if not raw_scores:
        raise ValueError("No patches supplied")

    cz = [score.cz_error_sum for score in raw_scores]
    readout = [score.readout_error_sum for score in raw_scores]
    one_qubit = [score.single_qubit_error_sum for score in raw_scores]
    t1 = [score.minimum_t1 for score in raw_scores]
    t2 = [score.minimum_t2 for score in raw_scores]

    combined = [
        -cz_z - readout_z - one_qubit_z + t1_z + t2_z
        for cz_z, readout_z, one_qubit_z, t1_z, t2_z in zip(
            _z_scores(cz),
            _z_scores(readout),
            _z_scores(one_qubit),
            _z_scores(t1),
            _z_scores(t2),
        )
    ]
    component_ranks = zip(
        _ranks(cz, higher_is_better=False),
        _ranks(readout, higher_is_better=False),
        _ranks(one_qubit, higher_is_better=False),
        _ranks(t1, higher_is_better=True),
        _ranks(t2, higher_is_better=True),
        _ranks(combined, higher_is_better=True),
        combined,
    )
    ranked = [
        replace(
            score,
            cz_rank=cz_rank,
            readout_rank=readout_rank,
            single_qubit_rank=one_qubit_rank,
            t1_rank=t1_rank,
            t2_rank=t2_rank,
            combined_rank=combined_rank,
            combined_z_score=combined_z_score,
        )
        for score, (
            cz_rank,
            readout_rank,
            one_qubit_rank,
            t1_rank,
            t2_rank,
            combined_rank,
            combined_z_score,
        ) in zip(raw_scores, component_ranks)
    ]
    return sorted(ranked, key=lambda score: score.combined_rank)


def select_representatives(ranking: Sequence[PatchScore]) -> dict[str, PatchScore]:
    """Choose P*, median, and worst from descending combined-score order."""
    if not ranking:
        raise ValueError("No ranking supplied")
    return {
        "P*": ranking[0],
        "median": ranking[(len(ranking) - 1) // 2],
        "worst": ranking[-1],
    }


def load_patches(path: str | Path) -> list[Patch]:
    """Load results/patches.json: [{"name": ..., "qubits": [...], "cz_pairs": [[...], ...]}]."""
    with Path(path).open(encoding="utf-8") as source:
        entries = json.load(source)
    return [
        Patch(
            name=entry["name"],
            qubits=tuple(entry["qubits"]),
            cz_pairs=tuple(tuple(pair) for pair in entry["cz_pairs"]),
        )
        for entry in entries
    ]


def load_snapshot(path: str | Path) -> Mapping[str, Mapping[str, float]]:
    """Extract the calibration metrics from a D1 snapshot JSON file."""
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)["calibration_data"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="D1 calibration snapshot JSON")
    parser.add_argument("patches", type=Path, help="patch-definition JSON (results/patches.json)")
    parser.add_argument("--output", type=Path, default=Path("patch_ranking.json"))
    arguments = parser.parse_args()

    ranking = rank_patches(load_patches(arguments.patches), load_snapshot(arguments.snapshot))
    report = {
        "ranking": [asdict(score) for score in ranking],
        "representatives": {name: asdict(score) for name, score in select_representatives(ranking).items()},
        "interpretation": "Calibration-only proxy; confirm or reject it with the later patch scan.",
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
