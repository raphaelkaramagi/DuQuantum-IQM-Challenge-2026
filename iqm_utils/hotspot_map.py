"""Describe detector hot spots for a physical d=3 patch.

The analysis is intentionally descriptive: a d=3 rotated patch has only eight
ancillas, so Spearman correlations are useful for triage, not evidence of a
reliable physical relationship.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _get(layout: Any, field: str) -> Any:
    if isinstance(layout, Mapping):
        return layout[field]
    return getattr(layout, field)


def _load_calibration(calib: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(calib, (str, Path)):
        with Path(calib).open(encoding="utf-8") as source:
            calib = json.load(source)
    return calib.get("calibration_data", calib)


def _pair_key(first: str, second: str) -> str:
    return str([first, second])


def _cz_fidelity(cz: Mapping[str, float], first: str, second: str) -> float:
    """Find a pair calibration, accepting a reverse key when necessary."""
    for key in (_pair_key(first, second), _pair_key(second, first)):
        if key in cz:
            return float(cz[key])
    raise KeyError(f"No CZ calibration for {first}, {second}")


def detector_to_ancilla(hw: Any) -> dict[int, int]:
    """Map each detector ID to a Stim ancilla by matching its x/y coordinates.

    Stim detector coordinates include time as an optional third component.  The
    first two components identify the stabilizer/ancilla; time is deliberately
    ignored so that all rounds aggregate onto one physical ancilla.
    """
    qubit_coordinates = hw.get_final_qubit_coordinates()
    coordinate_to_stim = {
        tuple(map(float, coordinates[:2])): int(stim_qubit)
        for stim_qubit, coordinates in qubit_coordinates.items()
    }
    result = {}
    for detector_id, coordinates in hw.get_detector_coordinates().items():
        coordinate = tuple(map(float, coordinates[:2]))
        if coordinate not in coordinate_to_stim:
            raise KeyError(
                f"Detector {detector_id} at {coordinate} does not match a circuit qubit coordinate"
            )
        result[int(detector_id)] = coordinate_to_stim[coordinate]
    return result


def analyze_hotspots(
    syndrome_result: Mapping[str, Any],
    hw: Any,
    layout: Mapping[str, Any] | Any,
    calib: Mapping[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """Aggregate detector rates per physical ancilla and attach calibration errors.

    ``layout`` must supply:

    * ``stim_to_physical``: Stim-index-to-``QB`` mapping (integer or string keys)
    * ``cz_pairs``: physical pairs used by this patch
    * ``grid``: a 5 by 5 list of QB labels (or ``None`` for unused locations)
    """
    firing_rates = np.asarray(syndrome_result["detector_firing_rate"], dtype=float)
    detector_map = detector_to_ancilla(hw)
    if len(firing_rates) != len(detector_map):
        raise ValueError("Firing-rate count does not match the circuit's detector count")

    raw_mapping = _get(layout, "stim_to_physical")
    stim_to_physical = {int(stim): str(physical) for stim, physical in raw_mapping.items()}
    cz_pairs = [tuple(map(str, pair)) for pair in _get(layout, "cz_pairs")]
    calibration = _load_calibration(calib)
    cz, readout = calibration["CZ"], calibration["readout"]

    rates_by_ancilla: dict[str, list[float]] = defaultdict(list)
    for detector_id, stim_ancilla in detector_map.items():
        if stim_ancilla not in stim_to_physical:
            raise KeyError(f"B's layout has no physical mapping for ancilla Stim index {stim_ancilla}")
        rates_by_ancilla[stim_to_physical[stim_ancilla]].append(float(firing_rates[detector_id]))

    rows = []
    for physical, rates in sorted(rates_by_ancilla.items()):
        if physical not in readout:
            raise KeyError(f"No readout calibration for {physical}")
        incident_pairs = [pair for pair in cz_pairs if physical in pair]
        if not incident_pairs:
            raise ValueError(f"Ancilla {physical} has no incident CZ pair in the supplied layout")
        cz_errors = [1.0 - _cz_fidelity(cz, *pair) for pair in incident_pairs]
        rows.append(
            {
                "physical_qubit": physical,
                "detector_count": len(rates),
                "mean_firing_rate": float(np.mean(rates)),
                "readout_error": 1.0 - float(readout[physical]),
                "mean_incident_cz_error": float(np.mean(cz_errors)),
            }
        )
    return rows


def spearman_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Correlate eight-ancilla firing rates with readout and incident CZ error."""
    from scipy.stats import spearmanr

    rates = [row["mean_firing_rate"] for row in rows]
    comparisons = {}
    for metric in ("readout_error", "mean_incident_cz_error"):
        result = spearmanr(rates, [row[metric] for row in rows])
        comparisons[metric] = {"rho": float(result.statistic), "p_value": float(result.pvalue)}
    return comparisons


def plot_hotspot_map(
    rows: Sequence[Mapping[str, Any]],
    grid: Sequence[Sequence[str | None]],
    *,
    output_path: str | Path,
) -> Path:
    """Plot 5×5 physical grid labels and mean ancilla detector firing rates."""
    import matplotlib.pyplot as plt

    if len(grid) != 5 or any(len(row) != 5 for row in grid):
        raise ValueError("D5 expects B's 5 by 5 grid")
    rate_by_qubit = {row["physical_qubit"]: row["mean_firing_rate"] for row in rows}
    values = np.array([[rate_by_qubit.get(qubit, np.nan) for qubit in row] for row in grid])

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(np.ma.masked_invalid(values), cmap="magma", vmin=0)
    figure.colorbar(image, ax=axis, label="Mean detector firing rate")
    axis.set_xticks(range(5), labels=[str(index) for index in range(5)])
    axis.set_yticks(range(5), labels=[str(index) for index in range(5)])
    axis.set_xlabel("Grid column")
    axis.set_ylabel("Grid row")
    axis.set_title("Ancilla detector hot spots (descriptive; n ≤ 8)")
    for row_index, grid_row in enumerate(grid):
        for column_index, qubit in enumerate(grid_row):
            if qubit is None:
                axis.text(column_index, row_index, "—", ha="center", va="center", color="gray")
            elif qubit in rate_by_qubit:
                axis.text(
                    column_index,
                    row_index,
                    f"{qubit}\n{rate_by_qubit[qubit]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
            else:
                axis.text(column_index, row_index, qubit, ha="center", va="center", color="black", fontsize=8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def write_hotspot_report(
    syndrome_result: Mapping[str, Any],
    hw: Any,
    layout: Mapping[str, Any] | Any,
    calib: Mapping[str, Any] | str | Path,
    *,
    image_path: str | Path = "hotspot_map.png",
    report_path: str | Path = "hotspot_report.json",
) -> dict[str, Any]:
    """Create the D5 heat map and a machine-readable descriptive report."""
    rows = analyze_hotspots(syndrome_result, hw, layout, calib)
    image = plot_hotspot_map(rows, _get(layout, "grid"), output_path=image_path)
    report = {
        "ancillas": rows,
        "spearman": spearman_comparisons(rows),
        "interpretation": "Descriptive only: a d=3 patch has at most eight ancillas, so correlation is not inferential evidence.",
        "heat_map": str(image),
    }
    destination = Path(report_path)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
