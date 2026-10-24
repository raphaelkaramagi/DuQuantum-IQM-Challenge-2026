"""Turn an IQM calibration snapshot into a per-patch Stim noise model.

The returned dictionary is intentionally explicit rather than the uniform
``noise`` dictionary accepted by ``stim.Circuit.generated``.  A caller that
constructs or rewrites the circuit can attach each entry to its physical qubit
or pair.  This preserves the patch-specific calibration information.

For the compiler decomposition used by this project, one Stim CX is modeled as
one native CZ plus two single-qubit Hadamards.  The model therefore provides:

* ``DEPOLARIZE2(1.25 * r_CZ)`` on its CZ pair;
* two ``DEPOLARIZE1(1.5 * r_1Q)`` channels, one for each Hadamard;
* ``X_ERROR(readout_error)`` immediately before each measurement.

``p_idle`` deliberately remains a caller-supplied parameter.  Calibration T1
and T2 values are retained as metadata but are not converted to an idle error
without the unavailable measurement/gate durations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _patch_field(layout: Any, name: str) -> Any:
    """Read a layout field from a mapping or an object such as D2's Patch."""
    if isinstance(layout, Mapping):
        return layout[name]
    return getattr(layout, name)


def _pair_key(first: str, second: str) -> str:
    return str([first, second])


def _as_probability(value: float, *, label: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1, got {probability}")
    return probability


def _fidelity_to_depolarize1(fidelity: float) -> float:
    """Convert average 1Q gate infidelity r to Stim DEPOLARIZE1 probability."""
    return _as_probability(1.5 * (1.0 - float(fidelity)), label="DEPOLARIZE1 probability")


def _fidelity_to_depolarize2(fidelity: float) -> float:
    """Convert average CZ infidelity r to Stim DEPOLARIZE2 probability."""
    return _as_probability(1.25 * (1.0 - float(fidelity)), label="DEPOLARIZE2 probability")


def _load_calibration(calib: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    """Accept either raw calibration data or a complete D1 snapshot path."""
    if isinstance(calib, (str, Path)):
        with Path(calib).open(encoding="utf-8") as source:
            calib = json.load(source)
    return calib.get("calibration_data", calib)


def noise_for_patch(
    layout: Mapping[str, Any] | Any,
    calib: Mapping[str, Any] | str | Path,
    *,
    p_idle: float | None = None,
) -> dict[str, Any]:
    """Return a calibration-derived, physical-qubit Stim noise specification.

    Args:
        layout: D2-style object/mapping with ``qubits`` and ``cz_pairs``.  Pair
            order is preserved because the saved calibration is directed.
        calib: D1's ``calibration_data`` mapping, full snapshot mapping, or its
            JSON path.
        p_idle: Optional trial value from the decoder prior sweep. If absent, idle noise stays
            unset because duration data is not available.

    Returns:
        A serializable dictionary. Attach each listed operation at its precise
        location in a custom Stim circuit/rewrite; do not collapse it to an
        average, which would discard the patch ranking information.
    """
    qubits = tuple(_patch_field(layout, "qubits"))
    cz_pairs = tuple(tuple(pair) for pair in _patch_field(layout, "cz_pairs"))
    calibration = _load_calibration(calib)
    required = ("CZ", "1Q", "readout")
    absent = [metric for metric in required if metric not in calibration]
    if absent:
        raise KeyError(f"Calibration data lacks: {', '.join(absent)}")
    if p_idle is not None:
        p_idle = _as_probability(p_idle, label="p_idle")

    invalid_pairs = [pair for pair in cz_pairs if len(pair) != 2 or not set(pair) <= set(qubits)]
    if invalid_pairs:
        raise ValueError(f"CZ pairs must contain two qubits from layout: {invalid_pairs}")

    cz, one_qubit, readout = calibration["CZ"], calibration["1Q"], calibration["readout"]
    missing_qubits = [qubit for qubit in qubits if qubit not in one_qubit or qubit not in readout]
    missing_pairs = [pair for pair in cz_pairs if _pair_key(*pair) not in cz]
    if missing_qubits or missing_pairs:
        raise KeyError(f"Missing calibration values; qubits={missing_qubits}, CZ pairs={missing_pairs}")

    single_qubit_noise = {
        qubit: {
            "DEPOLARIZE1": _fidelity_to_depolarize1(one_qubit[qubit]),
            "X_ERROR_before_measurement": _as_probability(
                1.0 - float(readout[qubit]), label=f"readout error for {qubit}"
            ),
        }
        for qubit in qubits
    }
    cx_noise = {
        _pair_key(*pair): {
            "qubits": list(pair),
            "DEPOLARIZE2_after_CZ": _fidelity_to_depolarize2(cz[_pair_key(*pair)]),
            # A Stim CX is treated as CZ plus H on each participating qubit.
            "DEPOLARIZE1_after_H_on_first_qubit": single_qubit_noise[pair[0]]["DEPOLARIZE1"],
            "DEPOLARIZE1_after_H_on_second_qubit": single_qubit_noise[pair[1]]["DEPOLARIZE1"],
        }
        for pair in cz_pairs
    }

    coherence = {
        metric: {qubit: float(calibration[metric][qubit]) for qubit in qubits}
        for metric in ("t1", "t2")
        if metric in calibration
    }
    return {
        "model": "calibration-derived proxy; validate against the patch scan",
        "qubits": list(qubits),
        "cz_pairs": [list(pair) for pair in cz_pairs],
        "single_qubit": single_qubit_noise,
        "cx": cx_noise,
        "p_idle": p_idle,
        "coherence_metadata_us": coherence,
        "notes": [
            "DEPOLARIZE2 p = 1.25 * CZ average gate infidelity.",
            "DEPOLARIZE1 p = 1.5 * 1Q average gate infidelity.",
            "X_ERROR uses the readout error immediately before measurement.",
            "A Stim CX is modeled as CZ plus two Hadamards, so each CX includes both 1Q terms.",
            "p_idle is intentionally open until C supplies a duration-calibrated sweep value.",
        ],
    }

