"""Calibration-informed circuit noise proxy for the saved Garnet P00 layout.

This preserves each physical-qubit and CZ-pair calibration while attaching
Stim-compatible channels to the d=3 circuit. Idle noise is deliberately absent:
the calibration snapshots do not include gate or measurement durations.
"""
from __future__ import annotations

import json
from pathlib import Path

import stim


def _pair_key(first: str, second: str) -> str:
    return str([first, second])


def _fidelity(mapping: dict[str, float], first: str, second: str) -> float:
    for key in (_pair_key(first, second), _pair_key(second, first)):
        if key in mapping:
            return float(mapping[key])
    raise KeyError(f"Missing CZ calibration for {first}, {second}")


def p00_stim_to_qb(root: Path) -> dict[int, str]:
    table = json.loads((root / "results" / "layout_table.json").read_text(encoding="utf-8"))
    return {int(row["stim"]): row["qb"] for row in table["qubits"] if row["stim"] is not None}


def add_calibrated_proxy_noise(
    circuit: stim.Circuit,
    snapshot: str | Path,
    stim_to_qb: dict[int, str],
    *,
    include_cz: bool = True,
    include_single_qubit: bool = True,
    include_readout: bool = True,
) -> stim.Circuit:
    """Attach calibration-derived CZ, 1Q, and readout channels to a no-reset circuit.

    A Stim CX is treated as one native CZ plus two Hadamards, matching the
    project's hardware decomposition. The inserted 1Q channels therefore
    include both explicit H instructions and the two H operations implicit in
    each CX. This is a simulation proxy, not a duration-calibrated idle model.
    """
    raw = json.loads(Path(snapshot).read_text(encoding="utf-8"))["calibration_data"]
    one_qubit, cz, readout = raw["1Q"], raw["CZ"], raw["readout"]
    result = stim.Circuit()

    for instruction in circuit.flattened():
        name = instruction.name
        targets = instruction.targets_copy()
        qubits = [target.value for target in targets if target.is_qubit_target]

        if name == "M" and include_readout:
            for qubit in qubits:
                qb = stim_to_qb[qubit]
                result.append("X_ERROR", [qubit], 1.0 - float(readout[qb]))

        result.append(name, targets, instruction.gate_args_copy())

        if name == "H" and include_single_qubit:
            for qubit in qubits:
                qb = stim_to_qb[qubit]
                result.append("DEPOLARIZE1", [qubit], 1.5 * (1.0 - float(one_qubit[qb])))

        if name == "CX":
            if len(qubits) % 2:
                raise ValueError("CX instruction must contain qubit pairs")
            for first, second in zip(qubits[::2], qubits[1::2]):
                first_qb, second_qb = stim_to_qb[first], stim_to_qb[second]
                if include_cz:
                    result.append("DEPOLARIZE2", [first, second],
                                  1.25 * (1.0 - _fidelity(cz, first_qb, second_qb)))
                if include_single_qubit:
                    result.append("DEPOLARIZE1", [first, second],
                                  1.5 * (1.0 - float(one_qubit[first_qb])))
                    result.append("DEPOLARIZE1", [second],
                                  1.5 * (1.0 - float(one_qubit[second_qb])))
    return result
