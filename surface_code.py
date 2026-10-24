# -*- coding: utf-8 -*-

from internal_helpers import *

"""
surface_code.py

Circuit generation for a rotated surface code memory experiment (Z or X basis)
on IQM Emerald. By default distance 3: 9 data + 8 ancilla = 17 qubits.

Pipeline via Qrisp OR Qiskit
------------------------------------
  Qrisp  : circuit → Qrisp DetectorExperiment → IQM Emerald
           → detection events → decoder
  Qiskit : circuit → Qiskit circuit → IQM Emerald → raw measurements
           → extract_syndromes() → decoder


Generators (this file): pick ONE starting point
-----------------------------------------------
  make_stim_circuit()      For both Qrisp and Qiskit. Use one of the converters (VERIFY their funcionality FIRST!) to run on hardware, as Stim circuits dont work.
                           (stim_to_qiskit.py / stim_to_qrisp.py)
  make_qiskit_circuit()    Qiskit-native alternative
  make_qrisp_experiment()  Qrisp-native alternative

Where the other stages live
---------------------------
  1. Circuit generation              this file
  2. Stim → Qiskit / Qrisp           stim_to_qiskit.py, stim_to_qrisp.py
  3. OPTIONAL Emerald qubit mapping  build_emerald_qubit_rotated.py  (only needed for Qiskit)
  4. Simulation                      TBD (section 4 below)
  5. Hardware execution              run_on_hardware.py: run_qiskit() / run_qrisp()
  6. Syndromes → Decoder             extract_syndromes.py (only needed for Qiskit),
                                     run_on_hardware.py: decode_hardware_results()
  Helpers                            internal_helpers.py

Usage
-----
  from surface_code import make_stim_circuit, DEFAULT_NOISE
  from run_on_hardware import run_qiskit, run_qrisp, decode_hardware_results


"""

import numpy as np
import stim
import pymatching
from qiskit import QuantumCircuit, transpile
from typing import Optional

from noreset_tools import to_noreset
from stim_to_qiskit import stim_to_qiskit


# ─────────────────────────────────────────────────────────────────────────────
#  1.  STIM CIRCUIT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

#: Default noise model matching IQM Emerald typical gate error rates (~0.1-0.5%).
#: Adjust these based on your latest calibration data from Resonance.
DEFAULT_NOISE = dict(
    after_clifford_depolarization  = 0.003,
    after_reset_flip_probability   = 0.003,
    before_measure_flip_probability= 0.003,
    before_round_data_depolarization = 0.003,
)


def make_stim_circuit(
    distance: int = 3,
    rounds: int = 1,
    basis: str = "Z",
    no_reset: bool = True,
    noise: Optional[dict] = None,
) -> stim.Circuit:
    """
    Generates a Stim circuit for the rotated surface code.

    Shared starting point for BOTH tracks. A Stim circuit does not run on hardware, you need to convert it first
    first:
        Qiskit track : stim_to_qiskit(stim_circ)               (stim_to_qiskit.py)
        Qrisp track  : stim_to_detector_experiment(stim_circ)  (stim_to_qrisp.py)
    Alternatively, you can start from make_qiskit_circuit() / make_qrisp_experiment().

    Parameters
    ----------
    distance : code distance
    rounds   : number of stabilizer rounds
    basis    : "Z" or "X" memory experiment NOTE: Stim's X-memory circuits use RX / MX,
             which the provided converters do not handle. Possible avenue for extension.
    no_reset : True -> ancillas are measured without reset (required on Emerald)
    noise    : None for the hardware circuit. For for simulation / DEM it can be  e.g. DEFAULT_NOISE or sophisticated choices.

    NOTE: Emerald does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant.
    With reset: ancilla starts fresh each round. Raw measurement m_k = syndrome s_k of round k. Detection event = s_k XOR s_(k-1) = m_k XOR m_(k-1).
    Without reset: ancilla accumulates. Raw measurement m_k = s_k XOR m_(k-1), i.e. the XOR of all syndromes up to round k.
    The detection event is the same quantity (s_k XOR s_(k-1)), but in raw measurements it becomes m_k XOR m_(k-2)  (with m_0 = 0).
    So the DETECTOR definitions must be rewritten for the no-reset circuit. stim_to_qrisp.py does this rewrite for you (PLEASE VERIFY!); on the Qiskit track it is on you.
    Trade-off: no-reset avoids reset errors (X_ERROR after R), but the ancilla stays "live" (potentially decohering) throughout, which causes amplitude damping.


    Returns
    -------
    stim.Circuit   (flattened when no_reset=True). For d=3: 8*rounds detectors,
                   8*rounds + 9 measurements, 17 qubits used (num_qubits is 26
                   because Stim indices are sparse), 1 observable.
    """
    if basis.upper() != "Z":
        raise NotImplementedError("X basis needs RX/MX support in the converters")
    n = noise or dict(after_clifford_depolarization=0, after_reset_flip_probability=0,
                      before_measure_flip_probability=0, before_round_data_depolarization=0)
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=rounds, **n)
    return to_noreset(c) if no_reset else c


def build_pair(
    distance: int = 3,
    rounds: int = 1,
    noise: Optional[dict] = None,
) -> tuple[stim.Circuit, stim.Circuit]:
    """
    Returns (hw, twin): the same no-reset Z-memory circuit twice.

    hw   : noiseless. This is what runs on hardware (via to_qiskit_hw) and what
           extract_syndromes() / m2d conversion should be given.
    twin : identical circuit with noise channels (default DEFAULT_NOISE).
           Use it for simulation and for building the DEM / matching graph.

    hw is derived as twin.without_noise(), so both have identical gates,
    measurement order and detector definitions (Stim formats p=0 and p>0
    circuits differently, so generating them separately would not be).
    """
    twin = make_stim_circuit(distance, rounds, basis="Z", no_reset=True, noise=noise or DEFAULT_NOISE)
    hw = twin.without_noise()
    assert hw.get_detector_coordinates() == twin.get_detector_coordinates()
    assert str(hw.without_noise()) == str(twin.without_noise())
    return hw, twin


def to_qiskit_hw(hw: stim.Circuit) -> tuple[QuantumCircuit, dict, list]:
    """
    Converts the hardware circuit from build_pair() into Qiskit.

    Initial R instructions are dropped first: qubits start in |0>, so the
    detectors stay deterministic without them, and Emerald gets no reset ops.
    A reset on an already-used qubit raises ValueError (Emerald cannot do it).

    Returns
    -------
    qc            : QuantumCircuit, dense qubits 0..16, one clbit per measurement
                    (clbit i = Stim measurement i)
    stim_to_dense : dict  stim qubit index -> dense Qiskit index
    meas_order    : list  stim qubit index of each measurement, in order
    """
    stripped, touched = stim.Circuit(), set()
    for inst in hw.flattened():
        qubits = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name == "R":
            late = [q for q in qubits if q in touched]
            if late:
                raise ValueError(f"mid-circuit reset on qubits {late}; use the no-reset circuit")
            continue
        if inst.name != "QUBIT_COORDS":
            touched.update(qubits)
        stripped.append(inst)
    return stim_to_qiskit(stripped)

def get_circuit_info(circuit: stim.Circuit) -> dict:
    """
    Returns a summary dict of qubit counts and measurement structure.
    Useful for sanity-checking before hardware runs.
    """
    data_q, anc_q = get_qubit_lists(circuit)
    return {
        "num_qubits"      : circuit.num_qubits,
        "num_data_qubits" : len(data_q),
        "num_anc_qubits"  : len(anc_q),
        "num_measurements": circuit.num_measurements,
        "num_detectors"   : circuit.num_detectors,
        "num_observables" : circuit.num_observables,
        "data_stim_indices": sorted(data_q),
        "anc_stim_indices" : sorted(anc_q),
        "meas_order"      : get_meas_order(circuit),
    }


def make_qiskit_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
) -> QuantumCircuit:
    """
    QISKIT TRACK, native alternative.
    Generates a Qiskit circuit for the rotated surface code.

    Pros (vs. Stim + converter)
    ---------------------------
    + Full control over gate basis (e.g. IQM native) and order (e.g. qubit placement). 
    + No converter before it is passed to the transpiler.

    Cons
    ----
    - No detectors or logical observables: Tracking needs to be done by user, across rounds and
      for the no-reset case.
    - No detector error model: for decoding you still need a Stim version of your circuit
      (or build the decoding graph by hand). extract_syndromes() only works with a
      Stim circuit that matches yours.

    NOTE: Emerald does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant.
    With reset: ancilla starts fresh each round. Raw measurement m_k = syndrome s_k of round k. Detection event = s_k XOR s_(k-1) = m_k XOR m_(k-1).
    Without reset: ancilla accumulates. Raw measurement m_k = s_k XOR m_(k-1), i.e. the XOR of all syndromes up to round k.
    The detection event is the same quantity (s_k XOR s_(k-1)), but in raw measurements it changes to m_k XOR m_(k-2)  (with m_0 = 0).


    Returns
    -------
    qiskit.QuantumCircuit
    """

    return

def make_qrisp_experiment(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    p: Optional[float] = None,
):
    """
    QRISP TRACK, native alternative to make_stim_circuit() + stim_to_detector_experiment().
    Generates a Qrisp DetectorExperiment for the rotated surface code.

    Pros (vs. Stim + converter)
    ---------------------------
    + Detectors and observables are part of the experiment (qrisp.parity). Qrisp derives the Stim circuit and DEM from it.
    + Noise model per gate via stim_noise annotations: easy to adjust according to calibration data.
    + Full control over gate choice, CX/CZ schedule and layout.

    Cons
    ----
    - Geometry, CX schedule and detector bookkeeping: Qrisp has no
      surface-code generator.
    - Expectation values must be correct. Wrong choice is Invisible in a
      noiseless Stim simulation, but breaks the hardware results.
    - The no-reset detector algebra (which raw measurements form a detector) needs to be implemented.

    Parameters
    ----------
    distance, rounds, basis, no_reset : as in make_stim_circuit()
    p        : noise strength for stim_noise annotations (decoder weights only;
               ignored on hardware). None -> noiseless.

    Returns
    -------
    iqm.qrisp_iqm.DetectorExperiment
    """

    return


##########################
# Below the necessary steps. No need to put everything in this file. 
##########################

# ─────────────────────────────────────────────────────────────────────────────
#  2.  STIM → Qrisp/QISKIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  3.  EMERALD QUBIT MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  4.  STIM/AER SIMULATION  (no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  5.  Hardware execution
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  6.  Offline Decoding
# ─────────────────────────────────────────────────────────────────────────────

# TBD