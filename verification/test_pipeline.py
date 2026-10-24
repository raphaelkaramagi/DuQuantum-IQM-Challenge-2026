"""End-to-end pipeline tests (Task 5): Qiskit/Aer -> counts -> syndromes -> decoder.

Collaborator APIs used here
---------------------------
B : counts_to_measurement_array(counts, num_measurements, total_shots) -> (shots, n_meas) bool
    (currently in internal_helpers.py)
C : decode_events(det_events, obs_flips, circuit) -> P  or (P, stderr)
    decode_ler(circuit, shots)                    -> P  or (P, stderr)
    ASSUMED signatures, not yet in the repo. C: if yours differ, adjust the two
    calls in test_roundtrip_matches_direct_stim (and add your module to TEAM_MODULES).
"""
import importlib
from collections import Counter

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from extract_syndromes import extract_syndromes
from surface_code import DEFAULT_NOISE, build_pair, to_qiskit_hw

D = 3
ROUNDS = [1, 2, 3, 4, 5]
TEAM_MODULES = ["decoder", "decoding", "decode", "run_on_hardware", "surface_code", "internal_helpers"]


def team_import(name):
    """Find a collaborator's function in the repo; fail clearly if it isn't there yet."""
    for mod in TEAM_MODULES:
        try:
            fn = getattr(importlib.import_module(mod), name, None)
        except ImportError:
            continue
        if fn is not None:
            return fn
    pytest.fail(f"{name}() not found in any of {TEAM_MODULES}; waiting on its owner", pytrace=False)


def p_and_se(result, shots):
    if isinstance(result, tuple):
        return float(result[0]), float(result[1])
    p = float(result)
    return p, np.sqrt(p * (1 - p) / shots)


@pytest.mark.parametrize("rounds", ROUNDS)
def test_noiseless_aer_all_silent(rounds):
    counts_to_measurement_array = team_import("counts_to_measurement_array")
    shots = 1_000
    hw, _ = build_pair(D, rounds)
    qc, _, _ = to_qiskit_hw(hw)

    counts = AerSimulator(method="stabilizer").run(qc, shots=shots, seed_simulator=1).result().get_counts()
    raw = counts_to_measurement_array(counts, hw.num_measurements, shots)
    syn = extract_syndromes(raw, hw, print_summary=False)

    # X-ancilla outcomes are random, so the test has teeth: a bit-order or
    # measurement-order bug would light up detectors.
    assert raw.any(axis=0).sum() >= 4
    assert not syn["det_events"].any()
    assert not syn["obs_flips"].any()


@pytest.mark.parametrize("rounds", ROUNDS)
def test_roundtrip_matches_direct_stim(rounds):
    counts_to_measurement_array = team_import("counts_to_measurement_array")
    decode_events = team_import("decode_events")
    decode_ler = team_import("decode_ler")
    shots = 20_000
    hw, twin = build_pair(D, rounds, DEFAULT_NOISE)

    # fake a Qiskit result: rightmost character is classical bit 0
    meas = twin.compile_sampler(seed=rounds).sample(shots)
    counts = Counter("".join("1" if b else "0" for b in row[::-1]) for row in meas)

    raw = counts_to_measurement_array(dict(counts), hw.num_measurements, shots)
    syn = extract_syndromes(raw, hw, print_summary=False)
    p_pipe, se_pipe = p_and_se(decode_events(syn["det_events"], syn["obs_flips"], twin), shots)
    p_stim, se_stim = p_and_se(decode_ler(twin, shots), shots)

    assert abs(p_pipe - p_stim) <= 3 * np.hypot(se_pipe, se_stim), (p_pipe, p_stim)
