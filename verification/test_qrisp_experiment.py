"""Qrisp path (Garnet plan): the DetectorExperiment built by stim_to_detector_experiment
must have the no-reset shape and reproduce the direct no-reset Stim LER.

The experiment is built from the noisy WITH-reset circuit; stim_to_detector_experiment
does the no-reset rewrite itself.
"""
import numpy as np
import pymatching
import pytest

from stim_to_qrisp import stim_to_detector_experiment
from surface_code import DEFAULT_NOISE, make_stim_circuit

D = 3
ROUNDS = [1, 2, 3, 4, 5]
SHOTS = 20_000


def experiment(rounds):
    return stim_to_detector_experiment(make_stim_circuit(D, rounds, no_reset=False, noise=DEFAULT_NOISE), no_reset=True)


def direct_ler(rounds, shots):
    c = make_stim_circuit(D, rounds, no_reset=True, noise=DEFAULT_NOISE)
    matcher = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    det, obs = c.compile_detector_sampler(seed=rounds).sample(shots, separate_observables=True)
    return float(np.mean(np.any(matcher.decode_batch(det) != obs, axis=1)))


@pytest.mark.parametrize("rounds", ROUNDS)
def test_experiment_shape(rounds):
    s = experiment(rounds).to_stim()
    assert s.num_detectors == 8 * rounds
    assert s.num_measurements == 8 * rounds + 9
    assert s.num_observables == 1


@pytest.mark.parametrize("rounds", ROUNDS)
def test_experiment_ler_matches_direct_stim(rounds):
    p_exp = experiment(rounds).compute_LER(shots=SHOTS)
    p_dir = direct_ler(rounds, SHOTS)
    se = np.sqrt(p_exp * (1 - p_exp) / SHOTS + p_dir * (1 - p_dir) / SHOTS)
    assert abs(p_exp - p_dir) < 3 * se, f"rounds={rounds}: experiment {p_exp:.4f} vs direct {p_dir:.4f} (3se={3*se:.4f})"
