"""Circuit-correctness tests for the d=3 no-reset Z-memory experiment (Tasks 2-4)."""
import pytest
import stim

from surface_code import DEFAULT_NOISE, build_pair, make_stim_circuit, to_qiskit_hw
from internal_helpers import get_meas_order

D = 3
ROUNDS = [1, 2, 3, 4, 5]
SHOTS = 2_000


@pytest.mark.parametrize("rounds", ROUNDS)
def test_noiseless_detectors_and_observable_silent(rounds):
    c = make_stim_circuit(D, rounds, no_reset=True)
    dets, obs = c.compile_detector_sampler().sample(SHOTS, separate_observables=True)
    assert dets.shape == (SHOTS, 8 * rounds) and not dets.any()
    assert obs.shape == (SHOTS, 1) and not obs.any()


@pytest.mark.parametrize("rounds", ROUNDS)
def test_counts(rounds):
    c = make_stim_circuit(D, rounds, no_reset=True)
    assert c.num_detectors == 8 * rounds
    assert c.num_measurements == 8 * rounds + 9
    assert c.num_observables == 1
    # 17 physical qubits; c.num_qubits is 26 because Stim indices are sparse
    assert len(c.get_final_qubit_coordinates()) == 17


@pytest.mark.parametrize("rounds", ROUNDS)
@pytest.mark.parametrize("no_reset", [True, False])
def test_circuit_level_distance(rounds, no_reset):
    c = make_stim_circuit(D, rounds, no_reset=no_reset, noise=DEFAULT_NOISE)
    assert len(c.shortest_graphlike_error()) == 3


@pytest.mark.parametrize("rounds", [2, 3, 4, 5])
def test_naive_mr_to_m_is_non_deterministic(rounds):
    # Negative control: drop the resets WITHOUT rewriting detectors -> Stim must refuse.
    # rounds=1 is excluded on purpose: each ancilla is measured once, so the reset and
    # no-reset circuits are identical and the naive swap is (correctly) still valid.
    naive = stim.Circuit(str(make_stim_circuit(D, rounds, no_reset=False).flattened()).replace("MR ", "M "))
    with pytest.raises(ValueError, match="non-deterministic detectors"):
        naive.detector_error_model()


@pytest.mark.parametrize("rounds", ROUNDS)
def test_no_reset_rewrite_matches_stim_to_qrisp(rounds):
    # to_noreset() must implement the same algebra as stim_to_qrisp._analyse()
    from stim_to_qrisp import _analyse

    reset_flat = make_stim_circuit(D, rounds, no_reset=False).flattened()
    _, qrisp_dets, qrisp_obs = _analyse(reset_flat, no_reset=True)

    ours, dets, obs, n = make_stim_circuit(D, rounds, no_reset=True), [], [], 0
    for inst in ours.flattened():
        if inst.name == "M":
            n += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            dets.append(sorted(n + t.value for t in inst.targets_copy()))
        elif inst.name == "OBSERVABLE_INCLUDE":
            obs.extend(n + t.value for t in inst.targets_copy())
    assert "MR" not in str(ours)
    assert dets == qrisp_dets
    assert sorted(obs) == qrisp_obs[0]


@pytest.mark.parametrize("rounds", ROUNDS)
def test_build_pair_twins_match(rounds):
    hw, twin = build_pair(D, rounds, DEFAULT_NOISE)
    assert hw.get_detector_coordinates() == twin.get_detector_coordinates()
    assert str(hw.without_noise()) == str(twin.without_noise())
    assert str(hw) == str(hw.without_noise())          # hw really is noiseless
    assert twin.detector_error_model().num_errors > 0  # twin really is noisy


@pytest.mark.parametrize("rounds", ROUNDS)
def test_to_qiskit_hw(rounds):
    hw, _ = build_pair(D, rounds)
    qc, stim_to_dense, meas_order = to_qiskit_hw(hw)
    ops = qc.count_ops()
    assert "reset" not in ops
    assert qc.num_qubits == 17 and len(stim_to_dense) == 17
    assert qc.num_clbits == hw.num_measurements == ops["measure"]
    assert meas_order == get_meas_order(hw)
