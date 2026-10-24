"""Layout and hardware-path tests (no QPU).

- layout.py on synthetic square grids: 24 pairs, cz == 24 * rounds, no SWAP, every
  two-qubit gate on an edge, agreement with IQM's vf2pp_layout, no patch on a too-small grid.
- Noiseless dry run on the saved Garnet connectivity through IQM's real converter and
  result parser (dryrun.FakeIQMBackend): zero detector firing and zero observable flips.
"""
import pytest
from iqm.qrisp_iqm import vf2pp_layout
from qrisp import PassManager, convert_to_cz, convert_to_prx

from dryrun import FakeIQMBackend, garnet_connectivity, grid
from layout import cx_pairs, enumerate_patches, transpile_qrisp, verify_layout
from run_on_hardware import build_experiment, run_batch


@pytest.mark.parametrize("rows, cols, n_patches", [(5, 5, 2), (6, 9, 20), (4, 5, 0)])
def test_patch_count_on_synthetic_grid(rows, cols, n_patches):
    pairs = cx_pairs(build_experiment(1).to_qc())
    assert len(pairs) == 24
    assert len(enumerate_patches(pairs, grid(rows, cols))) == n_patches


@pytest.mark.parametrize("rounds", [1, 3, 5])
@pytest.mark.parametrize("fuse", [True, False])
def test_fixed_layout_is_swap_free(rounds, fuse):
    conn = grid(5, 5)
    qc = build_experiment(rounds).to_qc()
    layout = enumerate_patches(cx_pairs(qc), conn)[0]["layout"]
    qc_t, _ = transpile_qrisp(qc, layout, 25, fuse)
    rep = verify_layout(qc_t, conn, expect_cz=24 * rounds)
    assert rep["swap"] == 0 and rep["off_edge"] == 0 and len(rep["physical_qubits_used"]) == 17


def test_matches_iqm_vf2pp_layout():
    conn = grid(5, 5)
    pm = PassManager()
    pm += vf2pp_layout(conn)
    pm += convert_to_cz()
    pm += convert_to_prx
    rep = verify_layout(pm.run(build_experiment(1).to_qc()), conn, expect_cz=24)
    patch_sets = [sorted(p["qubits"]) for p in enumerate_patches(cx_pairs(build_experiment(1).to_qc()), conn)]
    assert rep["physical_qubits_used"] in patch_sets


def test_noiseless_dry_run_on_garnet_edges():
    conn, _ = garnet_connectivity()
    patch = enumerate_patches(cx_pairs(build_experiment(1).to_qc()), conn)[0]
    for out in run_batch([1, 2], 200, patch["layout"], backend=FakeIQMBackend(conn), save=False):
        assert out["stats"]["cz"] == 24 * out["rounds"]
        assert not out["det_events"].any() and not out["obs_flips"].any()
