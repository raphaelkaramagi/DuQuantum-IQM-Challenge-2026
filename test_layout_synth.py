"""Offline check of layout.py on SYNTHETIC square grids (not the real Garnet map).

python test_layout_synth.py
"""
from iqm.qrisp_iqm import vf2pp_layout
from qrisp import PassManager, convert_to_cz, convert_to_prx

from layout import cx_pairs, enumerate_patches, transpile_qrisp, verify_layout
from stim_to_qrisp import stim_to_detector_experiment
from surface_code import DEFAULT_NOISE, make_stim_circuit


def grid(rows, cols):
    e = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols: e.append((i, i + 1))
            if r + 1 < rows: e.append((i, i + cols))
    return e


def experiment(rounds):
    return stim_to_detector_experiment(make_stim_circuit(3, rounds, no_reset=False, noise=DEFAULT_NOISE), no_reset=True)


for R, C in [(5, 5), (6, 9)]:
    conn = grid(R, C)
    for rounds in (1, 3, 5):
        qc = experiment(rounds).to_qc()
        pairs = cx_pairs(qc)
        assert len(pairs) == 24, len(pairs)
        patches = enumerate_patches(pairs, conn)
        qc_t, st = transpile_qrisp(qc, patches[0]["layout"], R * C)
        rep = verify_layout(qc_t, conn, expect_cz=24 * rounds)
        print(f"{R}x{C} rounds={rounds} patches={len(patches)} cz={rep['cz']} swap={rep['swap']} "
              f"prx={rep['prx']} measure={rep['measure']} depth={rep['depth']}")

    # cross-check against IQM's own vf2pp_layout pass on the same grid
    pm = PassManager()
    pm += vf2pp_layout(conn)
    pm += convert_to_cz()
    pm += convert_to_prx
    rep = verify_layout(pm.run(experiment(1).to_qc()), conn, expect_cz=24)
    print(f"{R}x{C} vf2pp_layout cross-check: cz={rep['cz']} swap={rep['swap']}")

# a grid too small for the 17-qubit pattern must give no patch
assert enumerate_patches(cx_pairs(experiment(1).to_qc()), grid(4, 5)) == []
print("4x5 grid: 0 patches (expected)")
