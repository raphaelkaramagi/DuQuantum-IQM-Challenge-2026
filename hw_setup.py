"""
hw_setup.py  
    python hw_setup.py [device]     # default garnet

1. Connect (tries resonance.iqm.tech, then resonance.meetiqm.com), save results/backend_info.json.
2. Enumerate zero-SWAP patches on the real connectivity, save results/patches.json, print the count.
3. Offline proof on the first patch: transpile + verify_layout for rounds 1-5, plus a
   cross-check with IQM's vf2pp_layout. Nothing is sent to the QPU.
"""
import json
import os
import sys

from layout import cx_pairs, enumerate_patches, verify_layout
from run_on_hardware import backend_info, build_experiment, connect, load_dotenv, prepare

SERVERS = ["https://resonance.iqm.tech", "https://resonance.meetiqm.com"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def patch_rows(patches, pairs, names):
    """patches.json schema: a plain list, one entry per patch.
    id/name: patch ID; layout: {logical: physical index}; layout_qb: {logical: QB name};
    qubits: the 17 QB names; cz_pairs: the 24 interacting pairs as QB names
    (the format verification/score_patch.py reads)."""
    qb = (lambda p: names[p]) if names else (lambda p: f"idx{p}")
    order = lambda s: int(s[2:]) if s[2:].isdigit() else s
    return [{"id": p["id"], "name": p["id"],
             "layout": {str(k): int(v) for k, v in sorted(p["layout"].items())},
             "layout_qb": {str(k): qb(v) for k, v in sorted(p["layout"].items())},
             "qubits": sorted((qb(v) for v in p["qubits"]), key=order),
             "cz_pairs": sorted((sorted((qb(p["layout"][a]), qb(p["layout"][b])), key=order) for a, b in pairs),
                                key=lambda pr: (order(pr[0]), order(pr[1])))}
            for p in patches]


def _clean(msg):
    tok = os.environ.get("IQM_TOKEN", "")
    return str(msg).replace(tok, "***")[:300] if tok else str(msg)[:300]


def main(device="garnet"):
    load_dotenv()
    assert os.environ.get("IQM_TOKEN"), "IQM_TOKEN not set in the environment or .env"
    backend = None
    for url in SERVERS:
        try:
            backend = connect(device, url)
            print(f"connected: {url} device={device}")
            break
        except Exception as e:  # report and try the next server
            print(f"FAILED {url}: {type(e).__name__}: {_clean(e)}")
    if backend is None:
        sys.exit("no server accepted the token")

    os.makedirs(OUT, exist_ok=True)
    info = backend_info(backend)
    with open(os.path.join(OUT, "backend_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    names = info["qubit_names"]
    print(f"qubits: {len(names) if names else '?'}  edges: {info['n_edges']}  calset: {info['calibration_set_id']}")
    print("connectivity:", sorted(tuple(sorted(e)) for e in backend.connectivity))

    pairs = cx_pairs(build_experiment(1).to_qc())
    assert len(pairs) == 24, len(pairs)
    patches = enumerate_patches(pairs, backend.connectivity)
    rows = patch_rows(patches, pairs, names)
    with open(os.path.join(OUT, "patches.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nZERO-SWAP PATCHES: {len(patches)}   (written to results/patches.json)")
    if not patches:
        print("STOP: no zero-SWAP embedding. Fallback = plasma_layout + plasma_route; tell the team.")
        return

    for p in rows:
        print(p["id"], p["qubits"])
    print("\nverify_layout on", patches[0]["id"], "(offline, nothing submitted):")
    for r in range(1, 6):
        rep = prepare(r, patches[0]["layout"], backend.connectivity)["verify"]
        print(f"  rounds={r} cz={rep['cz']} (expect {24 * r}) swap={rep['swap']} off_edge={rep['off_edge']} "
              f"prx={rep['prx']} depth={rep['depth']}")

    from iqm.qrisp_iqm import vf2pp_layout
    from qrisp import PassManager, convert_to_cz, convert_to_prx
    pm = PassManager()
    pm += vf2pp_layout(backend.connectivity)
    pm += convert_to_cz()
    pm += convert_to_prx
    rep = verify_layout(pm.run(build_experiment(1).to_qc()), backend.connectivity, expect_cz=24)
    print(f"vf2pp_layout cross-check: cz={rep['cz']} swap={rep['swap']} qubits={rep['physical_qubits_used']}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["garnet"]))
