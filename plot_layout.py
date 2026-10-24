"""plot_layout.py  (B, figure F2): the zero-SWAP patch on the Garnet grid.

python plot_layout.py   -> figures/garnet_layout.png, results/layout_table.json
Grid positions are inferred from backend_info.json connectivity; every edge is
asserted to be one grid step, so a wrong guess fails loudly.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surface_code import make_stim_circuit

ROOT = os.path.dirname(os.path.abspath(__file__))
# Garnet rows as physical indices (row, first column)
ROWS = [([0, 1], 1), ([2, 3, 4, 5, 6], 0), ([7, 8, 9, 10, 11], 0), ([12, 13, 14, 15, 16], 0), ([17, 18, 19], 1)]


def positions():
    return {q: (r, c0 + k) for r, (qs, c0) in enumerate(ROWS) for k, q in enumerate(qs)}


def qubit_roles():
    """Dense logical index -> (stim index, role). X ancillas are the ones that get H."""
    c = make_stim_circuit(3, 1, no_reset=False)
    flat = c.flattened()
    stim_q = sorted({t.value for inst in flat for t in inst.targets_copy() if t.is_qubit_target})
    hq = {t.value for inst in flat if inst.name == "H" for t in inst.targets_copy()}
    coords = c.get_final_qubit_coordinates()
    roles = {}
    for i, sq in enumerate(stim_q):
        x, y = coords[sq]
        roles[i] = (sq, "data" if int(x) % 2 == 1 and int(y) % 2 == 1 else ("X" if sq in hq else "Z"))
    return roles


def main():
    info = json.load(open(os.path.join(ROOT, "results", "backend_info.json")))
    patch = json.load(open(os.path.join(ROOT, "results", "patches.json")))[0]
    names, edges = info["qubit_names"], [tuple(e) for e in info["connectivity"]]
    pos = positions()
    assert sorted(pos) == list(range(len(names))), "grid does not cover every qubit"
    for a, b in edges:
        (ra, ca), (rb, cb) = pos[a], pos[b]
        assert abs(ra - rb) + abs(ca - cb) == 1, f"edge {a}-{b} is not one grid step"
    layout = {int(k): v for k, v in patch["layout"].items()}
    roles = qubit_roles()
    on = {p: l for l, p in layout.items()}
    pairs_phys = set()
    from layout import cx_pairs
    from run_on_hardware import build_experiment
    for a, b in cx_pairs(build_experiment(1).to_qc()):
        pairs_phys.add(tuple(sorted((layout[a], layout[b]))))

    colors = {"data": "#4C78A8", "X": "#E45756", "Z": "#54A24B", None: "#DDDDDD"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for a, b in edges:
        (ra, ca), (rb, cb) = pos[a], pos[b]
        used = (a, b) in pairs_phys or (b, a) in pairs_phys
        ax.plot([ca, cb], [-ra, -rb], color="#333333" if used else "#CCCCCC", lw=3 if used else 1, zorder=1)
    table = []
    for p, (r, c) in pos.items():
        role = roles[on[p]][1] if p in on else None
        ax.scatter(c, -r, s=1500, color=colors[role], zorder=2, edgecolor="black")
        label = names[p] + ("" if p not in on else f"\n{role} L{on[p]}")
        ax.text(c, -r, label, ha="center", va="center", fontsize=8, zorder=3,
                color="white" if role else "black")
        table.append({"physical": p, "qb": names[p], "logical": on.get(p), "stim": roles[on[p]][0] if p in on else None,
                      "role": role, "row": r, "col": c})
    for role, col in [("data qubit", colors["data"]), ("X ancilla", colors["X"]), ("Z ancilla", colors["Z"]), ("unused", colors[None])]:
        ax.scatter([], [], s=120, color=col, edgecolor="black", label=role)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)
    ax.set_title(f"d=3 rotated surface code on IQM Garnet: patch {patch['id']}\n"
                 f"24/24 CZ pairs on hardware edges (dark), 0 SWAPs; calset {info['calibration_set_id'][:8]}", fontsize=9)
    ax.set_xlim(-0.7, 4.7); ax.set_ylim(-4.7, 0.7)
    ax.set_aspect("equal"); ax.axis("off")
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "garnet_layout.png"), dpi=150, bbox_inches="tight")
    with open(os.path.join(ROOT, "results", "layout_table.json"), "w") as f:
        json.dump({"patch_id": patch["id"], "calibration_set_id": info["calibration_set_id"],
                   "n_cz_pairs_on_edges": len(pairs_phys & {tuple(sorted(e)) for e in edges}), "qubits": table}, f, indent=1)
    print("pairs on edges:", len(pairs_phys & {tuple(sorted(e)) for e in edges}), "/ 24")
    for t in sorted(table, key=lambda t: (t["row"], t["col"])):
        print(t)


if __name__ == "__main__":
    main()
