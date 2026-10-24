"""
analyze_hardware.py

Re-decodes every hardware job in results/raw/ (decode.decode_events), fits the logical
error rate per round (decode.fit_eps, P < 0.35 rule, with and without round 1, bootstrap
95% interval), compares with Stim simulations, and maps detector firing to physical
ancillas.

    python analyze_hardware.py

Writes results/tables/hardware_summary.csv, hardware_fits.csv, hotspots.csv and
figures/hardware_ler.png, figures/layout_cost.png.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
from scipy.stats import spearmanr

from decode import decode_events, decode_ler, fit_eps
from run_on_hardware import load_hardware
from surface_code import DEFAULT_NOISE, make_stim_circuit

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "figures")
P_MAX = 0.35
N_BOOT = 500

# arm name -> config_ids pooled into it
ARMS = {
    "no DD (E1+E2+E3, pooled)": ["P00-nodd-E1", "P00-nodd-E2nodd", "P00-nodd-E3"],
    "DD (E2)": ["P00-dd-E2dd"],
    "naive layout + SWAPs (E3)": ["NAIVE-nodd-E3"],
    "E1 only": ["P00-nodd-E1"],
    "E2 no DD only": ["P00-nodd-E2nodd"],
    "E3 zero-SWAP only": ["P00-nodd-E3"],
}


def load_all():
    """config_id -> rounds -> list of (record, failures bool array, det_events)."""
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*.json"))):
        rec, det, obs, circ = load_hardware(f)
        matcher = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
        fails = np.any(matcher.decode_batch(det) != obs, axis=1)
        assert abs(fails.mean() - decode_events(det, obs, circ)[0]) < 1e-12   # same as decode.py
        data[rec["config_id"]][rec["rounds"]].append((rec, fails, det))
    return data


def arm_points(data, config_ids):
    """rounds -> pooled failure array (all jobs of the listed configs)."""
    pooled = defaultdict(list)
    for cid in config_ids:
        for r, items in data.get(cid, {}).items():
            pooled[r].extend(f for _, f, _ in items)
    return {r: np.concatenate(v) for r, v in sorted(pooled.items())}


def fit_boot(points, rounds, seed=0):
    r = np.array(rounds)
    P = np.array([points[x].mean() for x in rounds])
    n = np.array([len(points[x]) for x in rounds])
    eps, se = fit_eps(r, P, n)
    rng = np.random.default_rng(seed)
    boots = [fit_eps(r, np.array([points[x][rng.integers(0, len(points[x]), len(points[x]))].mean()
                                  for x in rounds]), n)[0] for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return eps, se, lo, hi


def calibration_noise(snapshot, layout_qubits, cz_pairs):
    """Patch-averaged gate and readout error from a Garnet snapshot, as Stim's uniform knobs.
    Idle / measurement-induced dephasing is deliberately NOT included."""
    cal = json.load(open(snapshot))["calibration_data"]
    cz = np.mean([1 - cal["CZ"][str(list(p))] if str(list(p)) in cal["CZ"] else 1 - cal["CZ"][str(list(p[::-1]))]
                  for p in cz_pairs])
    ro = np.mean([1 - cal["readout"][q] for q in layout_qubits])
    sq = np.mean([1 - cal["1Q"][q] for q in layout_qubits])
    noise = dict(after_clifford_depolarization=1.25 * cz, after_reset_flip_probability=ro,
                 before_measure_flip_probability=ro, before_round_data_depolarization=0.0)
    return noise, {"cz_error": cz, "readout_error": ro, "1q_error": sq,
                   "t1_us": np.mean([cal["t1"][q] for q in layout_qubits]),
                   "t2_us": np.mean([cal["t2"][q] for q in layout_qubits])}


def simulate(noise, rounds, shots=20000):
    return {r: decode_ler(make_stim_circuit(3, r, no_reset=True, noise=noise), shots)[0] for r in rounds}


def hotspots(data, table, snapshot):
    """Firing rate of bulk detectors per ancilla (no-DD, zero-SWAP jobs, rounds >= 2)."""
    cal = json.load(open(snapshot))["calibration_data"]
    sc_xy = {tuple(map(int, v)): k for k, v in make_stim_circuit(3, 1, no_reset=False).get_final_qubit_coordinates().items()}
    by_stim = {t["stim"]: t for t in table["qubits"] if t["stim"] is not None}
    fired, total = defaultdict(float), defaultdict(int)
    for cid in ARMS["no DD (E1+E2+E3, pooled)"]:
        for r, items in data.get(cid, {}).items():
            if r < 2:
                continue
            coords = make_stim_circuit(3, r, no_reset=False).get_detector_coordinates()
            for _, _, det in items:
                for i, (x, y, t) in coords.items():
                    if 1 <= t <= r - 1:                         # bulk detectors only
                        qb = by_stim[sc_xy[(int(x), int(y))]]["qb"]
                        fired[qb] += det[:, i].sum(); total[qb] += len(det)
    rows = []
    for qb in sorted(fired, key=lambda q: -fired[q] / total[q]):
        role = next(t["role"] for t in table["qubits"] if t["qb"] == qb)
        pairs = [k for k in cal["CZ"] if f"'{qb}'" in k]
        rows.append({"ancilla": qb, "role": role, "bulk_firing": fired[qb] / total[qb],
                     "readout_error": 1 - cal["readout"][qb], "mean_cz_error": np.mean([1 - cal["CZ"][k] for k in pairs]),
                     "t1_us": cal["t1"][qb], "t2_us": cal["t2"][qb]})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(TABLES, exist_ok=True)
    data = load_all()
    patch = json.load(open(os.path.join(ROOT, "results", "patches.json")))[0]
    table = json.load(open(os.path.join(ROOT, "results", "layout_table.json")))
    snaps = sorted(f for f in glob.glob(os.path.join(ROOT, "calib", "*.json"))
                   if json.load(open(f)).get("quantum_computer") == "garnet")
    snapshot = snaps[0]

    summary, fits, curves = [], [], {}
    for arm, cids in ARMS.items():
        pts = arm_points(data, cids)
        if not pts:
            continue
        curves[arm] = pts
        for r, f in pts.items():
            summary.append({"arm": arm, "rounds": r, "shots": len(f), "P": f.mean(),
                            "sigma_P": np.sqrt(f.mean() * (1 - f.mean()) / len(f))})
        keep = [r for r in pts if pts[r].mean() < P_MAX]
        for label, rs in (("P<0.35", keep), ("P<0.35, without round 1", [r for r in keep if r > 1]),
                          ("rounds 1-3 (common set)", [r for r in (1, 2, 3) if r in pts])):
            if len(rs) >= 2:
                eps, se, lo, hi = fit_boot(pts, rs)
                fits.append({"arm": arm, "fit": label, "rounds": " ".join(map(str, rs)), "eps_L": eps,
                             "sigma_eps": se, "boot_lo": lo, "boot_hi": hi})
    write_csv(os.path.join(TABLES, "hardware_summary.csv"), summary)
    write_csv(os.path.join(TABLES, "hardware_fits.csv"), fits)

    rounds = [1, 2, 3, 4, 5]
    cal_noise, cal_info = calibration_noise(snapshot, patch["qubits"], patch["cz_pairs"])
    sims = {"Stim, uniform 0.3%": simulate(DEFAULT_NOISE, rounds),
            "Stim, Garnet gate+readout errors (no idling)": simulate(cal_noise, rounds)}
    print("calibration (patch mean):", {k: round(v, 5) for k, v in cal_info.items()}, "from", os.path.relpath(snapshot, ROOT))
    for name, s in sims.items():
        print(f"{name}: " + " ".join(f"r{r}={p:.4f}" for r, p in s.items()))
        summary_sim = fit_eps(np.array(rounds), np.array([s[r] for r in rounds]), 20000)
        fits.append({"arm": name, "fit": "all rounds", "rounds": "1 2 3 4 5", "eps_L": summary_sim[0],
                     "sigma_eps": summary_sim[1], "boot_lo": "", "boot_hi": ""})
    write_csv(os.path.join(TABLES, "hardware_fits.csv"), fits)

    hs = hotspots(data, table, snapshot)
    write_csv(os.path.join(TABLES, "hotspots.csv"), hs)
    rho_ro = spearmanr([h["bulk_firing"] for h in hs], [h["readout_error"] for h in hs])
    rho_t2 = spearmanr([h["bulk_firing"] for h in hs], [h["t2_us"] for h in hs])
    print("hot spots (bulk firing):", ", ".join(f"{h['ancilla']}({h['role']}) {h['bulk_firing']:.3f}" for h in hs))
    print(f"Spearman firing vs readout error: rho={rho_ro.statistic:.2f} p={rho_ro.pvalue:.2f}; "
          f"vs T2: rho={rho_t2.statistic:.2f} p={rho_t2.pvalue:.2f}  (8 ancillas, descriptive)")

    for f in fits:
        lo = f"[{f['boot_lo']:.4f}, {f['boot_hi']:.4f}]" if f["boot_lo"] != "" else ""
        print(f"{f['arm']:45s} {f['fit']:28s} r={f['rounds']:10s} eps_L={f['eps_L']:.4f} +- {f['sigma_eps']:.4f} {lo}")

    # Figure: P_L vs rounds and logical fidelity, hardware vs simulation
    style = {"no DD (E1+E2+E3, pooled)": ("C0", "o"), "DD (E2)": ("C2", "s")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rf = np.linspace(1, 5, 100)
    for arm, (col, mk) in style.items():
        pts = curves[arm]; r = np.array(list(pts)); P = np.array([pts[x].mean() for x in r])
        s = np.sqrt(P * (1 - P) / np.array([len(pts[x]) for x in r]))
        eps = next(f["eps_L"] for f in fits if f["arm"] == arm and f["fit"] == "P<0.35")
        ax[0].errorbar(r, P, s, fmt=mk, color=col, capsize=3, label=f"Garnet, {arm}")
        ax[0].plot(rf, (1 - (1 - 2 * eps) ** rf) / 2, color=col, alpha=0.6, label=f"fit, eps_L={eps:.3f}")
        ax[1].errorbar(r, 1 - 2 * P, 2 * s, fmt=mk, color=col, capsize=3)
        ax[1].plot(rf, (1 - 2 * eps) ** rf, color=col, alpha=0.6)
    for (name, sim), ls in zip(sims.items(), ("--", ":")):
        ax[0].plot(rounds, [sim[r] for r in rounds], ls, color="gray", label=name)
        ax[1].plot(rounds, [1 - 2 * sim[r] for r in rounds], ls, color="gray")
    ax[0].axhline(P_MAX, color="red", lw=0.8, alpha=0.5)
    ax[0].set(xlabel="rounds", ylabel="logical error probability P_L", title="d=3 Z memory on IQM Garnet (patch P00)")
    ax[1].set(xlabel="rounds", ylabel="logical fidelity 1 - 2P_L", yscale="log", title="fidelity decay")
    ax[0].legend(fontsize=7); [a.grid(alpha=0.3) for a in ax]
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "hardware_ler.png"), dpi=200); plt.close(fig)

    # Figure: layout cost (zero-SWAP vs naive + SWAPs), same session, interleaved
    fig, ax = plt.subplots(figsize=(5.5, 4))
    extra = {r: items[0][0]["stats"]["extra_cz"] for r, items in data["NAIVE-nodd-E3"].items()}
    for arm, col, lab in (("E3 zero-SWAP only", "C0", "zero-SWAP patch P00 (24 CZ/round)"),
                          ("naive layout + SWAPs (E3)", "C3", "naive layout, routed")):
        pts = curves[arm]; r = np.array(list(pts)); P = np.array([pts[x].mean() for x in r])
        ax.errorbar(r, P, np.sqrt(P * (1 - P) / 5000), fmt="o-", color=col, capsize=3, label=lab)
    for r, e in extra.items():
        ax.annotate(f"+{e} CZ", (r, curves["naive layout + SWAPs (E3)"][r].mean()), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="C3")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set(xlabel="rounds", ylabel="logical error probability P_L", title="Layout cost on Garnet (no DD, 5000 shots)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "layout_cost.png"), dpi=200); plt.close(fig)
    print("wrote results/tables/{hardware_summary,hardware_fits,hotspots}.csv, figures/{hardware_ler,layout_cost}.png")


if __name__ == "__main__":
    main()
