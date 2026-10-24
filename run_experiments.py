"""
run_experiments.py

Hardware run matrix on IQM Garnet, patch P00. Each experiment takes a calibration
snapshot (calib/) before and after, submits through run_on_hardware.run_batch (raw data
to results/raw/, one row per circuit to results/tables/results.csv) and appends one row
per job to results/tables/experiment_log.csv.

    python run_experiments.py E1     # baseline: rounds 1-5, no DD, one batch
    python run_experiments.py E2     # DD on vs off, interleaved by round count
    python run_experiments.py E3     # layout cost: naive layout + SWAP routing vs zero-SWAP P00
    python run_experiments.py E4     # drift replicate of E1 (run a few hours after E1)
"""
import csv
import json
import os
import sys

import calibration
from run_on_hardware import connect, load_dotenv, run_batch

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "results", "tables", "experiment_log.csv")
ROUNDS = [1, 2, 3, 4, 5]
SHOTS = 5000

calibration.QUANTUM_COMPUTER = "garnet"   # calibration.py still defaults to emerald


def patch():
    p = json.load(open(os.path.join(ROOT, "results", "patches.json")))[0]
    return p["id"], {int(k): v for k, v in p["layout"].items()}


def snapshot():
    return os.path.relpath(calibration.save_calibration_snapshot(), ROOT)


def log(experiment, outs, calib_before, calib_after, dd, notes=""):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["experiment", "job_id", "rounds", "shots", "dd", "P", "sigma_P",
                        "mean_detector_firing", "calib_before", "calib_after", "notes"])
        for o in outs:
            w.writerow([experiment, o["job_id"], o["rounds"], len(o["obs_flips"]), int(dd), o["P"], o["sigma"],
                        o["mean_detector_firing"], calib_before, calib_after, notes])


def show(outs):
    for o in outs:
        print(f"  r={o['rounds']} job={o['job_id'][:8]} P={o['P']:.4f}+-{o['sigma']:.4f} "
              f"firing={o['mean_detector_firing']:.3f} cz={o['stats']['cz']} extra_cz={o['stats']['extra_cz']} "
              f"depth={o['stats']['depth']}")


# Naive placement: logical qubit i on physical qubit i, made executable by IQM's router
NAIVE = ("NAIVE", {i: i for i in range(17)})


def run(experiment, jobs, backend):
    """jobs: list of (rounds_list, dd, tag[, (patch_id, layout)]). Jobs without a layout use P00
    with no routing; jobs with one are routed. All jobs share one before/after snapshot pair."""
    before = snapshot()
    results = []
    for rounds_list, dd, tag, *custom in jobs:
        pid, layout = custom[0] if custom else patch()
        outs = run_batch(rounds_list, SHOTS, layout, dd=dd, backend=backend, patch_id=pid, tag=tag,
                         calib_before=before, route=bool(custom))
        print(f"{experiment} {tag} dd={dd}"); show(outs)
        results.append((outs, dd))
    after = snapshot()
    for outs, dd in results:
        log(experiment, outs, before, after, dd)
    print(f"calibration: {before} -> {after}")
    return results


EXPERIMENTS = {
    "E1": [(ROUNDS, False, "E1")],
    # interleave by round count so drift hits both arms equally
    "E2": [([r], dd, "E2dd" if dd else "E2nodd") for r in ROUNDS for dd in (False, True)],
    "E3": [job for r in ROUNDS for job in (([r], False, "E3"), ([r], False, "E3", NAIVE))],
    "E4": [(ROUNDS, False, "E4")],
}

if __name__ == "__main__":
    load_dotenv()
    backend = connect()
    for name in sys.argv[1:]:
        run(name, EXPERIMENTS[name], backend)
