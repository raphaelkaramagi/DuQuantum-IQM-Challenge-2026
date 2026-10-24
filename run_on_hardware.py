"""
run_on_hardware.py

One call per hardware job, Qrisp path on IQM Garnet:

    backend = connect()                                   # token from IQM_TOKEN
    out = run_experiment(rounds, shots, layout, dd=False, backend=backend, patch_id="P00")

run_experiment builds the DetectorExperiment, applies a fixed zero-SWAP layout
(layout.py), checks it with verify_layout, submits, turns counts into detection
events exactly like iqm.qrisp_iqm DetectorExperiment._sample_via_backend (no
bitstring reversal), decodes with the DEM from exp.to_stim(), and writes
results/raw/{config_id}_r{rounds}_{timestamp}_{job}.json/.npz/.stim.
It never calls compute_LER, which returns only a float and drops the data.

The Qiskit stubs (run_qiskit / run_qrisp) were replaced by run_experiment;
the Qiskit fallback lives in surface_code.to_qiskit_hw + verification/test_pipeline.py.
"""
import datetime as _dt
import json
import os

import numpy as np

from layout import transpile_qrisp, verify_layout
from surface_code import DEFAULT_NOISE, make_stim_circuit

SERVER_URL = "https://resonance.iqm.tech"   # alternative to try: https://resonance.meetiqm.com
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw")
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tables", "results.csv")
# column order from the implementation plan ("Results table and IDs")
CSV_COLUMNS = ["timestamp", "config_id", "patch_id", "dd", "opt_level", "dem_variant", "rounds", "shots",
               "P", "sigma_P", "eps_L", "sigma_eps", "job_id", "notes"]


def _now():
    return _dt.datetime.now(_dt.timezone.utc)


def load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Load KEY=VALUE lines from .env into os.environ without overriding existing vars. Values are never printed."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def connect(device="garnet", server_url=SERVER_URL):
    """IQMBackend with an EMPTY pass manager; run_experiment transpiles itself."""
    from iqm.qrisp_iqm import IQMBackend
    from qrisp import PassManager

    load_dotenv()
    # IQM_TOKEN is read by the IQM client from the environment. Do not also pass token=,
    # the client rejects mixing the two sources.
    return IQMBackend(server_url, device_instance=device, pass_manager=PassManager())


def backend_info(backend):
    """Facts for results/backend_info.json. Physical index i <-> qubit_names[i]."""
    dqa = getattr(backend, "_dqa", None)
    return {
        "server_url": getattr(backend, "iqm_server_url", None),
        "device": getattr(backend, "quantum_computer", None),
        "calibration_set_id": str(getattr(backend, "_calibration_set_id", None)),
        "qubit_names": list(dqa.qubits) if dqa is not None else None,
        "connectivity": [list(e) for e in backend.connectivity],
        "n_edges": len(backend.connectivity),
        "timestamp_utc": _now().isoformat(),
    }


def build_experiment(rounds, distance=3, noise=None):
    """DetectorExperiment from the noisy WITH-reset circuit (CLAUDE.md section 4).
    noise only shapes the DEM used for decoding; Stim noise ops never reach hardware."""
    from stim_to_qrisp import stim_to_detector_experiment

    c = make_stim_circuit(distance, rounds, no_reset=False, noise=noise or DEFAULT_NOISE)
    return stim_to_detector_experiment(c, no_reset=True)


def prepare(rounds, layout, connectivity, distance=3, noise=None, fuse=True):
    """Trace, transpile with the fixed layout, and verify. No network access."""
    import jax
    from qrisp.jasp import make_jaspr

    exp = build_experiment(rounds, distance, noise)
    jaspr = make_jaspr(lambda: exp._func())()          # same as _sample_via_backend
    qc = jaspr.to_qc()[-1]
    post = jax.jit(jaspr.extract_post_processing())
    n_physical = max(max(e) for e in connectivity) + 1
    qc_t, stats = transpile_qrisp(qc, layout, n_physical, fuse)
    stats["fuse"] = bool(fuse)
    report = verify_layout(qc_t, connectivity, expect_cz=24 * rounds)
    return {"exp": exp, "stim": exp.to_stim(), "qc": qc, "qc_t": qc_t, "post": post,
            "stats": stats, "verify": report, "n_clbits": len(qc.clbits)}


def counts_to_events(counts, post, shots, n_clbits):
    """Exactly IQM's post-processing: bits = [c == "1" for c in bitstring], no reversal."""
    total = sum(counts.values())
    assert total == shots, f"counts sum to {total}, expected {shots}"
    det, obs = [], []
    for bitstring, count in counts.items():
        assert len(bitstring) == n_clbits, f"bitstring length {len(bitstring)} != {n_clbits}: {bitstring!r}"
        out = np.asarray(post(np.array([c == "1" for c in bitstring])))
        det.append(np.repeat(out[None, :-1], count, axis=0))
        obs.append(np.repeat(out[None, -1:], count, axis=0))
    return np.concatenate(det).astype(bool), np.concatenate(obs).astype(bool)


def decode_inline(stim_circuit, det_events, obs_flips):
    """PyMatching with the experiment's own DEM. Replace with C's decode_events when it lands."""
    import pymatching

    dem = stim_circuit.detector_error_model(decompose_errors=True)
    pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(det_events)
    fails = np.any(pred != obs_flips, axis=1)
    p = float(fails.mean())
    return p, float(np.sqrt(p * (1 - p) / len(fails)))


def _compilation_options(dd):
    from iqm.iqm_client import CircuitCompilationOptions, DDMode

    return CircuitCompilationOptions(dd_mode=DDMode.ENABLED if dd else DDMode.DISABLED)


def run_batch(rounds_list, shots, layout, dd=False, backend=None, patch_id="P??", tag=None,
              calib_before=None, calib_after=None, save=True, distance=3, noise=None, fuse=True):
    """One IQM job holding one circuit per entry of rounds_list (fewer queue waits).
    Returns a list with one run_experiment-style dict per circuit, all sharing job_id."""
    if backend is None:
        raise ValueError("pass backend=connect() (or a FakeIQMBackend for a dry run)")
    from qrisp import PassManager

    preps = [prepare(r, layout, backend.connectivity, distance, noise, fuse) for r in rounds_list]
    opts = _compilation_options(dd)

    old_pm, backend.pm = backend.pm, PassManager()     # circuits are already transpiled + verified
    try:
        submitted = _now()
        circuits = [p["qc_t"] for p in preps]
        job = backend.run_async(circuits if len(circuits) > 1 else circuits[0], shots, compilation_options=opts)
        all_counts = [dict(c) for c in job.result().all_counts]
        finished = _now()
    finally:
        backend.pm = old_pm
    assert len(all_counts) == len(preps), f"{len(all_counts)} results for {len(preps)} circuits"
    job_id = str(getattr(job, "job_id", None) or getattr(job, "_job_id", "nojobid"))

    info = {k: v for k, v in backend_info(backend).items() if k != "connectivity"}
    config_id = f"{patch_id}-{'dd' if dd else 'nodd'}" + (f"-{tag}" if tag else "")
    outs = []
    for i, (rounds, prep, counts) in enumerate(zip(rounds_list, preps, all_counts)):
        det, obs = counts_to_events(counts, prep["post"], shots, prep["n_clbits"])
        P, sigma = decode_inline(prep["stim"], det, obs)
        firing = det.mean(axis=0)
        record = {
            "config_id": config_id, "patch_id": patch_id, "rounds": rounds, "shots": shots, "dd": bool(dd),
            "job_id": job_id, "batch_index": i, "batch_rounds": list(rounds_list),
            "submitted_utc": submitted.isoformat(), "finished_utc": finished.isoformat(),
            "layout": {str(k): int(v) for k, v in sorted(layout.items())},
            "backend": info, "stats": prep["stats"], "verify": prep["verify"],
            "calib_before": calib_before, "calib_after": calib_after,
            "P": P, "sigma": sigma, "mean_detector_firing": float(firing.mean()),
            "detector_firing": [float(x) for x in firing],
            "dem_noise": noise or DEFAULT_NOISE,
            "counts": counts,
        }
        base = f"{config_id}_r{rounds}_{submitted:%Y%m%dT%H%M%SZ}_{job_id[:8]}"
        paths = _save(record, det, obs, str(prep["stim"]), base) if save else None
        if save:
            append_results_row(record, notes=os.path.basename(paths[0]))
        outs.append({"rounds": rounds, "counts": counts, "det_events": det, "obs_flips": obs, "P": P,
                     "sigma": sigma, "stats": prep["stats"], "verify": prep["verify"], "job_id": job_id,
                     "mean_detector_firing": record["mean_detector_firing"], "paths": paths})
    return outs


def run_experiment(rounds, shots, layout, dd=False, backend=None, patch_id="P??", tag=None,
                   calib_before=None, calib_after=None, save=True, distance=3, noise=None, fuse=True):
    """Run one job with one circuit and keep everything. Returns
    {"counts", "det_events", "obs_flips", "P", "sigma", "stats", "verify", "job_id", "paths"}."""
    return run_batch([rounds], shots, layout, dd, backend, patch_id, tag, calib_before, calib_after,
                     save, distance, noise, fuse)[0]


def append_results_row(record, notes="", csv_path=RESULTS_CSV):
    """Append one row per decoded circuit to results/tables/results.csv (append-only).
    eps_L / sigma_eps are left empty: the multi-round fit is done in decode.py."""
    import csv

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = not os.path.exists(csv_path)
    row = {"timestamp": record["submitted_utc"], "config_id": record["config_id"], "patch_id": record["patch_id"],
           "dd": int(record["dd"]), "opt_level": "fuse" if record["stats"].get("fuse") else "nofuse",
           "dem_variant": "uniform{:g}".format(record["dem_noise"]["after_clifford_depolarization"]),
           "rounds": record["rounds"], "shots": record["shots"], "P": record["P"], "sigma_P": record["sigma"],
           "eps_L": "", "sigma_eps": "", "job_id": record["job_id"], "notes": notes}
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_hardware(path):
    """Load one saved job from any of its three files (.json/.npz/.stim, same stem).
    Returns (record, det_events, obs_flips, stim_circuit); decode with
    decode.decode_events(det_events, obs_flips, stim_circuit)."""
    import stim

    stem = os.path.splitext(path)[0]
    with open(stem + ".json") as f:
        record = json.load(f)
    with np.load(stem + ".npz") as z:
        det, obs = z["det_events"], z["obs_flips"]
    with open(stem + ".stim") as f:
        circuit = stim.Circuit(f.read())
    assert det.shape[0] == obs.shape[0] == record["shots"], "shot count mismatch"
    assert det.shape[1] == circuit.num_detectors, "detector count mismatch"
    return record, det, obs, circuit


def _save(record, det, obs, stim_text, base, raw_dir=RAW_DIR):
    """Never overwrite: open in 'x' mode, so an existing file raises."""
    os.makedirs(raw_dir, exist_ok=True)
    stem = os.path.join(raw_dir, base)
    with open(stem + ".json", "x") as f:
        json.dump(record, f, indent=1)
    with open(stem + ".npz", "xb") as f:
        np.savez_compressed(f, det_events=det, obs_flips=obs)
    with open(stem + ".stim", "x") as f:
        f.write(stim_text)
    return [stem + ext for ext in (".json", ".npz", ".stim")]


def decode_hardware_results(
    syndromes: dict
) -> tuple[float, float]:
    """
    Decodes the output of run_qiskit() or run_qrisp() via your desired decoder.

    NOTE: Other parameters can be important for this function, such as a noise model.
    If you decode with a detector error model (DEM): the DEM’s detectors must match the circuit that ran.


    Parameters
    ----------
    syndromes           : dict  output of run_qiskit() / run_qrisp()

    Returns
    -------
    ler : float   estimated logical error probability for this round count (fit over several round counts to get the logical error rate, see README)
    err : float   OPTIONAL 1-sigma statistical error
    """

    # USE e.g. PyMatching to decode

    ler,err = [], []
    return ler, err