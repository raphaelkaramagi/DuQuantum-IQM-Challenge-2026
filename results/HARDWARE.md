# Hardware and layout (Person B)

Status as of Sat Oct 24, ~5:15 PM ET. Labels: **verified** = we ran it, **read** = from code or docs, **unknown** = not checked yet.

## Device facts (verified, `results/backend_info.json`)

- IQM Garnet via `https://resonance.iqm.tech`. Calibration set `a18a947f-0408-4628-81ea-1715637e434f`, read at 20:21 UTC.
- 20 qubits, 30 CZ edges. Physical index i is `QB{i+1}` for this calibration set.
- The edges form a 5x5 square grid with 5 cells missing (rows of 2, 5, 5, 5, 3 qubits). See `figures/garnet_layout.png`.

## Layout (verified)

- The d=3 code's 24 interacting qubit pairs have **exactly one** SWAP-free placement on Garnet: patch `P00` (`results/patches.json`). `rustworkx.vf2_mapping` lists every subgraph embedding, and they all land on the same 17 qubits.
- Unused qubits: QB3, QB17, QB18. Data: QB2, 4, 6, 8, 10, 12, 14, 16, 19. X ancillas: QB1, 9, 11, 20. Z ancillas: QB5, 7, 13, 15. Full table: `results/layout_table.json`.
- `verify_layout` on the transpiled circuits (`python hw_setup.py`, `python dryrun.py`):

| rounds | CZ | expected 24r | SWAP | gates off an edge | PRX | depth |
|---|---|---|---|---|---|---|
| 1 | 24 | 24 | 0 | 0 | 76 | 13 |
| 2 | 48 | 48 | 0 | 0 | 144 | 25 |
| 3 | 72 | 72 | 0 | 0 | 212 | 35 |
| 4 | 96 | 96 | 0 | 0 | 280 | 44 |
| 5 | 120 | 120 | 0 | 0 | 348 | 53 |

- IQM's `vf2pp_layout` independently picks the same 17 qubits.
- For comparison, IQM's default pipeline (`create_iqm_pass_manager`) crashes on the raw experiment circuit, because it cannot handle the `parity` markers. With the markers removed it also finds 0 SWAPs (the embedding is unique, so any good router lands there), and it removes 5 CZs per circuit (24r - 5). We kept exactly 24 CZ per round so the gate count can be checked against the circuit.
- Single-qubit gates are merged after the CX-to-CZ conversion (`fuse=True`, the default), which roughly halves depth: 105 -> 53 at 5 rounds, with the same CZs. Noise and parity markers are stripped first, because they block the merge. IQM's converter drops them anyway, so the circuit sent to the chip is unchanged by that step. A trailing global-phase gate is also removed, since the converter would send it as a `u` gate.

## Pipeline checks before hardware (verified)

- Stim-only: the Qrisp `DetectorExperiment` matches a direct Stim decode within 3 standard errors for rounds 1-5 (`verification/test_qrisp_experiment.py`). The earlier round-1 gap was statistical noise: 0.0026 vs 0.0025 at 200k shots.
- Noiseless dry run (`python dryrun.py`) on the real Garnet edges, rounds 1-5 batched in one fake job: **zero detector firing, zero observable flips**. The fake backend uses IQM's real `qrisp_to_iqm_converter` and IQM's real counts parser; only the QPU is replaced by Aer.
- Negative control: reversing the bitstrings gives 41-51% detector firing, so the check above does detect a wrong bit order.
- With Aer depolarizing noise (0.3% CZ, 0.1% 1q): decoded P = 0.002 / 0.004 / 0.004 / 0.004 / 0.008 for rounds 1-5 (500 shots each).

## Smoke job (verified, Sat 20:39 UTC, job `01a16fa4-3a7b-757d-abe2-0218f2bc861d`)

One job, two circuits (rounds 1 and 2), 1000 shots each, patch P00, no DD, fused circuit, calibration set `a18a947f`.

| rounds | P (decoded, uniform 0.3% prior) | mean detector firing | max detector firing |
|---|---|---|---|
| 1 | 0.033 +/- 0.006 | 8.9% | 15.3% |
| 2 | 0.166 +/- 0.012 | 16.3% | 38.8% |

- **Pass**: mean firing is under the 25% gate at both round counts, and P is far below 0.5.
- **Garnet accepts repeated measurement of the same qubit without reset**: the round-2 circuit measures every ancilla twice and ran without error.
- Counts sum to 1000 for both circuits. The events in the `.npz` files were re-derived from the raw counts with a fresh trace and match exactly, as does the saved Stim circuit. `decode.decode_events` on the saved files gives the same P.
- A two-point fit (`decode.fit_eps`, rounds 1-2) gives eps_L of about 0.057 per round. This is indicative only; it needs rounds 1-5 and the P < 0.35 rule.
- Round-2 firing is highest on bulk detectors, peaking at 38.8%. Without reset, each bulk detector combines measurements two rounds apart, so ancilla idling and readout errors count twice. This is a hypothesis for D's hot-spot map, not a finding yet.

## Data interfaces

- `results/patches.json`: a plain list, one entry per patch: `id`/`name`, `layout` (logical -> physical index), `layout_qb`, `qubits` (17 QB names), `cz_pairs` (24 QB-name pairs). `verification/score_patch.py` reads it directly, but needs a **Garnet** calibration snapshot (the snapshot in `calib/` is from Emerald).
- `results/raw/{config}_r{rounds}_{UTC}_{job}.{json,npz,stim}`: one file set per circuit.
  - `.json`: raw counts (source of truth), layout, transpile stats, verify report, job id, times, calibration set, P.
  - `.npz`: `det_events`, `obs_flips`.
  - `.stim`: the circuit whose DEM decodes it.
- `run_on_hardware.load_hardware(path) -> (record, det_events, obs_flips, stim_circuit)` loads one set; then `decode.decode_events(det_events, obs_flips, stim_circuit)`. Note: `decode.load_raw` expects the older Qiskit `raw_meas` format and does not read these files.
- `results/tables/results.csv`: one row appended per circuit by `run_batch`, columns from the plan. `eps_L` is left empty for the fit in `decode.py`.
- `results/verification_results.pdf`: Person A's verification report (moved from `verification/` so all outputs live under `results/`).

## How to run (token comes from `IQM_TOKEN` or `.env`, never committed)

```bash
python hw_setup.py garnet        # backend_info.json, patches.json, offline verify. No job submitted.
python dryrun.py                 # noiseless fake-hardware run, must print PASS
python plot_layout.py            # figures/garnet_layout.png, results/layout_table.json
```

```python
import json
from run_on_hardware import connect, run_batch
p = json.load(open("results/patches.json"))[0]
layout = {int(k): v for k, v in p["layout"].items()}
outs = run_batch([1, 2, 3, 4, 5], 5000, layout, dd=False, backend=connect(), patch_id=p["id"], tag="E1")
```

Dynamical decoupling: `run_batch(..., dd=True)` sends `CircuitCompilationOptions(dd_mode=DDMode.ENABLED)` with IQM's default strategy (read). Whether it changes anything on hardware is **unknown**.

## Unknown

- Whether 4-5 round circuits pass the server's duration check (`max_circuit_duration_over_t2`, read in the client source).
- Shot cap per job, batch size cap, credit budget. The smoke job finished about 4 s after submission, so the queue was empty at that time.
- Whether the Intro Track rubric rewards the zero-SWAP result.

## Impact on the plan

- **E3 (patch scan) cannot run as planned**: there is only one SWAP-free patch on Garnet. Per the plan change, its budget moves to E4 (drift replicate) and decoder-only experiments on E1 data. Optional: compare P00 with a layout that needs SWAPs.
- Patch ranking (D2) is moot; P* = P00.
- Next hardware step: E1, rounds 1-5 at 5000 shots as one batch.
