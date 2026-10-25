# Hardware and layout

Status as of Sat Oct 24, ~6:30 PM ET. Labels: **verified** = we ran it, **read** = from code or docs, **unknown** = not checked yet.

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
- Round-2 firing is highest on bulk detectors, peaking at 38.8%. Without reset, each bulk detector combines measurements two rounds apart, so ancilla idling and readout errors count twice. This is a hypothesis for the hot-spot map, not a finding yet.

## Run matrix (verified, Sat 22:04-22:14 UTC, calibration set `a18a947f`, 5000 shots per circuit)

`python run_experiments.py E1 E2 E3`; calibration snapshots before/after each experiment in `calib/`, job list in `results/tables/experiment_log.csv`, analysis with `python analyze_hardware.py`.

| experiment | what | rounds 1 / 2 / 3 / 4 / 5 (P_L) |
|---|---|---|
| E1 | baseline, no DD, one batch | 0.046 / 0.261 / 0.339 / 0.425 / 0.435 |
| E2 no DD | interleaved with DD by round count | 0.049 / 0.253 / 0.325 / 0.389 / 0.440 |
| E2 DD | `dd=True` | 0.040 / 0.127 / 0.210 / 0.271 / 0.335 |
| E3 P00 | zero-SWAP control, interleaved | 0.040 / 0.261 / 0.332 / 0.381 / 0.445 |
| E3 naive | logical i on QB(i+1), routed by `plasma_route` (+54 to +93 CZ) | 0.261 / 0.409 / 0.493 / 0.500 / 0.503 |
| E4 | E1 repeated Oct 25 12:20 UTC, calibration set `0fe32137` | 0.026 / 0.180 / 0.283 / 0.353 / 0.401 |

- Fits (P < 0.35, bootstrap 95%): on matched rounds 1-3, no DD (pooled) 0.102 [0.099, 0.106] vs DD 0.065 [0.061, 0.068]. DD over all rounds 1-5: 0.076 [0.074, 0.079]. Without round 1: 0.152 and 0.085.
- Gate + readout errors from the Garnet snapshot (patch mean CZ 0.62%, readout 1.8%, 1Q 0.09%; T1 38 us, T2 10 us) give a simulated 0.015 per round (0.012 with the per-qubit model in `run_d6_error_budget.py`), 7-8.5x below hardware. The gap is consistent with idle decoherence during mid-circuit readout: DD, which acts only on idling qubits, removes about 37% of it. Idling was not measured directly, so this is not causal proof.
- The smoke job (20:39 UTC) gave P = 0.166 at 2 rounds versus 0.258 in the 22:05 runs, on the same calibration set: a drift of about 7 sigma over 1.5 h. E4 (next day, new calibration set) gave eps_L 0.070 vs 0.102, lower at every round count (z 4.9-12.0; `results/tables/drift.csv`).
- Hot spots (bulk detectors, no DD): QB15 (Z) 54%, QB11 (X) 34%, QB5 (Z) 33%, QB9 (X) 28%, QB20 (X) 24%, QB13 (Z) 19%, QB1 (X) 15%, QB7 (Z) 12%. No significant rank correlation with readout error or T2.

## Data interfaces

- `results/patches.json`: a plain list, one entry per patch: `id`/`name`, `layout` (logical -> physical index), `layout_qb`, `qubits` (17 QB names), `cz_pairs` (24 QB-name pairs). `verification/score_patch.py` reads it directly, but and runs on the Garnet snapshots in `calib/` (verified).
- `results/raw/{config}_r{rounds}_{UTC}_{job}.{json,npz,stim}`: one file set per circuit.
  - `.json`: raw counts (source of truth), layout, transpile stats, verify report, job id, times, calibration set, P.
  - `.npz`: `det_events`, `obs_flips`.
  - `.stim`: the circuit whose DEM decodes it.
- `run_on_hardware.load_hardware(path) -> (record, det_events, obs_flips, stim_circuit)` loads one set; then `decode.decode_events(det_events, obs_flips, stim_circuit)`. Note: `decode.load_raw` expects the older Qiskit `raw_meas` format and does not read these files.
- `results/tables/results.csv`: one row appended per circuit by `run_batch`, columns from the plan. `eps_L` is left empty for the fit in `decode.py`.
- `results/verification_results.pdf`: the circuit verification report (moved from `verification/` so all outputs live under `results/`).

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

Dynamical decoupling: `run_batch(..., dd=True)` sends `CircuitCompilationOptions(dd_mode=DDMode.ENABLED)` with IQM's default strategy. In interleaved E2 hardware data, DD lowered P_L at all rounds; the paired DD-minus-no-DD differences are -0.009, -0.126, -0.116, -0.118, and -0.105 for rounds 1-5, respectively. See `results/tables/dd_comparison.csv`.

## Unknown

- Shot and batch caps (5000 shots and 5-circuit batches were accepted; credits are unlimited for the event). Jobs returned within seconds.
- Whether the Intro Track rubric rewards the zero-SWAP result.

## Impact on the plan

- The planned patch scan is impossible (one SWAP-free patch on Garnet); E3 was redefined as the layout-cost comparison above.
- Patch ranking is moot; the best patch is P00.
- All planned hardware experiments (E1-E4) are done.
