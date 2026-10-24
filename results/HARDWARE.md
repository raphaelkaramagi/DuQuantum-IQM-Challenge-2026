# Hardware and layout (Person B)

Status as of Sat Oct 24, ~4:45 PM ET. Labels: **verified** = we ran it, **read** = from code or docs, **unknown** = not checked yet.

## Device facts (verified, `results/backend_info.json`)

- IQM Garnet via `https://resonance.iqm.tech`. Calibration set `a18a947f-0408-4628-81ea-1715637e434f`, read at 20:21 UTC.
- 20 qubits, 30 CZ edges. Physical index i is `QB{i+1}` for this calibration set.
- The edges form a 5x5 square grid with 5 cells missing (rows of 2, 5, 5, 5, 3 qubits). See `figures/garnet_layout.png`.

## Layout (verified)

- The d=3 code's 24 interacting qubit pairs have **exactly one** SWAP-free placement on Garnet: patch `P00` (`results/patches.json`). `rustworkx.vf2_mapping` lists every subgraph embedding, and they all land on the same 17 qubits.
- Unused qubits: QB3, QB17, QB18. Data: QB2, 4, 6, 8, 10, 12, 14, 16, 19. X ancillas: QB1, 9, 11, 20. Z ancillas: QB5, 7, 13, 15. Full table: `results/layout_table.json`.
- `verify_layout` on the transpiled circuits, from `python hw_setup.py` and `python dryrun.py`:

| rounds | CZ | expected 24r | SWAP | gates off an edge | PRX | depth |
|---|---|---|---|---|---|---|
| 1 | 24 | 24 | 0 | 0 | 76 | 13 |
| 2 | 48 | 48 | 0 | 0 | 144 | 25 |
| 3 | 72 | 72 | 0 | 0 | 212 | 35 |
| 4 | 96 | 96 | 0 | 0 | 280 | 44 |
| 5 | 120 | 120 | 0 | 0 | 348 | 53 |

- IQM's `vf2pp_layout` independently picks the same 17 qubits.
- For comparison, IQM's default pipeline (`create_iqm_pass_manager`) crashes on the raw experiment circuit, because it cannot handle the `parity` markers. With the markers removed it also finds 0 SWAPs (the embedding is unique, so any good router lands there). It also removes 5 CZs per circuit (24r - 5). We did not adopt that, because the CZ-count proof needs exactly 24 per round.
- Single-qubit gates are merged after the CX-to-CZ conversion (`fuse=True`, the default), which roughly halves depth: 105 -> 53 at 5 rounds, with the same CZs. Noise and parity markers are stripped first, because they block the merge. IQM's converter drops them anyway, so the circuit sent to the chip is unchanged by that step.

## Pipeline checks before hardware (verified)

- Stim-only: the Qrisp `DetectorExperiment` matches a direct Stim decode within 3 standard errors for rounds 1-5 (`verification/test_qrisp_experiment.py`). The earlier round-1 gap was statistical noise: 0.0026 vs 0.0025 at 200k shots.
- Noiseless dry run (`python dryrun.py`) on the real Garnet edges, rounds 1-5 batched in one fake job: **zero detector firing, zero observable flips**. The fake backend uses IQM's real `qrisp_to_iqm_converter` and IQM's real counts parser; only the QPU is replaced by Aer.
- Negative control: reversing the bitstrings gives 41-51% detector firing, so the check above does detect a wrong bit order.
- With Aer depolarizing noise (0.3% CZ, 0.1% 1q): decoded P = 0.002 / 0.004 / 0.004 / 0.004 / 0.008 for rounds 1-5 (500 shots each).

## How to run (token comes from `IQM_TOKEN` or `.env`, never committed)

```bash
python hw_setup.py garnet        # B1+B2: backend_info.json, patches.json, offline verify. No job submitted.
python dryrun.py                 # noiseless fake-hardware run, must print PASS
python plot_layout.py            # figures/garnet_layout.png
```

```python
import json
from run_on_hardware import connect, run_batch
p = json.load(open("results/patches.json"))["patches"][0]
layout = {int(k): v for k, v in p["layout"].items()}
outs = run_batch([1, 2, 3, 4, 5], 5000, layout, dd=False, backend=connect(), patch_id=p["id"], tag="E1")
```

`run_batch` sends one job with one circuit per round count. It writes `results/raw/{config}_r{rounds}_{UTC}_{job}.json` (raw counts, layout, stats, verify report, job id, times, calibration set), `.npz` (`det_events`, `obs_flips`) and `.stim` (circuit for the DEM), and refuses to overwrite an existing file. Decoding inside it is plain PyMatching with the experiment's own DEM (uniform 0.3% prior); C should re-decode from the saved files.

Dynamical decoupling: `run_batch(..., dd=True)` sends `CircuitCompilationOptions(dd_mode=DDMode.ENABLED)` with IQM's default strategy (read). Whether it changes anything on hardware is **unknown**.

## Unknown

- Whether Garnet accepts repeated measurement of the same qubit without reset (first tested at rounds >= 2; 1 round measures each qubit once).
- Whether 4-5 round circuits pass the server's duration check (`max_circuit_duration_over_t2`, read in the client source).
- Shots or circuits per job limit, queue time, credit budget.
- Real detector firing rates and LER.

## Impact on the plan

- **E3 (patch scan) cannot be run as planned**: there is only one SWAP-free patch on Garnet. Options for D: (a) drop E3, or (b) compare P00 with a layout that needs SWAPs (a "layout cost" experiment), or (c) run E3 on Emerald if access exists.
- Patch ranking (D2) is moot; P* = P00.
