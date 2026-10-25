"""Generate D6's calibration-informed, non-idle error-budget proxy."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from decode import decode_ler, fit_eps
from iqm_utils.calibrated_proxy import add_calibrated_proxy_noise, p00_stim_to_qb
from iqm_utils.error_budget import plot_error_budget, write_report
from surface_code import make_stim_circuit


ROOT = Path(__file__).resolve().parent
ROUNDS = (1, 2, 3, 4, 5)
SHOTS = 20_000
SNAPSHOT = ROOT / "calib" / "20261024T220428.057329Z.json"
OUT_REPORT = ROOT / "results" / "tables" / "error_budget_proxy.json"
OUT_FIGURE = ROOT / "figures" / "error_budget_proxy.png"


def hardware_eps() -> float:
    with (ROOT / "results" / "tables" / "hardware_fits.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["arm"] == "no DD (E1+E2+E3, pooled)" and row["fit"] == "P<0.35":
                return float(row["eps_L"])
    raise ValueError("No pooled no-DD hardware fit found")


def simulate(*, include_cz: bool, include_single_qubit: bool, include_readout: bool, seed_offset: int):
    mapping = p00_stim_to_qb(ROOT)
    probabilities = []
    for rounds in ROUNDS:
        clean = make_stim_circuit(3, rounds, no_reset=True)
        noisy = add_calibrated_proxy_noise(
            clean, SNAPSHOT, mapping,
            include_cz=include_cz,
            include_single_qubit=include_single_qubit,
            include_readout=include_readout,
        )
        probabilities.append(decode_ler(noisy, SHOTS, seed=20261024 + seed_offset + rounds)[0])
    return probabilities


def main() -> None:
    variants = {
        "calibrated_baseline": (True, True, True, 0),
        "without_cz": (False, True, True, 100),
        "without_single_qubit": (True, False, True, 200),
        "without_readout": (True, True, False, 300),
    }
    all_variants = {}
    for name, (cz, one_qubit, readout, offset) in variants.items():
        probabilities = simulate(
            include_cz=cz,
            include_single_qubit=one_qubit,
            include_readout=readout,
            seed_offset=offset,
        )
        eps, sigma = fit_eps(ROUNDS, probabilities, SHOTS)
        all_variants[name] = {
            "rounds": list(ROUNDS), "P": probabilities, "shots": [SHOTS] * len(ROUNDS),
            "eps_L": eps, "sigma_eps": sigma,
        }

    baseline = all_variants["calibrated_baseline"]["eps_L"]
    ablations = {}
    for channel, key in (("cz", "without_cz"), ("single_qubit", "without_single_qubit"),
                         ("readout", "without_readout")):
        eps = all_variants[key]["eps_L"]
        ablations[channel] = {
            "eps_L_without_channel": eps,
            "eps_L_drop": baseline - eps,
            "fraction_of_baseline": (baseline - eps) / baseline if baseline else None,
        }

    hardware = hardware_eps()
    report = {
        "baseline": all_variants["calibrated_baseline"],
        "ablations": ablations,
        "all_variants": all_variants,
        "hardware_comparison": {
            "hardware_eps_L": hardware,
            "baseline_to_hardware_ratio": baseline / hardware,
            "within_factor_of_two": 0.5 <= baseline / hardware <= 2.0,
        },
        "interpretation": (
            "Calibration-informed circuit proxy; CZ, single-qubit, and readout channels are independently ablated. "
            "Idle noise is not modeled or ablated because durations are unavailable, so channel drops are not additive."
        ),
    }
    write_report(report, OUT_REPORT)
    plot_error_budget({**report, "ablations": ablations}, OUT_FIGURE)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
