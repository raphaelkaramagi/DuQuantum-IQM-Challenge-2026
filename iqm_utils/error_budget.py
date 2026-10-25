"""D6 calibrated-noise ablation runner and figure generator.

This module does not implement C's circuit-specific ``add_noise`` or decoder.
Instead it provides the experiment harness: pass a simulator callback that
consumes a D3 noise specification and returns the decoded points, plus C's
``fit_eps(rounds, P, shots)`` function.  This keeps the ablation reproducible
without silently substituting a different decoder or circuit.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = ("cz", "single_qubit", "readout", "idle")


def _zero_channel(noise: Mapping[str, Any], channel: str) -> dict[str, Any]:
    """Copy a D3 noise specification and zero exactly one physical channel."""
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    ablated = copy.deepcopy(noise)
    if channel == "cz":
        for pair in ablated["cx"].values():
            pair["DEPOLARIZE2_after_CZ"] = 0.0
    elif channel == "single_qubit":
        for qubit in ablated["single_qubit"].values():
            qubit["DEPOLARIZE1"] = 0.0
        for pair in ablated["cx"].values():
            pair["DEPOLARIZE1_after_H_on_first_qubit"] = 0.0
            pair["DEPOLARIZE1_after_H_on_second_qubit"] = 0.0
    elif channel == "readout":
        for qubit in ablated["single_qubit"].values():
            qubit["X_ERROR_before_measurement"] = 0.0
    else:
        if ablated.get("p_idle") is None:
            raise ValueError("Set p_idle from C's sweep before performing an idle ablation")
        ablated["p_idle"] = 0.0
    return ablated


def _normalise_simulation(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accept C's points as either uppercase/lowercase probability keys."""
    rounds = np.asarray(result["rounds"], dtype=float)
    probabilities = np.asarray(result.get("P", result.get("p")), dtype=float)
    shots = np.asarray(result["shots"], dtype=float)
    if not (len(rounds) == len(probabilities) == len(shots)):
        raise ValueError("simulator must return equally-sized rounds, P/p, and shots arrays")
    return rounds, probabilities, shots


def run_error_budget(
    calibrated_noise: Mapping[str, Any],
    simulate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    fit_eps: Callable[[Sequence[float], Sequence[float], Sequence[float]], tuple[float, float]],
    *,
    hardware_eps: float | None = None,
) -> dict[str, Any]:
    """Run calibrated baseline and one-channel-zero ablations.

    ``simulate`` is normally C's pipeline: it calls ``add_noise(hw, noise)``,
    samples the resulting circuit and decodes it. Its return value includes
    ``rounds``, ``P`` (or ``p``), and ``shots``.  The reported contribution is
    ``baseline eps_L - ablated eps_L``; positive means that channel raises LER.
    """
    variants = {"calibrated_baseline": copy.deepcopy(calibrated_noise)}
    variants.update({f"without_{channel}": _zero_channel(calibrated_noise, channel) for channel in CHANNELS})

    results = {}
    for name, noise in variants.items():
        rounds, probabilities, shots = _normalise_simulation(simulate(noise))
        eps, sigma_eps = fit_eps(rounds, probabilities, shots)
        results[name] = {
            "rounds": rounds.tolist(),
            "P": probabilities.tolist(),
            "shots": shots.astype(int).tolist(),
            "eps_L": float(eps),
            "sigma_eps": float(sigma_eps),
        }

    baseline_eps = results["calibrated_baseline"]["eps_L"]
    ablations = {}
    for channel in CHANNELS:
        eps = results[f"without_{channel}"]["eps_L"]
        ablations[channel] = {
            "eps_L_without_channel": eps,
            "eps_L_drop": baseline_eps - eps,
            "fraction_of_baseline": None if baseline_eps == 0 else (baseline_eps - eps) / baseline_eps,
        }
    report = {
        "baseline": results["calibrated_baseline"],
        "ablations": ablations,
        "all_variants": results,
        "interpretation": "Simulation-only, independent channel-zero ablations; channels are not additive.",
    }
    if hardware_eps is not None:
        if hardware_eps <= 0:
            raise ValueError("hardware_eps must be positive")
        ratio = baseline_eps / hardware_eps
        report["hardware_comparison"] = {
            "hardware_eps_L": hardware_eps,
            "baseline_to_hardware_ratio": ratio,
            "within_factor_of_two": 0.5 <= ratio <= 2.0,
            "interpretation": (
                "Baseline agrees within a factor of two."
                if 0.5 <= ratio <= 2.0
                else "Gap suggests unmodeled leakage, crosstalk, or idle effects; do not assign a cause without follow-up."
            ),
        }
    return report


def plot_error_budget(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write the D6 horizontal ablation chart (positive means worse with channel)."""
    import matplotlib.pyplot as plt

    labels = list(CHANNELS)
    drops = [report["ablations"][channel]["eps_L_drop"] for channel in labels]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.barh(labels, drops, color="#8b5cf6")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(r"$\epsilon_L$(baseline) − $\epsilon_L$(channel set to zero)")
    axis.set_title("Calibrated simulation error-budget ablation")
    axis.bar_label(bars, fmt="%.3g", padding=3)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
