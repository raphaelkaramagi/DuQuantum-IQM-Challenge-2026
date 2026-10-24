
"""
Synthetic Noise Prior Sweep

Run:
    python noise_sweep.py

Experiment:
    True physical noise p = 1.5%.
    Decode the SAME detector samples with assumed
    noise levels of 0.3%, 0.5%, 1%, 1.5%, and 3%.

Outputs:
    results/simulation/noise_prior_sweep.csv
    figures/noise_prior_sweep.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from surface_code import build_pair
from decode import decode_events


DISTANCE = 3
ROUNDS = 4
SHOTS = 20000

TRUE_P = 0.015
ASSUMED_PS = [0.003, 0.005, 0.01, 0.015, 0.03]


def make_noise(p):
    """Use the same probability for all four Stim noise channels."""
    return {
        "after_clifford_depolarization": p,
        "after_reset_flip_probability": p,
        "before_measure_flip_probability": p,
        "before_round_data_depolarization": p,
    }


def run_noise_prior_sweep():
    """Sample once at true p, then compare different decoder priors."""

    # Step 1: Generate ONE fixed dataset at true p = 1.5%.
    _, true_twin = build_pair(
        distance=DISTANCE,
        rounds=ROUNDS,
        noise=make_noise(TRUE_P),
    )

    sampler = true_twin.compile_detector_sampler(seed=42)
    det_events, obs_flips = sampler.sample(
        shots=SHOTS,
        separate_observables=True,
    )

    print(f"True noise: {TRUE_P:.1%}")
    print(f"Rounds: {ROUNDS}, Shots: {SHOTS}")
    print(f"Detectors: {det_events.shape[1]}")
    print()

    results = []

    # Step 2: Decode the SAME data with different assumed noise.
    for assumed_p in ASSUMED_PS:

        _, decoder_twin = build_pair(
            distance=DISTANCE,
            rounds=ROUNDS,
            noise=make_noise(assumed_p),
        )

        p_L, se = decode_events(
            det_events,
            obs_flips,
            decoder_twin,
        )

        results.append((assumed_p, p_L, se))

        print(
            f"Assumed p={assumed_p:.1%} | "
            f"P_L={p_L:.6f} | SE={se:.6f}"
        )

    # Step 3: Save summary statistics.
    output_dir = Path("results/simulation")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "noise_prior_sweep.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "true_p",
            "assumed_p",
            "rounds",
            "shots",
            "logical_failure",
            "standard_error",
        ])

        for assumed_p, p_L, se in results:
            writer.writerow([
                TRUE_P,
                assumed_p,
                ROUNDS,
                SHOTS,
                p_L,
                se,
            ])

    # Step 4: Plot logical failure vs assumed noise.
    x = [row[0] * 100 for row in results]
    y = [row[1] * 100 for row in results]
    yerr = [row[2] * 100 for row in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-",
        capsize=4,
        label="MWPM decoder",
    )

    ax.axvline(
        TRUE_P * 100,
        linestyle="--",
        label="True physical noise",
    )

    ax.set_xticks(x)
    ax.set_xlabel("Assumed Noise Probability (%)")
    ax.set_ylabel("Logical Failure Probability (%)")
    ax.set_title(
        f"Noise Prior Sweep (d={DISTANCE}, rounds={ROUNDS})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / "noise_prior_sweep.png"

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {figure_path}")

    return results


if __name__ == "__main__":
    run_noise_prior_sweep()
