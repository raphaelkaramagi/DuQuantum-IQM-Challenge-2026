"""
Surface-code decoding and logical-error-rate (LER) analysis.

Terminal commands:
    python decode.py --help
    python decode.py --simulate
    python decode.py --analyze
    python decode.py --all

Quick checks:
    python -m py_compile decode.py
    python -m pytest verification/test_pipeline.py -q
"""

import numpy as np
import pymatching
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
from surface_code import build_pair
from extract_syndromes import extract_syndromes



def decode_events(det_events, obs_flips, circuit):
    """
    Decode detection events using PyMatching.

    Returns:
        p_L: Logical error probability
        sigma: Standard error
    """

    dets = np.asarray(det_events, dtype=np.uint8)
    obs = np.asarray(obs_flips, dtype=np.uint8)

    if dets.ndim != 2 or obs.ndim != 2:
        raise ValueError("Expected 2D arrays")

    if len(dets) == 0 or len(dets) != len(obs):
        raise ValueError("Invalid shot count")

    # Step 1: Build detector error model
    dem = circuit.detector_error_model(
        decompose_errors=True
    )

    # Step 2: Build MWPM decoder
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # Step 3: Predict logical observable flips
    pred = matcher.decode_batch(dets)

    # Step 4: Compare prediction with actual results
    failures = np.any(pred != obs, axis=1)

    # Step 5: Calculate logical error probability
    p_L = float(np.mean(failures))

    # Step 6: Calculate statistical uncertainty
    shots = len(failures)
    sigma = float(np.sqrt(p_L * (1 - p_L) / shots))

    return p_L, sigma


def decode_ler(circuit, shots=20000):
    """
    Run Stim simulation and estimate logical error probability.
    """

    sampler = circuit.compile_detector_sampler()

    dets, obs = sampler.sample(
        shots=shots,
        separate_observables=True
    )

    return decode_events(dets, obs, circuit)

def fit_eps(rounds, P, shots):
    """
    Fit logical error rate per round.

    Parameters:
        rounds: array of round counts
        P: logical error probabilities
        shots: number of shots per round (scalar or array)

    Returns:
        eps: fitted logical error rate per round
        sigma_eps: 1-sigma fitting uncertainty
    """
    rounds = np.asarray(rounds, dtype=float)
    P = np.asarray(P, dtype=float)
    shots = np.broadcast_to(
        np.asarray(shots, dtype=float), P.shape
    )

    def model(r, eps):
        return (1 - (1 - 2 * eps) ** r) / 2

    # Binomial standard errors, with a small floor
    # so zero observed failures still receive finite weight.
    sigma_P = np.sqrt(
        np.maximum(P * (1 - P), 1 / shots) / shots
    )

    params, covariance = curve_fit(
        model,
        rounds,
        P,
        p0=[0.005],
        sigma=sigma_P,
        absolute_sigma=True,
        bounds=(0, 0.5)
    )

    eps = float(params[0])
    sigma_eps = float(np.sqrt(covariance[0, 0]))

    return eps, sigma_eps


def run_simulation_baseline(shots=20000):
    """
    Run a no-reset surface code simulation,
    decode the results, fit LER, and save figures.
    """

    rounds = np.array([1, 2, 3, 4, 6, 8])
    P_values = []

    # Create output directories
    Path("results/simulation").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)

    for r in rounds:
        # Generate Person A's no-reset circuit
        hw, twin = build_pair(distance=3, rounds=int(r))

        # Sample raw measurement records
        raw = twin.compile_sampler().sample(shots)

        # Save raw simulation data
        np.savez_compressed(
            f"results/simulation/rounds_{r}.npz",
            raw_meas=raw,
            rounds=int(r),
            shots=shots
        )

        # Extract detection events
        syn = extract_syndromes(raw, hw, print_summary=False)

        # Decode
        p_L, sigma = decode_events(
            syn["det_events"],
            syn["obs_flips"],
            twin
        )

        P_values.append(p_L)
        print(f"Rounds={r}, P_L={p_L:.6f}, SE={sigma:.6f}")

    P = np.array(P_values)

    # Fit logical error rate
    eps, sigma_eps = fit_eps(rounds, P, shots)

    print(f"\nLER = {eps:.6f} +/- {sigma_eps:.6f} per round")

    # Build fitted curve
    r_fit = np.linspace(1, 8, 200)
    P_fit = (1 - (1 - 2 * eps) ** r_fit) / 2
    sigma_P = np.sqrt(P * (1 - P) / shots)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Plot 1: Logical error probability
    axes[0].errorbar(
        rounds, P, yerr=sigma_P,
        fmt="o", capsize=4, label="Simulation"
    )
    axes[0].plot(r_fit, P_fit, label="Fit")
    axes[0].set_xlabel("Measurement Rounds")
    axes[0].set_ylabel("Logical Error Probability")
    axes[0].set_title("P_L vs Rounds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Logical fidelity
    axes[1].errorbar(
        rounds, 1 - 2 * P, yerr=2 * sigma_P,
        fmt="o", capsize=4, label="Simulation"
    )
    axes[1].plot(
        r_fit, (1 - 2 * eps) ** r_fit,
        label="Fit"
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Measurement Rounds")
    axes[1].set_ylabel("Logical Fidelity")
    axes[1].set_title("Logical Fidelity Decay")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/simulation_baseline.png", dpi=300)
    plt.close(fig)

    print("Saved figure: figures/simulation_baseline.png")

    return eps, sigma_eps


def load_raw(path):
    """
    Load saved Stim measurement data from an NPZ file.

    Returns:
        rounds: Number of measurement rounds
        raw: Raw measurement array
    """
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw_meas"]
        rounds = int(data["rounds"])
        shots = int(data["shots"])

    if raw.ndim != 2 or len(raw) != shots:
        raise ValueError(f"Invalid raw data in {path}")

    return rounds, raw

def bootstrap_eps(raw_by_round, twin_by_round, n=200, seed=42):
    """
    Estimate logical error rate and 95% bootstrap CI.

    Resample individual shot failures independently
    for each measurement-round count.

    Returns:
        eps: Original fitted LER per round
        lo: Lower 95% confidence bound
        hi: Upper 95% confidence bound
    """

    rounds = np.array(sorted(raw_by_round.keys()))
    failures_by_round = []
    shots_by_round = []

    # Step 1: Decode the original saved measurements
    for r in rounds:
        raw = raw_by_round[r]
        twin = twin_by_round[r]

        hw = twin.without_noise()

        syn = extract_syndromes(
            raw, hw, print_summary=False
        )

        dem = twin.detector_error_model(
            decompose_errors=True
        )
        matcher = pymatching.Matching.from_detector_error_model(dem)

        pred = matcher.decode_batch(
            np.asarray(syn["det_events"], dtype=np.uint8)
        )
        actual = np.asarray(
            syn["obs_flips"], dtype=np.uint8
        )

        failures = np.any(pred != actual, axis=1)

        failures_by_round.append(failures)
        shots_by_round.append(len(failures))

    shots = np.array(shots_by_round)

    # Step 2: Fit original LER
    P = np.array([
        np.mean(f) for f in failures_by_round
    ])

    eps, _ = fit_eps(rounds, P, shots)

    # Step 3: Bootstrap resampling
    rng = np.random.default_rng(seed)
    bootstrap_values = []

    for _ in range(n):
        sampled_P = []

        for failures in failures_by_round:
            indices = rng.integers(
                0, len(failures), size=len(failures)
            )

            sampled_failures = failures[indices]
            sampled_P.append(np.mean(sampled_failures))

        eps_boot, _ = fit_eps(
            rounds, np.array(sampled_P), shots
        )
        bootstrap_values.append(eps_boot)

    # Step 4: Calculate 95% CI
    lo, hi = np.percentile(
        bootstrap_values, [2.5, 97.5]
    )

    return float(eps), float(lo), float(hi)


def analyze_saved_baseline():
    """
    Reanalyze saved simulation data without
    running a new simulation.
    """

    folder = Path("results/simulation")
    paths = sorted(folder.glob("rounds_*.npz"))

    if len(paths) < 2:
        raise FileNotFoundError(
            "Need at least two saved round datasets."
        )

    raw_by_round = {}
    twin_by_round = {}

    for path in paths:
        r, raw = load_raw(path)

        raw_by_round[r] = raw

        # Reconstruct the decoder's noise model
        _, twin = build_pair(distance=3, rounds=r)
        twin_by_round[r] = twin

        print(f"Loaded round {r}: {len(raw)} shots")

    eps, lo, hi = bootstrap_eps(
        raw_by_round,
        twin_by_round,
        n=200,
        seed=42
    )

    print("\nBootstrap Results")
    print("------------------------------")
    print(f"LER per round: {eps:.6f}")
    print(f"95% CI: [{lo:.6f}, {hi:.6f}]")
    print(f"LER percentage: {eps * 100:.4f}%")

    return eps, lo, hi


import argparse

def plot_saved_baseline():
    """
    Regenerate LER figures from saved raw data.
    No new simulation is performed.
    """

    folder = Path("results/simulation")
    paths = list(folder.glob("rounds_*.npz"))

    if len(paths) < 2:
        raise FileNotFoundError("Not enough saved data.")

    results = []

    for path in paths:
        r, raw = load_raw(path)

        hw, twin = build_pair(distance=3, rounds=r)

        syn = extract_syndromes(
            raw, hw, print_summary=False
        )

        p_L, _ = decode_events(
            syn["det_events"],
            syn["obs_flips"],
            twin
        )

        results.append((r, p_L, len(raw)))

    results.sort()

    rounds = np.array([x[0] for x in results])
    P = np.array([x[1] for x in results])
    shots = np.array([x[2] for x in results])

    eps, sigma_eps = fit_eps(rounds, P, shots)

    sigma_P = np.sqrt(P * (1 - P) / shots)

    r_fit = np.linspace(
        rounds.min(), rounds.max(), 200
    )

    P_fit = (1 - (1 - 2 * eps) ** r_fit) / 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Logical error probability
    axes[0].errorbar(
        rounds, P, yerr=sigma_P,
        fmt="o", capsize=4, label="Simulation"
    )

    axes[0].plot(r_fit, P_fit, label="Fit")
    axes[0].set_xlabel("Measurement Rounds")
    axes[0].set_ylabel("Logical Error Probability")
    axes[0].set_title("P_L vs Rounds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Logical fidelity (1 - 2P_L)
    axes[1].errorbar(
        rounds, 1 - 2 * P,
        yerr=2 * sigma_P,
        fmt="o", capsize=4,
        label="Simulation"
    )

    axes[1].plot(
        r_fit, (1 - 2 * eps) ** r_fit,
        label="Fit"
    )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Measurement Rounds")
    axes[1].set_ylabel("Logical Fidelity (1 - 2P_L)")
    axes[1].set_title("Logical Fidelity Decay")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    Path("figures").mkdir(
        parents=True, exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(
        "figures/simulation_baseline.png",
        dpi=300
    )
    plt.close(fig)

    print("Regenerated figures/simulation_baseline.png")
    print(f"Fitted LER: {eps:.6f} +/- {sigma_eps:.6f}")

    return eps

def main():
    parser = argparse.ArgumentParser(
        description="Surface code decoding and LER analysis"
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run new Stim simulation and save results"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing saved simulation data"
    )

    parser.add_argument(
    "--all",
    action="store_true",
    help="Analyze saved data and regenerate figures"
)

    args = parser.parse_args()

    if sum([args.simulate, args.analyze, args.all]) > 1:
        parser.error("Choose only one mode")

    if args.simulate:
        run_simulation_baseline()
    elif args.analyze:
        analyze_saved_baseline()
    elif args.all:
        analyze_saved_baseline()
        plot_saved_baseline()
    else:
        analyze_saved_baseline()
        



if __name__ == "__main__":
    main()


