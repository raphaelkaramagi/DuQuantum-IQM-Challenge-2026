"""
D-owned analysis artifacts regenerated from saved hardware summaries.

Writes:
- results/tables/dd_comparison.csv
- results/tables/hotspot_correlations.csv
- figures/dd_comparison.png
- figures/hotspot_map.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


# This module lives under iqm_utils; artifacts belong at the repository root.
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def dd_comparison_rows() -> list[dict[str, float | int]]:
    """Paired E2 DD-on versus DD-off comparison at each round."""
    rows = read_csv(TABLES / "experiment_log.csv")
    e2 = [row for row in rows if row["experiment"] == "E2"]
    by_round_dd = {
        (int(row["rounds"]), int(row["dd"])): row
        for row in e2
    }

    comparison = []
    for rounds in range(1, 6):
        no_dd = by_round_dd[(rounds, 0)]
        dd = by_round_dd[(rounds, 1)]

        p_no_dd = float(no_dd["P"])
        p_dd = float(dd["P"])
        combined_se = np.hypot(
            float(no_dd["sigma_P"]),
            float(dd["sigma_P"]),
        )
        delta = p_dd - p_no_dd

        comparison.append(
            {
                "rounds": rounds,
                "P_no_DD": p_no_dd,
                "P_DD": p_dd,
                "delta_P_DD_minus_no_DD": delta,
                "combined_SE": combined_se,
                "z": delta / combined_se,
            }
        )
    return comparison


def plot_dd_comparison(rows: list[dict[str, float | int]]) -> None:
    rounds = np.array([row["rounds"] for row in rows])
    delta = np.array([row["delta_P_DD_minus_no_DD"] for row in rows])
    error = np.array([row["combined_SE"] for row in rows])

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.errorbar(
        rounds,
        delta,
        yerr=error,
        fmt="o",
        capsize=4,
        color="#148F77",
        label="DD − no DD",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.fill_between(
        [0.7, 5.3],
        [-2 * error.max(), -2 * error.max()],
        [2 * error.max(), 2 * error.max()],
        color="gray",
        alpha=0.12,
        label="largest ±2 combined-SE band",
    )
    axis.set(
        xlabel="rounds",
        ylabel=r"$\Delta P_L$ (DD − no DD)",
        title="Dynamical decoupling benefit on Garnet",
        xlim=(0.7, 5.3),
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "dd_comparison.png", dpi=200)
    plt.close(figure)


def hotspot_correlation_rows() -> list[dict[str, float | str]]:
    rows = read_csv(TABLES / "hotspots.csv")
    firing = np.array([float(row["bulk_firing"]) for row in rows])

    output = []
    for metric in ("readout_error", "mean_cz_error", "t1_us", "t2_us"):
        values = np.array([float(row[metric]) for row in rows])
        result = spearmanr(firing, values)
        output.append(
            {
                "metric": metric,
                "spearman_rho": float(result.statistic),
                "two_sided_p_value": float(result.pvalue),
                "n_ancillas": len(rows),
                "interpretation": (
                    "Descriptive only; eight ancillas are insufficient for "
                    "an inferential physical-noise claim."
                ),
            }
        )
    return output


def plot_hotspot_map() -> None:
    hotspot_rows = read_csv(TABLES / "hotspots.csv")
    firing_by_qb = {
        row["ancilla"]: float(row["bulk_firing"])
        for row in hotspot_rows
    }
    layout = json.loads((ROOT / "results" / "layout_table.json").read_text())
    qubits = layout["qubits"]

    values = np.full((5, 5), np.nan)
    for qubit in qubits:
        if qubit["role"] in ("X", "Z") and qubit["qb"] in firing_by_qb:
            values[qubit["row"], qubit["col"]] = firing_by_qb[qubit["qb"]]

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(np.ma.masked_invalid(values), cmap="magma", vmin=0)
    figure.colorbar(image, ax=axis, label="Bulk detector firing probability")

    for qubit in qubits:
        row, col = qubit["row"], qubit["col"]
        qb = qubit["qb"]
        rate = firing_by_qb.get(qb)

        if rate is None:
            label = qb
            color = "black"
        else:
            label = f"{qb}\n{rate:.3f}"
            color = "white"

        axis.text(
            col,
            row,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )

    axis.set(
        xticks=range(5),
        yticks=range(5),
        xlabel="Garnet grid column",
        ylabel="Garnet grid row",
        title="Ancilla detector hot spots on Garnet (descriptive)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "hotspot_map.png", dpi=200)
    plt.close(figure)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    dd_rows = dd_comparison_rows()
    write_csv(TABLES / "dd_comparison.csv", dd_rows)
    plot_dd_comparison(dd_rows)

    correlations = hotspot_correlation_rows()
    write_csv(TABLES / "hotspot_correlations.csv", correlations)
    plot_hotspot_map()

    for row in dd_rows:
        print(
            f"round {row['rounds']}: "
            f"ΔP={row['delta_P_DD_minus_no_DD']:+.4f}, "
            f"combined SE={row['combined_SE']:.4f}, z={row['z']:.1f}"
        )
    print("Wrote DD comparison and hotspot analysis artifacts.")


if __name__ == "__main__":
    main()
