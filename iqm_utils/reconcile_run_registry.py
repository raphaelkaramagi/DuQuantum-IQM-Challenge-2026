"""Rebuild D4's run-status registry from immutable saved experiment logs.

The registry is descriptive: it records every completed E1--E4 circuit submission
and marks E4 skipped only if no E4 results exist.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables" / "results.csv"
OUTPUT = ROOT / "results" / "tables" / "run_registry.csv"
FIELDS = ("run_id", "experiment", "config_id", "patch_id", "rounds", "shots", "dd",
          "sequence", "status", "job_id", "notes")


def completed_rows() -> list[dict[str, str | int]]:
    with RESULTS.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))

    rows = []
    records = [record for record in records if record["config_id"] != "P00-nodd-smoke"]
    for sequence, record in enumerate(records, start=1):
        config = record["config_id"]
        experiment = config.rsplit("-", 1)[-1][:2]   # "...-E1", "...-E2dd", "...-E3", "...-E4"
        rows.append({
            "run_id": f"{experiment}-{config}-r{record['rounds']}",
            "experiment": experiment,
            "config_id": config,
            "patch_id": record["patch_id"],
            "rounds": record["rounds"],
            "shots": record["shots"],
            "dd": record["dd"],
            "sequence": sequence,
            "status": "done",
            "job_id": record["job_id"],
            "notes": "Reconciled from results.csv; raw data retained under results/raw/.",
        })
    return rows


def skipped_e4_rows(sequence: int) -> list[dict[str, str | int]]:
    return [{
        "run_id": f"E4-P00-nodd-E4-r{rounds}",
        "experiment": "E4",
        "config_id": "P00-nodd-E4",
        "patch_id": "P00",
        "rounds": rounds,
        "shots": 5000,
        "dd": 0,
        "sequence": sequence + rounds - 1,
        "status": "skipped",
        "job_id": "",
        "notes": "Not run: drift replicate requires a new hardware submission, excluded from this non-hardware closeout.",
    } for rounds in range(1, 6)]


def main() -> None:
    rows = completed_rows()
    if not any(row["experiment"] == "E4" for row in rows):
        rows.extend(skipped_e4_rows(len(rows) + 1))
    with OUTPUT.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    done = sum(row["status"] == "done" for row in rows)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {done} done, {len(rows) - done} skipped.")


if __name__ == "__main__":
    main()
