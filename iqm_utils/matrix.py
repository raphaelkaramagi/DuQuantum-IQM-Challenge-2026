"""D4 hardware run matrix and append-only registry for the Garnet experiment.

This module creates the plan's E1--E4 run list but never submits jobs. B's
runner should update the generated registry after each submission/completion.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


Status = Literal["planned", "submitted", "done", "skipped", "failed"]


@dataclass(frozen=True)
class Run:
    """One hardware job point; raw-data paths and job ID are filled after submission."""

    run_id: str
    experiment: Literal["E1", "E2", "E3", "E4"]
    patch_id: str
    rounds: int
    shots: int
    dd: bool
    sequence: int
    status: Status = "planned"
    job_id: str = ""
    notes: str = ""


REGISTRY_FIELDS = tuple(Run.__dataclass_fields__)


def _config_id(patch_id: str, dd: bool) -> str:
    return f"{patch_id}-{'dd' if dd else 'nodd'}-o1"


def _run(
    experiment: Literal["E1", "E2", "E3", "E4"],
    patch_id: str,
    rounds: int,
    shots: int,
    dd: bool,
    sequence: int,
) -> Run:
    config_id = _config_id(patch_id, dd)
    return Run(
        run_id=f"{experiment}-{config_id}-r{rounds}",
        experiment=experiment,
        patch_id=patch_id,
        rounds=rounds,
        shots=shots,
        dd=dd,
        sequence=sequence,
    )


def build_run_matrix(
    best_patch: str,
    *,
    median_patch: str | None = None,
    worst_patch: str | None = None,
    shots: int = 5_000,
    include_e3: bool = True,
    include_e4: bool = True,
) -> list[Run]:
    """Build the plan's 23-job, 115k-shot matrix.

    E1's round-one point is the decode gate.  Once it is decoded end-to-end,
    E1/E2 are interleaved by round (off then on) to balance later drift. E3
    comes next, then E4.  E3 is automatically omitted when B finds fewer than
    three distinct patches (and median/worst are therefore unavailable).
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if include_e3 and (not median_patch or not worst_patch):
        raise ValueError("E3 needs the preselected median and worst patch IDs")

    matrix: list[Run] = []
    sequence = 1
    # The first point is the end-to-end smoke/decode gate before any DD job.
    matrix.append(_run("E1", best_patch, 1, shots, False, sequence))
    sequence += 1
    # After the E1 r=1 gate passes, DD-off/DD-on are paired by round.
    for rounds in range(1, 6):
        if rounds != 1:
            matrix.append(_run("E1", best_patch, rounds, shots, False, sequence))
            sequence += 1
        matrix.append(_run("E2", best_patch, rounds, shots, True, sequence))
        sequence += 1
    if include_e3:
        for patch_id in (median_patch, worst_patch):
            for rounds in range(1, 5):
                matrix.append(_run("E3", patch_id, rounds, shots, False, sequence))
                sequence += 1
    if include_e4:
        for rounds in range(1, 6):
            matrix.append(_run("E4", best_patch, rounds, shots, False, sequence))
            sequence += 1
    return matrix


def write_registry(matrix: Iterable[Run], path: str | Path) -> Path:
    """Write a new CSV registry; refuse to overwrite an existing job registry."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing registry: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as registry:
        writer = csv.DictWriter(registry, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(run) for run in matrix)
    return destination


def update_status(
    path: str | Path,
    run_id: str,
    status: Status,
    *,
    job_id: str | None = None,
    notes: str | None = None,
) -> None:
    """Update one registry row while preserving every previous row and job ID."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as registry:
        rows = list(csv.DictReader(registry))
    matched = [row for row in rows if row["run_id"] == run_id]
    if len(matched) != 1:
        raise KeyError(f"Expected exactly one run ID, found {len(matched)}: {run_id}")
    row = matched[0]
    row["status"] = status
    if job_id is not None:
        row["job_id"] = job_id
    if notes is not None:
        row["notes"] = notes
    with source.open("w", newline="", encoding="utf-8") as registry:
        writer = csv.DictWriter(registry, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary(matrix: Iterable[Run]) -> dict[str, int]:
    """Return simple planning totals for a matrix or a loaded registry."""
    runs = list(matrix)
    return {"jobs": len(runs), "shots": sum(run.shots for run in runs)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", required=True, help="P* patch ID chosen from calibration only")
    parser.add_argument("--median", help="Median patch ID for E3")
    parser.add_argument("--worst", help="Worst patch ID for E3")
    parser.add_argument("--shots", type=int, default=5_000)
    parser.add_argument("--without-e3", action="store_true")
    parser.add_argument("--without-e4", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/tables/run_matrix.csv"))
    args = parser.parse_args()

    matrix = build_run_matrix(
        args.best,
        median_patch=args.median,
        worst_patch=args.worst,
        shots=args.shots,
        include_e3=not args.without_e3,
        include_e4=not args.without_e4,
    )
    write_registry(matrix, args.output)
    totals = summary(matrix)
    print(f"Wrote {args.output}: {totals['jobs']} jobs, {totals['shots']:,} shots")
