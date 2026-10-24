"""Capture IQM Emerald calibration data alongside hardware job batches.

Set ``IQM_TOKEN`` in the environment before running this module.  The token is
read only when a calibration request is made and is never written to a file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TypeVar

from iqm.qiskit_iqm import IQMProvider
from iqm.qubit_selector.qubit_selector import CalibrationDataManager, CalibrationType


SERVER_URL = "https://resonance.iqm.tech"
QUANTUM_COMPUTER = "emerald"
CALIBRATION_DIRECTORY = Path(__file__).resolve().parent / "calib"

T = TypeVar("T")


def get_backend():
    """Create an authenticated Emerald backend using ``IQM_TOKEN``.

    The explicit ``token=`` argument is required by the installed IQM client;
    merely setting the environment variable does not pass it to IQMProvider.
    """
    token = os.environ.get("IQM_TOKEN")
    if not token:
        raise RuntimeError(
            "IQM_TOKEN is not set. Export a Resonance token before capturing calibration data."
        )

    provider = IQMProvider(
    SERVER_URL,
    quantum_computer=QUANTUM_COMPUTER,
    )
    return provider.get_backend()


def fetch_calibration_data() -> dict[str, dict[str, float]]:
    """Fetch current Emerald metrics and display the fields available to this client."""
    # Print this before the request so an API/schema issue is easy to diagnose.
    print(list(CalibrationType))
    calibration_data = CalibrationDataManager().get_calibration_fidelities(get_backend())
    print(calibration_data.keys())
    return calibration_data


def _stringify_key(key: object) -> str:
    """Return a stable JSON key, including for tuple/list-like qubit pairs."""
    if isinstance(key, tuple):
        return ",".join(map(str, key))
    return str(key)


def _make_json_safe(value: Any) -> Any:
    """Recursively convert calibration data to JSON-compatible values.

    IQM versions may expose two-qubit measurements with tuple keys. JSON only
    permits string keys, so those become comma-separated pair strings such as
    ``"QB1,QB4"``.
    """
    if isinstance(value, Mapping):
        return {_stringify_key(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        # Handles NumPy scalar values without importing NumPy.
        return value.item()
    return value


def save_calibration_snapshot(
    calibration_data: Mapping[str, Any] | None = None,
    *,
    directory: str | Path = CALIBRATION_DIRECTORY,
) -> Path:
    """Save a calibration snapshot to ``calib/{UTC timestamp}.json``.

    Supplying ``calibration_data`` permits serialization tests and offline use;
    omitting it fetches the current calibration from IQM.
    """
    captured_at = datetime.now(timezone.utc)
    timestamp = captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(directory) / f"{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if calibration_data is None:
        calibration_data = fetch_calibration_data()

    snapshot = {
        "captured_at_utc": captured_at.isoformat().replace("+00:00", "Z"),
        "server_url": SERVER_URL,
        "quantum_computer": QUANTUM_COMPUTER,
        "calibration_data": _make_json_safe(calibration_data),
    }
    with path.open("w", encoding="utf-8") as snapshot_file:
        json.dump(snapshot, snapshot_file, indent=2, sort_keys=True)
        snapshot_file.write("\n")
    return path


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load a snapshot saved by :func:`save_calibration_snapshot`."""
    with Path(path).open(encoding="utf-8") as snapshot_file:
        return json.load(snapshot_file)


@contextmanager
def calibration_batch(
    batch_name: str,
    *,
    directory: str | Path = CALIBRATION_DIRECTORY,
) -> Iterator[Path]:
    """Capture calibration immediately before and after a hardware job batch.

    The post-batch snapshot is attempted even if the batch raises. The yielded
    value is the pre-batch snapshot path; both resulting paths are printed.
    """
    before = save_calibration_snapshot(directory=directory)
    print(f"[{batch_name}] calibration before batch: {before}")
    try:
        yield before
    finally:
        after = save_calibration_snapshot(directory=directory)
        print(f"[{batch_name}] calibration after batch:  {after}")


def run_batch_with_calibration(
    batch_name: str,
    run_batch: Callable[[], T],
    *,
    directory: str | Path = CALIBRATION_DIRECTORY,
) -> T:
    """Run a caller-provided job batch with calibration snapshots on both sides.

    Example::

        results = run_batch_with_calibration(
            "distance-3", lambda: [backend.run(circuit) for circuit in circuits]
        )
    """
    with calibration_batch(batch_name, directory=directory):
        return run_batch()


if __name__ == "__main__":
    snapshot_path = save_calibration_snapshot()
    print(f"Saved calibration snapshot: {snapshot_path}")
