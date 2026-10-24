"""
dryrun.py

No-hardware dry run of run_experiment. FakeIQMBackend follows IQMBackend.run_async:
pass manager -> iqm.qrisp_iqm.qrisp_to_iqm_converter (the real one) -> execute the
IQM Circuit (prx / cz / measure) on Aer -> per-key shot results -> the real
IQMCircuitJob._parse_single_circuit_result. Only the QPU is replaced.

    python dryrun.py            # noiseless, rounds 1-5, synthetic 5x5 grid: firing must be 0
    python dryrun.py --noise    # small Aer depolarizing noise: P must be small but nonzero
"""
import sys
from types import SimpleNamespace

import numpy as np


def grid(rows, cols):
    e = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols: e.append((i, i + 1))
            if r + 1 < rows: e.append((i, i + cols))
    return e


class _Job:
    def __init__(self, job_id, all_counts):
        self.job_id, self._all_counts = job_id, all_counts

    def result(self):
        return SimpleNamespace(all_counts=self._all_counts)


class FakeIQMBackend:
    def __init__(self, connectivity, n_qubits=None, p2=0.0, p1=0.0, seed=1):
        from qrisp import PassManager

        self.connectivity = list(connectivity)
        n = n_qubits or max(max(e) for e in connectivity) + 1
        self._dqa = SimpleNamespace(qubits=[f"QB{i + 1}" for i in range(n)])
        self.iqm_server_url, self.quantum_computer, self._calibration_set_id = "fake", "fake-grid", None
        self.pm = PassManager()
        self.p2, self.p1, self.seed, self._n = p2, p1, seed, 0
        self.last_options = None

    def run_async(self, qc, shots, compilation_options=None):
        from iqm.qrisp_iqm import qrisp_to_iqm_converter
        from iqm.qrisp_iqm.backends.iqm_circuit_job import IQMCircuitJob

        self.last_options = compilation_options
        circuits = qc if isinstance(qc, (list, tuple)) else [qc]
        self._n += 1
        all_counts = []
        for c in circuits:
            iqm_circ = qrisp_to_iqm_converter(self.pm.run(c), self._dqa)
            all_counts.append(IQMCircuitJob._parse_single_circuit_result(self._execute(iqm_circ, shots)))
        return _Job(f"fake{self._n:04d}", all_counts)

    def _execute(self, iqm_circ, shots):
        """Run the IQM Circuit on Aer; return {key: [[bit]] * shots} like IQM's server."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import RGate
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        ops = iqm_circ.instructions
        names = sorted({q for op in ops for q in op.locus}, key=lambda s: int(s[2:]))
        idx = {q: i for i, q in enumerate(names)}
        keys = sorted({op.args["key"] for op in ops if op.name == "measure"})
        kidx = {k: i for i, k in enumerate(keys)}
        qk = QuantumCircuit(len(names), len(keys))
        for op in ops:
            qs = [idx[q] for q in op.locus]
            if op.name == "prx":       # IQM PRX(angle, phase) == Qiskit R(theta, phi)
                qk.append(RGate(op.args["angle"], op.args["phase"]), qs)
            elif op.name == "cz":
                qk.cz(*qs)
            elif op.name == "measure":
                qk.measure(qs[0], kidx[op.args["key"]])
            else:
                raise ValueError(f"unexpected IQM op {op.name}")
        nm = None
        if self.p1 or self.p2:
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(self.p1, 1), ["r"])
            nm.add_all_qubit_quantum_error(depolarizing_error(self.p2, 2), ["cz"])
        sim = AerSimulator(method="automatic", noise_model=nm, seed_simulator=self.seed + self._n)
        mem = sim.run(qk, shots=shots, memory=True).result().get_memory()
        bits = np.array([[int(c) for c in m[::-1]] for m in mem])   # Qiskit memory: rightmost = clbit 0
        return {k: bits[:, kidx[k]][:, None].tolist() for k in keys}


def garnet_connectivity():
    """Real Garnet edges saved by hw_setup.py, else a synthetic 5x5 grid."""
    import json, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "backend_info.json")
    if os.path.exists(path):
        info = json.load(open(path))
        return [tuple(e) for e in info["connectivity"]], f"{info['device']} ({info['calibration_set_id']})"
    return grid(5, 5), "synthetic 5x5"


def main(noise=False, rounds_list=(1, 2, 3, 4, 5), shots=500, fuse=True):
    from layout import cx_pairs, enumerate_patches
    from run_on_hardware import build_experiment, run_batch

    conn, label = garnet_connectivity()
    patches = enumerate_patches(cx_pairs(build_experiment(1).to_qc()), conn)
    print(f"connectivity: {label}, {len(patches)} patch(es), fuse={fuse}")
    be = FakeIQMBackend(conn, p2=0.003 if noise else 0.0, p1=0.001 if noise else 0.0)
    ok = True
    for out in run_batch(list(rounds_list), shots, patches[0]["layout"], backend=be,
                         patch_id=patches[0]["id"], save=False, fuse=fuse):
        r = out["rounds"]
        print(f"rounds={r} shots={shots} cz={out['stats']['cz']} swap={out['stats']['swap']} "
              f"prx={out['stats']['prx']} depth={out['stats']['depth']} "
              f"mean_firing={out['mean_detector_firing']:.4f} "
              f"obs_flip_rate={out['obs_flips'].mean():.4f} P={out['P']:.4f}±{out['sigma']:.4f}")
        if not noise:
            ok &= out["det_events"].sum() == 0 and out["obs_flips"].sum() == 0
    if not noise:
        print("NOISELESS DRY RUN:", "PASS (zero detector firing, zero observable flips)" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(noise="--noise" in sys.argv, fuse="--nofuse" not in sys.argv) else 1)
