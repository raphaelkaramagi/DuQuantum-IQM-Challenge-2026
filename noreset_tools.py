"""
noreset_tools.py

to_noreset(): rewrite a with-reset Stim circuit (ancillas use MR) into the
no-reset variant that Emerald can run (ancillas use plain M).

NOTE: stand-in implementation. The task brief references a team-provided
noreset_tools.py that was not in the repo; this one mirrors the rewrite in
stim_to_qrisp._analyse() exactly, and verification/test_circuit.py checks
that the two agree detector-for-detector. If the team version lands,
swap it in and rerun the tests.

Algebra
-------
With reset, raw outcome m_j of an ancilla is that round's syndrome s_j.
Without reset the ancilla keeps its previous outcome, so the raw outcome is
    m'_j = s_j XOR m'_prev(j)
and hence s_j = m'_j XOR m'_prev(j). Every measurement j referenced by a
DETECTOR / OBSERVABLE_INCLUDE is replaced by the set {j, prev(j)} (XOR /
symmetric difference). For a bulk detector s_k XOR s_(k-1) this collapses
to m'_k XOR m'_(k-2), i.e. "two rounds apart".
"""
import stim


def to_noreset(circuit: stim.Circuit) -> stim.Circuit:
    """
    Return the no-reset version of `circuit` (flattened, REPEAT unrolled).

    - every MR becomes M (gate args such as MR(p) are kept)
    - DETECTOR and OBSERVABLE_INCLUDE record targets are rewritten so the
      detectors stay deterministic (see module docstring)
    - noise channels, coordinates and TICKs are passed through unchanged

    Only Z-basis reset/measurement is supported (R, M, MR).
    """
    out = stim.Circuit()
    last_on_q = {}       # qubit -> absolute index of its latest measurement
    was_mr = set()       # absolute indices of measurements that were MR
    prev = {}            # j -> earlier measurement whose reset was removed
    n_meas = 0

    def rewrite(abs_idx):
        s = set()
        for j in abs_idx:
            s ^= {j}
            if j in prev:
                s ^= {prev[j]}
        return sorted(s)

    def as_rec(abs_idx):
        return [stim.target_rec(j - n_meas) for j in abs_idx]

    for inst in circuit.flattened():
        name = inst.name
        if name in ("MX", "MY", "MRX", "MRY", "MPP"):
            raise NotImplementedError(f"to_noreset: {name} not supported (Z basis only)")

        if name in ("M", "MR"):
            for t in inst.targets_copy():
                q = t.value
                i = last_on_q.get(q)
                if i is not None and i in was_mr:
                    prev[n_meas] = i
                if name == "MR":
                    was_mr.add(n_meas)
                last_on_q[q] = n_meas
                n_meas += 1
            out.append(stim.CircuitInstruction("M", inst.targets_copy(), inst.gate_args_copy()))

        elif name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            abs_idx = [n_meas + t.value for t in inst.targets_copy() if t.is_measurement_record_target]
            out.append(stim.CircuitInstruction(name, as_rec(rewrite(abs_idx)), inst.gate_args_copy()))

        else:
            out.append(inst)

    return out
