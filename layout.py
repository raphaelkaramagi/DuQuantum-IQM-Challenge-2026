"""
layout.py

Zero-SWAP placement of the d=3 rotated surface code on IQM Garnet (Qrisp path).

    exp     = stim_to_detector_experiment(make_stim_circuit(3, r, no_reset=False, noise=DEFAULT_NOISE))
    qc      = exp.to_qc()
    pairs   = cx_pairs(qc)                                     # 24 logical pairs
    patches = enumerate_patches(pairs, backend.connectivity)   # every exact embedding
    qc_t, stats = transpile_qrisp(qc, patches[0]["layout"], n_physical)
    verify_layout(qc_t, backend.connectivity, expect_cz=24 * r)  # raises if anything is off

`layout` is {logical_index: physical_index}. Logical index i is qc.qubits[i] of the
traced circuit (= i-th smallest Stim qubit index). Physical index i is position i in
backend.connectivity, which qrisp_to_iqm_converter maps to dqa.qubits[i] (a QB name).
"""
import rustworkx as rx

# ops that are annotations, not gates: Stim noise, parity checks, allocation markers
_NON_GATES = ("parity", "qb_alloc", "qb_dealloc", "barrier", "measure")


def _is_gate(name):
    return name not in _NON_GATES and not name.startswith("stim.")


def cx_pairs(qc):
    """Undirected logical qubit pairs that carry a two-qubit gate in a Qrisp circuit."""
    index = {q: i for i, q in enumerate(qc.qubits)}
    pairs = set()
    for instr in qc.data:
        if len(instr.qubits) == 2 and _is_gate(instr.op.name):
            pairs.add(tuple(sorted(index[q] for q in instr.qubits)))
    return sorted(pairs)


def _edges(connectivity):
    return {tuple(sorted(e)) for e in connectivity}


def enumerate_patches(pairs, connectivity, n_pattern=17, limit=None):
    """All distinct physical qubit sets that host the pattern graph with no SWAPs.

    connectivity: list of (a, b) physical index pairs (backend.connectivity).
    Returns [{"id": "P00", "layout": {logical: physical}, "qubits": frozenset}]
    One layout is kept per distinct qubit set (the code has symmetries, so many
    mappings land on the same 17 qubits).
    """
    edges = sorted(_edges(connectivity))
    host = rx.PyGraph()
    host.add_nodes_from(range(max(max(e) for e in edges) + 1))
    host.add_edges_from_no_data(edges)
    pat = rx.PyGraph()
    pat.add_nodes_from(range(n_pattern))
    pat.add_edges_from_no_data(sorted(pairs))
    found = {}
    for m in rx.vf2_mapping(host, pat, subgraph=True, induced=False):
        layout = {p: h for h, p in m.items()}
        found.setdefault(frozenset(layout.values()), layout)
        if limit and len(found) >= limit:
            break
    return [{"id": f"P{i:02d}", "layout": l, "qubits": q}
            for i, (q, l) in enumerate(found.items())]


def fixed_layout(layout, n_physical):
    """Qrisp CircuitPass placing logical qubit i at physical index layout[i].

    Mirrors the ending of iqm.qrisp_iqm.vf2pp_layout: copy the circuit, pad with
    amended_qb_* qubits up to n_physical, then reorder qc.qubits so list position
    p is physical qubit p.
    """
    from qrisp import Qubit
    from qrisp.circuit.pass_management.circuit_pass import CircuitPass

    @CircuitPass
    def _fixed_layout(qc):
        n = qc.num_qubits()
        missing = [i for i in range(n) if i not in layout]
        if missing:
            raise ValueError(f"layout has no physical qubit for logical qubits {missing}")
        phys = [layout[i] for i in range(n)]
        if len(set(phys)) != n or max(phys) >= n_physical or min(phys) < 0:
            raise ValueError(f"layout is not injective into range({n_physical}): {phys}")

        new_qc = qc.copy()
        k = 0
        while new_qc.num_qubits() < n_physical:
            new_qc.add_qubit(Qubit("amended_qb_" + str(k)))
            k += 1
        inv_map = [-1] * n_physical
        for i, p in enumerate(phys):
            inv_map[p] = i
        spare = iter(range(n, n_physical))
        inv_map = [i if i != -1 else next(spare) for i in inv_map]
        new_qc.qubits = [new_qc.qubits[inv_map[p]] for p in range(n_physical)]
        return new_qc

    _fixed_layout.__name__ = "fixed_layout"
    return _fixed_layout


def strip_annotations(qc):
    """Copy of qc with only real gates and measurements. Drops stim.* noise, parity and
    qb_alloc markers, which qrisp_to_iqm_converter drops anyway. Clbits are kept, so the
    post-processing from the same trace still applies. Needed because the markers block
    combine_single_qubit_gates and crash IQM's default plasma pipeline."""
    out = qc.copy()
    out.data = [i for i in out.data if _is_gate(i.op.name) or i.op.name == "measure"]
    return out


def _drop_global_phase():
    """convert_to_prx can append a GPhaseGate; qrisp_to_iqm_converter would send it as a 'u'
    gate. A global phase has no physical effect, so remove it."""
    from qrisp.circuit.pass_management.circuit_pass import CircuitPass

    @CircuitPass
    def _drop(qc):
        out = qc.copy()
        out.data = [i for i in out.data if i.op.name != "gphase"]
        return out

    _drop.__name__ = "drop_global_phase"
    return _drop


def layout_pass_manager(layout, n_physical, fuse=True, route=None):
    """PassManager: fixed layout -> [SWAP routing] -> CX to CZ -> [merge 1q gates] -> PRX.
    route: None (no routing; the layout must be an exact embedding) or the connectivity
    to route on with IQM's plasma_route (used only for the SWAP-cost comparison)."""
    from qrisp import PassManager, combine_single_qubit_gates, convert_to_cz, convert_to_prx

    pm = PassManager()
    pm += fixed_layout(layout, n_physical)
    if route is not None:
        from iqm.qrisp_iqm import plasma_route
        pm += plasma_route(route)
    pm += convert_to_cz()
    if fuse:
        pm += combine_single_qubit_gates
    pm += convert_to_prx
    pm += _drop_global_phase()
    return pm


def gate_stats(qc):
    counts = {}
    for instr in qc.data:
        counts[instr.op.name] = counts.get(instr.op.name, 0) + 1
    gates = {k: v for k, v in counts.items() if _is_gate(k)}
    return {"cz": counts.get("cz", 0), "prx": counts.get("prx", 0), "swap": counts.get("swap", 0),
            "measure": counts.get("measure", 0), "depth": qc.depth(), "gates": gates}


def transpile_qrisp(qc, layout, n_physical, fuse=True, route=None):
    """Apply layout_pass_manager and return (qc_t, stats). fuse=True strips annotations
    first and merges adjacent single-qubit gates (about half the depth, same CZs)."""
    if fuse:
        qc = strip_annotations(qc)
    qc_t = layout_pass_manager(layout, n_physical, fuse, route).run(qc)
    return qc_t, gate_stats(qc_t)


def verify_layout(qc_t, connectivity, expect_cz=None):
    """Hard checks on a transpiled Qrisp circuit. Raises AssertionError on failure.

    No SWAP, only native gates (cz, prx), every two-qubit gate on a hardware edge,
    and optionally cz == expect_cz. Returns a report dict for the README.
    """
    edges = _edges(connectivity)
    index = {q: i for i, q in enumerate(qc_t.qubits)}
    stats = gate_stats(qc_t)
    assert stats["swap"] == 0, f"SWAPs present: {stats['swap']}"
    non_native = {k: v for k, v in stats["gates"].items() if k not in ("cz", "prx")}
    assert not non_native, f"non-native gates left: {non_native}"
    used, off_edge = set(), []
    for instr in qc_t.data:
        qs = tuple(sorted(index[q] for q in instr.qubits))
        if _is_gate(instr.op.name) or instr.op.name == "measure":
            used.update(qs)
        if len(qs) == 2 and _is_gate(instr.op.name) and qs not in edges:
            off_edge.append((instr.op.name, qs))
    assert not off_edge, f"two-qubit gates on non-edges: {off_edge[:5]} ({len(off_edge)} total)"
    if expect_cz is not None:
        assert stats["cz"] == expect_cz, f"cz={stats['cz']} expected {expect_cz}"
    return {"swap": 0, "cz": stats["cz"], "expect_cz": expect_cz, "prx": stats["prx"],
            "measure": stats["measure"], "depth": stats["depth"], "off_edge": 0,
            "physical_qubits_used": sorted(used)}


def transpile_for_emerald(qc, backend, layout, optimization_level=1, seed=1):
    """QISKIT FALLBACK ONLY. Transpile a Qiskit circuit with a fixed layout -> (qc_t, stats)."""
    from qiskit import transpile

    initial = [layout[i] for i in range(qc.num_qubits)]
    qc_t = transpile(qc, backend=backend, initial_layout=initial,
                     optimization_level=optimization_level, seed_transpiler=seed)
    ops = qc_t.count_ops()
    stats = {"cz": ops.get("cz", 0), "prx": ops.get("prx", 0), "swap": ops.get("swap", 0),
             "measure": ops.get("measure", 0), "depth": qc_t.depth(), "all_ops": dict(ops)}
    return qc_t, stats
