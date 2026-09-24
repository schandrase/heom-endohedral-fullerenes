"""
phi_heom -- run PHI-style HEOM calculations with QuTiP 5.

Reads a PHI 1.0 parameter file (www.ks.uiuc.edu/Research/phi) and integrates the
hierarchy equations of motion for a Drude-Lorentz (overdamped Brownian) bath
using qutip.solver.heom.HEOMSolver.

Units follow PHI exactly:
    Hamiltonian, InitialDensityMatrix, lambda : cm^-1
    gamma                                     : ps^-1
    Temperature                               : K
    Timestep, Time                            : ps
Internally everything is converted to rad/ps with hbar = 1.

CLI (mirrors PHI's own usage):
    python phi_heom.py paramfile [rk4|rkf45|memory|print]
"""

from __future__ import annotations

import re
import sys
from math import comb
from pathlib import Path

import numpy as np
from qutip import Qobj, basis, ket2dm
from qutip.core.environment import DrudeLorentzEnvironment
from qutip.solver.heom import HEOMSolver

# ---------------------------------------------------------------- constants
C_CM_PER_PS = 2.99792458e-2            # speed of light, cm/ps
CM1_TO_RADPS = 2.0 * np.pi * C_CM_PER_PS   # 1 cm^-1 -> 0.1883652 rad/ps
KB_CM1_PER_K = 0.6950348                # Boltzmann constant, cm^-1 / K

SCALAR_DEFAULTS = {
    "numstates": 0,
    "numcouplingterms": 0,
    "temperature": 300.0,
    "matsubaraterms": 0,
    "hierarchytruncation": 0,
    "timelocal": 1,
    "timestep": 0.001,
    "time": 0.0,
    "runlength": None,          # PHI synonym for Time seen in older inputs
    "rkf45mindt": 1e-6,
    "rkf45tolerance": 1e-6,
    "filtertolerance": 1e-6,
    "outputfile": None,
}

_KEYLINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=|:)\s*(.*)$")
_TOKEN = re.compile(r"\(([^)]*)\)|([^,]+)")


# ---------------------------------------------------------------- parsing
def _num(tok: str):
    """Parse a PHI scalar token: 1.0, 1e-3, or a complex '(a,b)'."""
    tok = tok.strip()
    m = re.fullmatch(r"\(([^,]+),([^,]+)\)", tok)
    if m:
        return complex(float(m.group(1)), float(m.group(2)))
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def _row(line: str):
    out = []
    for m in _TOKEN.finditer(line):
        if m.group(1) is not None:
            a, b = m.group(1).split(",")
            out.append(complex(float(a), float(b)))
        else:
            tok = m.group(2).strip()
            if tok:
                out.append(_num(tok))
    return out


def parse_prm(path) -> dict:
    """Parse a PHI parameter file into a dict with lower-cased keys."""
    lines = Path(path).read_text().splitlines()
    params = dict(SCALAR_DEFAULTS)
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _KEYLINE.match(line)
        if not m:
            continue
        key, sep, rest = m.group(1).lower(), m.group(2), m.group(3).strip()
        if sep == "=":
            params[key] = _num(rest)
            continue
        # ':' -- vector or matrix; rows follow (possibly starting on this line)
        rows = []
        if rest:
            rows.append(_row(rest))
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("#"):
                i += 1
                continue
            if _KEYLINE.match(nxt):
                break
            rows.append(_row(nxt))
            i += 1
        params[key] = rows[0] if len(rows) == 1 else rows
    if params.get("time") in (0, 0.0, None) and params.get("runlength"):
        params["time"] = params["runlength"]
    return params


# ---------------------------------------------------------------- model build
def _as_matrix(val, n, name):
    arr = np.array(val, dtype=complex)
    if arr.shape != (n, n):
        raise ValueError(f"{name} must be {n}x{n}, got shape {arr.shape}")
    return arr


def build_system(p: dict):
    """Return (H, rho0, bath_list, meta) in rad/ps from parsed PHI parameters."""
    n = int(p["numstates"])
    if n < 1:
        raise ValueError("NumStates must be >= 1")

    H = Qobj(_as_matrix(p["hamiltonian"], n, "Hamiltonian") * CM1_TO_RADPS)

    if "initialdensitymatrix" in p:
        rho0 = Qobj(_as_matrix(p["initialdensitymatrix"], n, "InitialDensityMatrix"))
    else:
        rho0 = ket2dm(basis(n, 0))

    lam = np.atleast_1d(np.array(p["lambda"], dtype=float)) * CM1_TO_RADPS
    gam = np.atleast_1d(np.array(p["gamma"], dtype=float))          # already ps^-1
    m_terms = int(p["numcouplingterms"]) or len(lam)
    if len(lam) == 1 and m_terms > 1:
        lam = np.repeat(lam, m_terms)
    if len(gam) == 1 and m_terms > 1:
        gam = np.repeat(gam, m_terms)
    if not (len(lam) == len(gam) == m_terms):
        raise ValueError(
            f"lambda/gamma length ({len(lam)}/{len(gam)}) != NumCouplingTerms ({m_terms})"
        )

    # coupling operators V_m (PHI allows diagonal-independent or correlated forms)
    if "correlatedcouplingterms" in p:
        rows = np.array(p["correlatedcouplingterms"], dtype=float).reshape(m_terms, n)
        Qs = [Qobj(np.diag(r)) for r in rows]
        coupling = "correlated"
    else:
        if "diagonalcouplingindices" in p:
            idx = [int(k) for k in np.atleast_1d(p["diagonalcouplingindices"])]
        else:
            idx = list(range(m_terms))            # PHI default: term m -> state m
        if len(idx) != m_terms or max(idx) >= n:
            raise ValueError("DiagonalCouplingIndices inconsistent with NumStates")
        Qs = [ket2dm(basis(n, k)) for k in idx]
        coupling = "diagonal"

    T_energy = KB_CM1_PER_K * float(p["temperature"]) * CM1_TO_RADPS   # kB*T in rad/ps
    nk = int(p["matsubaraterms"])                 # PHI K; QuTiP uses K+1 exponents
    # BathExpansion is an extension beyond PHI: 'pade' converges much faster in
    # Nk than the Matsubara expansion at the same hierarchy cost.
    expansion = str(p.get("bathexpansion") or "matsubara").lower()
    if expansion not in ("matsubara", "pade"):
        raise ValueError("BathExpansion must be 'matsubara' or 'pade'")
    baths = []
    for lm, gm, Qop in zip(lam, gam, Qs):
        envr = DrudeLorentzEnvironment(T=T_energy, lam=lm, gamma=gm)
        baths.append((envr.approximate(expansion, Nk=nk), Qop))

    meta = dict(
        n_states=n, n_baths=m_terms, coupling=coupling, Nk=nk, expansion=expansion,
        max_depth=int(p["hierarchytruncation"]),
        kT_cm1=round(KB_CM1_PER_K * float(p["temperature"]), 4),
        lam_cm1=(lam / CM1_TO_RADPS).round(3).tolist(), gamma_ps=gam.round(4).tolist(),
    )
    return H, rho0, baths, meta


def n_ados(n_exponents: int, depth: int) -> int:
    """Number of auxiliary density matrices in a depth-L hierarchy."""
    return comb(n_exponents + depth, depth)


def memory_report(p: dict) -> str:
    n = int(p["numstates"])
    m = int(p["numcouplingterms"]) or len(np.atleast_1d(p["lambda"]))
    n_exp = m * (int(p["matsubaraterms"]) + 1)
    L = int(p["hierarchytruncation"])
    ados = n_ados(n_exp, L)
    gb = ados * n * n * 16 / 1024 ** 3
    return (f"states={n} baths={m} exponents/bath={int(p['matsubaraterms'])+1} "
            f"total exponents={n_exp} depth={L}\n"
            f"ADOs={ados}  hierarchy vector={ados * n * n} complex "
            f"({gb:.4f} GiB)\n"
            f"Liouvillian is sparse; peak RSS is typically a few x the vector size.")


# ---------------------------------------------------------------- integration
def run(p: dict, integrator: str = "rkf45", progress: bool = False):
    """Integrate the HEOM. Returns (times, rho_list, meta)."""
    H, rho0, baths, meta = build_system(p)
    depth = int(p["hierarchytruncation"])
    if depth < 1:
        raise ValueError("HierarchyTruncation must be >= 1")

    tmax = float(p["time"])
    dt = float(p["timestep"])
    if tmax <= 0:
        raise ValueError("Time (ps) must be > 0")
    times = np.arange(0.0, tmax + 0.5 * dt, dt)

    tol = float(p["rkf45tolerance"])
    options = {
        "progress_bar": "text" if progress else False,
        "store_states": True,
        "atol": tol * 1e-2,
        "rtol": tol,
        "nsteps": 100000,
    }
    if integrator == "rk4":
        # PHI's fixed-step RK4 has no QuTiP analogue; use a high-order explicit
        # integrator with max_step pinned to the requested timestep.
        options.update({"method": "vern9", "max_step": dt})
    else:
        options.update({"method": "adams"})

    solver = HEOMSolver(H, baths, max_depth=depth, options=options)
    res = solver.run(rho0, times)
    meta["n_ados"] = len(solver.ados.labels)
    meta["integrator"] = integrator
    return np.array(res.times), res.states, meta


# ---------------------------------------------------------------- output
def write_output(path, times, states, meta=None):
    """PHI-style plain-text output: time then populations then coherences."""
    n = states[0].shape[0]
    pops = [f"rho{i}{i}" for i in range(n)]
    cohs = [f"Re_rho{i}{j}" for i in range(n) for j in range(i + 1, n)]
    cohs += [f"Im_rho{i}{j}" for i in range(n) for j in range(i + 1, n)]
    cols = ["time_ps"] + pops + cohs
    rows = []
    for t, rho in zip(times, states):
        R = rho.full()
        row = [t] + [R[i, i].real for i in range(n)]
        row += [R[i, j].real for i in range(n) for j in range(i + 1, n)]
        row += [R[i, j].imag for i in range(n) for j in range(i + 1, n)]
        rows.append(row)
    arr = np.array(rows)
    hdr = ""
    if meta:
        hdr += "\n".join(f"{k} = {v}" for k, v in meta.items()) + "\n"
    hdr += " ".join(cols)
    np.savetxt(path, arr, header=hdr, fmt="%.10e")
    return arr, cols


# ---------------------------------------------------------------- CLI
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: python phi_heom.py paramfile [rk4|rkf45|memory|print]")
        return 1
    p = parse_prm(argv[1])
    mode = argv[2] if len(argv) > 2 else "rkf45"
    if mode == "memory":
        print(memory_report(p))
        return 0
    if mode == "print":
        for k, v in p.items():
            if v is not None:
                print(f"{k} = {v}")
        return 0
    if mode == "steadystate":
        H, rho0, baths, meta = build_system(p)
        solver = HEOMSolver(H, baths, max_depth=int(p["hierarchytruncation"]),
                            options={"progress_bar": False})
        rho_ss, _ = solver.steady_state()
        print("steady-state density matrix (site basis):")
        print(np.array2string(rho_ss.full(), precision=6, suppress_small=True))
        return 0
    times, states, meta = run(p, integrator=mode, progress=True)
    out = p.get("outputfile") or Path(argv[1]).with_suffix(".out").name
    write_output(out, times, states, meta)
    print(f"wrote {out}  ({len(times)} steps, {meta['n_ados']} ADOs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
