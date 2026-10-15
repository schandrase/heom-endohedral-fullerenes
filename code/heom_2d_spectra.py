#!/usr/bin/env python3
"""
Absorptive two-dimensional electronic spectra of the all-system endohedral
fullerenes (C60, CH4@C60, N2@C60, H2O@C60) from the hierarchical
equations of motion (HEOM), computed with QuTiP 5 (qutip.solver.heom.HEOMSolver).

Model
-----
Frenkel exciton Hamiltonian in an 11-state space: the ground state |0>, the
four single-exciton states |m>, and the six two-exciton states |mn> (m < n).
Two-exciton energies are the sums of the two site energies (no biexciton
binding) and two-exciton states sharing one site are coupled by the coupling
between the two remaining sites.  Site energies (cm^-1): 28911, 28884, 28867,
28910; couplings: Table 1 of the main text (TrESP).  The constant mean of the
site energies (28893 cm^-1) is removed; it commutes with the Liouvillian and
only defines the carrier frequency of the 2D axes.

Environment: one Drude-Lorentz bath per site, J(w) = 2 lam gam w/(w^2+gam^2),
coupled through the projector onto every single- and two-exciton state that
contains the site, plus one bath of the same parameters on the ground-state
projector (ground-state dephasing).  lam = 7 cm^-1, gam = 10 ps^-1
(53.09 cm^-1), T = 300 K, no Matsubara terms (K = 0), hierarchy depth L = 5
(252 ADOs; converged to 1% of the map maximum against L = 7).

Response: impulsive, rotating-wave third-order response with the six
Liouville pathways of Kramer et al. (DM-HEOM): ground-state bleach (GB),
stimulated emission (SE) and excited-state absorption (ESA), each rephasing
(RP) and non-rephasing (NR).  Each dipole interaction is applied to every
auxiliary density operator of the hierarchy, and the hierarchy is propagated
through t1, t2 and t3.  The transition-dipole operator for lab polarisation e
has elements mu_m.e (ground -> single) and mu_n.e (|m> -> |mn>).  All four
pulses share one polarisation; the spectrum is averaged over the six axes of a
regular icosahedron, which reproduces the exact isotropic average of a
fourth-rank dipole product (1/15 formula) to machine precision.

Numerics: t1, t3 = 64 points x 25 fs (1.6 ps), Gaussian apodisation of width
0.9 ps on both axes, 2D FFT, frequency spacing 20.8 cm^-1.  Each 2D map is
normalised to its own maximum when plotted.

Usage
-----
    python heom_2d_spectra.py [outdir] [--t2 0,300,400,500,1000] [--quick]

--t2     comma-separated waiting times in fs (default: the five in the paper)
--quick  16-point grid and one polarisation: a smoke test only

Outputs: tetramer_2D_series_full.npz (maps, axes, exciton energies, raw
time-domain pathway signals) and tetramer_2D_series.{png,pdf}.
Run time on a desktop CPU: about 50 min per laboratory axis (six axes,
five waiting times) for the full grid; the axes are independent and can be run
in parallel.

Environment: python 3.12, qutip 5.x, numpy, scipy, matplotlib
    conda create -n heom -c conda-forge python=3.12 qutip numpy scipy matplotlib
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
from qutip import Qobj, basis
from qutip.core.environment import DrudeLorentzEnvironment
from qutip.solver.heom import HEOMSolver

# ------------------------------------------------------------------ units
CM1 = 2 * np.pi * 2.99792458e-2      # 1 cm^-1 in rad/ps (hbar = 1)
KB = 0.6950348                       # Boltzmann constant, cm^-1 K^-1

# ------------------------------------------------------------------ system
NAMES = ["C60", "CH4@C60", "N2@C60", "H2O@C60"]
E_SITE = np.array([28911.0, 28884.0, 28867.0, 28910.0])          # cm^-1
COUPLINGS = {(0, 1): -45.141, (0, 2): -71.890, (0, 3): -33.437,   # cm^-1
             (1, 2): -79.414, (1, 3): -10.867, (2, 3): 32.601}
MU_SITE = np.array([[0.05323, 0.08427, 0.15893],                  # C60
                    [0.05463, -0.18887, 0.12274],                 # CH4@C60
                    [0.05718, -0.14546, 0.04954],                 # N2@C60
                    [0.03667, 0.14952, 0.09379]])                 # H2O@C60

LAM, WIDTH_CM, TEMP, DEPTH = 7.0, 53.0884, 300.0, 5     # width = gamma = 10 ps^-1, as for the dynamics
_PHI = (1 + 5 ** 0.5) / 2
ISO_AXES = np.array([[0, 1, _PHI], [0, 1, -_PHI], [1, _PHI, 0], [-1, _PHI, 0],
                     [_PHI, 0, 1], [_PHI, 0, -1]], float)
ISO_AXES /= np.linalg.norm(ISO_AXES, axis=1)[:, None]
DT, NT, APOD_PS = 0.025, 64, 0.9            # ps, points per axis, ps
T2_DEFAULT_FS = [0, 300, 400, 500, 1000]

PAIRS = list(itertools.combinations(range(4), 2))
NS = 1 + 4 + len(PAIRS)
IDX_S = {m: 1 + m for m in range(4)}
IDX_D = {p: 5 + k for k, p in enumerate(PAIRS)}

# (name, [(operator, side) for interactions 1-3], sign);  "p" = mu+, "m" = mu-
PATHS = [("GB_RP",  [("m", "R"), ("p", "R"), ("p", "L")], +1),
         ("SE_RP",  [("m", "R"), ("p", "L"), ("p", "R")], +1),
         ("ESA_RP", [("m", "R"), ("p", "L"), ("p", "L")], -1),
         ("GB_NR",  [("p", "L"), ("m", "L"), ("p", "L")], -1),
         ("SE_NR",  [("p", "L"), ("m", "R"), ("p", "R")], -1),
         ("ESA_NR", [("p", "L"), ("m", "R"), ("p", "L")], +1)]


def hamiltonian():
    """11 x 11 Hamiltonian in cm^-1, referenced to the mean site energy."""
    J = np.zeros((4, 4))
    for (a, b), v in COUPLINGS.items():
        J[a, b] = J[b, a] = v
    ref = E_SITE.mean()
    H = np.zeros((NS, NS))
    for m in range(4):
        H[IDX_S[m], IDX_S[m]] = E_SITE[m] - ref
    for a in range(4):
        for b in range(4):
            if a != b:
                H[IDX_S[a], IDX_S[b]] = J[a, b]
    for p in PAIRS:
        H[IDX_D[p], IDX_D[p]] = E_SITE[p[0]] + E_SITE[p[1]] - 2 * ref
    for p, q in itertools.combinations(PAIRS, 2):
        common = set(p) & set(q)
        if len(common) == 1:
            (x,) = set(p) - common
            (y,) = set(q) - common
            H[IDX_D[p], IDX_D[q]] = H[IDX_D[q], IDX_D[p]] = J[x, y]
    return H


def bath_operators():
    """Site projectors on the single- and two-exciton manifolds + ground state."""
    ops = []
    for m in range(4):
        d = np.zeros(NS)
        d[IDX_S[m]] = 1.0
        for p in PAIRS:
            if m in p:
                d[IDX_D[p]] = 1.0
        ops.append(Qobj(np.diag(d)))
    g = np.zeros(NS)
    g[0] = 1.0
    ops.append(Qobj(np.diag(g)))        # ground-state dephasing
    return ops


def dipole_operators(e):
    """mu+ (raising) and mu- for lab polarisation e."""
    d = MU_SITE @ e
    M = np.zeros((NS, NS))
    for m in range(4):
        M[IDX_S[m], 0] = d[m]
    for a, b in PAIRS:
        M[IDX_D[(a, b)], IDX_S[a]] = d[b]
        M[IDX_D[(a, b)], IDX_S[b]] = d[a]
    return Qobj(M), Qobj(M).dag()


def build_solver():
    env = DrudeLorentzEnvironment(T=KB * TEMP * CM1, lam=LAM * CM1,
                                  gamma=WIDTH_CM * CM1)
    baths = [(env.approximate("matsubara", Nk=0), Q) for Q in bath_operators()]
    return HEOMSolver(Qobj(hamiltonian() * CM1), baths, max_depth=DEPTH,
                      options={"progress_bar": False, "store_ados": True})


class Response:
    """Third-order response from one HEOM solver."""

    def __init__(self, solver, nt):
        self.sol = solver
        self.labels = list(solver.ados.labels)
        self.nado = len(self.labels)
        self.tgrid = np.arange(nt) * DT
        self.rho0 = basis(NS, 0) * basis(NS, 0).dag()

    # QuTiP's array input for a hierarchy state is the transpose of the Qobj
    # convention, hence the .T on every operator loaded into the hierarchy.
    def _initial(self, op):
        a = np.zeros((self.nado, NS, NS), dtype=complex)
        a[0] = op.full().T
        return a

    def _apply(self, state, op, side):
        a = np.zeros((self.nado, NS, NS), dtype=complex)
        for i, lab in enumerate(self.labels):
            r = state.extract(lab)
            a[i] = ((op * r) if side == "L" else (r * op)).full().T
        return a

    def pathways(self, t2, e):
        """Time-domain signal S(t1, t3) of every pathway at waiting time t2 (ps)."""
        mp, mm = dipole_operators(e)
        ops = {"p": mp, "m": mm}
        g = self.tgrid
        out = {}
        for name, seq, sign in PATHS:
            (o1, s1), (o2, s2), (o3, s3) = seq
            first_op = (self.rho0 * ops[o1]) if s1 == "R" else (ops[o1] * self.rho0)
            first = self.sol.run(self._initial(first_op), g)
            sig = np.zeros((len(g), len(g)), dtype=complex)
            for i in range(len(g)):
                mid = self.sol.run(self._apply(first.ado_states[i], ops[o2], s2),
                                   [0.0, max(t2, 1e-6)])
                third = self.sol.run(self._apply(mid.ado_states[-1], ops[o3], s3), g)
                sig[i] = [complex((mm * st).tr()) for st in third.states]
            out[name] = sign * sig
        return out


def absorptive(S, nt):
    """Sum of rephasing (frequency-flipped) and non-rephasing spectra.

    The flip-and-roll of the rephasing term and the sign/conjugation of the
    non-rephasing term were fixed on a single chromophore at +100 cm^-1, whose
    spectrum must show one positive diagonal peak at (100, 100) cm^-1.
    """
    g = np.arange(nt) * DT
    apod = np.exp(-0.5 * (g / APOD_PS) ** 2)
    win = apod[:, None] * apod[None, :]
    R = sum(S[k] for k in ("GB_RP", "SE_RP", "ESA_RP")) * win
    NR = sum(S[k] for k in ("GB_NR", "SE_NR", "ESA_NR")) * win
    FR = np.roll(np.fft.fftshift(np.fft.fft2(np.conj(R)))[::-1, :], 1, axis=0)
    FNR = -np.fft.fftshift(np.fft.fft2(np.conj(NR)))
    return (FR + FNR).real


def plot_series(W, t2_fs, maps, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(1, len(t2_fs), figsize=(1.9 * len(t2_fs), 2.2),
                            sharey=True, squeeze=False)
    levels = np.linspace(-0.2, 1.0, 25)
    for ax, t2, M in zip(axs[0], t2_fs, maps):
        Z = M / np.abs(M).max()
        cf = ax.contourf(W, W, Z.T, levels=levels, cmap="jet", extend="both")
        ax.contour(W, W, Z.T, levels=levels[::3], colors="k", linewidths=0.2)
        ax.plot([-300, 300], [-300, 300], "k--", lw=0.5)
        ax.set_xlim(-300, 300); ax.set_ylim(-300, 300); ax.set_box_aspect(1)
        ax.set_xlabel(r"$\omega_1$ (cm$^{-1}$)")
        ax.set_title(f"$t_2$ = {t2:g} fs", fontsize=7)
    axs[0][0].set_ylabel(r"$\omega_3$ (cm$^{-1}$)")
    fig.colorbar(cf, ax=axs[0].tolist(), fraction=0.015, pad=0.02,
                 label="normalised amplitude")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")


def parse_args(argv):
    outdir, t2_fs, quick, rest = Path("."), list(T2_DEFAULT_FS), False, iter(argv[1:])
    for a in rest:
        if a == "--quick":
            quick = True
        elif a == "--t2":
            t2_fs = [float(x) for x in next(rest).split(",")]
        else:
            outdir = Path(a)
    return outdir, t2_fs, quick


def main(argv):
    outdir, t2_fs, quick = parse_args(argv)
    outdir.mkdir(parents=True, exist_ok=True)
    nt = 16 if quick else NT
    pols = ISO_AXES[:1] if quick else ISO_AXES

    H = hamiltonian()
    resp = Response(build_solver(), nt)
    W = np.fft.fftshift(np.fft.fftfreq(nt, d=DT)) * 2 * np.pi / CM1
    sel = np.abs(W) <= 320
    print(f"states {NS} | ADOs {resp.nado} | grid {nt} x {DT * 1000:.0f} fs | "
          f"dw = {W[1] - W[0]:.1f} cm^-1 | polarisations {len(pols)}")

    maps, raw, t0 = [], {}, time.time()
    for t2 in t2_fs:
        acc = []
        for k, e in enumerate(pols):
            S = resp.pathways(t2 / 1000.0, e)
            raw.update({f"raw_{int(t2)}_{k}_{nm}": S[nm] for nm in S})
            acc.append(absorptive(S, nt)[np.ix_(sel, sel)])
        maps.append(np.mean(acc, axis=0))
        print(f"t2 = {t2:6.0f} fs done ({time.time() - t0:.0f} s)", flush=True)

    np.savez_compressed(outdir / "tetramer_2D_series_full.npz",
                        W=W[sel], t2=np.array(t2_fs) / 1000.0, maps=np.array(maps),
                        single_exciton=np.linalg.eigvalsh(H[1:5, 1:5]),
                        double_exciton=np.linalg.eigvalsh(H[5:, 5:]),
                        t_grid=resp.tgrid, **raw)
    plot_series(W[sel], t2_fs, maps, str(outdir / "tetramer_2D_series"))
    print("single-exciton energies (cm^-1):",
          np.round(np.linalg.eigvalsh(H[1:5, 1:5]), 1))
    print("map maxima:", [round(float(np.abs(M).max()), 4) for M in maps])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
