#!/usr/bin/env python3
"""
Figure of the all-system endohedral fullerenes: site-resolved population dynamics (a)-(d),
with 0-600 fs insets, and absorptive 2D spectra at five waiting times (e)-(i).

Dynamics: QuTiP 5 HEOM, four Drude-Lorentz baths (lam = 7 cm^-1,
gamma = 10 ps^-1, T = 300 K, K = 0, L = 10), each cage excited in turn,
0-3 ps with 5 fs output.  Hamiltonian as in heom_2d_spectra.py.
2D spectra: read from the .npz written by heom_2d_spectra.py.

Usage
-----
    python heom_2d_spectra.py out            # ~25 min, writes out/tetramer_2D_series_full.npz
    python plot_tetramer_figure.py out       # ~20 s, writes out/tetramer_combined_178mm.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qutip import Qobj, basis
from qutip.core.environment import DrudeLorentzEnvironment
from qutip.solver.heom import HEOMSolver

from heom_2d_spectra import CM1, KB, E_SITE, COUPLINGS

LABELS = [r"$C_{60}$", r"$CH_4@C_{60}$", r"$N_2@C_{60}$", r"$H_2O@C_{60}$"]
COLOURS = ["#FF0000", "#00FF00", "#4169E1", "#FF8C00"]


def site_dynamics(start, lam=7.0, gamma=10.0, temp=300.0, depth=10, tmax=3.0, dt=0.005):
    n = len(E_SITE)
    H = np.diag(E_SITE - E_SITE.mean())
    for (a, b), v in COUPLINGS.items():
        H[a, b] = H[b, a] = v
    env = DrudeLorentzEnvironment(T=KB * temp * CM1, lam=lam * CM1, gamma=gamma)
    baths = [(env.approximate("matsubara", Nk=0),
              Qobj(np.diag([1.0 if j == i else 0.0 for j in range(n)]))) for i in range(n)]
    sol = HEOMSolver(Qobj(H * CM1), baths, max_depth=depth, options={"progress_bar": False})
    t = np.arange(0.0, tmax + dt / 2, dt)
    res = sol.run(basis(n, start) * basis(n, start).dag(), t)
    return t, np.array([[s.full()[i, i].real for i in range(n)] for s in res.states])


def letter(ax, text, dx=0.0, y=1.10):
    ax.text(-0.18 + dx, y, text, transform=ax.transAxes, fontsize=7,
            fontweight="bold", va="bottom", ha="left")


def main(argv):
    outdir = Path(argv[1]) if len(argv) > 1 else Path(".")
    d2 = np.load(outdir / "tetramer_2D_series_full.npz")
    W, t2, maps = d2["W"], d2["t2"], d2["maps"]

    plt.rcParams.update({"font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.01, 3.78))                 # 178 mm wide
    gs = fig.add_gridspec(2, 20, hspace=0.46, wspace=2.1, left=0.072,
                          right=0.905, top=0.905, bottom=0.10)

    for s in range(4):
        t, P = site_dynamics(s)
        ax = fig.add_subplot(gs[0, 5 * s:5 * s + 5])
        for i in range(4):
            ax.plot(t, P[:, i], color=COLOURS[i], lw=0.9)
        ax.set_xlim(0, 3); ax.set_ylim(0, 1.03); ax.set_box_aspect(1)
        ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1.0"])
        ax.set_xlabel("t (ps)", labelpad=1.5)
        if s == 0:
            ax.set_ylabel(r"$\rho_{ii}(t)$", labelpad=1.5)
        ax.text(0.95, 0.95, f"excite {LABELS[s]}", transform=ax.transAxes,
                fontsize=5.4, ha="right", va="top")
        letter(ax, f"({'abcd'[s]})", y=1.03)
        ins = ax.inset_axes([0.42, 0.46, 0.54, 0.36])
        for i in range(4):
            ins.plot(t * 1000, P[:, i], color=COLOURS[i], lw=0.7)
        ins.set_xlim(0, 600); ins.set_ylim(0, 1.03)
        ins.set_xticks([0, 300, 600]); ins.set_xticklabels(["0", "300", "600 fs"])
        ins.set_yticks([0, 0.5, 1.0]); ins.set_yticklabels(["", "0.5", "1"])
        ins.tick_params(labelsize=5, length=1.5, pad=1, width=0.5)
        for sp in ins.spines.values():
            sp.set_linewidth(0.5)
        ins.patch.set_facecolor("white")
    fig.legend(handles=[Line2D([], [], color=COLOURS[i], lw=1.1, label=LABELS[i]) for i in range(4)],
               loc="upper center", ncol=4, frameon=False, fontsize=6,
               bbox_to_anchor=(0.49, 1.0), handlelength=1.8, columnspacing=1.8)

    levels = np.linspace(-0.8, 1.0, 37)
    for k in range(len(t2)):
        ax = fig.add_subplot(gs[1, 4 * k:4 * k + 4])
        Z = maps[k] / np.abs(maps[k]).max()
        cf = ax.contourf(W, W, Z.T, levels=levels, cmap="jet", extend="both")
        ax.contour(W, W, Z.T, levels=levels[::3], colors="k", linewidths=0.15, alpha=0.6)
        ax.plot([-300, 300], [-300, 300], color="k", lw=0.4, ls=(0, (4, 2)), alpha=0.7)
        ax.set_xlim(-300, 300); ax.set_ylim(-300, 300); ax.set_box_aspect(1)
        ax.set_xticks([-200, 0, 200]); ax.set_yticks([-200, 0, 200])
        if k > 0:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$\omega_1$ (cm$^{-1}$)", labelpad=1.5)
        if k == 0:
            ax.set_ylabel(r"$\omega_3$ (cm$^{-1}$)", labelpad=1.5)
        ax.text(0.94, 0.94, f"$t_2$={t2[k] * 1000:.0f} fs", transform=ax.transAxes,
                fontsize=5.4, ha="right", va="top", color="w")
        letter(ax, f"({'efghi'[k]})", dx=-0.06)
        if k == len(t2) - 1:
            p = ax.get_position()
            cb = fig.colorbar(cf, cax=fig.add_axes([0.915, p.y0, 0.011, p.height]),
                              ticks=[-0.8, -0.4, 0, 0.5, 1.0])
            cb.set_label("norm. amplitude", fontsize=5.5)
            cb.ax.tick_params(labelsize=5)

    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"tetramer_combined_178mm.{ext}", dpi=600)
    print("wrote", outdir / "tetramer_combined_178mm.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
