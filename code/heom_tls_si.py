#!/usr/bin/env python3
"""
Supporting Information script
=============================
Hierarchical equations of motion (HEOM) simulation of excitation dynamics in
two electronic two-level systems coupled to independent Drude-Lorentz
(overdamped Brownian oscillator) baths, and generation of the figures reported
in the main text.

Method
------
Each site n is coupled to its own harmonic bath through V_n = |n><n| with the
Drude-Lorentz spectral density

    J(w) = 2 * lambda * gamma * w / (w^2 + gamma^2),

whose correlation function is decomposed into a Drude term plus K Matsubara
terms.  The resulting hierarchy is propagated with the time-non-local HEOM
implementation of QuTiP 5 (qutip.solver.heom.HEOMSolver, the BoFiN solver of
Lambert et al., Phys. Rev. Research 5, 013181 (2023)), which integrates the
same equations as the PHI code of Strumpfer and Schulten, J. Chem. Theory
Comput. 8, 2808 (2012).  Parameter names and units follow PHI:

    Hamiltonian, lambda   cm^-1
    gamma                 ps^-1
    temperature           K
    time, timestep        ps

Internally all quantities are converted to rad/ps with hbar = 1.  A constant
energy offset (the mean of the diagonal) is subtracted from each Hamiltonian;
because it is proportional to the identity it commutes with the Liouvillian and
leaves the density matrix unchanged, while improving the conditioning of the
integration.

Numerical settings
------------------
Hierarchy truncation depth L = 10, K = 0 Matsubara terms (one bath exponent per
site, 66 auxiliary density matrices), adaptive Adams integrator with
rtol = 1e-6 / atol = 1e-8, trajectories reported every 0.1 fs to 10 ps.  Trace
conservation is checked for every trajectory and is better than 1e-14
throughout.

Environment
-----------
python 3.12, qutip 5.3.1, numpy 2.5.3, scipy 1.18.1, matplotlib 3.10
    conda create -n heom -c conda-forge python=3.12 qutip numpy scipy matplotlib

Usage
-----
    python heom_tls_si.py [output_directory]

Outputs (written to the output directory, default the current directory):
    TLS-Dynamics_300K_lambda<L>.out   density-matrix trajectories, one per run
    tls_beats_10ps.{png,pdf,eps}      Figure: system A, lambda = 1, 5, 9 cm^-1
    tls2_lambda_sweep_10ps.{...}      Figure: system B, lambda = 3.9, 7, 10 cm^-1
    summary printed to stdout (damping times, first-swing minima, long-time
    populations)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from qutip import Qobj, basis, ket2dm
from qutip.core.environment import DrudeLorentzEnvironment
from qutip.solver.heom import HEOMSolver

# --------------------------------------------------------------- constants
C_CM_PER_PS = 2.99792458e-2                 # speed of light, cm/ps
CM1_TO_RADPS = 2.0 * np.pi * C_CM_PER_PS    # 1 cm^-1 = 0.1883652 rad/ps
KB_CM1_PER_K = 0.6950348                    # Boltzmann constant, cm^-1/K

# --------------------------------------------------------------- model input
TEMPERATURE = 300.0        # K
GAMMA = 10.0               # ps^-1, bath response frequency (same on both sites)
MATSUBARA_TERMS = 0        # K in the Matsubara expansion
DEPTH = 10                 # hierarchy truncation level
TIMESTEP = 1.0e-4          # ps, reporting interval
TMAX = 10.0                # ps

# System A: site-energy gap 41.9 cm^-1, electronic coupling 32.6014 cm^-1
SYSTEM_A = np.array([[28910.0, 32.6014],
                     [32.6014, 28868.1]])
LAMBDAS_A = [1.0, 5.0, 9.0]

# System B: site-energy gap 19.1 cm^-1, electronic coupling -79.4138 cm^-1
SYSTEM_B = np.array([[28887.5, -79.4138],
                     [-79.4138, 28868.4]])
LAMBDAS_B = [3.9, 7.0, 10.0]

# line colours (red / green / royal blue), one per lambda in each sweep
COLOURS = ["#FF0000", "#00FF00", "#4169E1"]


# --------------------------------------------------------------- simulation
def run_heom(h_cm1, lam_cm1, gamma_ps=GAMMA, temperature=TEMPERATURE,
             nk=MATSUBARA_TERMS, depth=DEPTH, tmax=TMAX, dt=TIMESTEP):
    """Propagate the HEOM for a two-level system with one Drude bath per site.

    Returns (times [ps], rho [n_t, 2, 2] complex, n_ados).
    """
    n = h_cm1.shape[0]
    h_cm1 = h_cm1 - np.eye(n) * np.trace(h_cm1) / n      # identity shift: no effect
    H = Qobj(h_cm1 * CM1_TO_RADPS)
    rho0 = ket2dm(basis(n, 0))                            # excitation on site 1

    kT = KB_CM1_PER_K * temperature * CM1_TO_RADPS        # rad/ps
    baths = []
    for site in range(n):
        env = DrudeLorentzEnvironment(T=kT, lam=lam_cm1 * CM1_TO_RADPS,
                                      gamma=gamma_ps)
        baths.append((env.approximate("matsubara", Nk=nk), ket2dm(basis(n, site))))

    solver = HEOMSolver(H, baths, max_depth=depth,
                        options={"progress_bar": False, "store_states": True,
                                 "method": "adams", "rtol": 1e-6, "atol": 1e-8,
                                 "nsteps": 100000})
    times = np.arange(0.0, tmax + 0.5 * dt, dt)
    result = solver.run(rho0, times)
    rho = np.array([s.full() for s in result.states])
    return np.array(result.times), rho, len(solver.ados.labels)


def write_trajectory(path, times, rho, header):
    """PHI-style output: time, populations, then Re/Im of the coherences."""
    n = rho.shape[1]
    cols = ["time_ps"] + [f"rho{i}{i}" for i in range(n)]
    cols += [f"Re_rho{i}{j}" for i in range(n) for j in range(i + 1, n)]
    cols += [f"Im_rho{i}{j}" for i in range(n) for j in range(i + 1, n)]
    data = [times]
    data += [rho[:, i, i].real for i in range(n)]
    data += [rho[:, i, j].real for i in range(n) for j in range(i + 1, n)]
    data += [rho[:, i, j].imag for i in range(n) for j in range(i + 1, n)]
    np.savetxt(path, np.column_stack(data), fmt="%.10e",
               header=header + "\n" + " ".join(cols))


# --------------------------------------------------------------- analysis
def beat_statistics(times, pop, n_windows=20, n_fit=12):
    """Damping time of the population beats, from the windowed peak-to-peak
    amplitude of rho00(t) fitted to a single exponential."""
    w = len(times) // n_windows
    centres = np.array([times[i * w:(i + 1) * w].mean() for i in range(n_windows)])
    amp = np.array([np.ptp(pop[i * w:(i + 1) * w]) for i in range(n_windows)])
    keep = amp > 1e-6
    slope, _ = np.polyfit(centres[keep][:n_fit], np.log(amp[keep][:n_fit]), 1)
    return -1.0 / slope, centres, amp


def exciton_splitting(h_cm1):
    gap = h_cm1[0, 0] - h_cm1[1, 1]
    return float(np.hypot(gap, 2 * h_cm1[0, 1]))


# --------------------------------------------------------------- figure
def plot_sweep(results, out_stem, outdir):
    """One panel: rho00(t) for each lambda, house style, vector + raster."""
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "savefig.bbox": None,
    })
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    rho_inf = np.mean([r["pop"][-1] for r in results.values()])
    ax.axhline(rho_inf, color="0.55", lw=0.6, ls=(0, (4, 2)), zorder=0)
    # fastest-damping trace drawn last so no curve is hidden
    for z, (lam, r) in enumerate(sorted(results.items())):
        ax.plot(r["t"][::10], r["pop"][::10], color=COLOURS[z % len(COLOURS)],
                lw=1.2, zorder=3 + z, label=rf"$\lambda$={lam:g} cm$^{{-1}}$")
    ax.set_xlim(0, TMAX)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"$\rho_{00}(t)$")
    ax.legend(frameon=False, loc="upper right", handlelength=1.5,
              labelspacing=0.35, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(Path(outdir) / f"{out_stem}.{ext}", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------- driver
def run_sweep(h_cm1, lambdas, label, outdir):
    print(f"\n{label}: splitting {exciton_splitting(h_cm1):.2f} cm^-1, "
          f"beat period {2 * np.pi / (exciton_splitting(h_cm1) * CM1_TO_RADPS):.4f} ps")
    print(f"{'lambda':>8} {'ADOs':>5} {'tau (ps)':>9} {'min rho00':>10} "
          f"{'rho00(10ps)':>12} {'max|tr-1|':>10}")
    results = {}
    for lam in lambdas:
        times, rho, n_ados = run_heom(h_cm1, lam)
        pop = rho[:, 0, 0].real
        trace_err = np.abs(rho[:, 0, 0].real + rho[:, 1, 1].real - 1.0).max()
        tau, _, _ = beat_statistics(times, pop)
        name = f"TLS-Dynamics_300K_lambda{lam:g}"
        write_trajectory(Path(outdir) / f"{name}.out", times, rho,
                         f"{label}  lambda={lam:g} cm^-1  gamma={GAMMA} ps^-1  "
                         f"T={TEMPERATURE} K  K={MATSUBARA_TERMS}  L={DEPTH}  "
                         f"ADOs={n_ados}")
        results[lam] = {"t": times, "pop": pop}
        print(f"{lam:8.3g} {n_ados:5d} {tau:9.2f} {pop.min():10.4f} "
              f"{pop[-1]:12.4f} {trace_err:10.1e}")
    return results


def main(argv):
    outdir = Path(argv[1]) if len(argv) > 1 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    res_a = run_sweep(SYSTEM_A, LAMBDAS_A, "System A", outdir)
    plot_sweep(res_a, "tls_beats_10ps", outdir)

    res_b = run_sweep(SYSTEM_B, LAMBDAS_B, "System B", outdir)
    plot_sweep(res_b, "tls2_lambda_sweep_10ps", outdir)

    print(f"\nfigures and trajectories written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
