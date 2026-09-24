# Exciton coherence in endohedral fullerene dimers: HEOM dynamics and a neural-network inverse model

Code, data and figures accompanying the manuscript on guest-controlled exciton coherence in
endohedral fullerene dimers. Two things are provided: the hierarchical-equations-of-motion (HEOM)
simulations of the exciton dynamics, and the neural network that recovers the system–bath
parameters from a simulated trajectory.

## What the work does

Each site of a fullerene dimer is coupled to its own Drude–Lorentz bath,

    J(w) = 2 * lambda * gamma * w / (w^2 + gamma^2),

and the reduced density matrix is propagated with the bosonic HEOM solver of QuTiP 5. Parameters
follow the convention of the PHI code: site energies, excitonic couplings and reorganisation
energies in cm^-1, the bath response frequency `gamma` in ps^-1, times in ps, temperature in K.
Because a term proportional to the identity commutes with the Liouvillian, only the *difference*
of the two site energies is physical; the mean site energy is subtracted before propagation.

Two dimers are studied:

| system | site-energy gap Δ | coupling 𝒱₁₂ | exciton splitting | beat period |
|---|---|---|---|---|
| A | 41.9 cm⁻¹ | +32.6014 cm⁻¹ | 77.50 cm⁻¹ | 430 fs |
| B | 19.1 cm⁻¹ | −79.4138 cm⁻¹ | 159.97 cm⁻¹ | 209 fs |

Both are propagated at 300 K with `gamma = 10` ps⁻¹ (100 fs bath correlation time), hierarchy
truncation `L = 10` and `K = 0` Matsubara terms, which is 66 auxiliary density matrices. Trace is
conserved to better than 1e-14 over 10 ps in every run.

The inverse model is a fully connected feed-forward neural network (512-256-128 ReLU,
781,059 parameters) trained on 3000 HEOM trajectories. Given the three channels ρ₁₁(t),
Re ρ₁₂(t) and Im ρ₁₂(t) sampled every 5 fs over 2 ps, it recovers

| target | mean absolute error | R² |
|---|---|---|
| site-energy gap Δ | 3.45 cm⁻¹ | 0.997 |
| excitonic coupling 𝒱₁₂ (signed) | 1.87 cm⁻¹ | 0.997 |
| reorganisation energy λ | 1.13 cm⁻¹ | 0.990 |

The sign of the coupling is recovered in 100% of held-out cases when the coherence is supplied and
is **not recoverable at all** from populations alone: the gauge transformation |1><1| − |2><2|
reverses the sign of 𝒱₁₂ while leaving ρ₁₁(t) unchanged.

## Repository layout

    code/heom_tls_si.py            standalone HEOM propagation of both dimers + figures
    code/ann_inverse.py            training-set construction and the inverse neural network
    code/phi_compat/phi_heom.py    reads PHI-format .prm parameter files and runs them through QuTiP
    data/trajectories/*.prm        PHI-format input files, one per reorganisation energy
    data/trajectories/*_1fs.dat    trajectories on a 1 fs grid (see note below)
    data/ann/ann_dataset.npz       population-only training corpus  (X: 3000x401,  theta: 3000x3)
    data/ann/ann_dataset_coherence.npz  three-channel corpus        (X: 3000x1203, theta: 3000x3)
    figures/                       figures as published (PDF) plus raster previews

## Quick start

    conda env create -f environment.yml
    conda activate heom

Reproduce every trajectory and both population figures from scratch (about 4 minutes):

    python code/heom_tls_si.py output/

Train the inverse model. Loading the shipped corpus skips 155 s of HEOM propagation; the fit itself
takes roughly 30 minutes on a CPU:

    python -c "
    import numpy as np; exec(open('code/ann_inverse.py').read())
    d = np.load('data/ann/ann_dataset_coherence.npz')
    net, xs, ys, (yte, pred) = fit_model3(d['X'], d['theta'])
    print(np.abs(pred - yte).mean(axis=0))"

Run a PHI-format input file through the QuTiP engine:

    python code/phi_compat/phi_heom.py data/trajectories/TLS-Dynamics_300K_lambda3.9.prm rkf45

## Notes on the data

`theta` columns are (Δ, 𝒱₁₂, λ) in cm⁻¹. In `ann_dataset.npz` the coupling is positive by
construction and only its magnitude is learnable; in `ann_dataset_coherence.npz` it is signed.
Input vectors are the raw trajectory channels, unnormalised — the standardisation is fitted inside
`fit_model3` on the training partition only.

The shipped trajectories are decimated to a 1 fs grid (10,001 rows) to keep the repository small.
The full 0.1 fs output written by the simulations is 100,001 rows per run and is regenerated exactly
by `code/heom_tls_si.py`, since all random seeds and integrator tolerances are fixed. Columns are
`time_ps rho11 rho22 Re_rho12 Im_rho12`.

## Known limits

The network was trained at fixed `gamma = 10` ps⁻¹ and 300 K, with λ sampled from 1 cm⁻¹ upward, so
it extrapolates below that and would misread a trajectory generated with a different bath
correlation time. The training corpus uses an unstructured Drude spectral density and therefore does
not represent underdamped guest modes. Inputs are noise-free simulated trajectories; applying the
model to measured data requires retraining with realistic noise and finite time resolution, and
reporting uncertainty rather than a point estimate.

## Citation

If you use this code or data, please cite the accompanying paper (see `CITATION.cff` once the DOI is
assigned) and the underlying software: QuTiP (Johansson, Nation & Nori, *Comput. Phys. Commun.* 184,
1234 (2013)), its HEOM solver (Lambert et al., *Phys. Rev. Research* 5, 013181 (2023)) and
scikit-learn (Pedregosa et al., *J. Mach. Learn. Res.* 12, 2825 (2011)).
