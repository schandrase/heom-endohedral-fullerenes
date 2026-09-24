"""
Pilot: recover Hamiltonian parameters from an HEOM population trajectory.

Forward model: two-level system, one Drude bath per site (gamma = 10 ps^-1,
T = 300 K, K = 0 Matsubara terms, hierarchy depth L = 10), propagated for 2 ps
and sampled every 5 fs (401 points).

Inverse model: a feed-forward neural network (sklearn MLPRegressor) maps the
401-point rho00(t) trace onto three targets

    Delta  = E1 - E2   site-energy gap        cm^-1
    J      = |H_12|    electronic coupling    cm^-1
    lambda             reorganisation energy  cm^-1

Only |J| is learnable from populations alone: the gauge transformation
diag(1,-1) flips the sign of J while leaving rho00(t) invariant, so the sign
has to come from the coherence.
"""
import time

import numpy as np
from qutip import Qobj, basis, ket2dm
from qutip.core.environment import DrudeLorentzEnvironment
from qutip.solver.heom import HEOMSolver
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CM1 = 2.0 * np.pi * 2.99792458e-2
KB = 0.6950348
GAMMA, TEMP, NK, DEPTH = 10.0, 300.0, 0, 10
TMAX, DT = 2.0, 0.005
TIMES = np.arange(0.0, TMAX + 0.5 * DT, DT)


def trajectory(delta, j, lam):
    """rho00(t) for a dimer with gap `delta`, coupling `j`, reorg energy `lam`."""
    H = Qobj(np.array([[delta / 2, j], [j, -delta / 2]]) * CM1)
    kT = KB * TEMP * CM1
    baths = [(DrudeLorentzEnvironment(T=kT, lam=lam * CM1, gamma=GAMMA)
              .approximate("matsubara", Nk=NK), ket2dm(basis(2, s)))
             for s in (0, 1)]
    solver = HEOMSolver(H, baths, max_depth=DEPTH,
                        options={"progress_bar": False, "store_states": True,
                                 "method": "adams", "rtol": 1e-6, "atol": 1e-8,
                                 "nsteps": 100000})
    res = solver.run(ket2dm(basis(2, 0)), TIMES)
    return np.array([s.full()[0, 0].real for s in res.states])


def trajectory3(delta, j, lam):
    """Three-channel trace: rho00(t), Re rho01(t), Im rho01(t).

    Re rho01 changes sign with J, so the signed coupling is identifiable from
    this input while it is not from the population alone.
    """
    H = Qobj(np.array([[delta / 2, j], [j, -delta / 2]]) * CM1)
    kT = KB * TEMP * CM1
    baths = [(DrudeLorentzEnvironment(T=kT, lam=lam * CM1, gamma=GAMMA)
              .approximate("matsubara", Nk=NK), ket2dm(basis(2, s)))
             for s in (0, 1)]
    solver = HEOMSolver(H, baths, max_depth=DEPTH,
                        options={"progress_bar": False, "store_states": True,
                                 "method": "adams", "rtol": 1e-6, "atol": 1e-8,
                                 "nsteps": 100000})
    res = solver.run(ket2dm(basis(2, 0)), TIMES)
    rho = np.array([s.full() for s in res.states])
    return np.concatenate([rho[:, 0, 0].real, rho[:, 0, 1].real, rho[:, 0, 1].imag])


def build_dataset3(n, seed=0):
    """Signed-J dataset with population + coherence channels."""
    rng = np.random.default_rng(seed)
    theta = np.column_stack([
        rng.uniform(-200.0, 200.0, n),                               # Delta
        rng.choice([-1.0, 1.0], n) * rng.uniform(10.0, 120.0, n),    # signed J
        rng.uniform(1.0, 60.0, n),                                   # lambda
    ])
    X = np.empty((n, 3 * TIMES.size))
    t0 = time.time()
    for i, (d, j, lam) in enumerate(theta):
        X[i] = trajectory3(d, j, lam)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return X, theta


def build_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.column_stack([
        rng.uniform(-200.0, 200.0, n),    # Delta
        rng.uniform(10.0, 120.0, n),      # |J|
        rng.uniform(1.0, 60.0, n),        # lambda
    ])
    X = np.empty((n, TIMES.size))
    t0 = time.time()
    for i, (d, j, lam) in enumerate(theta):
        X[i] = trajectory(d, j, lam)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return X, theta


def fit_model(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(ytr)
    net = MLPRegressor(hidden_layer_sizes=(512, 256, 128), activation="relu",
                       alpha=1e-4, learning_rate_init=1e-3, batch_size=64,
                       max_iter=600, early_stopping=True, n_iter_no_change=25,
                       random_state=seed)
    net.fit(xs.transform(Xtr), ys.transform(ytr))
    pred = ys.inverse_transform(net.predict(xs.transform(Xte)))
    return net, xs, ys, (yte, pred)


def fit_model3(X, y, seed=2):
    """Fit the three-channel (population + coherence) model.

    The 1203-feature input needs a longer schedule than the population-only
    model: with the short patience used in `fit_model` training stops after
    ~50 epochs and the coupling magnitude is a factor of two worse.
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    xs, ys_ = StandardScaler().fit(Xtr), StandardScaler().fit(ytr)
    net = MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=1e-4,
                       batch_size=64, learning_rate_init=1e-3, max_iter=1500,
                       early_stopping=True, n_iter_no_change=60,
                       validation_fraction=0.12, random_state=seed)
    net.fit(xs.transform(Xtr), ys_.transform(ytr))
    pred = ys_.inverse_transform(net.predict(xs.transform(Xte)))
    return net, xs, ys_, (yte, pred)


def predict(net, xs, ys, traces):
    return ys.inverse_transform(net.predict(xs.transform(np.atleast_2d(traces))))
