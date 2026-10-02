import numpy as np

from esmda_egg.esmda import default_alphas, esmda_update
from esmda_egg.simulator import ReservoirConfig, WaterfloodSimulator


def test_alphas_sum_to_one():
    for na in (1, 2, 4, 8):
        assert np.isclose(np.sum(1.0 / default_alphas(na)), 1.0)


def test_esmda_matches_kalman_on_linear_gaussian_problem():
    """For a linear forward model, ES-MDA with a large ensemble must reproduce the exact
    Bayesian (Kalman) posterior mean, whatever the number of assimilations."""
    rng = np.random.default_rng(0)
    nm, nd, ne = 5, 8, 20000
    G = rng.normal(size=(nd, nm))
    m_true = rng.normal(size=nm)
    sd = np.full(nd, 0.5)
    d_obs = G @ m_true + sd * rng.normal(size=nd)
    # exact posterior mean for prior N(0, I)
    Cd = np.diag(sd ** 2)
    post_exact = G.T @ np.linalg.solve(G @ G.T + Cd, d_obs)
    M = rng.normal(size=(nm, ne))
    for a in default_alphas(4):
        M = esmda_update(M, G @ M, d_obs, sd, a, rng)
    assert np.allclose(M.mean(1), post_exact, atol=0.05)


def test_simulator_mass_balance():
    """Incompressible flow: total produced volume rate equals total injection rate."""
    cfg = ReservoirConfig(nx=20, ny=20, injectors={"I1": (2, 2)}, producers={"P1": (19, 19)})
    out = WaterfloodSimulator(cfg).run(np.full((20, 20), 300.0), np.arange(30, 601, 30.0))
    total = out["qo"].sum(1) + out["qw"].sum(1)
    assert np.allclose(total, cfg.inj_rate, rtol=1e-6)
    assert out["qw"][0, 0] < out["qw"][-1, 0]  # water arrives over time
