"""
Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple data
assimilation. Computers & Geosciences, 55, 3-15.

Idea in one line: instead of assimilating all data once (ES), assimilate the same
data Na times with the observation-error covariance inflated by alpha_i, where
sum(1/alpha_i) = 1. Each small step keeps the update closer to linear.

Update for every ensemble member j, at each step i:
    d_uc,j = d_obs + sqrt(alpha_i) * C_D^(1/2) * z_j,   z_j ~ N(0, I)
    m_j   <- m_j + C_MD (C_DD + alpha_i C_D)^-1 (d_uc,j - d_j)

  m     = model parameters (here: log-permeability of every grid cell)
  d     = simulated data g(m);  d_obs = observed (historical) data
  C_D   = observation-error covariance (diagonal here)
  C_MD  = cross-covariance parameters <-> predicted data (from the ensemble)
  C_DD  = auto-covariance of predicted data (from the ensemble)
"""
import numpy as np


def default_alphas(na=4):
    """Inflation factors. For Na=4 we use the values recommended by Emerick & Reynolds (2013)."""
    if na == 4:
        a = np.array([9.333, 7.0, 4.0, 2.0])
    else:
        a = np.full(na, float(na))
    a = a * np.sum(1.0 / a)  # rescale so that sum(1/alpha) == 1 exactly
    return a


def esmda_update(M, D, d_obs, sd_obs, alpha, rng):
    """
    One ES-MDA analysis step.

    M      : (Nm, Ne) parameter ensemble
    D      : (Nd, Ne) predicted data ensemble
    d_obs  : (Nd,)    observed data
    sd_obs : (Nd,)    observation-error standard deviations (C_D = diag(sd_obs**2))
    alpha  : inflation factor for this step
    returns updated (Nm, Ne) ensemble
    """
    Nd, Ne = D.shape
    # Perturb observations with inflated noise
    D_uc = d_obs[:, None] + np.sqrt(alpha) * sd_obs[:, None] * rng.standard_normal((Nd, Ne))
    # Ensemble anomalies (deviations from the mean), scaled by 1/sqrt(Ne-1)
    dM = (M - M.mean(1, keepdims=True)) / np.sqrt(Ne - 1)
    dD = (D - D.mean(1, keepdims=True)) / np.sqrt(Ne - 1)
    # Work with data scaled by C_D^(-1/2) so every datum is dimensionless
    S = dD / sd_obs[:, None]                       # (Nd, Ne)
    R = (D_uc - D) / sd_obs[:, None]               # scaled innovations
    # X = (S S^T + alpha I)^-1 R, solved in whichever space is smaller
    if Ne < Nd:
        # ensemble space, Ne x Ne system (Woodbury identity)
        small = S.T @ S + alpha * np.eye(Ne)
        X = (R - S @ np.linalg.solve(small, S.T @ R)) / alpha
    else:
        # data space, Nd x Nd system
        X = np.linalg.solve(S @ S.T + alpha * np.eye(Nd), R)
    # C_MD (C_DD + alpha C_D)^-1 (d_uc - d) = dM S^T X
    return M + dM @ (S.T @ X)
