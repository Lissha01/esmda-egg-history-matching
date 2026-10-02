"""
History matching of an Egg-type waterflood with ES-MDA.

Twin experiment:
  1. Pick a "true" permeability field and simulate it -> add noise -> "observed" data.
  2. Start from a prior ensemble of permeability fields that does NOT contain the truth.
  3. Assimilate the first HISTORY_DAYS of well data with ES-MDA.
  4. Compare prior vs. posterior ensembles: data match, permeability, and forecasts.

Usage:
  python run_history_match.py                     # synthetic Egg-type prior (default)
  python run_history_match.py --egg-dir data/egg  # real Egg Model realizations
  python run_history_match.py --ne 40 --quick     # fast test run
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from esmda_egg.esmda import default_alphas, esmda_update
from esmda_egg.geomodel import load_egg_ensemble, synthetic_ensemble
from esmda_egg.simulator import ReservoirConfig, WaterfloodSimulator
from esmda_egg import plots

LNK_MIN, LNK_MAX = np.log(1.0), np.log(20000.0)   # bounds on ln(k [mD])

_SIM = None


def _init_worker(cfg):
    global _SIM
    _SIM = WaterfloodSimulator(cfg)


def _simulate(args):
    lnk, times = args
    return _SIM.run(np.exp(lnk), times)


def to_vector(out, n_hist):
    """Stack the history-period well data into one data vector d."""
    return np.concatenate([out["qo"][:n_hist].ravel(),
                           out["qw"][:n_hist].ravel(),
                           out["bhp_inj"][:n_hist].ravel()])


def run_ensemble(pool, lnk_ens, times):
    return pool.map(_simulate, [(m, times) for m in lnk_ens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=100, help="ensemble size")
    ap.add_argument("--na", type=int, default=4, help="number of ES-MDA iterations")
    ap.add_argument("--egg-dir", default=None, help="folder with the unzipped Egg Model data")
    ap.add_argument("--history-days", type=float, default=1800.0)
    ap.add_argument("--total-days", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="coarser time steps for a fast check")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---------------- geological models -----------------------------------------
    if args.egg_dir:
        ens, active, _ = load_egg_ensemble(args.egg_dir)
        lnk_true, prior = ens[0], ens[1:args.ne + 1]
        source = f"Egg Model realizations ({len(prior)} prior members, realization 0 as truth)"
    else:
        active = np.ones((60, 60), bool)
        lnk_true = synthetic_ensemble(1, seed=999)[0]          # different seed: truth not in prior
        prior = synthetic_ensemble(args.ne, seed=args.seed)
        source = f"synthetic Egg-type Gaussian fields ({args.ne} prior members)"
    ne, ny, nx = prior.shape
    lnk_inactive = np.log(1e-3)
    lnk_true = np.where(active, lnk_true, lnk_inactive)
    cfg = ReservoirConfig(active=active)
    step = 60.0 if args.quick else 30.0
    times = np.arange(step, args.total_days + 1e-9, step)
    n_hist = int(np.sum(times <= args.history_days + 1e-9))
    print(f"Prior: {source}; {len(times)} report steps, {n_hist} in history")

    with Pool(os.cpu_count(), initializer=_init_worker, initargs=(cfg,)) as pool:
        # ---------------- synthetic "observed" data from the truth ----------------
        truth = pool.map(_simulate, [(lnk_true.ravel(), times)])[0]
        d_true = to_vector(truth, n_hist)
        nprod, ninj = truth["qo"].shape[1], truth["bhp_inj"].shape[1]
        n_rate = 2 * n_hist * nprod
        sd = np.concatenate([np.maximum(0.05 * np.abs(d_true[:n_rate]), 2.0),  # rates: 5%, min 2 m3/d
                             np.full(n_hist * ninj, 1.0)])                        # BHP: 1 bar
        d_obs = d_true + sd * rng.standard_normal(d_true.size)

        # ---------------- ES-MDA loop -------------------------------------------------
        act = active.ravel()
        M = prior.reshape(ne, -1).T.copy()          # (Nm, Ne), one column per member
        M[~act] = lnk_inactive
        alphas = default_alphas(args.na)
        history, mismatch = [], []
        t0 = time.time()
        for it in range(args.na + 1):
            outs = run_ensemble(pool, M.T, times)
            D = np.column_stack([to_vector(o, n_hist) for o in outs])
            obj = np.mean(((D - d_obs[:, None]) / sd[:, None]) ** 2, axis=0)  # normalised misfit per member
            mismatch.append(obj)
            history.append(outs)
            print(f"  iter {it}: mean normalised misfit = {obj.mean():9.2f}   ({time.time() - t0:5.0f} s)")
            if it == args.na:
                break
            M_new = esmda_update(M[act], D, d_obs, sd, alphas[it], rng)
            M[act] = np.clip(M_new, LNK_MIN, LNK_MAX)

    post = M.T.reshape(ne, ny, nx)
    prior_mean, post_mean = prior.mean(0), post.mean(0)
    rmse = lambda a: float(np.sqrt(np.mean((a[active] - lnk_true[active]) ** 2)))

    def forecast_err(outs):
        fut = slice(n_hist, None)
        tq = truth["qo"][fut].sum(1)
        return float(np.mean([np.sqrt(np.mean((o["qo"][fut].sum(1) - tq) ** 2)) for o in outs]))

    metrics = {
        "prior_source": source, "ensemble_size": ne, "esmda_iterations": args.na,
        "alphas": [round(a, 3) for a in alphas],
        "history_days": args.history_days, "total_days": args.total_days,
        "n_observations": int(d_obs.size),
        "mean_normalised_misfit_prior": float(mismatch[0].mean()),
        "mean_normalised_misfit_posterior": float(mismatch[-1].mean()),
        "lnk_rmse_prior_mean_vs_truth": rmse(prior_mean),
        "lnk_rmse_posterior_mean_vs_truth": rmse(post_mean),
        "field_oil_rate_forecast_rmse_prior_m3d": forecast_err(history[0]),
        "field_oil_rate_forecast_rmse_posterior_m3d": forecast_err(history[-1]),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))

    np.savez_compressed(os.path.join(args.out, "ensembles.npz"), lnk_true=lnk_true, prior=prior,
                        posterior=post, active=active, misfit=np.array(mismatch), d_obs=d_obs, sd=sd)
    plots.make_all(times, args.history_days, truth, d_obs, sd, n_hist, history[0], history[-1],
                   lnk_true, prior, post, active, mismatch, cfg, out_dir="figures")


if __name__ == "__main__":
    main()
