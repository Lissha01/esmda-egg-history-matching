"""Figures for the README: permeability maps, well-data fans, misfit reduction."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIOR = "#a3a29c"      # neutral gray for the prior ensemble
POST = "#2a78d6"       # blue for the posterior ensemble
TRUTH = "#0b0b0b"      # truth in ink
OBS = "#eb6834"        # observed data (orange)
INK2 = "#52514e"
SEQ = "Blues"          # sequential, single hue: magnitude of ln(k)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.6, "figure.dpi": 130, "savefig.bbox": "tight",
})


def _wells(ax, cfg):
    for name, (i, j) in cfg.injectors.items():
        ax.plot(i - 1, j - 1, "v", ms=6, mfc="white", mec=TRUTH, mew=1.2)
    for name, (i, j) in cfg.producers.items():
        ax.plot(i - 1, j - 1, "o", ms=6, mfc=OBS, mec="white", mew=1.0)
        ax.annotate(name.replace("PROD", "P"), (i - 1, j - 1), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=TRUTH)


def perm_maps(lnk_true, prior, post, active, cfg, path):
    panels = [("Truth", lnk_true), ("Prior mean", prior.mean(0)), ("Posterior mean", post.mean(0)),
              ("Prior member 1", prior[0]), ("Posterior member 1", post[0]),
              ("Posterior std. dev.", post.std(0))]
    vmin, vmax = np.percentile(lnk_true[active], [2, 98])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, (title, f) in zip(axes.ravel(), panels):
        is_sd = "std" in title
        img = np.ma.masked_where(~active, f)
        im = ax.imshow(img, origin="lower", cmap="Oranges" if is_sd else SEQ,
                       vmin=None if is_sd else vmin, vmax=None if is_sd else vmax)
        _wells(ax, cfg)
        ax.set_title(title, fontsize=10, color=TRUTH, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label="std. dev. of ln k" if is_sd else "ln k  (k in mD)")
    fig.suptitle("Log-permeability: truth, prior and ES-MDA posterior   "
                 "(▽ injector, ● producer)", x=0.01, ha="left", fontsize=11)
    fig.savefig(path); plt.close(fig)


def well_fans(times, t_hist, truth, d_obs, n_hist, prior_out, post_out, path):
    nprod = truth["qw"].shape[1]
    nh = n_hist
    obs_qo = d_obs[:nh * nprod].reshape(nh, nprod)
    obs_qw = d_obs[nh * nprod:2 * nh * nprod].reshape(nh, nprod)
    yrs = times / 365.25
    fig, axes = plt.subplots(2, nprod, figsize=(12, 5.6), sharex=True)
    for row, (key, obs, lab) in enumerate([("qo", obs_qo, "Oil rate (m³/d)"),
                                           ("qw", obs_qw, "Water rate (m³/d)")]):
        for w in range(nprod):
            ax = axes[row, w]
            for o in prior_out:
                ax.plot(yrs, o[key][:, w], color=PRIOR, lw=0.6, alpha=0.35)
            for o in post_out:
                ax.plot(yrs, o[key][:, w], color=POST, lw=0.6, alpha=0.35)
            ax.plot(yrs, truth[key][:, w], color=TRUTH, lw=2)
            ax.plot(yrs[:nh], obs[:, w], "o", ms=3, color=OBS, mec="white", mew=0.4)
            ax.axvline(t_hist / 365.25, color=INK2, lw=1, ls="--")
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(f"PROD{w + 1}", loc="left", fontsize=10, color=TRUTH)
            if w == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel("Time (years)")
    axes[0, 0].text(t_hist / 365.25 - 0.15, axes[0, 0].get_ylim()[1] * 0.95, "history ← | → forecast",
                    ha="center", va="top", fontsize=7, color=INK2,
                    bbox=dict(fc="white", ec="none", pad=1))
    handles = [plt.Line2D([], [], color=PRIOR, lw=2, label="Prior ensemble"),
               plt.Line2D([], [], color=POST, lw=2, label="Posterior ensemble (ES-MDA)"),
               plt.Line2D([], [], color=TRUTH, lw=2, label="Truth"),
               plt.Line2D([], [], color=OBS, marker="o", ls="", label="Observed (noisy)")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.03))
    fig.savefig(path); plt.close(fig)


def misfit(mismatch, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    data = [np.log10(m) for m in mismatch]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(PRIOR if i == 0 else POST); b.set_edgecolor("white"); b.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color(TRUTH)
    ax.set_xticks(range(1, len(data) + 1), ["Prior"] + [f"Iter {i}" for i in range(1, len(data))])
    ax.set_ylabel("log₁₀ normalised data misfit")
    ax.set_title("Data misfit across ES-MDA iterations", loc="left", fontsize=10, color=TRUTH)
    ax.axhline(0, color=OBS, lw=1, ls="--")
    ax.text(0.55, 0.04, "misfit ≈ 1: matched to noise level", color=INK2, fontsize=7, ha="left", va="bottom")
    fig.savefig(path); plt.close(fig)


def make_all(times, t_hist, truth, d_obs, sd, n_hist, prior_out, post_out,
             lnk_true, prior, post, active, mismatch, cfg, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    perm_maps(lnk_true, prior, post, active, cfg, os.path.join(out_dir, "permeability_maps.png"))
    well_fans(times, t_hist, truth, d_obs, n_hist, prior_out, post_out,
              os.path.join(out_dir, "well_rates.png"))
    misfit(mismatch, os.path.join(out_dir, "misfit.png"))
