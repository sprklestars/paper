"""
Generates fig/silence.pdf -- the figure the whole paper turns on.

Left:   L1 view. Test MSE over rounds, attacked vs clean. They overlap.
Right:  L3 view. Adversary reward share over rounds, attacked vs clean.
        4.6x separation.

Same run, same seeds, two observables. Anything monitoring the left panel
sees a healthy system.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from e2_silence import CFG

import matplotlib.ticker as mt

seeds = list(range(12))
S_ADV, DELTA = 0.35, 0.05
R = CFG["rounds"]


def trace(dm, sa, k, honest):
    """Re-run recording full per-round traces rather than tail means."""
    out_mse, out_rew = [], []
    for seed in seeds:
        cfg = dict(CFG); cfg["s_adv"] = sa; cfg["k_adv"] = k
        rng = np.random.default_rng(seed)
        d, N = cfg["d"], cfg["N"]
        adv, hon = np.arange(k), np.arange(k, N)
        w_star = rng.normal(size=d)
        dl = rng.normal(size=d); dl = dm * dl / (np.linalg.norm(dl) + 1e-12)
        sigma = np.empty(N)
        sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
        sigma[adv] = sa if not honest else rng.uniform(cfg["s_lo"], cfg["s_hi"], size=k)
        Xte = rng.normal(size=(cfg["n_test"], d)); yte = Xte @ w_star
        w = np.zeros(d)
        prev = np.zeros((N, d)); ev = np.full(N, np.nan)
        cum = np.zeros(N)
        ms, rw = [], []
        for t in range(cfg["rounds"]):
            V = np.empty((N, d))
            for i in range(N):
                tgt = w_star + dl if (i < k and not honest) else w_star
                X = rng.normal(size=(cfg["n_local"], d))
                y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
                V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])
            if t > 0:
                inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
                ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
            prev = V.copy()
            wt = (np.ones(N) / N if t == 0
                  else (lambda z: z / z.sum())(1.0 / np.maximum(ev, 1e-12)))
            w = w + wt @ V
            cum += wt
            ms.append(float(np.mean((Xte @ w - yte) ** 2)))
            rw.append(cum[adv].sum() / cum.sum())
        out_mse.append(ms); out_rew.append(rw)
    return np.array(out_mse), np.array(out_rew)


mse_a, rew_a = trace(DELTA, S_ADV, 1, honest=False)
mse_c, rew_c = trace(0.0, 1.0, 1, honest=True)

x = np.arange(1, R + 1)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))

C_A, C_C = "#c0392b", "#2c3e50"


def band(a, arr, c, lab, ls="-"):
    m = arr.mean(0)
    lo, hi = np.percentile(arr, 10, axis=0), np.percentile(arr, 90, axis=0)
    a.plot(x, m, color=c, lw=1.6, ls=ls, label=lab, zorder=3)
    a.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)


band(ax[0], mse_c, C_C, "No attack", ls="--")
band(ax[0], mse_a, C_A, "TSP-VAR")
ax[0].set_yscale("log")
ax[0].set_xlabel("Round")
ax[0].set_ylabel("Test MSE")
ax[0].set_title(r"(a) L1: model layer", fontsize=9.5, pad=6)
ax[0].legend(frameon=False, fontsize=8, loc="upper right")

band(ax[1], rew_c, C_C, "No attack", ls="--")
band(ax[1], rew_a, C_A, "TSP-VAR")
ax[1].axhline(1 / CFG["N"], color="#7f8c8d", lw=0.9, ls=":", zorder=1)
ax[1].annotate("uniform share $1/N$", xy=(R * 0.5, 1 / CFG["N"]),
               xytext=(0, -13), textcoords="offset points",
               fontsize=7.5, color="#7f8c8d")
ax[1].set_xlabel("Round")
ax[1].set_ylabel("Cumulative reward share")
ax[1].set_title(r"(b) L3: trust decision layer", fontsize=9.5, pad=6)
ax[1].legend(frameon=False, fontsize=8, loc="center right")

f, fc = rew_a.mean(0)[-1], rew_c.mean(0)[-1]
ax[1].annotate(f"{f / fc:.1f}$\\times$ its honest\ncounterfactual",
               xy=(R, f), xytext=(-60, 1), textcoords="offset points",
               fontsize=8, color=C_A, weight="bold", linespacing=1.15)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
    a.xaxis.set_major_locator(mt.MultipleLocator(50))
    a.grid(axis="y", lw=0.4, color="#dfe4e6", zorder=0)
    a.set_axisbelow(True)

fig.tight_layout(w_pad=2.0)
import os
os.makedirs("../paper/fig", exist_ok=True)
fig.savefig("../paper/fig/silence.pdf", bbox_inches="tight")
fig.savefig("results/silence-preview.png", dpi=200, bbox_inches="tight")
print(f"wrote ../paper/fig/silence.pdf")
print(f"final  MSE  attacked={mse_a.mean(0)[-1]:.5f}  clean={mse_c.mean(0)[-1]:.5f}"
      f"   ratio={mse_a.mean(0)[-1]/mse_c.mean(0)[-1]:.3f}")
print(f"final  reward attacked={rew_a.mean(0)[-1]:.4f}  clean={rew_c.mean(0)[-1]:.4f}"
      f"   x-fair={rew_a.mean(0)[-1]*CFG['N']:.2f}")
