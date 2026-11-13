"""
E2d: the harm is at L3, not L1.

E2c showed that inside the silent band the model MSE is 0.88-1.01x baseline,
i.e. accuracy is NOT degraded. So the paper's harm claim cannot be "the model
gets worse". It has to be "the trust state is wrong while the model looks fine".

This script measures that directly:
  - cumulative reward share   (Shapley-free proxy: sum of assigned weights)
  - selection frequency       (top-K by score, as in Oort / client selection)
  - honest displacement eps   (fraction of rightful weight lost)
against detector excess and MSE ratio, as k_adv grows.
"""

import numpy as np
from e2_silence import SCORERS, CFG
import e2b_sweep as m

seeds = [0, 1, 2, 3, 4, 5, 6, 7]
TOPK = 10


def sim_reward(dm, sa, k, seed, rho=None):
    """Same dynamics as e2b.sim, but also logs selection and cumulative share."""
    cfg = dict(CFG); cfg["s_adv"] = sa; cfg["k_adv"] = k
    if rho is not None:
        cfg["rho"] = rho
    rng = np.random.default_rng(seed)
    d, N = cfg["d"], cfg["N"]
    adv, hon = np.arange(k), np.arange(k, N)

    w_star = rng.normal(size=d)
    dl = rng.normal(size=d); dl = dm * dl / (np.linalg.norm(dl) + 1e-12)
    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv] = sa

    Xte = rng.normal(size=(cfg["n_test"], d)); yte = Xte @ w_star
    w = np.zeros(d)
    state = {"var": np.full(N, 1.0), "hist": np.zeros((N, d)),
             "anchor_ref": np.zeros(d)}
    prev = np.zeros((N, d)); ev = np.full(N, np.nan)
    cum = np.zeros(N); sel = np.zeros(N)
    hw, aw, mse, alarms = [], [], [], []

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if i < k else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        A = []
        for _ in range(cfg["a_anchor"]):
            Xa = rng.normal(size=(cfg["n_anchor"], d))
            ya = Xa @ w_star + rng.normal(0, cfg["s_anchor"], size=cfg["n_anchor"])
            A.append(-cfg["lr"] * (Xa.T @ (Xa @ w - ya) / cfg["n_anchor"]))
        state["anchor_ref"] = np.mean(A, axis=0)

        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
            state["var"] = ev
        prev = V.copy()
        state["hist"] = (1 - cfg["rho"]) * state["hist"] + cfg["rho"] * V

        wt = SCORERS["Inverse-variance"](V, state, cfg)
        w = w + wt @ V
        cum += wt
        sel[np.argsort(-wt)[:TOPK]] += 1
        hw.append(wt[hon].mean()); aw.append(wt[adv].sum())
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))
        alarms.append(m.detectors2(V, adv, state, cfg))

    R = cfg["rounds"]
    return dict(
        aw=float(np.mean(aw[-20:])), hw=float(np.mean(hw[-20:])),
        mse=float(np.mean(mse[-20:])),
        rew_adv=float(cum[adv].sum() / cum.sum()),
        sel_adv=float(sel[adv].sum() / (R * TOPK)),
        sel_hon=float(sel[hon].sum() / (R * TOPK)),
        al={kk: float(np.mean([a[kk] for a in alarms])) for kk in alarms[0]},
    )


def agg(dm, sa, k, rho=None):
    R = [sim_reward(dm, sa, k, s, rho) for s in seeds]
    return {kk: (float(np.mean([r[kk] for r in R])) if kk != "al"
                 else {j: float(np.mean([r["al"][j] for r in R]))
                       for j in R[0]["al"]})
            for kk in R[0]}


print("=" * 88)
print("L3 harm at the SILENT operating point (s_adv=0.35, delta=0.05)")
print("=" * 88)
print(f"{'k':>3} {'fair':>6} {'advW':>7} {'reward':>7} {'x-fair':>7} "
      f"{'sel_adv':>8} {'eps':>7} {'mse/base':>9}  max-excess")
for k in [0, 1, 2, 3, 5]:
    if k == 0:
        base = agg(0.0, 1.0, 1)
        fp = base["al"]
        print(f"{'ctl':>3} {1/CFG['N']:6.3f} {base['aw']:7.3f} "
              f"{base['rew_adv']:7.3f} {base['rew_adv']*CFG['N']:7.2f} "
              f"{base['sel_adv']:8.3f} {0.0:7.3f} {1.0:9.2f}  "
              f"FP={ {j: round(v,2) for j,v in fp.items()} }")
        continue
    r = agg(0.05, 0.35, k)
    ex = max(r["al"][j] - fp.get(j, 0.0) for j in r["al"])
    fair = k / CFG["N"]
    print(f"{k:3d} {fair:6.3f} {r['aw']:7.3f} {r['rew_adv']:7.3f} "
          f"{r['rew_adv']/fair:7.2f} {r['sel_adv']:8.3f} "
          f"{1 - r['hw']/base['hw']:7.3f} {r['mse']/base['mse']:9.2f}  "
          f"{ex:+.2f}" + ("  <-- SILENT" if ex < 0.05 else ""))

print()
print("=" * 88)
print("Frontier: largest reward distortion achievable at each detectability level")
print("=" * 88)
print(f"{'s_adv':>6} {'delta':>6} {'k':>3} {'reward':>7} {'x-fair':>7} "
      f"{'mse/base':>9}  max-excess")
rows = []
for sa in [0.5, 0.4, 0.35, 0.3, 0.2]:
    for dm in [0.02, 0.05, 0.1]:
        for k in [1, 3]:
            r = agg(dm, sa, k)
            ex = max(r["al"][j] - fp.get(j, 0.0) for j in r["al"])
            rows.append((sa, dm, k, r, ex))
for sa, dm, k, r, ex in rows:
    if ex < 0.05:
        fair = k / CFG["N"]
        print(f"{sa:6.2f} {dm:6.2f} {k:3d} {r['rew_adv']:7.3f} "
              f"{r['rew_adv']/fair:7.2f} {r['mse']/base['mse']:9.2f}  {ex:+.2f}")
