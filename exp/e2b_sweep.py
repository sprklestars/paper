"""
E2b: fixed detectors + (sigma_adv, delta) sweep to locate the silent regime.

Fixes from E2:
  - krum_excl was definitionally broken: multi-Krum keeps n-k of n clients, so
    "some honest client excluded" fires every round even with no adversary.
    Replaced by krum_excl_adv (is the ADVERSARY excluded), which is what
    "the defence caught it" actually means.
  - added a no-attack control run per seed so every detector is scored as an
    EXCESS over its own false-positive rate, not as a raw rate.
"""

import numpy as np
from e2_silence import SCORERS, CFG, detectors


def detectors2(V, adv_idx, state, cfg):
    """保留旧导入入口；检测定义统一由 E2 提供。"""
    return detectors(V, adv_idx, state, cfg)


def sim(scorer_name, delta_mag, s_adv, cfg, seed, rho=None):
    rng = np.random.default_rng(seed)
    cfg = dict(cfg)
    cfg["s_adv"] = s_adv
    if rho is not None:
        cfg["rho"] = rho
    d, N, k = cfg["d"], cfg["N"], cfg["k_adv"]
    adv, hon = np.arange(k), np.arange(k, N)

    w_star = rng.normal(size=d)
    dl = rng.normal(size=d)
    dl = delta_mag * dl / np.linalg.norm(dl)

    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv] = s_adv

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    scorer = SCORERS[scorer_name]
    state = {"var": np.full(N, 1.0), "hist": np.zeros((N, d)),
             "anchor_ref": np.zeros(d)}
    prev = np.zeros((N, d))
    ev = np.full(N, np.nan)
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
            X = rng.normal(size=(cfg["n_anchor"], d))
            y = X @ w_star + rng.normal(0, cfg["s_anchor"], size=cfg["n_anchor"])
            A.append(-cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_anchor"]))
        state["anchor_ref"] = np.mean(A, axis=0)

        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
            state["var"] = ev
        prev = V.copy()
        state["hist"] = (1 - cfg["rho"]) * state["hist"] + cfg["rho"] * V

        wt = scorer(V, state, cfg)
        w = w + wt @ V
        hw.append(wt[hon].mean()); aw.append(wt[adv].sum())
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))
        alarms.append(detectors2(V, adv, state, cfg))

    return dict(hw=np.array(hw), aw=np.array(aw), mse=np.array(mse), al=alarms)


def agg(name, dm, sa, seeds, rho=None):
    R = [sim(name, dm, sa, CFG, s, rho) for s in seeds]
    keys = R[0]["al"][0].keys()
    return dict(
        aw=float(np.mean([r["aw"][-20:].mean() for r in R])),
        hw=float(np.mean([r["hw"][-20:].mean() for r in R])),
        mse=float(np.mean([r["mse"][-20:].mean() for r in R])),
        al={k: float(np.mean([np.mean([a[k] for a in r["al"]]) for r in R]))
            for k in keys},
    )


if __name__ == "__main__":
    import sys
    seeds = [0, 1, 2, 3, 4]

    print("=" * 78)
    print("CONTROL: no attack (delta=0, s_adv = honest range)")
    print("=" * 78)
    base = {}
    for nm in SCORERS:
        b = agg(nm, 0.0, 1.0, seeds)
        base[nm] = b
        print(f"{nm:20s} advW={b['aw']:.3f} honW={b['hw']:.4f} "
              f"mse={b['mse']:.4f} FP={ {k: round(v,2) for k,v in b['al'].items() if v>0.01} }")

    print()
    print("=" * 78)
    print("SWEEP: inverse-variance, capture vs detectability")
    print("=" * 78)
    print(f"{'s_adv':>7} {'delta':>6} {'advW':>7} {'honW':>8} {'eps':>6} "
          f"{'mse':>8} {'mse/base':>9}  alarms(excess over FP)")
    mse_base = base["Inverse-variance"]["mse"]
    hw_base = base["Inverse-variance"]["hw"]
    fp = base["Inverse-variance"]["al"]
    for sa in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
        for dm in [0.05, 0.1]:
            r = agg("Inverse-variance", dm, sa, seeds)
            ex = {k: round(r["al"][k] - fp.get(k, 0.0), 2)
                  for k in r["al"] if r["al"][k] - fp.get(k, 0.0) > 0.05}
            print(f"{sa:7.2f} {dm:6.2f} {r['aw']:7.3f} {r['hw']:8.5f} "
                  f"{1 - r['hw']/hw_base:6.3f} {r['mse']:8.4f} "
                  f"{r['mse']/mse_base:9.2f}  {ex}")
            sys.stdout.flush()

    print()
    print("=" * 78)
    print("ABLATION: rho (memory).  s_adv=0.1, delta=0.05")
    print("=" * 78)
    for rho in [0.0, 0.05, 0.2, 0.5, 1.0]:
        r = agg("Inverse-variance", 0.05, 0.1, seeds, rho=rho)
        print(f"rho={rho:4.2f}  advW={r['aw']:.3f}  honW={r['hw']:.5f}  "
              f"mse={r['mse']:.4f}")
        sys.stdout.flush()
