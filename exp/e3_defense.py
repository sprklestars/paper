"""
E3: do M1 (variance floor) and M2 (anchor mixing) actually bound the harm?

Measured at the silent operating point established by E2d: s_adv=0.35, delta=0.05,
k=1, where the undefended system gives the adversary 4.6x its fair reward share
with NO detector excess and NO accuracy loss.

M1  variance floor c:      w_i propto 1 / max(var_i, c)
M2  anchor mixing lambda:  w = (1-lam) * uniform + lam * invvar   (lam<1 caps
    any single client's weight at (1-lam)/N + lam, independent of its claimed
    variance -- this is the mechanism behind the trust-displacement bound)

Cost is the point: a defence that removes the capture by destroying the
variance weighting is not a defence, it is FedAvg. So every row reports MSE
against BOTH the undefended-clean baseline and FedAvg.
"""

import numpy as np
from e2_silence import CFG
import e2b_sweep as m

seeds = [0, 1, 2, 3, 4, 5, 6, 7]
S_ADV, DELTA, K = 0.35, 0.05, 1


def sim_def(dm, sa, floor, lam, seed, honest_only=False):
    cfg = dict(CFG); cfg["s_adv"] = sa; cfg["k_adv"] = K
    rng = np.random.default_rng(seed)
    d, N = cfg["d"], cfg["N"]
    adv, hon = np.arange(K), np.arange(K, N)

    w_star = rng.normal(size=d)
    dl = rng.normal(size=d); dl = dm * dl / (np.linalg.norm(dl) + 1e-12)
    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - K)
    sigma[adv] = sa if not honest_only else rng.uniform(cfg["s_lo"], cfg["s_hi"])

    Xte = rng.normal(size=(cfg["n_test"], d)); yte = Xte @ w_star
    w = np.zeros(d)
    prev = np.zeros((N, d)); ev = np.full(N, np.nan)
    cum = np.zeros(N)
    aw, hw, mse = [], [], []

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if (i < K and not honest_only) else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
        prev = V.copy()

        if t == 0:
            wt = np.ones(N) / N
        else:
            iv = 1.0 / np.maximum(ev, floor)
            iv = iv / iv.sum()
            wt = (1.0 - lam) * (np.ones(N) / N) + lam * iv

        w = w + wt @ V
        cum += wt
        aw.append(wt[adv].sum()); hw.append(wt[hon].mean())
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))

    return dict(aw=float(np.mean(aw[-20:])), hw=float(np.mean(hw[-20:])),
                mse=float(np.mean(mse[-20:])),
                rew=float(cum[adv].sum() / cum.sum()),
                wmax=float(max(aw[-20:])))


def agg(floor, lam, dm=DELTA, sa=S_ADV, honest_only=False):
    R = [sim_def(dm, sa, floor, lam, s, honest_only) for s in seeds]
    return {k: float(np.mean([r[k] for r in R])) for k in R[0]}


fair = K / CFG["N"]
# references: FedAvg on an all-honest population, and invvar on all-honest
ref_fedavg = agg(1e-12, 0.0, dm=0.0, honest_only=True)
ref_iv = agg(1e-12, 1.0, dm=0.0, honest_only=True)

print("=" * 84)
print("REFERENCES (no adversary)")
print("=" * 84)
print(f"FedAvg           mse={ref_fedavg['mse']:.4f}")
print(f"Inverse-variance mse={ref_iv['mse']:.4f}   "
      f"(gain from variance weighting: {ref_fedavg['mse']/ref_iv['mse']:.2f}x)")

print()
print("=" * 84)
print(f"M1  variance floor    (s_adv={S_ADV}, delta={DELTA}, k=1, fair={fair:.3f})")
print("=" * 84)
print(f"{'floor':>10} {'advW':>7} {'reward':>7} {'x-fair':>7} {'eps':>7} "
      f"{'mse':>8} {'vs FedAvg':>10}")
for fl in [1e-12, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    r = agg(fl, 1.0)
    print(f"{fl:10.0e} {r['aw']:7.3f} {r['rew']:7.3f} {r['rew']/fair:7.2f} "
          f"{1 - r['hw']/(1/CFG['N']):7.3f} {r['mse']:8.4f} "
          f"{r['mse']/ref_fedavg['mse']:10.2f}")

print()
print("=" * 84)
print("M2  anchor mixing lambda   (bound: any client's weight <= (1-lam)/N + lam)")
print("=" * 84)
print(f"{'lambda':>8} {'advW':>7} {'bound':>7} {'reward':>7} {'x-fair':>7} "
      f"{'eps':>7} {'mse':>8} {'vs FedAvg':>10}")
for lam in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    r = agg(1e-12, lam)
    bound = (1 - lam) / CFG["N"] + lam
    print(f"{lam:8.2f} {r['aw']:7.3f} {bound:7.3f} {r['rew']:7.3f} "
          f"{r['rew']/fair:7.2f} {1 - r['hw']/(1/CFG['N']):7.3f} "
          f"{r['mse']:8.4f} {r['mse']/ref_fedavg['mse']:10.2f}")

print()
print("=" * 84)
print("M1+M2 combined, and the utility cost on an ALL-HONEST population")
print("=" * 84)
print(f"{'floor':>10} {'lam':>6} {'x-fair':>7} {'mse_atk':>8} "
      f"{'mse_honest':>11} {'cost vs iv':>11}")
for fl, lam in [(1e-12, 1.0), (3e-3, 1.0), (1e-12, 0.75),
                (3e-3, 0.75), (1e-2, 0.5), (3e-3, 0.9)]:
    ra = agg(fl, lam)
    rh = agg(fl, lam, dm=0.0, honest_only=True)
    print(f"{fl:10.0e} {lam:6.2f} {ra['rew']/fair:7.2f} {ra['mse']:8.4f} "
          f"{rh['mse']:11.4f} {rh['mse']/ref_iv['mse']:11.2f}")
