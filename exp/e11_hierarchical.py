"""
E11: Hierarchical (client-edge-cloud) aggregation, and why the protection it
appears to offer is an unverified claim.

Deployed edge FL is not flat. In the client-edge-cloud arrangement, edge
servers perform PARTIAL aggregation over their own cluster and the cloud
combines the edge results. This changes the algebra of trust-state capture in
a way that looks like a defence:

  effective weight of client i  =  (1 / m) * (its weight WITHIN its cluster)

so an adversary confined to one cluster cannot take more than 1/m of the
global model no matter what dispersion it reports. Capture is contained by
the topology rather than by any security mechanism.

The containment is real but conditional, and the condition is the thing this
paper is about: it holds only while the adversary is CONFINED to one cluster.
Cluster membership is a function of which edge server a device attaches to,
which is network topology -- reported, not verified. We therefore measure
three settings at a FIXED malicious fraction, so placement is the only
variable:

  flat       : one aggregation over all N clients
  clustered  : m edge clusters, all k adversaries inside one cluster
  spread     : m edge clusters, adversaries spread one per cluster

Generative model is identical to E2 so the numbers are comparable:
  honest i : y = X @ w_star           + N(0, sigma_i^2), sigma_i in [s_lo,s_hi]
  adversary: y = X @ (w_star + delta) + N(0, s_adv^2),   s_adv << s_lo

Note on faithfulness: HierFAVG runs tau local steps between edge
aggregations. We use tau = 1 because what is under test is the WEIGHT algebra
of two-level aggregation, not its convergence rate; tau enters the
communication cost, not the influence allocation.
"""

import numpy as np

# ---------------------------------------------------------------- weighting


def invvar_within(var, idx, floor):
    """Family I inverse-variance, normalised within a client subset."""
    v = np.maximum(var[idx], floor ** 2)
    w = 1.0 / v
    return w / w.sum()


def make_groups(N, m, k, placement, rng):
    """Partition clients 0..N-1 into m equal clusters.

    Adversaries are indices 0..k-1. `placement` decides where they land.
    """
    n = N // m
    assert n * m == N, "N must divide evenly into m clusters"
    adv = np.arange(k)
    honest = np.arange(k, N)
    honest = rng.permutation(honest)

    groups = []
    if placement == "clustered":
        # all adversaries in cluster 0, padded with honest clients
        assert k <= n, "clustered placement needs k <= cluster size"
        g0 = np.concatenate([adv, honest[: n - k]])
        groups.append(g0)
        rest = honest[n - k :]
        for j in range(m - 1):
            groups.append(rest[j * n : (j + 1) * n])
    elif placement == "spread":
        assert k <= m, "spread placement needs k <= number of clusters"
        pos = 0
        for j in range(m):
            take = n - 1 if j < k else n
            body = honest[pos : pos + take]
            pos += take
            groups.append(np.concatenate([[adv[j]], body]) if j < k else body)
    else:
        raise ValueError(placement)

    allidx = np.sort(np.concatenate(groups))
    assert np.array_equal(allidx, np.arange(N)), "partition is not a partition"
    assert all(len(g) == n for g in groups), "clusters not equal size"
    return groups


def effective_weights(V, var, groups, cfg, mode):
    """Return per-client effective weight in the global model."""
    N = V.shape[0]
    if mode == "flat":
        w = np.zeros(N)
        w[:] = invvar_within(var, np.arange(N), cfg["var_floor"])
        return w
    m = len(groups)
    w = np.zeros(N)
    for g in groups:
        w[g] = invvar_within(var, g, cfg["var_floor"]) / m
    return w


# ---------------------------------------------------------------- simulation


def run(mode, placement, m, cfg, seed):
    rng = np.random.default_rng(seed)
    d, N, k = cfg["d"], cfg["N"], cfg["k_adv"]
    adv_idx = np.arange(k)

    w_star = rng.normal(size=d)
    delta = rng.normal(size=d)
    delta = cfg["delta_mag"] * delta / np.linalg.norm(delta)

    sigma = np.empty(N)
    sigma[k:] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv_idx] = cfg["s_adv"]

    groups = make_groups(N, m, k, placement, rng) if mode == "hier" else None

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    var = np.full(N, 1.0)
    prev = np.zeros((N, d))
    ema = np.full(N, np.nan)

    aw, mse, ceil_ok = [], [], True
    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + delta if i < k else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ema = inst if t == 1 else (1 - cfg["rho"]) * ema + cfg["rho"] * inst
            var = ema
        prev = V.copy()

        wt = effective_weights(V, var, groups, cfg, mode)
        assert abs(wt.sum() - 1.0) < 1e-10, "effective weights must sum to 1"
        w = w + wt @ V

        a = float(wt[adv_idx].sum())
        aw.append(a)
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))
        # topology ceiling: adversaries confined to one cluster hold <= 1/m
        if mode == "hier" and placement == "clustered" and a > 1.0 / m + 1e-9:
            ceil_ok = False

    tail = slice(cfg["rounds"] // 2, None)
    return dict(
        adv_weight=float(np.mean(aw[tail])),
        mse=float(np.mean(mse[tail])),
        ceiling_held=ceil_ok,
    )


CFG = dict(
    d=20, N=30, k_adv=3, rounds=200, lr=0.30,
    n_local=50, n_test=2000, delta_mag=0.2,
    s_lo=0.5, s_hi=1.5, s_adv=0.02,
    rho=0.2, var_floor=1e-8,
)

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5]
    fair = CFG["k_adv"] / CFG["N"]
    M_LIST = [2, 3, 5, 6, 10]

    def agg(mode, placement, m):
        R = [run(mode, placement, m, CFG, s) for s in seeds]
        return (
            float(np.median([r["adv_weight"] for r in R])),
            float(np.median([r["mse"] for r in R])),
            all(r["ceiling_held"] for r in R),
        )

    print("=" * 74)
    print("E11: hierarchical edge aggregation vs adversary placement")
    print(f"N={CFG['N']} k={CFG['k_adv']} fair share={fair:.4f} "
          f"seeds={len(seeds)} rounds={CFG['rounds']}")
    print("=" * 74)

    fw, fmse, _ = agg("flat", None, 1)
    print(f"\nflat (one aggregation over N={CFG['N']})")
    print(f"  adv weight {fw:.4f}   capture {fw/fair:6.2f}x   mse {fmse:.4f}")

    print(f"\n{'m':>3} {'n':>3} | {'clustered':>22} | {'spread':>22}")
    print(f"{'':>3} {'':>3} | {'weight':>8} {'capture':>8} {'ceil':>4} |"
          f" {'weight':>8} {'capture':>8} {'ratio':>4}")
    rows = []
    for m in M_LIST:
        n = CFG["N"] // m
        cw, cmse, ceil = agg("hier", "clustered", m)
        if CFG["k_adv"] <= m:
            sw, smse, _ = agg("hier", "spread", m)
        else:
            sw = smse = float("nan")
        rows.append((m, n, cw, sw, ceil))
        sp = (f"{sw:8.4f} {sw/fair:8.2f}x {sw/cw:4.1f}"
              if sw == sw else f"{'--':>8} {'--':>9} {'--':>4}")
        print(f"{m:3d} {n:3d} | {cw:8.4f} {cw/fair:8.2f}x "
              f"{'ok' if ceil else 'FAIL':>4} | {sp}")

    # ---- claims -----------------------------------------------------------
    print("\n" + "-" * 74)
    for m, n, cw, sw, ceil in rows:
        assert ceil, f"topology ceiling 1/m violated at m={m}"
        assert cw < fw, f"clustering should contain capture (m={m})"
        assert cw <= 1.0 / m + 1e-9, f"clustered share exceeds 1/m at m={m}"
    print("PASS: clustered capture is contained, and bounded by 1/m, for all m")

    sp_rows = [(m, cw, sw) for m, n, cw, sw, ceil in rows if sw == sw]
    for m, cw, sw in sp_rows:
        assert sw > cw, f"spread must beat clustered at m={m}"
    print("PASS: spreading the SAME adversaries across clusters beats "
          "confining them, at every m")

    m5 = [r for r in rows if r[0] == 5][0]
    amp = m5[3] / m5[2]
    print(f"\nE11 CLAIMS: flat {fw/fair:.2f}x | m=5 clustered "
          f"{m5[2]/fair:.2f}x | m=5 spread {m5[3]/fair:.2f}x "
          f"| placement amplification {amp:.2f}x")
