"""A5: the trust score is not a function of the model.

A1-A4 were an attempt to build an attack out of the network's symmetry group,
and it failed.  A2 (hide a backdoor by gauge descent) failed, A3 (manufacture
apparent stability) failed against an AM-GM floor, and A4.2 showed detectability
outruns damage by three orders of magnitude.  Distance-based scoring is robust
to gauge manipulation.

The reason it is robust is the defect.  Distance scoring is hypersensitive to
the gauge -- A4.2 moves the adversary from Krum rank 4 to rank 29 at a
per-unit log-sd of 0.05, with the aggregate unharmed at 1.00x clean.  A scorer
that reacts that violently to a transformation the model is invariant under is
not measuring the model.  Turned around:

    an HONEST client whose parameters sit in a different gauge is rejected,
    with no adversary anywhere in the federation.

This is specific to neural networks.  A linear model has no orbit, so two
clients that fit the same function have the same parameters up to estimation
noise.  Two networks that fit the same function need not have parameters that
are close at all -- and nothing in FL forces them into a common gauge except
the round-by-round synchronisation, which is exactly what breaks under the
conditions IoT federations actually run in: partial participation, stale
clients, local restarts, and heterogeneous local epochs.

  A5.1  honest clients only: what does drifting out of sync cost in trust?
  A5.2  stragglers -- the realistic mechanism, no adversary
  A5.3  who gets excluded, and is the exclusion correlated with anything real?
  A5.4  the linear control: same protocol, no orbit
  A5.5  does gauge canonicalisation repair it, and what does it cost?
"""

import numpy as np

from a1_nn_symmetry import (D_IN, H1, H2, flat, forward, gauge_permute,
                            init_net, krum_score, sgd)
from a4_aggregate_gauge import gauge_canon, scale_units, unflat


def fed(N=20, T=12, n_per=150, seed=0, local=80, straggle=0.0,
        canon=False, rng_drift=0.0):
    """Honest federation.  No adversary anywhere.

    straggle   fraction of clients that skip each round (partial
               participation), so their parameters age out of the common gauge
    rng_drift  per-unit gauge drift applied by clients as an artefact of local
               reparameterisation (e.g. differing normalisation conventions)
    """
    r = np.random.default_rng(seed)
    teacher = init_net(r)
    glob = init_net(r)
    data = []
    for _ in range(N):
        X = r.normal(size=(n_per, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.2
        data.append((X, y))

    Xp = r.normal(size=(1500, D_IN))
    yp = forward(teacher, Xp)[0]
    local_state = [None] * N
    excluded = np.zeros(N)

    for t in range(T):
        subs = []
        for i in range(N):
            if r.random() < straggle and local_state[i] is not None:
                start = local_state[i]              # stale: never resynced
            else:
                start = glob
            p = sgd(start, data[i][0], data[i][1], steps=local, lr=0.03, rng=r)
            if rng_drift > 0:
                p = scale_units(p, np.exp(r.normal(size=H1) * rng_drift),
                                np.exp(r.normal(size=H2) * rng_drift))
            local_state[i] = p
            subs.append(flat(gauge_canon(p) if canon else p))
        V = np.array(subs)

        # Krum-style selection: keep the N-f closest, drop the rest
        sc = np.array([krum_score(V, i) for i in range(len(V))])
        keep = np.argsort(sc)[:int(0.7 * N)]
        excluded += np.array([0.0 if i in set(keep.tolist()) else 1.0
                              for i in range(N)])
        glob = unflat(V[keep].mean(axis=0), glob)

    mse = float(np.mean((forward(glob, Xp)[0] - yp) ** 2))
    # per-client honest quality, to test whether exclusion tracks anything real
    qual = np.array([float(np.mean((forward(local_state[i], Xp)[0] - yp) ** 2))
                     for i in range(N)])
    return excluded / T, mse, qual


# =========================================================== A5.1 =========
def a5_1_gauge_drift():
    print("=" * 76)
    print("A5.1  honest clients only: what does gauge drift cost in trust?")
    print("=" * 76)
    print(f"{'per-unit drift':>16} {'max excl rate':>15} {'clients excl >50%':>19} "
          f"{'final MSE':>11}")
    for dr in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        ex, mse, _ = fed(rng_drift=dr, seed=0)
        print(f"{dr:16.2f} {ex.max():15.2f} {int((ex > 0.5).sum()):19d} {mse:11.4f}")
    print("\n  Every client here is honest and trains on identically distributed")
    print("  data.  Exclusion is caused by the choice of coordinates alone.\n")


# =========================================================== A5.2 =========
def a5_2_stragglers():
    print("=" * 76)
    print("A5.2  stragglers: the realistic mechanism, still no adversary")
    print("=" * 76)
    print(f"{'straggle rate':>15} {'max excl':>10} {'mean excl':>11} "
          f"{'excl >50%':>11} {'final MSE':>11}")
    for s in [0.0, 0.1, 0.2, 0.3, 0.5]:
        e, m, q = [], [], []
        for sd in range(4):
            ex, mse, _ = fed(straggle=s, seed=sd)
            e.append(ex)
            m.append(mse)
        e = np.array(e)
        print(f"{s:15.2f} {e.max():10.2f} {e.mean():11.3f} "
              f"{np.mean([(x > 0.5).sum() for x in e]):11.2f} {np.mean(m):11.4f}")
    print("\n  A device that misses rounds -- battery, duty cycle, radio -- is")
    print("  penalised for having drifted, not for being wrong.\n")


# =========================================================== A5.3 =========
def a5_3_exclusion_vs_quality():
    print("=" * 76)
    print("A5.3  is exclusion correlated with anything REAL?")
    print("=" * 76)
    print(f"{'setting':>28} {'corr(excl, own MSE)':>21} {'p-ish':>9}")
    for name, kw in [("in sync, no drift", dict()),
                     ("gauge drift 0.1", dict(rng_drift=0.1)),
                     ("stragglers 0.3", dict(straggle=0.3))]:
        cs = []
        for sd in range(6):
            ex, _, qual = fed(seed=sd, **kw)
            if ex.std() > 1e-9 and qual.std() > 1e-9:
                cs.append(np.corrcoef(ex, qual)[0, 1])
        cs = np.array(cs)
        se = cs.std() / max(1, np.sqrt(len(cs)))
        print(f"{name:>28} {cs.mean():21.3f} {se:9.3f}")
    print("\n  Correlation near zero means the scorer is excluding clients for")
    print("  reasons unrelated to the quality of what they computed.\n")


# =========================================================== A5.4 =========
def a5_4_linear_control():
    print("=" * 76)
    print("A5.4  the control: identical protocol on a model with no orbit")
    print("=" * 76)
    d, N, T, n = 20, 20, 12, 150
    print(f"{'straggle rate':>15} {'max excl':>10} {'mean excl':>11} {'final MSE':>11}")
    for s in [0.0, 0.1, 0.3, 0.5]:
        E, M = [], []
        for sd in range(4):
            r = np.random.default_rng(sd)
            wt = r.normal(size=d)
            w = np.zeros(d)
            data = [(r.normal(size=(n, d)), None) for _ in range(N)]
            data = [(X, X @ wt + r.normal(size=n) * 0.2) for X, _ in data]
            state = [None] * N
            ex = np.zeros(N)
            for t in range(T):
                V = []
                for i in range(N):
                    start = state[i] if (r.random() < s and state[i] is not None) else w
                    X, y = data[i]
                    g = X.T @ (X @ start - y) / n
                    p = start - 0.02 * g
                    for _ in range(79):
                        p = p - 0.02 * (X.T @ (X @ p - y) / n)
                    state[i] = p
                    V.append(p)
                V = np.array(V)
                sc = np.array([krum_score(V, i) for i in range(N)])
                keep = np.argsort(sc)[:int(0.7 * N)]
                ex += np.array([0.0 if i in set(keep.tolist()) else 1.0
                                for i in range(N)])
                w = V[keep].mean(axis=0)
            Xp = r.normal(size=(1500, d))
            E.append(ex / T)
            M.append(float(np.mean((Xp @ w - Xp @ wt) ** 2)))
        E = np.array(E)
        print(f"{s:15.2f} {E.max():10.2f} {E.mean():11.3f} {np.mean(M):11.6f}")
    print("\n  Same straggling, same scorer, no orbit.  Whatever exclusion")
    print("  appears here is the baseline cost of staleness; the excess in")
    print("  A5.2 over this row is what the symmetry group adds.\n")


# =========================================================== A5.5 =========
def a5_5_repair():
    print("=" * 76)
    print("A5.5  does gauge canonicalisation repair it, and what does it cost?")
    print("=" * 76)
    print(f"{'setting':>24} {'max excl':>10} {'MSE':>10} | "
          f"{'canon excl':>11} {'canon MSE':>11}")
    for name, kw in [("in sync", dict()),
                     ("gauge drift 0.1", dict(rng_drift=0.1)),
                     ("gauge drift 0.4", dict(rng_drift=0.4)),
                     ("stragglers 0.3", dict(straggle=0.3)),
                     ("stragglers 0.5", dict(straggle=0.5))]:
        a = [fed(seed=sd, canon=False, **kw) for sd in range(4)]
        b = [fed(seed=sd, canon=True, **kw) for sd in range(4)]
        print(f"{name:>24} {np.mean([x[0].max() for x in a]):10.2f} "
              f"{np.mean([x[1] for x in a]):10.4f} | "
              f"{np.mean([x[0].max() for x in b]):11.2f} "
              f"{np.mean([x[1] for x in b]):11.4f}")
    print("\n  Canonicalisation is a gauge move, so it is free: it changes no")
    print("  client's function.  It addresses the continuous group only.\n")


if __name__ == "__main__":
    a5_1_gauge_drift()
    a5_2_stragglers()
    a5_3_exclusion_vs_quality()
    a5_4_linear_control()
    a5_5_repair()
