"""A13: the audit gap on real data, with real non-IID heterogeneity.

Every result up to A12 runs on synthetic data, and the paper defends that
choice on the grounds that only a synthetic generator lets bias_i and sigma_i^2
be varied independently (sec:setup).  That defence is sound for the TSP-var
mechanism, but it does not cover prop:auditgap, which is a statement about the
parameter space of a network and should therefore hold on any data at all.  If
it only held on Gaussian designs it would be an artefact.

So this script re-runs the two claims that do not depend on controlling the
noise model, on two real datasets, with the client partition taken from a real
covariate rather than drawn:

  A13.1  California housing (20640 x 8, sklearn), partitioned into N clients by
         GEOGRAPHY -- a k-means over (latitude, longitude).  Client shards are
         genuinely non-IID in both features and label, and unequal in size, in
         the way a real federation's are.  Claim: the gauge move leaves each
         client's function bit-identical while the aggregate degrades.
  A13.2  UCI handwritten digits (1797 x 64, sklearn, bundled), partitioned by
         LABEL: each client is dominated by one digit class.  Task is 3-vs-rest
         so the scalar-output harness applies unchanged.  Same claim.
  A13.3  M4 on both, including the honest cost.

What this can and cannot show.  It cannot show an effect size for a deployed
model -- the networks here are small, as sec:validity already says.  What it
shows is that the audit gap is not an artefact of the synthetic generator: the
mechanism is a property of the parameter space, so real, skewed, non-IID data
reproduces it without any tuning.

The dispersion-based trust attack (TSP-var) is deliberately NOT re-run here.
On real shards sigma_i^2 and bias_i move together, which is exactly why the
paper uses a synthetic generator for it, and running it here would produce a
number that confounds the two.  Reporting it would be worse than omitting it.
That was an assertion when this script was written; a26_real_confound.py
measures it on these same two partitions and it holds -- rho = +0.711 and
+0.554 against a synthetic control of 0.017, and at matched captured share the
damage is 1.087x and 1.005x here against 4.114x synthetic.  So the omission is
justified rather than merely convenient, and the confound understates the
attack rather than flattering it.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import fetch_california_housing, load_digits

from mlp import (copy_net, flat, forward, gauge_canon, init_mlp, rand_units,
                 scale_units, sgd, unflat)

SEEDS = range(6)


# ----------------------------------------------------------------- data ----
def california(n_clients, rng):
    """Partition real housing data by geography (k-means on lat/lon).

    Geography is the partition variable a real deployment would have: devices
    are where they are.  It induces heterogeneity we did not choose -- coastal
    shards differ from inland ones in both features and price level.
    """
    d = fetch_california_housing()
    X, y = d.data.astype(np.float64), d.target.astype(np.float64)
    # cap the tail; the target is censored at 5.0 and the top bin is a spike
    keep = y < 5.0
    X, y = X[keep], y[keep]
    lat_lon = X[:, 6:8]

    # Lloyd's algorithm, seeded from random points; no sklearn dependency for
    # the clustering itself so the partition is reproducible from this file.
    c = lat_lon[rng.choice(len(lat_lon), n_clients, replace=False)]
    for _ in range(25):
        lab = np.argmin(((lat_lon[:, None, :] - c[None]) ** 2).sum(2), axis=1)
        for j in range(n_clients):
            if (lab == j).any():
                c[j] = lat_lon[lab == j].mean(0)
    # recompute after the last centroid move, or the labels are one step stale
    lab = np.argmin(((lat_lon[:, None, :] - c[None]) ** 2).sum(2), axis=1)

    # standardise on the pooled data, as a real federation would with public
    # feature statistics; drop lat/lon so the label is not trivially separable
    F = X[:, :6].copy()
    # AveRooms/AveBedrms/Population/AveOccup are ratios with extreme tails
    # (AveOccup reaches 117 sd raw); log1p then clip at 5 sd.  Without this the
    # tail dominates the gradient and training diverges at any usable lr -- a
    # property of the real data, and the reason a synthetic Gaussian design
    # needs no such step.
    for j in (2, 3, 4, 5):
        F[:, j] = np.log1p(F[:, j])
    F = (F - F.mean(0)) / F.std(0)
    F = np.clip(F, -5.0, 5.0)
    y = (y - y.mean()) / y.std()
    shards = [(F[lab == j], y[lab == j]) for j in range(n_clients)]
    # an empty or near-empty cell would silently break sgd; California is dense
    # enough that this never fires, but a partition we did not choose is not a
    # partition we may assume is balanced
    assert all(len(s[1]) >= 32 for s in shards), \
        [len(s[1]) for s in shards]
    return shards, F.shape[1]


def digits(n_clients, rng):
    """Partition real digit images by label: client j is dominated by class j.

    Task is 3-vs-rest, so the scalar-output MSE harness needs no change.  Label
    skew is the standard non-IID model in the FL literature and here it is
    applied to real pixels rather than to a generator.
    """
    d = load_digits()
    X, t = d.data.astype(np.float64) / 16.0, d.target
    y = (t == 3).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    shards = []
    for j in range(n_clients):
        main = np.where(t == j % 10)[0]
        rest = rng.choice(np.where(t != j % 10)[0], size=len(main) // 2,
                          replace=False)
        idx = np.concatenate([main, rest])
        shards.append((X[idx], y[idx]))
    return shards, X.shape[1]


# ------------------------------------------------------------- the group ----
def align_to(p, ref):
    """Hungarian alignment for the generic (list-of-layers) MLP format.

    a11's align_to is written against the fixed three-layer dict; this is the
    same construction for arbitrary depth.  Unit j of a hidden layer is
    described by its incoming row [W_j, b_j], and the columns of the next
    layer's W are reindexed to match, which is a member of S2 -- so the
    function is preserved by construction, and A13.3 checks that in floats.
    """
    q = copy_net(p)
    for i in range(len(q) - 1):
        f_q = np.concatenate([q[i]["W"], q[i]["b"][:, None]], axis=1)
        f_r = np.concatenate([ref[i]["W"], ref[i]["b"][:, None]], axis=1)
        cost = ((f_q[:, None, :] - f_r[None, :, :]) ** 2).sum(2)
        row, col = linear_sum_assignment(cost)
        perm = np.empty(len(row), dtype=int)
        perm[col] = row
        for k in ("W", "b", "g", "beta"):
            if k in q[i]:
                q[i][k] = q[i][k][perm]
        q[i + 1]["W"] = q[i + 1]["W"][:, perm]
    return q


def full_canon(p, ref):
    """M4: quotient S1 by scale canonicalisation, then S2 by alignment."""
    return align_to(gauge_canon(p), ref)


def gauge_move(p, rng, sd=1.5):
    """The attack: a random element of S1, at the log-sd used throughout.

    rand_units returns the per-layer scale vectors; scale_units applies them.
    log-sd 1.5 is the same magnitude as A6/A11, so the real-data numbers are
    comparable to the synthetic ones rather than separately tuned.
    """
    return scale_units(p, rand_units(p, rng, sd))


# The attack draw is keyed on the run seed alone, so A13.1/A13.2 and A13.3
# hit the SAME model with the SAME gauge element.  Otherwise the two halves of
# the script would report two different damage figures for one quantity.
def atk_seed(sd):
    return 900 + sd


# ----------------------------------------------------------- aggregation ----
def agg_err(params, Xp, yp, replace=None):
    ps = list(params)
    if replace is not None:
        ps[0] = replace
    V = np.array([flat(c) for c in ps])
    g = unflat(V.mean(axis=0), ps[0])
    return float(np.mean((forward(g, Xp)[0] - yp) ** 2))


def train(shards, d_in, seed, hidden=(24, 24), steps=3000, lr=0.004,
          batch=64):
    """Common init, one local run per client.  Returns clients + a probe set.

    lr is an order of magnitude below the synthetic runs' 0.05: on real shards
    with residual heavy tails, 0.01 and above diverges to NaN.  Chosen by
    sweeping {0.02, 0.01, 0.005, 0.002, 0.001} at batch 32, then {0.002, 0.004}
    x {300, 1000, 3000, 6000} at batch 64, taking the largest stable lr and
    enough steps that the aggregate is well clear of the trivial predictor.
    Both targets are z-scored, so MSE 1.0 is exactly the score of predicting
    the mean; the clean aggregate has to beat that, or the damage ratio would
    be measured against a model that had learnt nothing.
    """
    r = np.random.default_rng(seed)
    g0 = init_mlp([d_in, *hidden, 1], r)
    clients = [sgd(g0, Xs, ys, steps=steps, lr=lr, batch=batch, rng=r)
               for Xs, ys in shards]
    Xp = np.concatenate([s[0] for s in shards])
    yp = np.concatenate([s[1] for s in shards])
    sel = r.choice(len(yp), size=min(2000, len(yp)), replace=False)
    assert np.isfinite([flat(c) for c in clients]).all(), "training diverged"
    return clients, Xp[sel], yp[sel], r


# ============================================================ A13.1 =======
def a13_1_california():
    print("=" * 78)
    print("A13.1  audit gap on California housing, clients partitioned by "
          "GEOGRAPHY")
    print("=" * 78)
    r0 = np.random.default_rng(0)
    shards, d_in = california(12, r0)
    sizes = [len(s[1]) for s in shards]
    print(f"  real data: {sum(sizes)} rows, {d_in} features, 12 geographic "
          f"clients")
    print(f"  shard sizes {min(sizes)}-{max(sizes)} (ratio "
          f"{max(sizes)/min(sizes):.1f}x), mean label per shard "
          f"{np.std([s[1].mean() for s in shards]):.3f} sd -- non-IID and "
          f"unbalanced, neither of which we chose")

    print(f"\n{'seed':>5} {'max|df|':>10} {'own MSE':>10} {'own after':>10} "
          f"{'agg clean':>10} {'agg after':>10} {'x':>8}")
    ratios, audits, cleans = [], [], []
    for sd in SEEDS:
        clients, Xp, yp, r = train(shards, d_in, sd)
        f0 = forward(clients[0], Xp)[0]
        q = gauge_move(clients[0], np.random.default_rng(atk_seed(sd)))
        df = float(np.abs(forward(q, Xp)[0] - f0).max())
        own0 = float(np.mean((f0 - yp) ** 2))
        own1 = float(np.mean((forward(q, Xp)[0] - yp) ** 2))
        c = agg_err(clients, Xp, yp)
        a = agg_err(clients, Xp, yp, replace=q)
        ratios.append(a / c)
        audits.append(df)
        cleans.append(c)
        print(f"{sd:5d} {df:10.1e} {own0:10.4f} {own1:10.4f} "
              f"{c:10.4f} {a:10.4f} {a/c:8.2f}")
    rr = np.array(ratios)
    print(f"\n  aggregate damage  median {np.median(rr):.2f}x  "
          f"min {rr.min():.2f}x  max {rr.max():.2f}x")
    print(f"  clean aggregate MSE {np.mean(cleans):.3f} vs 1.000 for "
          f"predicting the mean (y is z-scored) -- the baseline the ratio is\n"
          f"  measured against has genuinely learnt the task")
    print(f"  functional audit  max over seeds {max(audits):.1e} -- the "
          f"client's function is unchanged")
    print("  On real, geographically non-IID shards the gap is present with no")
    print("  tuning: prop:auditgap is a property of the parameter space, not "
          "of\n  the data-generating process.\n")
    return rr, max(audits)


# ============================================================ A13.2 =======
def a13_2_digits():
    print("=" * 78)
    print("A13.2  audit gap on UCI digits (3-vs-rest), clients by LABEL skew")
    print("=" * 78)
    r0 = np.random.default_rng(0)
    shards, d_in = digits(10, r0)
    sizes = [len(s[1]) for s in shards]
    pos = [float(s[1].mean()) for s in shards]
    print(f"  real data: {d_in} pixel features, 10 label-skewed clients, "
          f"shards {min(sizes)}-{max(sizes)}")
    print(f"  positive rate per client {min(pos):.2f}-{max(pos):.2f} -- "
          f"client 3 holds almost all positives")

    print(f"\n{'seed':>5} {'max|df|':>10} {'own MSE':>10} {'own after':>10} "
          f"{'agg clean':>10} {'agg after':>10} {'x':>8}")
    ratios, audits, cleans = [], [], []
    for sd in SEEDS:
        clients, Xp, yp, r = train(shards, d_in, sd, hidden=(24, 24))
        f0 = forward(clients[0], Xp)[0]
        q = gauge_move(clients[0], np.random.default_rng(atk_seed(sd)))
        df = float(np.abs(forward(q, Xp)[0] - f0).max())
        own0 = float(np.mean((f0 - yp) ** 2))
        own1 = float(np.mean((forward(q, Xp)[0] - yp) ** 2))
        c = agg_err(clients, Xp, yp)
        a = agg_err(clients, Xp, yp, replace=q)
        ratios.append(a / c)
        audits.append(df)
        cleans.append(c)
        print(f"{sd:5d} {df:10.1e} {own0:10.4f} {own1:10.4f} "
              f"{c:10.4f} {a:10.4f} {a/c:8.2f}")
    rr = np.array(ratios)
    print(f"\n  aggregate damage  median {np.median(rr):.2f}x  "
          f"min {rr.min():.2f}x  max {rr.max():.2f}x")
    print(f"  clean aggregate MSE {np.mean(cleans):.3f}; predicting the base "
          f"rate scores {np.mean(yp) * (1 - np.mean(yp)):.3f}")
    print(f"  functional audit  max over seeds {max(audits):.1e}")
    print("  Real pixels, real label skew, a classification objective: same\n"
          "  separation between P1 and P2.\n")
    return rr, max(audits)


# ============================================================ A13.3 =======
def a13_3_defence():
    print("=" * 78)
    print("A13.3  M4 on real data: repair under attack, and cost without one")
    print("=" * 78)
    out = {}
    for name, loader, nc in (("california", california, 12),
                             ("digits", digits, 10)):
        shards, d_in = loader(nc, np.random.default_rng(0))
        print(f"\n  -- {name} --")
        print(f"  {'seed':>5} {'undef x':>9} {'M4 x':>8} {'honest x':>9} "
              f"{'canon |df|':>11}")
        und, dfd, hon, dfs = [], [], [], []
        for sd in SEEDS:
            clients, Xp, yp, r = train(shards, d_in, sd)
            ref = clients[1]
            q = gauge_move(clients[0], np.random.default_rng(atk_seed(sd)))
            c = agg_err(clients, Xp, yp)
            a = agg_err(clients, Xp, yp, replace=q)

            # M4: every client is mapped into the quotient, attacker included
            cq = [full_canon(p, ref) for p in clients]
            qq = full_canon(q, ref)
            base_q = agg_err(cq, Xp, yp)
            att_q = agg_err(cq, Xp, yp, replace=qq)

            # the canon must not change any client's function
            df = max(float(np.abs(forward(full_canon(p, ref), Xp)[0]
                                  - forward(p, Xp)[0]).max())
                     for p in clients[:4])
            und.append(a / c)
            dfd.append(att_q / base_q)
            hon.append(base_q / c)
            dfs.append(df)
            print(f"  {sd:5d} {a/c:9.2f} {att_q/base_q:8.4f} "
                  f"{base_q/c:9.4f} {df:11.1e}")
        u, d_, h = np.array(und), np.array(dfd), np.array(hon)
        print(f"    undefended median {np.median(u):.2f}x -> under M4 "
              f"max {d_.max():.4f}x")
        print(f"    honest-case cost  max {h.max():.4f}x   "
              f"canon audit delta max {max(dfs):.1e}")
        out[name] = (u, d_, h, max(dfs))
    print("\n  M4 is data-agnostic by construction -- it never looks at the "
          "data --\n  and the measurement confirms it on both real sets.\n")
    return out


if __name__ == "__main__":
    ca, ca_a = a13_1_california()
    di, di_a = a13_2_digits()
    de = a13_3_defence()

    print("=" * 78)
    print("SUMMARY -- real-data replication of prop:auditgap and M4")
    print("=" * 78)
    print(f"  california  damage median {np.median(ca):.2f}x  "
          f"audit <= {ca_a:.1e}")
    print(f"  digits      damage median {np.median(di):.2f}x  "
          f"audit <= {di_a:.1e}")
    for k, (u, d_, h, a) in de.items():
        print(f"  M4 {k:<11} {np.median(u):.2f}x -> {d_.max():.4f}x   "
              f"honest cost <= {h.max():.4f}x")
