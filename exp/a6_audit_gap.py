"""A6: the audit gap.  "Correct model" and "correct update" come apart.

This is the one AI-specific result that survived A1-A5.  The failures matter and
are reported alongside it, because they bound what the claim can be:

    A2  hiding a backdoor by gauge descent            FAILED
    A3  manufacturing apparent stability              FAILED (AM-GM floor, 1.5x)
    A4.2 gauge damage as a stealth attack             FAILED (z-score 2.5 at
                                                      1.00x damage)
    A5  honest clients excluded by gauge drift        NULL vs linear control

So the network's symmetry group does NOT give an adversary a way to look better
than it is.  What it gives is a separation between two properties that a
federation implicitly treats as the same one:

    P1  the client holds a good model      -- gauge INVARIANT
    P2  the client sent a good update      -- gauge DEPENDENT

Any audit that evaluates the submitted model as a function -- held-out accuracy,
loss on a probe set, behavioural testing, a challenge-response on predictions --
tests P1.  A federation needs P2.  For a linear model P1 implies P2, because
there is no orbit: two parameter vectors computing the same function ARE the
same vector.  For a network they are independent, and the gap is unbounded.

The consequence is not a stealth attack.  It is that functional auditing --
which is the natural and widely proposed answer to "how do we know this client
is honest" -- is unsound as a basis for admitting updates, and the failure is
silent: the audit passes exactly, at machine precision, every time.

  A6.1  the gap, across seeds: audit passes bit-exactly, aggregate degrades
  A6.2  how large can it get?  scan the continuous group
  A6.3  the linear control -- the same scan where no orbit exists
  A6.4  which audits are fooled, and which are not
  A6.5  gauge canonicalisation as the repair, and its limit (permutation)
"""

import numpy as np

from a1_nn_symmetry import (D_IN, H1, H2, flat, forward, gauge_permute,
                            init_net, krum_score, sgd)
from a4_aggregate_gauge import (agg_mse, build, gauge_canon, scale_units,
                                unflat, zscore)


# =========================================================== A6.1 =========
def a6_1_gap_across_seeds():
    print("=" * 78)
    print("A6.1  the audit gap across seeds (per-unit gauge, log-sd 1.5)")
    print("=" * 78)
    print(f"{'seed':>5} {'own MSE':>11} {'own MSE after':>15} {'max|df|':>10} "
          f"{'agg clean':>11} {'agg after':>11} {'x':>8}")
    ratios = []
    for sd in range(8):
        clients, teacher, Xp, _ = build(seed=sd)
        ft = forward(teacher, Xp)[0]
        f0 = forward(clients[0], Xp)[0]
        rr = np.random.default_rng(500 + sd)
        q = scale_units(clients[0], np.exp(rr.normal(size=H1) * 1.5),
                        np.exp(rr.normal(size=H2) * 1.5))
        own0 = float(np.mean((f0 - ft) ** 2))
        own1 = float(np.mean((forward(q, Xp)[0] - ft) ** 2))
        df = np.abs(forward(q, Xp)[0] - f0).max()
        c = agg_mse(clients, teacher, Xp)
        a = agg_mse(clients, teacher, Xp, q)
        ratios.append(a / c)
        print(f"{sd:5d} {own0:11.6f} {own1:15.6f} {df:10.1e} "
              f"{c:11.6f} {a:11.6f} {a/c:8.2f}")
    r = np.array(ratios)
    print(f"\n  aggregate damage  mean {r.mean():.2f}x  median {np.median(r):.2f}x"
          f"  min {r.min():.2f}x  max {r.max():.2f}x")
    print("  own-model error is identical to 1e-15 in every row: the client's")
    print("  function is untouched, so every functional audit passes exactly.\n")


# =========================================================== A6.2 =========
def a6_2_magnitude():
    print("=" * 78)
    print("A6.2  how large is the gap as a function of gauge magnitude?")
    print("=" * 78)
    print(f"{'per-unit log-sd':>16} {'own MSE delta':>15} {'agg x clean':>13} "
          f"{'z-score':>10} {'Krum rank':>11}")
    for s in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        dm, ag, zs, rk = [], [], [], []
        for sd in range(5):
            clients, teacher, Xp, _ = build(seed=sd)
            ft = forward(teacher, Xp)[0]
            f0 = forward(clients[0], Xp)[0]
            rr = np.random.default_rng(900 + sd)
            q = scale_units(clients[0], np.exp(rr.normal(size=H1) * s),
                            np.exp(rr.normal(size=H2) * s))
            dm.append(abs(float(np.mean((forward(q, Xp)[0] - ft) ** 2))
                          - float(np.mean((f0 - ft) ** 2))))
            ag.append(agg_mse(clients, teacher, Xp, q)
                      / agg_mse(clients, teacher, Xp))
            zs.append(zscore(clients, q))
            V = np.array([flat(c) for c in clients])
            V[0] = flat(q)
            rk.append(int(np.argsort(np.argsort(
                np.array([krum_score(V, i) for i in range(len(V))])))[0]) + 1)
        print(f"{s:16.2f} {np.mean(dm):15.2e} {np.mean(ag):13.2f} "
              f"{np.mean(zs):10.2f} {np.mean(rk):11.1f}")
    print("\n  Column 2 is the audit's view and stays at machine zero throughout.")
    print("  Column 3 is the federation's.  Columns 4-5 are why this is an")
    print("  auditing result and not a stealth attack: a distance-based scorer")
    print("  does see it, and sees it early.  The two views never agree.\n")


# =========================================================== A6.3 =========
def a6_3_linear_control():
    print("=" * 78)
    print("A6.3  the control: no orbit, so no gap")
    print("=" * 78)
    d, N, n = 20, 30, 200
    print(f"{'client 0 transform':>26} {'own MSE':>12} {'own MSE after':>15} "
          f"{'agg x clean':>13}")
    r = np.random.default_rng(5)
    wt = r.normal(size=d)
    W = []
    for _ in range(N):
        X = r.normal(size=(n, d))
        W.append(np.linalg.lstsq(X, X @ wt + r.normal(size=n) * 0.1,
                                 rcond=None)[0])
    W = np.array(W)
    Xp = r.normal(size=(2000, d))
    yp = Xp @ wt

    def mse(w):
        return float(np.mean((Xp @ w - yp) ** 2))

    clean = mse(W.mean(axis=0))
    own0 = mse(W[0])
    for name, w0 in [("none", W[0]),
                     ("scale by 2", W[0] * 2.0),
                     ("scale by 8", W[0] * 8.0),
                     ("permute coefficients", W[0][r.permutation(d)]),
                     ("add gaussian 0.5", W[0] + r.normal(size=d) * 0.5)]:
        V = W.copy()
        V[0] = w0
        print(f"{name:>26} {own0:12.6f} {mse(w0):15.6f} "
              f"{mse(V.mean(axis=0))/clean:13.2f}")
    print("\n  Every transform that damages the aggregate also damages the")
    print("  client's own model by a comparable factor, so a functional audit")
    print("  catches all of them.  P1 implies P2.  This is the property the")
    print("  network loses, and losing it is what makes the defect AI-specific.\n")


# =========================================================== A6.4 =========
def a6_4_which_audits():
    print("=" * 78)
    print("A6.4  which audits are fooled?")
    print("=" * 78)
    clients, teacher, Xp, r = build(seed=0)
    rr = np.random.default_rng(77)
    q = scale_units(clients[0], np.exp(rr.normal(size=H1) * 1.5),
                    np.exp(rr.normal(size=H2) * 1.5))
    p = clients[0]
    ft = forward(teacher, Xp)[0]
    Xc = rr.normal(size=(500, D_IN))

    checks = [
        ("held-out MSE", lambda m: float(np.mean((forward(m, Xp)[0] - ft) ** 2))),
        ("max abs prediction error",
         lambda m: float(np.abs(forward(m, Xp)[0] - ft).max())),
        ("challenge-response digest",
         lambda m: float(np.sum(forward(m, Xc)[0] ** 2))),
        ("prediction on trigger region",
         lambda m: float(np.mean(forward(m, Xp[Xp[:, 0] > 1.0])[0]))),
        ("parameter L2 norm", lambda m: float(np.linalg.norm(flat(m)))),
        ("distance to peer mean", lambda m: float(np.linalg.norm(
            flat(m) - np.array([flat(c) for c in clients[1:]]).mean(axis=0)))),
    ]
    print(f"{'audit':>30} {'honest':>14} {'gauge-moved':>14} {'detects?':>10}")
    for name, fn in checks:
        a, b = fn(p), fn(q)
        det = "no" if abs(b - a) <= 1e-9 * max(1.0, abs(a)) else "YES"
        print(f"{name:>30} {a:14.6f} {b:14.6f} {det:>10}")
    print("\n  Everything computed from the model's OUTPUTS is blind, exactly.")
    print("  Only checks that read the parameter vector see anything -- and")
    print("  those are precisely the checks that cannot distinguish a gauge")
    print("  move from honest heterogeneity, which is what A5 tested.\n")


# =========================================================== A6.5 =========
def a6_5_repair():
    print("=" * 78)
    print("A6.5  gauge canonicalisation as the repair, and its limit")
    print("=" * 78)
    print("  Canonicalisation normalises each hidden unit's incoming weights and")
    print("  pushes the scale forward.  It is itself a gauge move, so it changes")
    print("  no client's function -- verified below -- and it maps the whole")
    print("  continuous orbit to one representative.")
    print()
    print(f"{'gauge move':>24} {'raw agg':>10} {'canon agg':>11} {'canon max|df|':>15}")
    clients, teacher, Xp, r = build(seed=0)
    gc = [gauge_canon(c) for c in clients]
    base_raw = agg_mse(clients, teacher, Xp)
    base_can = agg_mse(gc, teacher, Xp)
    print(f"{'(clean baseline)':>24} {base_raw:10.4f} {base_can:11.4f} "
          f"{'-':>15}")
    for name, q in [
        ("per-unit sd=0.5", scale_units(clients[0],
                                        np.exp(r.normal(size=H1) * 0.5),
                                        np.exp(r.normal(size=H2) * 0.5))),
        ("per-unit sd=1.5", scale_units(clients[0],
                                        np.exp(r.normal(size=H1) * 1.5),
                                        np.exp(r.normal(size=H2) * 1.5))),
        ("per-unit sd=3.0", scale_units(clients[0],
                                        np.exp(r.normal(size=H1) * 3.0),
                                        np.exp(r.normal(size=H2) * 3.0))),
        ("permutation", gauge_permute(clients[0], r)),
    ]:
        df = np.abs(forward(gauge_canon(q), Xp)[0] - forward(q, Xp)[0]).max()
        print(f"{name:>24} {agg_mse(clients, teacher, Xp, q):10.4f} "
              f"{agg_mse(gc, teacher, Xp, gauge_canon(q)):11.4f} {df:15.1e}")
    print("\n  The continuous group is closed at no cost.  The permutation")
    print("  subgroup is not: canonicalising it means choosing a consistent")
    print("  labelling of hidden units across clients, which is an assignment")
    print("  problem per client per round, and the row above shows why it is")
    print("  lower priority -- permutation alone barely moves the aggregate.\n")


if __name__ == "__main__":
    a6_1_gap_across_seeds()
    a6_2_magnitude()
    a6_3_linear_control()
    a6_4_which_audits()
    a6_5_repair()
