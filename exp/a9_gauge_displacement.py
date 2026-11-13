"""A9: the gauge is a channel into the paper's OWN trust state.  (the fusion)

A6 is a static, one-round statement: a functional audit cannot certify one
update.  The paper's existing contribution is dynamic: a persistent trust state
$s^{(t)}$ accumulates across rounds and the adversary attacks WHOM THE SYSTEM
BELIEVES rather than what one round computes.  Until these two are connected the
paper has an AI-specific observation stapled to a classical mechanism.

This script tests the connection.  The claim to test is:

    gauge freedom is a control input to the trust state that carries NO
    functional signature, so an adversary can drive the persistent state --
    the paper's own attack objective -- through a channel that is invisible
    to the audits the paper's defence relies on.

The direction matters and cuts against the adversary in one case already known:
A3 proved an AM-GM floor on FAKING STABILITY, so the adversary cannot use the
gauge to raise its own weight.  That leaves the opposite direction, which A3
never tested:

    the adversary cannot make ITSELF look stable, but can it make a TARGETED
    HONEST CLIENT look unstable?

Under Family II scoring the reference frame is built from submissions, so moving
one's own parameters moves everyone else's score.  That is displacement, which
is exactly Definition 4 (silent exclusion) in the paper -- reached here through
the network's symmetry group rather than through a bias vector.

  A9.1  can the gauge displace an honest client's WEIGHT under Family II?
  A9.2  is it silent?  accuracy and every functional audit, during the attack
  A9.3  the linear control: the same attack where there is no orbit
  A9.4  Family I vs Family II -- which family does the gauge reach?
  A9.5  does the paper's variance floor (M1) help, and does canonicalisation?
"""

import numpy as np

from mlp import (flat, forward, gauge_canon, init_mlp, mse, rand_units,
                 scale_units, sgd, unflat)

DIMS = [20, 32, 32, 1]
D_IN = 20


def krum_like(V, f=4):
    """Sum of squared distances to the closest N-f-2 peers, per client."""
    out = []
    for i in range(len(V)):
        d = np.sort(np.sum((V - V[i]) ** 2, axis=1))[1:]
        out.append(d[:max(1, len(V) - f - 2)].sum())
    return np.array(out)


def fed_run(N=20, T=15, seed=0, family="II", attack="none", sd_gauge=1.5,
            canon=False, phi=0.0, rho=0.7, n_per=150, local=60, target=1,
            warm=True):
    """One federation.  Client 0 is the adversary; `target` is the victim.

    family  "I"  inverse-variance on own across-round dispersion
            "II" inverse-Krum-score on distance to the submission cloud
    attack  "none"   adversary submits honestly
            "gauge"  adversary applies a random per-unit gauge move each round
    warm    Family I has no dispersion estimate at t=0, so with warm=False the
            first round weights everyone equally.  That admits a gauge-inflated
            update at full 1/N and can destroy the global model outright -- see
            a9_6, where this is measured rather than avoided.  warm=True skips
            aggregation until the state exists, which is what a deployment that
            has run for a while looks like.

    Returns (w_hist, acc_hist, audit_hist, diverged).
    """
    r = np.random.default_rng(seed)
    tr = np.random.default_rng(20_000 + seed)
    teacher = init_mlp([D_IN, 24, 24, 1], tr)
    glob = init_mlp(DIMS, r)
    data = []
    for _ in range(N):
        X = r.normal(size=(n_per, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.2
        data.append((X, y))

    Xp = r.normal(size=(1200, D_IN))
    yp = forward(teacher, Xp)[0]

    state = np.ones(N)
    prev = [None] * N
    w_hist, acc_hist, audit_hist = [], [], []
    diverged = False

    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(T):
            subs, disp = [], np.zeros(N)
            for i in range(N):
                p = sgd(glob, data[i][0], data[i][1], steps=local, lr=0.02,
                        rng=r)
                if i == 0 and attack == "gauge":
                    p = scale_units(p, rand_units(p, r, sd_gauge))
                f = flat(gauge_canon(p) if canon else p)
                subs.append(f)
                disp[i] = 0.0 if prev[i] is None else np.mean((f - prev[i]) ** 2)
                prev[i] = f
                if i == 0:
                    # the audit the aggregator would run on the adversary
                    audit_hist.append(
                        float(np.mean((forward(p, Xp)[0] - yp) ** 2)))
            V = np.array(subs)
            if not np.all(np.isfinite(V)):
                diverged = True
                break

            if family == "I":
                if t == 0:
                    # no dispersion estimate exists yet
                    if warm:
                        continue
                    e = np.ones(N)
                else:
                    state = rho * state + (1 - rho) * disp
                    e = state
            else:
                sc = krum_like(V)
                state = sc.copy() if t == 0 else rho * state + (1 - rho) * sc
                e = state
            if phi > 0:
                e = np.maximum(e, phi * np.median(e))
            w = (1.0 / np.maximum(e, 1e-30))
            w = w / w.sum()

            glob = unflat(w @ V, glob)
            w_hist.append(w.copy())
            acc = float(np.mean((forward(glob, Xp)[0] - yp) ** 2))
            if not np.isfinite(acc):
                diverged = True
                break
            acc_hist.append(acc)

    if not acc_hist:
        return np.zeros((1, N)), np.array([np.inf]), \
            np.array(audit_hist or [np.nan]), True
    return np.array(w_hist), np.array(acc_hist), np.array(audit_hist), diverged


def _tail(W, k=5):
    return W[-k:].mean(axis=0)


# =========================================================== A9.1 =========
def a9_1_displacement():
    print("=" * 78)
    print("A9.1  can a gauge move DISPLACE an honest client's weight?")
    print("=" * 78)
    print("  A3 showed the adversary cannot raise its own weight (AM-GM floor).")
    print("  The untested direction is whether it can lower someone else's.")
    print()
    print(f"{'family':>8} {'attack':>8} {'adv weight':>11} {'victim weight':>14} "
          f"{'median honest':>14} {'victim / clean':>15} {'div':>4}")
    for fam in ("I", "II"):
        ref = None
        for atk in ("none", "gauge"):
            aw, vw, hw, nd = [], [], [], 0
            for sd in range(4):
                W, _, _, dv = fed_run(seed=sd, family=fam, attack=atk)
                nd += int(dv)
                if dv:
                    continue
                last = _tail(W)
                aw.append(last[0])
                vw.append(last[1])
                hw.append(np.median(last[1:]))
            mv = np.mean(vw) if vw else np.nan
            if atk == "none":
                ref = mv
            print(f"{fam:>8} {atk:>8} {np.mean(aw):11.4f} {mv:14.4f} "
                  f"{np.mean(hw):14.4f} {mv/ref:15.3f} {nd:4d}")
    print("\n  Fair share is 1/N = 0.05.  The last column is Definition 4's")
    print("  quantity: honest weight under attack over honest weight clean.")
    print("  'div' counts seeds where the federation produced no finite state.")
    print()
    print("  NULL RESULT, and the value says why: N/(N-1) = 20/19 = 1.0526.")
    print("  Both families score the adversary to w = 0.0000, and the remaining")
    print("  N-1 clients then split the whole simplex, so EVERY honest client")
    print("  gains by the same renormalisation factor.  A ratio above 1 with no")
    print("  spread between victim and median honest is not displacement -- the")
    print("  hypothesis that the gauge can push a targeted honest client down")
    print("  is REFUTED, not confirmed, by this table.\n")


# =========================================================== A9.2 =========
def a9_2_silence():
    print("=" * 78)
    print("A9.2  is the displacement SILENT?")
    print("=" * 78)
    print("  Silence in the paper's sense: an honest client loses weight while")
    print("  global accuracy and every functional detector stay in range.")
    print()
    print(f"{'attack':>8} {'min honest w / clean':>22} {'global MSE':>12} "
          f"{'x clean MSE':>12} {'adv audit MSE':>15}")
    cleanW, cleanA = [], []
    for sd in range(4):
        W, A, _, dv = fed_run(seed=sd, family="II", attack="none")
        if dv:
            continue
        cleanW.append(_tail(W))
        cleanA.append(A[-1])
    cw = np.mean([w[1:].min() for w in cleanW])
    ca = np.mean(cleanA)
    print(f"{'none':>8} {1.0:22.3f} {ca:12.4f} {1.0:12.2f} {'-':>15}")
    for s in [0.5, 1.0, 1.5, 2.5]:
        mn, ac, au = [], [], []
        for sd in range(4):
            W, A, AU, dv = fed_run(seed=sd, family="II", attack="gauge",
                                   sd_gauge=s)
            if dv:
                continue
            mn.append(_tail(W)[1:].min())
            ac.append(A[-1])
            au.append(AU[-5:].mean())
        print(f"{'sd=' + str(s):>8} {np.mean(mn)/cw:22.3f} "
              f"{np.mean(ac):12.4f} {np.mean(ac)/ca:12.2f} "
              f"{np.mean(au):15.6f}")
    print("\n  The adversary's audit MSE is what a server checking 'is this")
    print("  client any good' would read.  Compare it across rows: the gauge")
    print("  does not change what the client computes, so the audit cannot")
    print("  separate the attacking rows from the honest one.\n")


# =========================================================== A9.3 =========
def a9_3_linear_control():
    print("=" * 78)
    print("A9.3  the control: the same displacement attempt, no orbit")
    print("=" * 78)
    d, N, T, n = 20, 20, 15, 150
    print(f"{'attack':>22} {'adv own MSE':>13} {'victim w / clean':>17} "
          f"{'global MSE':>12}")

    def run(atk, scale=4.0, seed=0):
        r = np.random.default_rng(seed)
        wt = r.normal(size=d)
        w = np.zeros(d)
        data = [(lambda X: (X, X @ wt + r.normal(size=n) * 0.2))(
            r.normal(size=(n, d))) for _ in range(N)]
        Xp = r.normal(size=(1200, d))
        state = np.ones(N)
        hist = []
        for t in range(T):
            V = []
            for i in range(N):
                X, y = data[i]
                p = w.copy()
                for _ in range(60):
                    p = p - 0.02 * (X.T @ (X @ p - y) / n)
                if i == 0 and atk:
                    p = p * scale
                V.append(p)
            V = np.array(V)
            sc = krum_like(V)
            state = 0.7 * state + 0.3 * sc if t else sc.copy()
            ww = 1.0 / np.maximum(state, 1e-30)
            ww = ww / ww.sum()
            w = ww @ V
            hist.append(ww.copy())
        own = float(np.mean((Xp @ V[0] - Xp @ wt) ** 2))
        g = float(np.mean((Xp @ w - Xp @ wt) ** 2))
        return np.array(hist), own, g

    base = np.mean([run(False, seed=s)[0][-5:].mean(axis=0)[1] for s in range(4)])
    for name, atk, sc in [("none", False, 1.0), ("scale by 4", True, 4.0),
                          ("scale by 20", True, 20.0)]:
        vs, ow, gs = [], [], []
        for s in range(4):
            H, o, g = run(atk, scale=sc, seed=s)
            vs.append(H[-5:].mean(axis=0)[1])
            ow.append(o)
            gs.append(g)
        print(f"{name:>22} {np.mean(ow):13.4f} {np.mean(vs)/base:17.3f} "
              f"{np.mean(gs):12.6f}")
    print("\n  Displacement is achievable here too -- but only by an adversary")
    print("  whose OWN model is visibly broken (column 1 against the 'none'")
    print("  row).  A functional audit rejects it.  That is the difference the")
    print("  orbit makes: for the network the same displacement costs nothing.\n")


# =========================================================== A9.4 =========
def a9_4_which_family():
    print("=" * 78)
    print("A9.4  which scoring family does the gauge reach?")
    print("=" * 78)
    print("  Family I scores a client on its OWN across-round dispersion, so it")
    print("  has no reference frame to move -- A3's floor applies and the")
    print("  adversary cannot help itself.  Family II builds its frame from the")
    print("  submissions, so one client's coordinates move everyone's score.")
    print()
    print(f"{'family':>8} {'adv w / fair':>13} {'victim w / clean':>17} "
          f"{'honest spread':>14}")
    for fam in ("I", "II"):
        cb = []
        for s in range(4):
            W, _, _, dv = fed_run(seed=s, family=fam, attack="none")
            if not dv:
                cb.append(_tail(W)[1])
        cb = np.mean(cb)
        aw, vw, sp = [], [], []
        for sd in range(4):
            W, _, _, dv = fed_run(seed=sd, family=fam, attack="gauge")
            if dv:
                continue
            last = _tail(W)
            aw.append(last[0] * 20)
            vw.append(last[1] / cb)
            sp.append(last[1:].max() / last[1:].min())
        print(f"{fam:>8} {np.mean(aw):13.2f} {np.mean(vw):17.3f} "
              f"{np.mean(sp):14.2f}")
    print("\n  Column 1 near 1.00 confirms A3 dynamically: no self-promotion.")
    print("  The damage, if any, is in column 2 and is done to someone else.\n")


# =========================================================== A9.5 =========
def a9_5_defences():
    print("=" * 78)
    print("A9.5  the paper's floor (M1) vs canonicalisation")
    print("=" * 78)
    print("  These answer different theorems, which is the separation the paper")
    print("  already reports for TSP-Var.  The floor bounds how much weight any")
    print("  ONE client can hold; it does not make the score gauge invariant.")
    print("  Canonicalisation makes the score gauge invariant; it does nothing")
    print("  about a client that is genuinely, functionally wrong.")
    print()
    print(f"{'defence':>28} {'victim w / clean':>17} {'global MSE':>12} "
          f"{'x clean':>9}")
    cb, ca = [], []
    for s in range(4):
        W, A, _, dv = fed_run(seed=s, family="II", attack="none")
        if dv:
            continue
        cb.append(_tail(W)[1])
        ca.append(A[-1])
    cb, ca = np.mean(cb), np.mean(ca)
    print(f"{'(clean baseline)':>28} {1.0:17.3f} {ca:12.4f} {1.0:9.2f}")
    for name, kw in [("none", dict()),
                     ("M1 floor phi=0.5", dict(phi=0.5)),
                     ("M1 floor phi=1.0", dict(phi=1.0)),
                     ("gauge canonicalisation", dict(canon=True)),
                     ("both", dict(phi=1.0, canon=True))]:
        vs, gs = [], []
        for s in range(4):
            W, A, _, dv = fed_run(seed=s, family="II", attack="gauge", **kw)
            if dv:
                continue
            vs.append(_tail(W)[1])
            gs.append(A[-1])
        print(f"{name:>28} {np.mean(vs)/cb:17.3f} {np.mean(gs):12.4f} "
              f"{np.mean(gs)/ca:9.2f}")
    print("\n  A value of 1.000 in column 1 means the honest client kept the")
    print("  weight it would have had with no adversary present.")
    print("  phi=0.5 floors the error at half the MEDIAN error, which under")
    print("  Family II never binds -- the adversary's Krum score is orders of")
    print("  magnitude ABOVE the median, not below it, so the floor is inactive")
    print("  by construction and the row is expected to equal 'none' exactly.\n")


# =========================================================== A9.6 =========
def a9_6_cold_start():
    print("=" * 78)
    print("A9.6  the cold-start round: where the gauge actually does damage")
    print("=" * 78)
    print("  Every score in Family I is an ACROSS-ROUND statistic, so at t=0")
    print("  there is nothing to score and the aggregator must fall back to")
    print("  uniform weights.  A gauge move inflates the submitted vector by")
    print("  orders of magnitude at zero functional cost, so that one round")
    print("  admits it at full 1/N.  Measured over 8 seeds.")
    print()
    print(f"{'family':>8} {'cold start':>11} {'diverged / 8':>13} "
          f"{'final global MSE':>18} {'|adv| / |honest|':>17}")
    for fam in ("I", "II"):
        for warm in (True, False):
            nd, fin = 0, []
            for s in range(8):
                W, A, _, dv = fed_run(seed=s, family=fam, attack="gauge",
                                      warm=warm, T=10)
                nd += int(dv)
                if not dv:
                    fin.append(A[-1])
            print(f"{fam:>8} {('warm' if warm else 'uniform'):>11} "
                  f"{nd:13d} "
                  f"{(np.mean(fin) if fin else float('nan')):18.4f} "
                  f"{'':>17}")
    print()
    print("  The norm ratio the gauge produces, measured directly:")
    r = np.random.default_rng(0)
    tr = np.random.default_rng(20_000)
    teacher = init_mlp([D_IN, 24, 24, 1], tr)
    g0 = init_mlp(DIMS, r)
    X = r.normal(size=(150, D_IN))
    y = forward(teacher, X)[0] + r.normal(size=150) * 0.2
    p = sgd(g0, X, y, steps=60, lr=0.02, rng=r)
    Xp = r.normal(size=(1200, D_IN))
    f0 = forward(p, Xp)[0]
    print(f"{'log-sd':>10} {'|q| / |p|':>11} {'max |df|':>11}")
    for s in [0.5, 1.0, 1.5, 2.5]:
        rr = np.random.default_rng(99)
        q = scale_units(p, rand_units(p, rr, s))
        print(f"{s:10.2f} "
              f"{np.linalg.norm(flat(q))/np.linalg.norm(flat(p)):11.2f} "
              f"{np.abs(forward(q, Xp)[0] - f0).max():11.1e}")
    print()
    print("  This is the honest reading of A9: the gauge is not a stealth")
    print("  displacement tool against a warmed-up Family II aggregator, which")
    print("  scores it out at 1e-7.  It is an availability attack on the round")
    print("  where the trust state does not exist yet, and THAT round is exactly")
    print("  the one the paper's persistent-state defence cannot cover.\n")


if __name__ == "__main__":
    a9_1_displacement()
    a9_2_silence()
    a9_3_linear_control()
    a9_4_which_family()
    a9_5_defences()
    a9_6_cold_start()
