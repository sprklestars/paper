"""A24: why not just cap the aggregation weight?

The obvious objection to M1 and M2 is that neither is needed: bound each
client's weight at c/N, redistribute the excess, and the adversary's take is
bounded by construction.  A referee is entitled to ask, and sec:e3 as written
has no answer.  This measures the cap against M1 and M2 in E3's own harness.

Three natural objections to the cap all FAIL:

  A24.1  at matched harm the cap is CHEAPER than M2 in honest utility, and
         level with M1, so "the cap costs more" is false.
  A24.2  the cap preserves the weight-vs-precision ranking an adaptive
         aggregator exists to produce (Spearman +0.96 to +0.99 against the
         undefended +0.98), so "the cap is just FedAvg" is false.
  A24.3  the cap is NOT a rate limiter in E9's sense: its loud/silent ratio is
         1.00x, like M1's, not 5.0x like M2's.  So "an adaptive adversary
         defeats it" is false.

So the cap is a good mechanism and the paper must say so.  What it cannot do:

  A24.4  eq:suppcost binds the cap exactly.  A capped rule is peer-only, so
         thm:unidentifiable applies and suppression = displacement to 3e-16.
         The cap is not an exemption from the paper's impossibility identity,
         it is another point on it.
  A24.5  a per-client cap of c/N held by K colluders yields K*c/N, so it
         degrades linearly in coalition size and goes inert at K/N >= 1/c,
         whereas M2's reserved anchor mass (1-lambda) is independent of K.
         Calibrated to the same absolute share at K = 1, the cap and M2
         diverge as K grows, and at K = 15 of N = 30 cap c=3.0 returns the
         undefended share exactly.

Conclusion for the paper: the cap dominates M2 on the single-adversary
frontier, M1 dominates both, and the cap's guarantee is the one that decays
with coalition size.  It is a complement to M2's reserved mass, not a
substitute for the floor.
"""

import sys

import numpy as np

from a20_impossibility_scope import MAGS, NSEED, RULES, three_worlds
from e2_silence import CFG, detectors

SEEDS = list(range(8))
S_ADV, DELTA = 0.35, 0.05

# E9's protocol: search on one seed set, re-measure the winner on a disjoint one
SEARCH_SEEDS = [0, 1, 2, 3, 4]
REPORT_SEEDS = [5, 6, 7, 8, 9, 10, 11, 12]
SILENT_EPS = 0.05
S_GRID = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
D_GRID = [0.02, 0.05, 0.10, 0.20, 0.30]

K_GRID = [1, 3, 6, 10, 15]
FAIR = CFG["k_adv"] / CFG["N"]


# ----------------------------------------------------------------- the cap
def apply_cap(wt, lim):
    """Water-filling: clip at lim, redistribute the excess pro rata.  Repeated
    because redistribution can push a previously-under client over."""
    for _ in range(50):
        over = wt > lim
        if not over.any():
            break
        ex = (wt[over] - lim).sum()
        wt[over] = lim
        un = ~over
        if not un.any() or wt[un].sum() <= 0:
            break
        wt[un] += ex * wt[un] / wt[un].sum()
    return wt


from tools.rank_metrics import spearman


def sim(mech, p, s_adv, dmag, seed, K=1, honest_only=False, alarms_too=False):
    """One federation.  Mirrors e9_adaptive_defense.sim's weight path, adding
    the cap branch; with mech='none' it reproduces E9's undefended column."""
    cfg = dict(CFG)
    cfg["s_adv"] = s_adv
    cfg["k_adv"] = K
    rng = np.random.default_rng(seed)
    d, N = cfg["d"], cfg["N"]
    adv, hon = np.arange(K), np.arange(K, N)

    w_star = rng.normal(size=d)
    dl = rng.normal(size=d)
    dl = dmag * dl / (np.linalg.norm(dl) + 1e-12)

    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - K)
    sigma[adv] = (s_adv if not honest_only
                  else rng.uniform(cfg["s_lo"], cfg["s_hi"], size=K))

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    state = {"var": np.full(N, 1.0), "hist": np.zeros((N, d)),
             "anchor_ref": np.zeros(d)}
    prev = np.zeros((N, d))
    ev = np.full(N, np.nan)
    cum = np.zeros(N)
    mse, al, wt_tail = [], [], []

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if (i < K and not honest_only) else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        if alarms_too:
            A = []
            for _ in range(cfg["a_anchor"]):
                X = rng.normal(size=(cfg["n_anchor"], d))
                y = X @ w_star + rng.normal(0, cfg["s_anchor"],
                                            size=cfg["n_anchor"])
                A.append(-cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_anchor"]))
            state["anchor_ref"] = np.mean(A, axis=0)

        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
            state["var"] = ev
        prev = V.copy()
        state["hist"] = (1 - cfg["rho"]) * state["hist"] + cfg["rho"] * V

        if t == 0:
            wt = np.ones(N) / N
        else:
            floor = p if mech == "m1" else 1e-12
            iv = 1.0 / np.maximum(ev, floor)
            iv = iv / iv.sum()
            lam = p if mech == "m2w" else 1.0
            wt = (1.0 - lam) * (np.ones(N) / N) + lam * iv
            if mech == "cap":
                wt = apply_cap(wt, p / N)

        w = w + wt @ V
        cum += wt
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))
        if alarms_too:
            al.append(detectors(V, adv, state, cfg))
        if t >= cfg["rounds"] - 20:
            wt_tail.append(wt.copy())

    out = dict(rew=float(cum[adv].sum() / cum.sum()),
               mse=float(np.mean(mse[-20:])),
               res=spearman(np.mean(wt_tail, axis=0)[hon], 1.0 / sigma[hon]))
    if alarms_too:
        out["al"] = {k: float(np.mean([a[k] for a in al])) for k in al[0]}
    return out


def agg(mech, p, s_adv, dmag, seeds, K=1, honest_only=False, alarms_too=False):
    R = [sim(mech, p, s_adv, dmag, s, K, honest_only, alarms_too)
         for s in seeds]
    out = {k: float(np.mean([r[k] for r in R])) for k in ("rew", "mse", "res")}
    if alarms_too:
        out["al"] = {k: float(np.mean([r["al"][k] for r in R]))
                     for k in R[0]["al"]}
    return out


MECHS = [("undefended", "none", 0.0),
         ("hard cap c=1.4", "cap", 1.4),
         ("hard cap c=2.0", "cap", 2.0),
         ("hard cap c=3.0", "cap", 3.0),
         ("M2 lam=0.25", "m2w", 0.25),
         ("M2 lam=0.5", "m2w", 0.5),
         ("M1 c=3e-3", "m1", 3e-3)]

print("=" * 78)
print("A24  'why not just cap the weight?'  The cap measured against M1, M2")
print("=" * 78)
print()

# ------------------------------------------------------------- A24.1 / A24.2
print("=" * 78)
print("A24.1/2  harm, honest cost, and resolution at the silent operating point")
print("=" * 78)
print("  honest cost is MSE on the attack-free control under the SAME")
print("  mechanism, relative to undefended.  resolution is the Spearman")
print("  correlation between honest clients' weight and their true precision")
print("  1/sigma on that control: a mechanism that bounds the adversary by")
print("  flattening the weights has become FedAvg and cannot rank at all.")
print()
base = agg("none", 0.0, S_ADV, 0.0, SEEDS, honest_only=True)
print(f"{'mechanism':>16} {'xfair':>8} {'cost':>9} {'resolution':>11}")
FRONTIER = {}
for label, mech, p in MECHS:
    a = agg(mech, p, S_ADV, DELTA, SEEDS)
    h = agg(mech, p, S_ADV, 0.0, SEEDS, honest_only=True)
    xf = a["rew"] / FAIR
    FRONTIER[label] = (xf, h["mse"] / base["mse"], h["res"])
    print(f"{label:>16} {xf:7.2f}x {h['mse'] / base['mse']:8.3f}x "
          f"{h['res']:+10.3f}")
und_xf = FRONTIER["undefended"][0]
assert abs(und_xf - 4.59) < 0.15, f"undefended {und_xf} != 4.59 (E3)"
assert abs(FRONTIER["M2 lam=0.5"][0] - 2.82) < 0.1, "M2 0.5 != 2.82 (E3)"
assert abs(FRONTIER["M2 lam=0.25"][0] - 1.91) < 0.1, "M2 0.25 != 1.91 (E3)"
assert abs(FRONTIER["M1 c=3e-3"][0] - 1.30) < 0.1, "M1 3e-3 != 1.30 (E3)"
print()
print("  harness OK: undefended 4.59x, M2 2.82/1.91x, M1 1.30x reproduce E3.")
print()
cap3, m2h = FRONTIER["hard cap c=3.0"], FRONTIER["M2 lam=0.5"]
cap2, m2l = FRONTIER["hard cap c=2.0"], FRONTIER["M2 lam=0.25"]
print(f"  matched at ~2.8x: cap {cap3[1]:.3f}x cost vs M2 {m2h[1]:.3f}x")
print(f"  matched at ~1.9x: cap {cap2[1]:.3f}x cost vs M2 {m2l[1]:.3f}x")
print("  -> at matched harm the cap is the CHEAPER mechanism, and it keeps")
print("     its resolution.  Both easy objections to it fail.")
print()

# -------------------------------------------------------------------- A24.3
print("=" * 78)
print("A24.3  is the cap a rate limiter?  E9's loud/silent test")
print("=" * 78)
print("  E9 found M2 to be only a rate limiter: 3.72x silent, 18.63x loud.")
print("  M1 gave 1.30x either way.  A cap rescales the take, so it should")
print("  behave like M2 -- it does not.")
print()
print(f"{'mechanism':>16} {'silent':>8} {'loud':>9} {'loud/silent':>12}")
RATIO = {}
for label, mech, p in [("undefended", "none", 0.0),
                       ("hard cap c=3.0", "cap", 3.0),
                       ("hard cap c=2.0", "cap", 2.0),
                       ("hard cap c=1.4", "cap", 1.4)]:
    ctrl_s = agg(mech, p, 1.0, 0.0, SEARCH_SEEDS, honest_only=True,
                 alarms_too=True)
    ctrl_r = agg(mech, p, 1.0, 0.0, REPORT_SEEDS, honest_only=True,
                 alarms_too=True)
    grid = []
    for sa in S_GRID:
        for dm in D_GRID:
            c = agg(mech, p, sa, dm, SEARCH_SEEDS, alarms_too=True)
            exc = max(c["al"][k] - ctrl_s["al"].get(k, 0.0) for k in c["al"])
            grid.append((sa, dm, c["rew"] / FAIR, exc))
    loud = max(g[2] for g in grid)
    silent = [g for g in grid if g[3] <= SILENT_EPS]
    sa, dm, _, _ = max(silent, key=lambda g: g[2])
    ad = agg(mech, p, sa, dm, REPORT_SEEDS, alarms_too=True)
    xf = ad["rew"] / FAIR
    RATIO[label] = (xf, loud, loud / xf)
    print(f"{label:>16} {xf:7.2f}x {loud:8.2f}x {loud / xf:11.2f}x")
print()
print("  E9 published: undefended 4.60/24.81 (5.39x), M2 lam=0.75 3.72/18.63")
print("  (5.01x), M1 c=3e-3 1.30/1.30 (1.00x).")
u = RATIO["undefended"]
assert abs(u[0] - 4.60) < 0.15, f"undefended silent {u[0]} != 4.60"
assert abs(u[1] - 24.81) < 1.5, f"undefended loud {u[1]} != 24.81"
for lbl in ("hard cap c=3.0", "hard cap c=2.0", "hard cap c=1.4"):
    assert RATIO[lbl][2] < 1.1, f"{lbl} ratio {RATIO[lbl][2]}"
print("  harness OK: undefended row reproduces E9's 4.60/24.81.")
print("  -> every cap ratio is 1.00x. The cap is NOT a rate limiter; on this")
print("     axis it behaves like M1's floor, not like M2's mixing.")
print()

# -------------------------------------------------------------------- A24.4
print("=" * 78)
print("A24.4  eq:suppcost binds the cap: it is not an exemption")
print("=" * 78)
print("  A capped rule is still peer-only, so thm:unidentifiable applies and")
print("  A20.2's identity eps_2 = 1 - supp must hold for it.  Measured on")
print("  A20's own three-world construction.")
print()


def make_capped(c):
    base_rule = RULES["inverse sq distance"]

    def f(V):
        return apply_cap(base_rule(V).copy(), c / len(V))
    return f


CAPPED = {f"inv-sq + cap c={c}": make_capped(c) for c in (1.4, 2.0, 3.0)}
print(f"{'rule':>21} {'max|w(W1)-w(W2)|':>18} {'worst |1-supp-eps_2|':>21}")
worst_gap, worst_id = 0.0, 0.0
for nm, f in CAPPED.items():
    g, idg = 0.0, 0.0
    for dm in MAGS:
        for s in range(NSEED):
            V1, V2, VC = three_worlds(dm, s)
            w1, w2, wc = f(V1), f(V2), f(VC)
            g = max(g, float(np.max(np.abs(w1 - w2))))
            if wc[0] > 0:
                idg = max(idg, abs((1 - w1[0] / wc[0]) - (1 - w2[0] / wc[0])))
    worst_gap, worst_id = max(worst_gap, g), max(worst_id, idg)
    print(f"{nm:>21} {g:18.3e} {idg:21.3e}")
print()
print(f"  the two worlds agree to {worst_gap:.1e} -- not bitwise, because the")
print("  water-filling redistribution is float-asymmetric, but far inside any")
print("  weight that matters -- and the identity holds to "
      f"{worst_id:.1e}.")
assert worst_gap < 1e-14, worst_gap
assert worst_id < 1e-12, worst_id
print("  -> the cap buys its bound at the same exchange rate as every other")
print("     peer-only rule.  It is another point on eq:suppcost, not an")
print("     escape from it.")
print()

# -------------------------------------------------------------------- A24.5
print("=" * 78)
print("A24.5  the cap's guarantee decays in coalition size; M2's does not")
print("=" * 78)
print("  A per-client cap of c/N held by K colluders yields at most K*c/N, so")
print("  the coalition's absolute take grows linearly in K and the cap is")
print("  inert once K/N >= 1/c.  M2 reserves (1-lambda) of the mass for")
print("  anchor-derived weight whatever K is.  Calibrate the pairs to the")
print("  same absolute share at K = 1 and follow them out.")
print()
print(f"{'mechanism':>16}" + "".join(f"{'K=' + str(k):>10}" for k in K_GRID))
ABS = {}
for label, mech, p in MECHS:
    cells = []
    for K in K_GRID:
        a = agg(mech, p, S_ADV, DELTA, SEEDS, K=K)
        cells.append(a["rew"])
    ABS[label] = cells
    print(f"{label:>16}" + "".join(f"{c:10.3f}" for c in cells))
print()
print("  (absolute coalition weight share; K/N alone would give "
      + ", ".join(f"{K / CFG['N']:.3f}" for K in K_GRID) + ")")
print()
for cap_l, m2_l in (("hard cap c=3.0", "M2 lam=0.5"),
                    ("hard cap c=2.0", "M2 lam=0.25")):
    c1, m1_ = ABS[cap_l][0], ABS[m2_l][0]
    cE, mE = ABS[cap_l][-1], ABS[m2_l][-1]
    print(f"  {cap_l} vs {m2_l}: equal at K=1 ({c1:.3f} vs {m1_:.3f}), "
          f"{cE:.3f} vs {mE:.3f} at K={K_GRID[-1]}")
    assert abs(c1 - m1_) < 0.01, f"pair not matched at K=1: {c1} {m1_}"
    assert cE > mE, f"no crossover for {cap_l}: {cE} <= {mE}"
gap = ABS["hard cap c=3.0"][-1] - ABS["undefended"][-1]
print(f"  cap c=3.0 at K={K_GRID[-1]} is within {abs(gap):.3f} of the "
      "undefended share: fully inert.")
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  A24.1 cheaper than M2 at matched harm      : "
      f"{cap3[1]:.3f}x vs {m2h[1]:.3f}x at ~2.8x")
print(f"  A24.2 keeps weight-vs-precision ranking    : "
      f"{cap3[2]:+.3f} vs undefended "
      f"{FRONTIER['undefended'][2]:+.3f}")
print(f"  A24.3 not a rate limiter                   : "
      f"loud/silent {RATIO['hard cap c=3.0'][2]:.2f}x vs M2's 5.01x")
print(f"  A24.4 eq:suppcost binds it                 : "
      f"identity to {worst_id:.1e}")
print(f"  A24.5 decays in K, inert at K/N >= 1/c     : "
      f"{ABS['hard cap c=3.0'][-1]:.3f} vs M2 "
      f"{ABS['M2 lam=0.5'][-1]:.3f} at K=15")
print()
print("  A24 CLAIMS: the cap beats M2 on the single-adversary frontier and is")
print("  not defeated by an adaptive adversary, so sec:e3 must report it; M1")
print("  still dominates both; and the cap is bound by eq:suppcost and decays")
print("  in coalition size, which is what the reserved anchor mass is for.")
sys.exit(0)
