"""E9: the defences against an adversary that KNOWS them.

E3 measures M1 and M2 at one operating point, (s_adv, delta) = (0.35, 0.05),
and that point was chosen against the UNDEFENDED aggregator.  A referee will
object immediately: a defence evaluated only at the attacker's old optimum is
not evaluated at all.  So here the adversary observes the deployed defence
parameters (c, lambda) and re-solves for its own best silent operating point.

The threat model does not change.  The adversary still fabricates nothing --
it reports its true local gradients with its true noise level s_adv and a
target shifted by delta -- and it is still bound by silence: an operating
point counts only if no public detector's alarm rate exceeds its own
false-positive rate under the SAME defence by more than SILENT_EPS.  Raising
s_adv or delta until the detectors fire is not an attack, it is a different
paper.

Why M1 and M2 should differ.  M2 caps any client's weight at
(1-lam)/N + lam by construction, with no reference to the claimed variance,
so no choice of (s_adv, delta) can move that ceiling -- the bound is
worst-case already.  M1's floor is a cap too, but on the ESTIMATE:
w_i propto 1/max(ev_i, c), so an adversary sitting anywhere below the floor
gets 1/c, the same as one sitting exactly at it.  The interesting question is
therefore not whether the adversary can beat the floor by shrinking s_adv
(it cannot; the floor is flat below c) but whether the delta knob buys
anything once s_adv is pinned, since delta shifts the target and only weakly
inflates the movement estimate.

Selection and reporting run on DISJOINT seeds.  The grid is searched on
seeds 0-4 and the winner is re-measured on seeds 5-12, because picking the
argmax of a noisy grid and then quoting that same cell is how a search
procedure manufactures an effect.  The re-measured number is the one that
goes in the paper, and it is allowed to be lower than the grid's.
"""

import sys

import numpy as np

from e2_silence import CFG, detectors, w_krum

SEARCH_SEEDS = [0, 1, 2, 3, 4]
REPORT_SEEDS = [5, 6, 7, 8, 9, 10, 11, 12]
SILENT_EPS = 0.05           # same threshold as E2b/E2c
STATIC = (0.35, 0.05)       # the operating point E3 was tuned at

S_GRID = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
D_GRID = [0.02, 0.05, 0.10, 0.20, 0.30]

# (label, floor c, lambda) -- the settings tab:defense reports
DEFENCES = [
    ("none",              1e-12, 1.00),
    ("M1 c=1e-3",         1e-03, 1.00),
    ("M1 c=3e-3",         3e-03, 1.00),
    ("M1 c=1e-2",         1e-02, 1.00),
    ("M2 lam=0.75",       1e-12, 0.75),
    ("M2 lam=0.50",       1e-12, 0.50),
    ("M2 lam=0.25",       1e-12, 0.25),
    ("M1+M2 3e-3,0.75",   3e-03, 0.75),
]


def sim(floor, lam, s_adv, delta_mag, seed, honest_only=False):
    """One defended federation.  Mirrors e3_defense.sim_def, plus detectors.

    honest_only replaces the adversary by an ordinary honest client, which is
    the per-defence false-positive control: the same defence, the same seed,
    the same client count, no attack.
    """
    cfg = dict(CFG)
    cfg["s_adv"] = s_adv
    k = cfg["k_adv"]
    rng = np.random.default_rng(seed)
    d, N = cfg["d"], cfg["N"]
    adv, hon = np.arange(k), np.arange(k, N)

    w_star = rng.normal(size=d)
    dl = rng.normal(size=d)
    dl = delta_mag * dl / (np.linalg.norm(dl) + 1e-12)

    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv] = s_adv if not honest_only else rng.uniform(cfg["s_lo"],
                                                           cfg["s_hi"])

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    state = {"var": np.full(N, 1.0), "hist": np.zeros((N, d)),
             "anchor_ref": np.zeros(d)}
    prev = np.zeros((N, d))
    ev = np.full(N, np.nan)
    cum = np.zeros(N)
    aw, hw, mse, alarms = [], [], [], []

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if (i < k and not honest_only) else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

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
            iv = 1.0 / np.maximum(ev, floor)
            iv = iv / iv.sum()
            wt = (1.0 - lam) * (np.ones(N) / N) + lam * iv

        w = w + wt @ V
        cum += wt
        aw.append(wt[adv].sum())
        hw.append(wt[hon].mean())
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))
        alarms.append(detectors(V, adv, state, cfg))

    return dict(aw=float(np.mean(aw[-20:])), hw=float(np.mean(hw[-20:])),
                mse=float(np.mean(mse[-20:])),
                rew=float(cum[adv].sum() / cum.sum()),
                al={k_: float(np.mean([a[k_] for a in alarms]))
                    for k_ in alarms[0]})


def agg(floor, lam, s_adv, delta_mag, seeds, honest_only=False):
    R = [sim(floor, lam, s_adv, delta_mag, s, honest_only) for s in seeds]
    return dict(
        aw=float(np.mean([r["aw"] for r in R])),
        hw=float(np.mean([r["hw"] for r in R])),
        mse=float(np.mean([r["mse"] for r in R])),
        rew=float(np.mean([r["rew"] for r in R])),
        al={k: float(np.mean([r["al"][k] for r in R])) for k in R[0]["al"]},
    )


def excess(cell, ctrl):
    """Worst detector alarm rate above that detector's own false-positive rate
    under the same defence.  Negative excess is clipped: a detector that fires
    LESS often under attack is not evidence of the attack."""
    return max(cell["al"][k] - ctrl["al"].get(k, 0.0) for k in cell["al"])


FAIR = CFG["k_adv"] / CFG["N"]

print("=" * 90)
print("E9  the adaptive adversary: it observes (c, lambda) and re-optimises")
print(f"     grid {len(S_GRID)}x{len(D_GRID)} on seeds {SEARCH_SEEDS}, "
      f"winner re-measured on the disjoint seeds {REPORT_SEEDS}")
print(f"     silent means worst detector excess <= {SILENT_EPS} against the "
      "same defence's own control")
print("=" * 90)

# reference: all-honest utility per defence, for the cost column
ROWS = []
for label, floor, lam in DEFENCES:
    ctrl_s = agg(floor, lam, 1.0, 0.0, SEARCH_SEEDS, honest_only=True)
    ctrl_r = agg(floor, lam, 1.0, 0.0, REPORT_SEEDS, honest_only=True)

    # the static attacker E3 reported, re-measured on the reporting seeds
    st = agg(floor, lam, STATIC[0], STATIC[1], REPORT_SEEDS)
    st_x, st_e = st["rew"] / FAIR, excess(st, ctrl_r)

    # ---- the search, on its own seeds
    best = None
    grid = []
    for sa in S_GRID:
        for dm in D_GRID:
            c = agg(floor, lam, sa, dm, SEARCH_SEEDS)
            e = excess(c, ctrl_s)
            x = c["rew"] / FAIR
            grid.append((sa, dm, x, e))
            if e <= SILENT_EPS and (best is None or x > best[2]):
                best = (sa, dm, x, e)
    loud = max(g[2] for g in grid)          # what silence costs the adversary

    if best is None:
        print(f"\n{label:18} no silent cell in the grid at all")
        ROWS.append((label, st_x, st_e, None, None, None, None, loud))
        continue

    # ---- the winner, re-measured out of sample
    sa, dm = best[0], best[1]
    ad = agg(floor, lam, sa, dm, REPORT_SEEDS)
    ad_x, ad_e = ad["rew"] / FAIR, excess(ad, ctrl_r)
    print(f"\n{label:18} static ({STATIC[0]}, {STATIC[1]}) -> "
          f"{st_x:.2f}x (excess {st_e:+.3f})")
    print(f"{'':18} adaptive argmax (s_adv={sa}, delta={dm}) -> "
          f"grid {best[2]:.2f}x, held out {ad_x:.2f}x (excess {ad_e:+.3f})")
    print(f"{'':18} gain from adapting {ad_x - st_x:+.2f}x   "
          f"best cell ignoring silence {loud:.2f}x   "
          f"honest-cost {ad['mse'] / ctrl_r['mse']:.2f}x")
    sys.stdout.flush()
    ROWS.append((label, st_x, st_e, sa, dm, ad_x, ad_e, loud))

print()
print("=" * 90)
print("SUMMARY  reward / fair share, k=1")
print("=" * 90)
print(f"{'defence':18} {'static':>8} {'adaptive':>9} {'gain':>7} "
      f"{'argmax (s,delta)':>18} {'excess':>8} {'if loud':>8}")
for label, st_x, st_e, sa, dm, ad_x, ad_e, loud in ROWS:
    if ad_x is None:
        print(f"{label:18} {st_x:8.2f} {'--':>9} {'--':>7} "
              f"{'no silent cell':>18} {'--':>8} {loud:8.2f}")
        continue
    print(f"{label:18} {st_x:8.2f} {ad_x:9.2f} {ad_x - st_x:+7.2f} "
          f"{f'({sa}, {dm})':>18} {ad_e:+8.3f} {loud:8.2f}")

# ---------------------------------------------------------------- assertions
res = {r[0]: r for r in ROWS}

# 1. The search must be able to find the undefended point E2c already found,
#    or the search is not exploring the same space as the rest of the paper.
assert 4.0 <= res["none"][5] <= 6.5, res["none"][5]

# 2. M2's ceiling is analytic, so no adaptive choice may cross it.  This is
#    the claim tab:defense makes; if it fails, lem:budget is wrong.
for label, lam in [("M2 lam=0.75", 0.75), ("M2 lam=0.50", 0.50),
                   ("M2 lam=0.25", 0.25)]:
    bound = ((1 - lam) / CFG["N"] + lam) / FAIR
    got = res[label][5]
    assert got <= bound + 1e-9, (label, got, bound)

# 3. Every defence must still reduce the capture against the ADAPTIVE
#    adversary, not merely against the static one.  A defence that only holds
#    at the old operating point does not belong in the paper.
base = res["none"][5]
for label in [l for l, _, _ in DEFENCES if l != "none"]:
    assert res[label][5] < base, (label, res[label][5], base)

print("\nOK: the search reproduces the undefended ceiling, M2's analytic bound "
      "holds under re-optimisation, and every defence still binds.")
print("E9 CLAIMS: " + " | ".join(
    f"{l} {r[5]:.2f}x (static {r[1]:.2f}x)" for l, r in res.items()
    if r[5] is not None))
