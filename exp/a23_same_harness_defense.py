"""A23: two claims in sec:e3 are cross-harness comparisons presented as if
they came from one harness, and both of the numbers involved were measured on
a single seed.

Claim 1.  'Frame mixing ... performs worse still: the adversary holds 5.79x
fair share at lam = 0.5 and 4.77x even at lam = 0.'  Those two figures are
V3.3's, whose own undefended reference is 6.85x, while the 3.71/2.82/1.91
weight-mixing figures they are contrasted with are E3's, whose undefended
reference is 4.59x.  Read in its own harness 5.79x is an IMPROVEMENT on 6.85x,
not a degradation relative to 4.59x.  Worse, V3.3's three M1 rows are all
inert at exactly 6.85x, so that harness's floor never binds and it cannot
support a comparison between mechanisms at all.

Claim 2.  'Sweeping a from 1 to 64 at lam = 0.5 leaves epsilon flat.'  V3.2
measures eps RISING monotonically-ish from 0.1487 at a=1 to 0.1817 at a=64,
against its own stated prediction that it falls as 1/sqrt(a).  One seed.

Both are settled here by running M1, M2-weight and M2-frame in E3's harness --
the real regression harness at the E2d silent operating point, over E3's eight
seeds -- so every number shares one undefended baseline.
"""

import numpy as np
from e2_silence import CFG

seeds = [0, 1, 2, 3, 4, 5, 6, 7]
S_ADV, DELTA, K = 0.35, 0.05, 1


def sim(floor, lam, mode, a=3, s_anch=0.3, seed=0, dm=DELTA, sa=S_ADV,
        honest_only=False):
    """E3's sim_def, extended with eq:refmix FRAME mixing as a third mode.

    mode="none"   : plain inverse variance
    mode="weight" : wt = (1-lam) uniform + lam invvar        (what E3 ran)
    mode="frame"  : mu  = lam mu_peer + (1-lam) mu_anchor    (eq:refmix, M2)

    The anchor frame is the mean of `a` trusted updates, each computed by the
    SAME local step on the true target with noise s_anch, so it is an unbiased
    estimate of the honest update.  This matters: E3's updates are gradient
    steps and are not centred on the origin, so a zero-centred draw would act
    as a large common offset and dilute every dispersion equally, which looks
    like a defence but is an artefact of the wrong reference.
    """
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
    aw, hwm, mse = [], [], []
    wt = np.ones(N) / N

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if (i < K and not honest_only) else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        if t > 0:
            if mode == "frame":
                mu_p = wt @ V
                if a > 0:
                    A = np.empty((a, d))
                    for j in range(a):
                        Xa = rng.normal(size=(cfg["n_local"], d))
                        ya = Xa @ w_star + rng.normal(0, s_anch,
                                                      size=cfg["n_local"])
                        A[j] = -cfg["lr"] * (Xa.T @ (Xa @ w - ya)
                                             / cfg["n_local"])
                    ref = lam * mu_p + (1 - lam) * A.mean(0)
                else:
                    ref = mu_p
                inst = np.linalg.norm(V - ref, axis=1) ** 2 / d
            else:
                inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
        prev = V.copy()

        if t == 0:
            wt = np.ones(N) / N
        else:
            iv = 1.0 / np.maximum(ev, floor)
            iv = iv / iv.sum()
            wt = (1.0 - lam) * (np.ones(N) / N) + lam * iv \
                if mode == "weight" else iv

        w = w + wt @ V
        cum += wt
        aw.append(wt[adv].sum()); hwm.append(wt[hon].mean())
        mse.append(float(np.mean((Xte @ w - yte) ** 2)))

    return dict(aw=float(np.mean(aw[-20:])), hw=float(np.mean(hwm[-20:])),
                mse=float(np.mean(mse[-20:])),
                rew=float(cum[adv].sum() / cum.sum()))


def agg(**kw):
    R = [sim(seed=s, **kw) for s in seeds]
    return {k: float(np.mean([r[k] for r in R])) for k in R[0]}, \
           {k: float(np.std([r[k] for r in R])) for k in R[0]}


fair = K / CFG["N"]
hon_ref = agg(floor=1e-12, lam=1.0, mode="none", dm=0.0, honest_only=True)[0]

print("=" * 78)
print("A23.1  every mechanism in ONE harness, against ONE baseline")
print("=" * 78)
print("  E3's harness at the E2d silent point, 8 seeds.  x fair is the")
print("  cumulative reward share over fair share; cost is honest-population")
print("  MSE against inverse variance on the same population.")
print()
print(f"{'mechanism':>32} {'x fair':>9} {'sd':>6} {'eps':>8} {'cost':>7}")


def row(nm, **kw):
    m, s = agg(**kw)
    xf = m["rew"] / fair
    eps = (m["aw"] - fair) / (1 - fair)
    hm = agg(dm=0.0, honest_only=True,
             **{k: v for k, v in kw.items()})[0]["mse"]
    print(f"{nm:>32} {xf:8.2f}x {s['rew']/fair:6.2f} {eps:8.4f} "
          f"{hm/hon_ref['mse']:6.2f}x")
    return xf


und = row("undefended (invvar, no floor)", floor=1e-12, lam=1.0, mode="none")
m1 = {}
for fl in [1e-3, 3e-3, 1e-2]:
    m1[fl] = row(f"M1 floor={fl:.0e}", floor=fl, lam=1.0, mode="none")
wm = {}
for lm in [0.75, 0.5, 0.25, 0.0]:
    wm[lm] = row(f"M2 WEIGHT mixing lam={lm:.2f}", floor=1e-12, lam=lm,
                 mode="weight")
fm = {}
for lm in [1.0, 0.75, 0.5, 0.25, 0.0]:
    fm[lm] = row(f"M2 FRAME mixing lam={lm:.2f} (a=3)", floor=1e-12, lam=lm,
                 mode="frame")

print()
print(f"  undefended baseline in THIS harness: {und:.2f}x fair")
print(f"  V3.3's baseline was 6.85x and E3's was 4.59x, which is why the two")
print("  sets of figures could not be compared: they are different systems.")
print()
print("  Comparing the two M2 variants at equal lam, same harness, same seeds:")
print(f"{'lam':>6} {'weight mixing':>15} {'frame mixing':>14} {'which is worse':>16}")
worse_frame = 0
for lm in [0.75, 0.5, 0.25, 0.0]:
    tag = "frame" if fm[lm] > wm[lm] else "weight"
    worse_frame += int(fm[lm] > wm[lm])
    print(f"{lm:6.2f} {wm[lm]:14.2f}x {fm[lm]:13.2f}x {tag:>16}")
print(f"\n  frame mixing is the worse of the two at {worse_frame} of 4 settings.")
print("  That is the claim sec:e3 makes, and this is the first measurement of")
print("  it in which both mechanisms saw the same clients.")
assert und > 3.0, ("the harness must reproduce E3's capture", und)
assert abs(und - 4.59) < 0.15, ("must reproduce E3's baseline exactly", und)
assert abs(wm[0.75] - 3.71) < 0.1 and abs(wm[0.5] - 2.82) < 0.1, \
    "weight-mixing rows must reproduce E3's"
assert worse_frame == 4, ("sec:e3's direction claim requires frame mixing to "
                          "be worse at every lam", worse_frame)

print()
print("=" * 78)
print("A23.2  does M1 still bind here, and does frame mixing reach it?")
print("=" * 78)
print("  V3.3's M1 rows were all inert at 6.85x -- its floor never bound, so")
print("  that harness could not rank the mechanisms.  Here:")
print()
best_m1 = min(m1.values())
best_fm = min(fm.values())
print(f"  best M1 row          {best_m1:.2f}x fair")
print(f"  best frame-mix row   {best_fm:.2f}x fair")
print(f"  best weight-mix row  {min(wm.values()):.2f}x fair")
print(f"  M1 range across the three floors: "
      f"{max(m1.values())-best_m1:.2f}x, so the floor BINDS.")
assert max(m1.values()) - best_m1 > 0.5, "M1 inert again; harness unusable"
print("  M1 is load-bearing and the frame mix is not, which is the design rule")
print("  sec:e3 states -- now supported by rows that share a baseline.")

print()
print("=" * 78)
print("A23.3  the anchor-budget sweep, with seeds")
print("=" * 78)
print("  V3.2 swept a on ONE seed and eps rose 0.1487 -> 0.1817 against its")
print("  own prediction of a 1/sqrt(a) fall; sec:e3 calls it flat.  Eight")
print("  seeds, E3's harness, lam = 0.5, frame mixing:")
print()
print(f"{'a':>4} {'x fair':>9} {'eps':>9} {'sd(eps)':>9} {'eps*sqrt(a)':>12}")
epsm, epss = [], []
for a in [1, 2, 4, 8, 16, 32, 64]:
    m, s = agg(floor=1e-12, lam=0.5, mode="frame", a=a)
    e = (m["aw"] - fair) / (1 - fair)
    epsm.append(e); epss.append(s["aw"] / (1 - fair))
    print(f"{a:4d} {m['rew']/fair:8.2f}x {e:9.4f} {s['aw']/(1-fair):9.4f} "
          f"{e*np.sqrt(a):12.4f}")
epsm, epss = np.array(epsm), np.array(epss)
aa = np.array([1, 2, 4, 8, 16, 32, 64], float)
sl = np.polyfit(np.log(aa), np.log(np.maximum(epsm, 1e-12)), 1)[0]
spread = epsm.max() - epsm.min()
print(f"\n  log eps vs log a slope {sl:+.4f}; the prediction eps ~ 1/sqrt(a)")
print(f"  needs -0.5, and a flat eps needs 0.")
print(f"  spread {spread:.4f} against a mean seed sd of {epss.mean():.4f}, i.e.")
print(f"  {spread/epss.mean():.1f} seed sd.")
if spread < 2 * epss.mean():
    print("  The variation across a is inside the seed noise, so 'flat' is the")
    print("  honest description and the 1/sqrt(a) term is not detectable here.")
else:
    print("  The variation across a EXCEEDS the seed noise, so 'flat' is not")
    print("  the honest description; the text must report the trend and sign.")
print("  Either way the load-bearing claim is the one that survives: under")
print("  TSP-Var the anchor budget does not govern displacement, Pi_max does.")
assert spread < 2 * epss.mean(), (
    "sec:e3 calls this flat; if it stops being flat the text must change",
    spread, epss.mean())
assert abs(sl) < 0.2, ("eps must not acquire a trend in a", sl)

print()
print("=" * 78)
print("A23 conclusion: the 5.79x/4.77x figures belong to a harness whose own")
print("undefended point is 6.85x and whose M1 rows are inert, so they cannot")
print("be set against E3's 4.59x.  A23.1 replaces them with same-harness")
print("numbers.  A23.3 replaces the single-seed 'flat' with a seeded")
print("comparison against the seed noise.")
print("=" * 78)
