"""
B1: published attack baselines against the same decision makers.

Every comparison in section VI so far is internal -- our attack against
scorers we implemented. This script closes that by reimplementing three
published attacks in the identical harness (same generator, same scorers,
same detectors, same seeds) and reporting them on the same three axes:

    L3 harm     adversary weight share            (what TSP targets)
    L1 harm     test MSE / no-attack baseline     (what the literature targets)
    silence     max detector excess over its own false-positive rate

Baselines, all full-knowledge (strictly stronger than our threat model,
which forbids fabricating an update at all):

  fang-trim   Fang et al., USENIX Security 2020, Sec. 5 -- the directed
              deviation attack tailored to trimmed mean / median. Per
              coordinate, malicious clients report values just outside the
              benign range on the side opposite the honest direction, so
              trimming removes honest values instead.
  fang-krum   Fang et al., USENIX Security 2020, Sec. 4 -- craft u = -lam*s
              with s the sign of the benign mean, halving lam until Krum
              selects a malicious client.
  shej-minmax Shejwalkar & Houmansadr, NDSS 2021, Alg. 1 -- u = mu + gam*p
              with p the negated per-coordinate std of the benign updates,
              gam maximised subject to
                  max_i ||u - u_i|| <= max_{i,j} ||u_i - u_j||.
  shej-minsum same paper, the min-sum variant:
                  sum_i ||u - u_i||^2 <= max_j sum_i ||u_j - u_i||^2.
  freerider   Fraboni et al., AISTATS 2021 -- resubmit the previous global
              update; the disguised variant adds noise of std sigma*gam^t.
  ace         Xu et al., USENIX Security 2024 -- the one published attack whose
              TARGET is the contribution/trust decision and not accuracy, so it
              is the nearest prior work and the comparison that matters. Predict
              the next global update by the Cauchy mean value theorem,
              ghat = g_{t-1} + H_t (w_t - w_{t-1}), the Hessian-vector product
              approximated by L-BFGS two-loop recursion over buffers of the last
              m global models and updates; fall back to honest local training
              when ||H_t dw|| > l*||dw|| (their Eq. 3, l = 1); amplify by c > 1
              for cosine-distance scorers (their Sec. 4.2.3). Same full
              knowledge of the broadcast history as the others.
  ace-amp     ACE with the cosine amplification at c = 2, their tuned variant.
  ace-tuned   ACE with the threshold filter relaxed. At their l = 1 the filter
              rejects 96% of rounds HERE, because in this harness dw is itself a
              noisy global update and ||H dw||/||dw|| has median 1.49, so l = 1
              is not the right tuning for this generator and reporting only that
              row would understate their attack. l is theirs to tune (their Sec.
              4.2.2 says so), and the row saturates from l = 3 on, so this row
              is ACE at its best and it is the row to compare against. B1.5
              measures the filter statistic and the l sweep rather than asserting
              them.
  tsp-var     ours. The adversary crafts nothing: it trains honestly on
              low-variance biased data of its own choosing.

Run:  python3 -u b1_published_baselines.py
"""

import json
import sys

import numpy as np

from e2_silence import SCORERS, w_krum, CFG
from e2b_sweep import detectors2

# TSP-Var operating point: the silent one used by fig:silence and tab:decouple.
TSP_S_ADV, TSP_DELTA = 0.35, 0.05

# Fang's expansion factor b, and the halving budget for the Krum attack.
FANG_B = 2.0
KRUM_LAM0, KRUM_ITERS = 10.0, 30

# Shejwalkar's gamma search: initial value and halving budget (their Alg. 1).
SHEJ_GAM0, SHEJ_ITERS = 10.0, 30

# Fraboni's disguise: noise std relative to benign dispersion, decay per round.
FR_SIGMA, FR_GAMMA = 1.0, 0.5

# ACE: buffer depth m, filter coefficient l, preliminary rounds, amplifier c.
# m = 3 and l = 1 are the values Xu et al. report as sufficient; the preliminary
# phase is their own remedy for an empty buffer.
ACE_M, ACE_L, ACE_PRELIM, ACE_C = 3, 1.0, 3, 2.0


# ------------------------------------------------------------------ crafting

def craft_fang_trim(Vh, k, rng, b=FANG_B):
    """Directed deviation, full knowledge of the benign updates."""
    d = Vh.shape[1]
    s = np.sign(Vh.mean(axis=0))
    vmax, vmin = Vh.max(axis=0), Vh.min(axis=0)
    out = np.empty((k, d))
    for j in range(d):
        if s[j] > 0:
            # honest direction positive -> report below the benign minimum
            lo, hi = (vmin[j] / b, vmin[j]) if vmin[j] > 0 else (b * vmin[j], vmin[j])
        else:
            lo, hi = (vmax[j], b * vmax[j]) if vmax[j] > 0 else (vmax[j], vmax[j] / b)
        out[:, j] = rng.uniform(min(lo, hi), max(lo, hi), size=k)
    return out


def craft_fang_krum(V, adv, hon, state, cfg):
    """Craft -lam*s and halve lam until multi-Krum admits a malicious client."""
    s = np.sign(V[hon].mean(axis=0))
    lam = KRUM_LAM0
    best = -lam * s
    for _ in range(KRUM_ITERS):
        Vt = V.copy()
        Vt[adv] = -lam * s
        if np.any(w_krum(Vt, state, cfg)[adv] > 0.0):
            best = -lam * s
            break
        lam /= 2.0
        best = -lam * s
    return np.tile(best, (len(adv), 1))


def craft_shej(Vh, k, kind):
    """AGR-agnostic min-max / min-sum with the 'std' perturbation direction."""
    mu = Vh.mean(axis=0)
    p = -Vh.std(axis=0)
    pn = np.linalg.norm(p)
    if pn < 1e-12:
        return np.tile(mu, (k, 1))
    p = p / pn

    D = np.linalg.norm(Vh[:, None, :] - Vh[None, :, :], axis=2)
    if kind == "minmax":
        budget = D.max()

        def ok(g):
            return np.linalg.norm(Vh - (mu + g * p), axis=1).max() <= budget
    else:
        budget = (D ** 2).sum(axis=1).max()

        def ok(g):
            return float(((Vh - (mu + g * p)) ** 2).sum()) <= budget

    gam, step, best = SHEJ_GAM0, SHEJ_GAM0 / 2.0, 0.0
    for _ in range(SHEJ_ITERS):
        if ok(gam):
            best = max(best, gam)
            gam += step
        else:
            gam -= step
        step /= 2.0
        gam = max(gam, 0.0)
    return np.tile(mu + best * p, (k, 1))


def lbfgs_hv(dW, dG, v):
    """Approximate the integrated-Hessian product H_t v, as ACE's Eq. (2) needs.

    dW[j] = w_{t-j} - w_{t-j-1} and dG[j] the matching update delta, newest last.

    The two-loop recursion over pairs (s, y) = (step, gradient difference) returns
    an INVERSE-Hessian product, so feeding it (dW, dG) would give H^-1 v and the
    threshold filter of Eq. (3) would reject nearly every round. Running it on the
    swapped pairs (s, y) = (dG, dW) inverts the map it approximates and yields
    H v. ace_hv_selfcheck() verifies this against an exact quadratic: on the
    vector ACE actually passes, w_t - w_{t-1}, the swapped form is exact to
    1e-15 by the secant condition, and the unswapped form is 71% wrong.

    Curvature pairs with non-positive inner product are skipped, the standard
    damping.
    """
    q = v.astype(float).copy()
    alpha, use = [], []
    for sk, yk in zip(reversed(dG), reversed(dW)):
        sy = float(sk @ yk)
        if sy <= 1e-12:
            continue
        a = float(sk @ q) / sy
        q -= a * yk
        alpha.append(a)
        use.append((sk, yk, sy))
    if not use:
        return v.astype(float).copy()
    sk0, yk0, sy0 = use[0]
    q *= sy0 / max(float(yk0 @ yk0), 1e-12)
    for (sk, yk, sy), a in zip(reversed(use), reversed(alpha)):
        b = float(yk @ q) / sy
        q += (a - b) * sk
    return q


def craft_ace(hist, k, amplify, l=ACE_L):
    """ACE: submit the PREDICTED next global update, or fall back to honest.

    hist holds only what a client learns from broadcasts it already receives, the
    global models w and the global updates g. Returns None when the buffer is too
    short or the filter rejects, and the caller then leaves the honest update in
    place, which is ACE's own fallback.
    """
    W, G = hist["w"], hist["g"]
    if len(W) < 2 or len(G) < 2:
        return None
    dW = [W[j] - W[j - 1] for j in range(1, len(W))][-ACE_M:]
    dG = [G[j] - G[j - 1] for j in range(1, len(G))][-ACE_M:]
    dw = W[-1] - W[-2]
    hv = lbfgs_hv(dW, dG, dw)                                  # their Eq. (2)
    if np.linalg.norm(hv) > l * np.linalg.norm(dw):            # their Eq. (3)
        return None
    g = G[-1] + hv
    return np.tile(ACE_C * g if amplify else g, (k, 1))


def craft_freerider(prev_agg, Vh, k, t, rng, disguise):
    """Resubmit the previous global update, optionally with decaying noise."""
    u = np.tile(prev_agg, (k, 1))
    if disguise:
        sd = float(np.median(Vh.std(axis=0)))
        u = u + rng.normal(0.0, FR_SIGMA * sd * FR_GAMMA ** t, size=u.shape)
    return u


ATTACKS = [
    "none",
    "fang-trim",
    "fang-krum",
    "shej-minmax",
    "shej-minsum",
    "freerider",
    "freerider-dis",
    "ace",
    "ace-amp",
    "ace-tuned",
    "tsp-var",
]

CRAFTED = {"fang-trim", "fang-krum", "shej-minmax", "shej-minsum",
           "freerider", "freerider-dis", "ace", "ace-amp", "ace-tuned"}


# ---------------------------------------------------------------- simulation

def sim(attack, scorer_name, cfg, seed):
    """One federated run. Identical to e2b_sweep.sim except for the attack."""
    rng = np.random.default_rng(seed)
    cfg = dict(cfg)
    d, N, k = cfg["d"], cfg["N"], cfg["k_adv"]
    adv, hon = np.arange(k), np.arange(k, N)

    # TSP-Var is the only attack that acts through the data-generating process.
    s_adv = TSP_S_ADV if attack == "tsp-var" else None
    dl = np.zeros(d)
    if attack == "tsp-var":
        g = rng.normal(size=d)
        dl = TSP_DELTA * g / np.linalg.norm(g)

    w_star = rng.normal(size=d)
    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv] = s_adv if s_adv is not None else rng.uniform(
        cfg["s_lo"], cfg["s_hi"], size=k)

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    scorer = SCORERS[scorer_name]
    state = {"var": np.full(N, 1.0), "hist": np.zeros((N, d)),
             "anchor_ref": np.zeros(d)}
    prev = np.zeros((N, d))
    ev = np.full(N, np.nan)
    prev_agg = np.zeros(d)
    hist = {"w": [w.copy()], "g": []}    # all ACE reads: the broadcast history
    hw, aw, mse, alarms = [], [], [], []

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if i < k else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        if attack in CRAFTED:
            Vh = V[hon]
            if attack == "fang-trim":
                V[adv] = craft_fang_trim(Vh, k, rng)
            elif attack == "fang-krum":
                V[adv] = craft_fang_krum(V, adv, hon, state, cfg)
            elif attack == "shej-minmax":
                V[adv] = craft_shej(Vh, k, "minmax")
            elif attack == "shej-minsum":
                V[adv] = craft_shej(Vh, k, "minsum")
            elif attack.startswith("ace"):
                if t >= ACE_PRELIM:
                    u = craft_ace(hist, k, attack == "ace-amp",
                                  l=np.inf if attack == "ace-tuned" else ACE_L)
                    if u is not None:            # else honest, per their rule
                        V[adv] = u
            else:
                V[adv] = craft_freerider(prev_agg, Vh, k, t, rng,
                                         attack.endswith("-dis"))

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
        step = wt @ V
        w = w + step
        prev_agg = step
        hist["g"].append(step.copy())
        hist["w"].append(w.copy())
        if len(hist["w"]) > ACE_M + 2:           # ACE keeps m snapshots only
            hist["w"] = hist["w"][-(ACE_M + 2):]
            hist["g"] = hist["g"][-(ACE_M + 2):]

        hw.append(wt[hon].mean())
        aw.append(wt[adv].sum())
        m = float(np.mean((Xte @ w - yte) ** 2))
        mse.append(m if np.isfinite(m) else np.inf)
        alarms.append(detectors2(V, adv, state, cfg))

    return dict(hw=np.array(hw), aw=np.array(aw), mse=np.array(mse), al=alarms)


def agg(attack, scorer, seeds, cfg=CFG):
    R = [sim(attack, scorer, cfg, s) for s in seeds]
    keys = R[0]["al"][0].keys()
    return dict(
        aw=float(np.mean([r["aw"][-20:].mean() for r in R])),
        hw=float(np.mean([r["hw"][-20:].mean() for r in R])),
        mse=float(np.mean([r["mse"][-20:].mean() for r in R])),
        al={kk: float(np.mean([np.mean([a[kk] for a in r["al"]]) for r in R]))
            for kk in keys},
    )


def ace_hv_selfcheck():
    """The Hessian product must be right or ACE is being strawmanned.

    On a quadratic with Hessian A the update is g(w) = A w, so dg = A dw exactly
    and the true answer is A v. Checked at v = the newest step, which is the
    vector ACE's Eq. (2) passes.
    """
    r = np.random.default_rng(0)
    d = 20
    M = r.normal(size=(d, d))
    A = M @ M.T / d + np.eye(d)
    W = [r.normal(size=d)]
    for _ in range(10):
        W.append(W[-1] + r.normal(size=d) * 0.1)
    G = [A @ w for w in W]
    dW = [W[j] - W[j - 1] for j in range(1, len(W))][-ACE_M:]
    dG = [G[j] - G[j - 1] for j in range(1, len(G))][-ACE_M:]
    v = dW[-1]
    tgt = A @ v
    err = np.linalg.norm(lbfgs_hv(dW, dG, v) - tgt) / np.linalg.norm(tgt)
    wrong = np.linalg.norm(lbfgs_hv(dG, dW, v) - tgt) / np.linalg.norm(tgt)
    print(f"B1.ace  L-BFGS Hessian product: rel err {err:.2e} "
          f"(buffers reversed: {wrong:.2f}) -- must be < 1e-10")
    assert err < 1e-10, err
    return err


# --------------------------------------------------------------------- main

if __name__ == "__main__":
    seeds = list(range(8))
    scorers = list(SCORERS)
    out = {}

    ace_hv_selfcheck()
    print()

    # ---- baselines per scorer, from the no-attack run of the same scorer
    print("=" * 78)
    print("B1.0  no-attack control per scorer (denominator of every ratio)")
    print("=" * 78)
    base = {}
    for sc in scorers:
        base[sc] = agg("none", sc, seeds)
        print(f"{sc:20s} advW={base[sc]['aw']:.4f} honW={base[sc]['hw']:.5f} "
              f"mse={base[sc]['mse']:.4f} "
              f"FP={ {k: round(v, 2) for k, v in base[sc]['al'].items()} }")
        sys.stdout.flush()

    print()
    print("=" * 78)
    print("B1.1  every attack against every scorer")
    print("=" * 78)
    hdr = (f"{'attack':14s} {'scorer':20s} {'advW':>7} {'x-fair':>7} "
           f"{'mse/base':>9} {'eps':>7}  max-excess  detectors-that-fired")
    print(hdr)
    fair = CFG["k_adv"] / CFG["N"]
    for at in ATTACKS:
        for sc in scorers:
            r = agg(at, sc, seeds)
            ex = {k: r["al"][k] - base[sc]["al"].get(k, 0.0) for k in r["al"]}
            mx = max(ex.values())
            fired = sorted(k for k, v in ex.items() if v > 0.05)
            ratio = r["mse"] / base[sc]["mse"] if np.isfinite(r["mse"]) else np.inf
            out[f"{at}|{sc}"] = dict(
                aw=r["aw"], hw=r["hw"], mse=r["mse"], mse_ratio=float(ratio),
                eps=float(1 - r["hw"] / base[sc]["hw"]),
                excess={k: float(v) for k, v in ex.items()},
                max_excess=float(mx), fired=fired,
            )
            print(f"{at:14s} {sc:20s} {r['aw']:7.4f} {r['aw']/fair:7.2f} "
                  f"{ratio:9.2f} {1 - r['hw']/base[sc]['hw']:7.3f}  "
                  f"{mx:+.2f}       {','.join(fired) if fired else '-'}")
            sys.stdout.flush()
        print()

    # ---- faithfulness check: does each baseline work on its intended target?
    print("=" * 78)
    print("B1.2  faithfulness: each baseline against the AGR it was designed for")
    print("=" * 78)
    intended = {
        "fang-trim": ["Trimmed mean", "FedAvg (uniform)"],
        "fang-krum": ["Krum"],
        "shej-minmax": ["Krum", "Trimmed mean"],
        "shej-minsum": ["Krum", "Trimmed mean"],
        "freerider": ["FedAvg (uniform)"],
        "freerider-dis": ["FedAvg (uniform)"],
    }
    ok = True
    for at, scs in intended.items():
        for sc in scs:
            r = out[f"{at}|{sc}"]
            verdict = "damages" if r["mse_ratio"] > 1.5 else "no L1 damage"
            print(f"{at:14s} vs {sc:20s} mse/base={r['mse_ratio']:10.2f}  "
                  f"advW={r['aw']:.4f}  {verdict}")
    print()

    # ---- the headline: L3 harm vs L1 harm vs silence, under the L3 scorer
    print("=" * 78)
    print("B1.3  headline comparison under the inverse-variance decision maker")
    print("=" * 78)
    print(f"{'attack':16s} {'x-fair':>8} {'mse/base':>10} {'max-excess':>11}  "
          f"{'silent?':>8}  compliant?")
    compliant = {"none": "n/a", "tsp-var": "yes"}
    for at in ATTACKS:
        r = out[f"{at}|Inverse-variance"]
        silent = r["max_excess"] < 0.05
        print(f"{at:16s} {r['aw']/fair:8.2f} {r['mse_ratio']:10.2f} "
              f"{r['max_excess']:+11.2f}  {str(silent):>8}  "
              f"{compliant.get(at, 'no (fabricates)')}")

    # ---- fairness: the published attacks assume 20% malicious, we assume k=1.
    # Running them only at k=1 would be crippling them, so report both. The
    # question this settles is whether the published attacks become silent when
    # given their intended coalition, or merely more damaging.
    print()
    print("=" * 78)
    print("B1.4  malicious fraction: k=1 (matched to TSP-Var) vs k=6 (20%, the")
    print("      fraction Fang et al. and Shejwalkar & Houmansadr assume)")
    print("=" * 78)
    print(f"{'attack':16s} {'scorer':16s} {'k':>2} {'x-fair':>8} {'mse/base':>10} "
          f"{'max-excess':>11}  silent?")
    ksweep = {}
    pairs = [("fang-trim", "Trimmed mean"), ("fang-krum", "Krum"),
             ("shej-minmax", "Krum"), ("shej-minsum", "Krum"),
             ("freerider-dis", "Inverse-variance"),
             ("tsp-var", "Inverse-variance")]
    for at, sc in pairs:
        for k in [1, 6]:
            c = dict(CFG); c["k_adv"] = k
            b = agg("none", sc, seeds, c)
            r = agg(at, sc, seeds, c)
            ex = max(r["al"][kk] - b["al"].get(kk, 0.0) for kk in r["al"])
            ratio = r["mse"] / b["mse"] if np.isfinite(r["mse"]) else np.inf
            xf = r["aw"] / (k / c["N"])
            ksweep[f"{at}|{sc}|{k}"] = dict(
                x_fair=float(xf), mse_ratio=float(ratio), max_excess=float(ex))
            print(f"{at:16s} {sc:16s} {k:2d} {xf:8.2f} {ratio:10.2f} "
                  f"{ex:+11.2f}  {ex < 0.05}")
            sys.stdout.flush()
    out["_ksweep"] = ksweep

    # every published attack must damage its intended target at its own
    # assumed fraction -- otherwise the reimplementation is not faithful
    for at, sc in pairs[:4]:
        r = ksweep[f"{at}|{sc}|6"]
        assert r["mse_ratio"] > 1.5, (at, sc, r)

    # ---- B1.5  ACE's own threshold filter, measured rather than asserted
    # The l = 1 row is honest-fallback almost everywhere, and reporting only it
    # would understate ACE. That claim needs the rejection rate and the ratio
    # distribution, and the ratio is TRAJECTORY-DEPENDENT: rejecting sends the
    # round back to honest training, so the l = 1 run visits different w than
    # the l = inf run and its ||H dw||/||dw|| distribution differs. Both are
    # reported; the l = 1 one is what justifies retuning.
    print()
    print("=" * 78)
    print("B1.5  ACE's Eq. (3) filter: rejection rate and the l sweep")
    print("=" * 78)
    orig_craft = craft_ace

    def _spy(store):
        def f(hist, k, amplify, l=ACE_L):
            W, G = hist["w"], hist["g"]
            if len(W) >= 2 and len(G) >= 2:
                dW = [W[j] - W[j - 1] for j in range(1, len(W))][-ACE_M:]
                dG = [G[j] - G[j - 1] for j in range(1, len(G))][-ACE_M:]
                dw = W[-1] - W[-2]
                nd = float(np.linalg.norm(dw))
                if nd > 0:
                    store.append(float(np.linalg.norm(lbfgs_hv(dW, dG, dw)) / nd))
            return orig_craft(hist, k, amplify, l)
        return f

    filt = {}
    for label, at in (("l = 1", "ace"), ("l = inf", "ace-tuned")):
        store = []
        globals()["craft_ace"] = _spy(store)
        for sd in seeds:
            sim(at, "Inverse-variance", CFG, sd)
        r = np.array(store)
        filt[label] = dict(n=len(r), median=float(np.median(r)),
                           rej1=float((r > 1.0).mean()),
                           rej2=float((r > 2.0).mean()))
        print(f"{label:8s} trajectory: n={len(r):5d} rounds  "
              f"||H dw||/||dw|| median={np.median(r):.4f}  "
              f"rejected at l=1: {100 * (r > 1.0).mean():5.1f}%  "
              f"at l=2: {100 * (r > 2.0).mean():4.1f}%")
        sys.stdout.flush()
    globals()["craft_ace"] = orig_craft
    out["_ace_filter"] = filt
    # l = 1 must reject the large majority, or reporting only that row would be
    # a fair reading of ACE and the retuning would need no justification
    assert filt["l = 1"]["rej1"] > 0.9, filt["l = 1"]["rej1"]

    print()
    print(f"{'l':>6} {'advW':>8} {'x-fair':>8} {'mse/base':>10} {'max-excess':>11}")
    lsweep = {}
    saved_L = ACE_L
    for lv in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, np.inf):
        globals()["ACE_L"] = lv
        r = agg("ace", "Inverse-variance", seeds)
        ex = max(r["al"][kk] - base["Inverse-variance"]["al"][kk] for kk in r["al"])
        lsweep[str(lv)] = dict(aw=r["aw"], x_fair=r["aw"] / fair,
                               mse_ratio=r["mse"] / base["Inverse-variance"]["mse"],
                               max_excess=float(ex))
        print(f"{str(lv):>6} {r['aw']:8.4f} {r['aw'] / fair:8.2f} "
              f"{r['mse'] / base['Inverse-variance']['mse']:10.3f} {ex:+11.3f}")
        sys.stdout.flush()
    globals()["ACE_L"] = saved_L
    out["_ace_lsweep"] = lsweep
    # ace-tuned (l = inf) must equal the saturated end of the sweep, and the
    # saturation must NOT already be reached at l = 2 -- an earlier version of
    # this file claimed it was
    assert abs(lsweep["inf"]["x_fair"]
               - out["ace-tuned|Inverse-variance"]["aw"] / fair) < 0.01
    assert lsweep["2.0"]["x_fair"] < lsweep["inf"]["x_fair"] - 0.1, lsweep

    # ---- B1.6  which statistic fires on ACE, and why it does not fire on us
    # B1.1 reports WHICH detectors trip but not the value that trips them, and
    # sec:e2b explains the difference between the two attacks by those values.
    #
    # The first version of that explanation was wrong in a way only this table
    # shows. It said ACE trips the norm statistic "because a magnitude set by a
    # Hessian estimate rather than by local data lands at 0.38x the median
    # update norm". True, but it proves too much: TSP-Var lands at 0.40x and
    # stays quiet. A ratio is not what the detector reads. It reads a deviation
    # scored against the population MAD, so what separates the two is how far
    # the ratio wanders relative to its own level -- an L-BFGS extrapolation is
    # as noisy as its curvature estimate, while a variance-shaped update tracks
    # the honest scale round to round. Relative is the operative word and the
    # control row is why: with no attacker the ratio's absolute spread is the
    # largest of the three, because it sits at 1 rather than at 0.4.
    #
    # The cosine row is a genuine sign claim and it does separate them: the
    # extrapolation error is a direction no honest gradient has, so ACE points
    # LESS at the anchor than an honest client while TSP-Var points MORE.
    print()
    print("=" * 78)
    print("B1.6  the two statistics behind the ACE rows (inverse variance)")
    print("=" * 78)
    orig_det = detectors2

    def _det_spy(store):
        def f(V, adv_idx, state, cfg):
            nrm = np.linalg.norm(V, axis=1)
            med = np.median(nrm)
            mad = np.median(np.abs(nrm - med)) + 1e-12
            store["nrm"].append(float(nrm[adv_idx].mean() / med))
            store["z"].append(float(np.abs(nrm[adv_idx] - med).mean() / mad))
            ref = state["anchor_ref"]
            cos = (V @ ref) / (nrm * np.linalg.norm(ref) + 1e-12)
            store["adv"].append(float(cos[adv_idx].mean()))
            store["hon"].append(float(cos[np.arange(cfg["k_adv"],
                                                    cfg["N"])].mean()))
            return orig_det(V, adv_idx, state, cfg)
        return f

    print(f"{'attack':10s} {'norm/med':>9} {'sd':>7} {'sd/mean':>8} "
          f"{'norm fires':>11} {'cos adv':>9} {'cos honest':>11}")
    diag = {}
    for at in ("none", "ace-tuned", "tsp-var"):
        store = {k: [] for k in ("nrm", "z", "adv", "hon")}
        globals()["detectors2"] = _det_spy(store)
        try:
            for sd in seeds:
                sim(at, "Inverse-variance", CFG, sd)
        finally:
            globals()["detectors2"] = orig_det
        d = {k: float(np.mean(v)) for k, v in store.items()}
        d["nrm_sd"] = float(np.std(store["nrm"]))
        d["nrm_cv"] = d["nrm_sd"] / d["nrm"]
        d["fires"] = float(np.mean(np.array(store["z"]) > CFG["z_norm"]))
        diag[at] = d
        print(f"{at:10s} {d['nrm']:9.3f} {d['nrm_sd']:7.3f} {d['nrm_cv']:8.3f} "
              f"{d['fires']:11.3f} {d['adv']:+9.4f} {d['hon']:+11.4f}")
        sys.stdout.flush()
    out["_ace_diagnostics"] = diag
    ace, tsp_d, ctl = diag["ace-tuned"], diag["tsp-var"], diag["none"]
    # both attacks submit updates far under the median, so the LEVEL of the
    # ratio cannot be what the detector separates them by
    assert ace["nrm"] < 0.5 and tsp_d["nrm"] < 0.5, diag
    assert abs(ace["nrm"] - tsp_d["nrm"]) < 0.1, diag
    # the relative spread is: only ACE's is elevated above the no-attack run
    assert ace["nrm_cv"] > 2 * tsp_d["nrm_cv"], diag
    assert ace["nrm_cv"] > 2 * ctl["nrm_cv"], diag
    assert abs(tsp_d["nrm_cv"] - ctl["nrm_cv"]) < 0.1, diag
    # read against the control, ACE's norm alarm is real and ours is not
    assert ace["fires"] - ctl["fires"] > 0.05, diag
    assert tsp_d["fires"] - ctl["fires"] < 0.05, diag
    # and the directions fall on OPPOSITE sides of the honest baseline
    assert ace["adv"] < ace["hon"], diag
    assert tsp_d["adv"] > tsp_d["hon"], diag

    # ---- assertions that pin the paper's claims
    tsp = out["tsp-var|Inverse-variance"]
    assert tsp["max_excess"] < 0.06, tsp["max_excess"]
    assert tsp["mse_ratio"] < 1.0, tsp["mse_ratio"]
    assert tsp["aw"] / fair > 3.0, tsp["aw"] / fair
    # at least one published baseline must be caught by the same detector suite
    caught = [a for a in ATTACKS if a not in ("none", "tsp-var")
              and out[f"{a}|Inverse-variance"]["max_excess"] >= 0.05]
    assert caught, "no published baseline fired any detector -- suite is inert"
    print()
    print(f"detected by the same suite that stays quiet on TSP-Var: {caught}")

    json.dump(out, open("results/b1_baselines.json", "w"), indent=1)
    print("wrote results/b1_baselines.json")
    print("B1 OK")
