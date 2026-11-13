"""A15: how much gauge damage survives the detector suite the paper defines?

A6.2 reports the two views separately -- the functional audit at machine zero,
the consensus-distance z-score at 9.01 already by log-sd 0.10 -- and sec:negatives
N3 states that gauge damage is not stealthy against a warmed-up distance-based
aggregator.  What neither reports is the joint quantity: the largest aggregate
degradation attainable while EVERY detector in the sec:setup suite stays within
its own false-positive rate.  Constraint (C4) is stated globally in sec:threat,
so that number is the one (C4) actually licenses for TSP-Gauge, and it belongs in
the paper next to the unconstrained magnitudes.

The suite is the one sec:setup defines, at the thresholds e2_silence.py uses:

  D1  robust norm check      |q| more than 3.5 MADs from the honest median norm
  D2  consensus distance     distance-to-mean, same 3.5-MAD test
  D3  Krum exclusion         adversary is the argmax Krum score (would be cut)
  D4  negative cosine        cos(q, mean) < 0, as in FLTrust

Each is evaluated against its own no-attack behaviour on the same seed, so the
comparison is excess over false-positive rate and not an absolute rate, exactly
as sec:setup specifies.  A15.1 sweeps magnitude and reports where the suite first
fires; A15.2 bisects for the largest silent damage; A15.3 reports the seed spread
of the scale-only attack at log-sd 3.0, and A15.4 the seed spread of the
scale-then-permute composition that A11.4 quotes as 9648x -- a different attack in
a different harness, so the two distributions are reported separately.
"""

import numpy as np

from a1_nn_symmetry import H1, H2, flat, forward, gauge_permute, krum_score
from a4_aggregate_gauge import agg_mse, build, scale_units
import a11_permutation_canon as a11

MAD_K = 3.5          # e2_silence.py: z_norm = z_dist = 3.5
SEEDS = range(8)


def mad(x):
    med = np.median(x)
    return med, float(np.median(np.abs(x - med)))


def detectors(clients, q):
    """Four public statistics, each as a boolean 'fires on client 0'.

    Thresholds are calibrated on the honest population of the same round, which
    is what a deployment can do without a trusted reference.
    """
    V = np.array([flat(c) for c in clients])
    V[0] = flat(q)

    norms = np.linalg.norm(V, axis=1)
    med, m = mad(norms[1:])
    d1 = abs(norms[0] - med) > MAD_K * (m + 1e-30)

    mean = V.mean(axis=0)
    dist = np.linalg.norm(V - mean, axis=1)
    med, m = mad(dist[1:])
    d2 = abs(dist[0] - med) > MAD_K * (m + 1e-30)

    ks = np.array([krum_score(V, i) for i in range(len(V))])
    d3 = bool(np.argmax(ks) == 0)

    c0 = float(V[0] @ mean / (np.linalg.norm(V[0]) * np.linalg.norm(mean) + 1e-30))
    d4 = c0 < 0.0

    return np.array([d1, d2, d3, d4])


def gauge(clients, seed, s):
    rr = np.random.default_rng(900 + seed)
    return scale_units(clients[0], np.exp(rr.normal(size=H1) * s),
                       np.exp(rr.normal(size=H2) * s))


_CACHE = {}


def _world(seed):
    """build() retrains N clients, so cache it: the bisection calls run() ~40
    times per seed and the world must be identical across those calls anyway."""
    if seed not in _CACHE:
        clients, teacher, Xp, _ = build(seed=seed)
        _CACHE[seed] = (clients, teacher, Xp, agg_mse(clients, teacher, Xp),
                        detectors(clients, clients[0]))
    return _CACHE[seed]


def run(seed, s):
    clients, teacher, Xp, clean, base = _world(seed)
    q = gauge(clients, seed, s)
    fired = detectors(clients, q)
    excess = fired.astype(int) - base.astype(int)
    return agg_mse(clients, teacher, Xp, q) / clean, fired, base, excess


# =========================================================== A15.1 ========
def a15_1_sweep():
    print("=" * 78)
    print("A15.1  where the four-detector suite first fires")
    print("=" * 78)
    print("\nfired = any detector above its own no-attack verdict, 8 seeds.")
    print(f"\n{'log-sd':>7} {'damage (mean)':>14} {'median':>9} "
          f"{'seeds silent':>13}  {'first detector to fire'}")
    print("-" * 78)
    names = ["norm", "dist", "krum", "cos"]
    for s in [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        dmg, silent, who = [], 0, set()
        for sd in SEEDS:
            d, fired, base, excess = run(sd, s)
            dmg.append(d)
            if not excess.any():
                silent += 1
            else:
                who |= {names[i] for i in np.where(excess > 0)[0]}
        print(f"{s:>7.2f} {np.mean(dmg):>14.4f} {np.median(dmg):>9.4f} "
              f"{silent:>9}/{len(list(SEEDS))}  "
              f"{','.join(sorted(who)) if who else '-'}")
    print("\nThe consensus-distance and Krum statistics fire long before the")
    print("magnitude that produces the headline damage: this is an auditing")
    print("result, not a stealth result, and A15.2 gives the silent bound.")


# =========================================================== A15.2 ========
def a15_2_silent_bound():
    print("\n" + "=" * 78)
    print("A15.2  largest damage with the whole suite at its false-positive rate")
    print("=" * 78)

    def silent(sd, s):
        _, _, _, excess = run(sd, s)
        return not excess.any()

    print(f"\n{'seed':>5} {'max silent log-sd':>18} {'damage there':>13}")
    print("-" * 40)
    got = []
    for sd in SEEDS:
        lo, hi = 0.0, 2.0
        if not silent(sd, lo):
            print(f"{sd:>5} {'(fires at s=0)':>18} {'-':>13}")
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if silent(sd, mid):
                lo = mid
            else:
                hi = mid
        d, _, _, _ = run(sd, lo)
        got.append((lo, d))
        print(f"{sd:>5} {lo:>18.4f} {d:>13.4f}")

    ds = np.array([g[1] for g in got])
    print(f"\nsilent damage over {len(got)} seeds: "
          f"mean {ds.mean():.4f}x  median {np.median(ds):.4f}x  "
          f"max {ds.max():.4f}x")
    print("\nThe honest reading: subject to (C4) against the full suite, the")
    print("gauge attack buys almost nothing.  What it buys is not damage but")
    print("BLINDNESS -- the functional audit cannot see any of it, at any")
    print("magnitude (A14), which is why prop:auditgap is an admission result")
    print("and why M4 is a canonicalisation and not a detector.")
    return ds


# =========================================================== A15.3 ========
def a15_3_combined_seeds():
    print("\n" + "=" * 78)
    print("A15.3  the SCALE-ONLY attack at log-sd 3.0, across seeds")
    print("=" * 78)
    print("\neq:maglaw is fitted over log-sd 0.5-2.0, so 3.0 is outside the")
    print("fitted band and the distribution is heavy-right-tailed; the spread")
    print("belongs in the paper wherever a single draw is quoted.")
    print(f"\n{'seed':>5} {'damage':>14} {'suite fires':>12}")
    print("-" * 34)
    out = []
    for sd in SEEDS:
        d, fired, base, excess = run(sd, 3.0)
        out.append(d)
        print(f"{sd:>5} {d:>14.2f} {str(bool(excess.any())):>12}")
    a = np.array(out)
    print(f"\nmean {a.mean():.2f}x  median {np.median(a):.2f}x  "
          f"min {a.min():.2f}x  max {a.max():.2f}x")
    print("\nThe mean is carried by the tail, so quoting a single draw as the")
    print("attack magnitude is not defensible; quote the median and give the")
    print("maximum as the maximum.")
    return a


# =========================================================== A15.4 ========
def a15_4_composed_seeds(nseed=8):
    """The 9648x figure itself: scale log-sd 3.0 THEN permute, in A11's harness.

    A15.1-3 run in a4's harness (N=30, scale only).  The number the abstract and
    sec:defense quote comes from A11.4, which composes S1 with S2 at N=24 and
    reports one RNG draw.  R2-m6 asks for its distribution, so it has to be this
    attack in this harness, not the scale-only proxy above.
    """
    print("\n" + "=" * 78)
    print("A15.4  the COMPOSED attack (scale log-sd 3.0 + permute) across seeds")
    print("=" * 78)
    print("\nA11.4's harness (N=24, common init), which is where 9648.12x comes")
    print("from; that run is seed 0 with RNG draws taken after three earlier")
    print("moves, so it is one sample of this distribution and not its centre.")
    print(f"\n{'seed':>5} {'undefended':>14} {'under M4+':>11} {'own MSE':>10}")
    print("-" * 44)
    out, dfn = [], []
    for sd in range(nseed):
        clients, teacher, g0, Xp, _ = a11.build(seed=sd)
        clean = a11.agg_mse(clients, teacher, Xp)
        rp = np.random.default_rng(7 + sd)
        a = np.exp(rp.normal(size=H1) * 3.0)
        d = np.exp(rp.normal(size=H2) * 3.0)
        q = gauge_permute(scale_units(clients[0], a, d), rp)
        atk = [q] + clients[1:]
        und = a11.agg_mse(atk, teacher, Xp) / clean
        dfd = a11.agg_mse([a11.full_canon(c, g0) for c in atk],
                          teacher, Xp) / clean
        out.append(und)
        dfn.append(dfd)
        print(f"{sd:>5} {und:>14.2f} {dfd:>11.4f} "
              f"{a11.own_mse(q, teacher, Xp):>10.4f}")
    u, v = np.array(out), np.array(dfn)
    print(f"\nundefended: mean {u.mean():.1f}x  median {np.median(u):.1f}x  "
          f"min {u.min():.1f}x  max {u.max():.1f}x")
    print(f"under M4+ : mean {v.mean():.4f}x  median {np.median(v):.4f}x  "
          f"max {v.max():.4f}x")
    print("\nThe defended column is the claim that matters, and it is flat")
    print("across seeds; the undefended column spans orders of magnitude, so")
    print("the paper should quote its median and range, not one draw.")
    return u, v


if __name__ == "__main__":
    a15_1_sweep()
    ds = a15_2_silent_bound()
    a = a15_3_combined_seeds()
    u, v = a15_4_composed_seeds()
    print("\n" + "=" * 78)
    print("A15 conclusion: (C4) against the full four-detector suite caps")
    print(f"TSP-Gauge damage at a median {np.median(ds):.2f}x, against the")
    print("unconstrained magnitudes the paper quotes.  Those magnitudes are")
    print("the reach of the GROUP, which is what prop:auditgap is about; the")
    print("silent figure is the reach of an adversary facing distance-based")
    print("monitoring.  Both belong in the paper, labelled as what they are.")
    print(f"The composed attack the abstract quotes has median "
          f"{np.median(u):.0f}x undefended")
    print(f"(range {u.min():.0f}-{u.max():.0f}x over {len(u)} seeds) and "
          f"median {np.median(v):.2f}x under M4+.")
    print("=" * 78)
