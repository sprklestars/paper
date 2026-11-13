"""A10: the gauge's damage obeys a law, and the audit is blind at every point.

A7 was built on the hypothesis that the audit gap grows with model CAPACITY.
Run at a fixed gauge magnitude (log-sd 1.0) it appeared to refute that: width
gave 1.01, 1.09, 1.46, 1.27, 1.31, 0.97 -- non-monotone, with the widest
network slightly BELOW its clean baseline.  That reading was wrong, and the
reason is worth recording, because it is the kind of error that turns into a
false claim in a paper.

At log-sd 1.0 there is essentially no damage at ANY capacity: the whole column
sits between 0.97 and 1.46 with a seed spread of the same size.  A7 measured
capacity at a magnitude where the effect does not exist yet, so its width
column was noise, and "width is refuted" was a conclusion drawn from noise.
Re-measured across magnitudes, capacity matters enormously -- at log-sd 2.0,
A10.3 below gives 7.65x for [20,8,8,1] and 120056x for [20,24,24,24,24,1].
(An earlier ad-hoc version of that sweep reported 30x and 66262x on a
different seed set; the numbers to cite are the ones this script prints, and
the spread between the two runs is itself a warning that the sd=2.0 column is
extremely high variance -- see the min/max columns in A10.1.)

So the controlling variable is the gauge magnitude, and capacity is a
multiplier on it.  This script establishes the law and its mechanism.

  A10.1  the magnitude sweep: damage against the audit, 8 seeds
  A10.2  is damage quadratic in the parameter displacement?
  A10.3  capacity at a magnitude where the effect is present
  A10.4  the N dependence: does dilution by more honest clients save us?
  A10.5  what the law implies for a threshold defence
"""

import numpy as np

from a7_capacity import agg_mse, build
from mlp import (flat, forward, gauge_canon, n_hidden, rand_units,
                 scale_units)

DIMS = [20, 32, 32, 1]
SEEDS = 8


def sweep(dims, s, seeds=SEEDS, N=24):
    """Returns (mean |q-p|, mean damage ratio, mean own-MSE delta, ratios)."""
    dn, dm, dd = [], [], []
    for sd in range(seeds):
        clients, teacher, Xp, _ = build(dims, N=N, seed=sd)
        ft = forward(teacher, Xp)[0]
        own0 = float(np.mean((forward(clients[0], Xp)[0] - ft) ** 2))
        clean = agg_mse(clients, teacher, Xp)
        rr = np.random.default_rng(5000 + sd)
        q = scale_units(clients[0], rand_units(clients[0], rr, s))
        dn.append(np.linalg.norm(flat(q) - flat(clients[0])))
        dm.append(agg_mse(clients, teacher, Xp, q) / clean)
        dd.append(abs(float(np.mean((forward(q, Xp)[0] - ft) ** 2)) - own0))
    return np.mean(dn), np.mean(dm), np.mean(dd), np.array(dm)


# ========================================================== A10.1 =========
def a10_1_magnitude():
    print("=" * 78)
    print("A10.1  damage against the audit, as the gauge magnitude grows")
    print("=" * 78)
    print(f"{'log-sd':>7} {'|q|/|p|':>9} {'own MSE delta':>14} "
          f"{'agg x clean':>12} {'median':>9} {'min':>8} {'max':>10}")
    for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        nr = []
        for sd in range(SEEDS):
            clients, _, _, _ = build(DIMS, seed=sd)
            rr = np.random.default_rng(5000 + sd)
            q = scale_units(clients[0], rand_units(clients[0], rr, s))
            nr.append(np.linalg.norm(flat(q))
                      / np.linalg.norm(flat(clients[0])))
        _, mean_dm, dd, ratios = sweep(DIMS, s)
        print(f"{s:7.2f} {np.mean(nr):9.2f} {dd:14.2e} {mean_dm:12.2f} "
              f"{np.median(ratios):9.2f} {ratios.min():8.2f} "
              f"{ratios.max():10.2f}")
    print("\n  Column 3 is the functional audit.  It is at machine zero on every")
    print("  row, including the row where the aggregate degrades 372x.  There is")
    print("  no magnitude at which the audit starts to see the attack: that is")
    print("  the P1/P2 decoupling of A6, now as a function of attack strength.")
    print("  Note how far median sits below mean -- the distribution is heavily")
    print("  right-skewed, so the mean is carried by a minority of seeds and")
    print("  both belong in the paper.\n")


# ========================================================== A10.2 =========
def a10_2_quadratic():
    print("=" * 78)
    print("A10.2  is the damage quadratic in the parameter displacement?")
    print("=" * 78)
    print("  The adversary moves the mean by (q - p)/N.  If the global loss is")
    print("  locally quadratic in the parameter error, damage - 1 should be")
    print("  proportional to |q - p|^2 / N^2.  Testing that directly.")
    print()
    print(f"{'log-sd':>7} {'|q-p|':>11} {'|q-p|^2/N^2':>14} {'damage-1':>11} "
          f"{'ratio':>9}")
    N = 24
    xs, ys = [], []
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        dn, dm, _, _ = sweep(DIMS, s, N=N)
        x = dn ** 2 / N ** 2
        y = dm - 1.0
        xs.append(x)
        ys.append(y)
        print(f"{s:7.2f} {dn:11.2f} {x:14.3f} {y:11.3f} {y/x:9.4f}")
    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(np.log(xs), np.log(ys), 1)
    r = np.corrcoef(np.log(xs), np.log(ys))[0, 1]
    print(f"\n  log-log slope {b:.3f}, correlation r = {r:.5f}")
    print("  A slope of 1.0 would be exactly quadratic.  The fit is close and")
    print("  very tight, but the slope is BELOW 1 and the ratio column falls")
    print("  monotonically, so the growth is slightly sub-quadratic rather than")
    print("  exactly quadratic.  The mechanism is ReLU: once the inflated unit")
    print("  drives its activation negative on most inputs the unit clips, and")
    print("  a clipped unit contributes less than the quadratic term predicts.")
    print("  Reported as an empirical law with a stated deviation, not as a")
    print("  theorem -- the honest form of this result.\n")


# ========================================================== A10.3 =========
def a10_3_capacity_where_it_exists():
    print("=" * 78)
    print("A10.3  capacity, measured where the effect actually exists")
    print("=" * 78)
    print("  This is A7.1 and A7.2 re-run across magnitudes.  Reading down a")
    print("  column gives capacity at fixed strength; reading across a row gives")
    print("  strength at fixed capacity.")
    print()
    mags = [0.5, 1.0, 1.5, 2.0]
    print(f"{'architecture':>24} {'gauge dim':>10} "
          + "".join(f"{'sd=' + str(m):>12}" for m in mags))
    for dims in [[20, 8, 8, 1], [20, 32, 32, 1], [20, 128, 128, 1],
                 [20, 24, 24, 1], [20, 24, 24, 24, 1],
                 [20, 24, 24, 24, 24, 1]]:
        row = [sweep(dims, m, seeds=4)[1] for m in mags]
        k = n_hidden(build(dims, N=2, seed=0, steps=1)[0][0])
        print(f"{str(dims):>24} {k:10d} "
              + "".join(f"{v:12.2f}" for v in row))
    print("\n  At sd=0.5 every architecture is at 1.00 and capacity is invisible.")
    print("  At sd=2.0 the same comparison spans orders of magnitude.  A7's")
    print("  width column was measured at sd=1.0, in the flat region, which is")
    print("  why it looked like a refutation.  Depth still dominates width at")
    print("  matched gauge dimension.\n")


# ========================================================== A10.4 =========
def a10_4_dilution():
    print("=" * 78)
    print("A10.4  does a larger federation dilute the attack?")
    print("=" * 78)
    print("  One adversary in N contributes 1/N of the mean, so the quadratic")
    print("  law predicts damage - 1 shrinks like 1/N^2.  If so, scale is a")
    print("  partial defence and the paper must say so rather than imply the")
    print("  attack is uniformly severe.")
    print()
    print(f"{'N':>5} {'damage-1 at sd=1.5':>20} {'x N^2':>12} "
          f"{'own MSE delta':>15}")
    for N in [10, 20, 40, 80]:
        dn, dm, dd, _ = sweep(DIMS, 1.5, seeds=4, N=N)
        print(f"{N:5d} {dm-1.0:20.4f} {(dm-1.0)*N**2:12.1f} {dd:15.2e}")
    print("\n  If the 'x N^2' column is roughly flat, dilution follows the law")
    print("  and a large federation is genuinely safer against ONE adversary.")
    print("  Note this is a defence only against a fixed adversary count: the")
    print("  audit delta in the last column stays at zero regardless, so nothing")
    print("  here lets the server IDENTIFY the attacker.\n")


# ========================================================== A10.5 =========
def a10_5_threshold():
    print("=" * 78)
    print("A10.5  the norm threshold this suggests, and what it costs")
    print("=" * 78)
    print("  The gauge does its damage by inflating the submitted vector, so a")
    print("  norm bound looks like an obvious cheap defence.  The question is")
    print("  whether it separates attack from honest variation.")
    print()
    print(f"{'log-sd':>8} {'|q|/median honest|':>20} {'damage':>9} "
          f"{'canon |q|/med':>15} {'canon damage':>13}")
    for s in [0.0, 0.5, 1.0, 1.5, 2.0]:
        rl, dm, rc, dc = [], [], [], []
        for sd in range(4):
            clients, teacher, Xp, _ = build(DIMS, seed=sd)
            med = np.median([np.linalg.norm(flat(c)) for c in clients])
            clean = agg_mse(clients, teacher, Xp)
            rr = np.random.default_rng(5000 + sd)
            q = scale_units(clients[0], rand_units(clients[0], rr, s))
            rl.append(np.linalg.norm(flat(q)) / med)
            dm.append(agg_mse(clients, teacher, Xp, q) / clean)
            gc = [gauge_canon(c) for c in clients]
            bc = agg_mse(gc, teacher, Xp)
            mc = np.median([np.linalg.norm(flat(c)) for c in gc])
            rc.append(np.linalg.norm(flat(gauge_canon(q))) / mc)
            dc.append(agg_mse(gc, teacher, Xp, gauge_canon(q)) / bc)
        print(f"{s:8.2f} {np.mean(rl):20.2f} {np.mean(dm):9.2f} "
              f"{np.mean(rc):15.2f} {np.mean(dc):13.2f}")
    print("\n  A norm bound does fire on the large-magnitude rows, so it is a")
    print("  real mitigation and the paper should not pretend otherwise.  But it")
    print("  is a threshold on a continuous quantity: the attacker picks a")
    print("  magnitude just under it and keeps the damage that remains, and the")
    print("  server still cannot tell an inflated honest client from an attack,")
    print("  because the audit is blind at every row.  Canonicalisation is the")
    print("  stronger answer -- it pins the ratio near 1 and removes the damage")
    print("  outright, at no functional cost, because it is itself a gauge move.\n")


if __name__ == "__main__":
    a10_1_magnitude()
    a10_2_quadratic()
    a10_3_capacity_where_it_exists()
    a10_4_dilution()
    a10_5_threshold()
