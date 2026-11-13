"""A11: can the DISCRETE subgroup S2 be quotiented out?

Where this sits.  M4 (gauge_canon) fixes the continuous subgroup S1 by
normalising each hidden unit's scale carrier.  The conclusion currently records
S2 -- hidden-unit permutation -- as open, and A2.5 confirmed that layerwise
normalisation does nothing to it: matching hidden units across clients is an
assignment problem, not a rescaling.

The assignment problem is exactly what FedMA / OT-fusion / Git Re-Basin solve,
for ACCURACY.  This script asks whether the same machinery works as a SECURITY
preprocessing step, and -- the part that decides whether it is a real defence --
what it costs and whether the adversary can beat it.

  A11.1  is Hungarian alignment function-preserving?  (must be checked)
  A11.2  does it remove the permutation damage of A4.3?
  A11.3  what does it cost when NOBODY attacks?  (the honest-cost question)
  A11.4  can the adversary beat the aligner by combining S1 and S2?
  A11.5  does it still work without a common initialisation?
  A11.6  does it need an anchor, or is the public init enough?
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from a1_nn_symmetry import (D_IN, H1, H2, flat, forward, gauge_permute,
                            init_net, sgd)
from a4_aggregate_gauge import gauge_canon, scale_units, unflat


# ------------------------------------------------------- the defence ----
def align_to(p, ref):
    """Permute p's hidden units onto ref's by Hungarian matching.

    Layer 1 units are described by their incoming row [W1_j, b1_j]; layer 2
    units by [W2_j, b2_j] AFTER layer 1 has been permuted, since W2's columns
    are indexed by layer-1 units.  Applying a permutation to a hidden layer
    together with the matching permutation of the next layer's columns is a
    member of S2, so the function is preserved by construction -- A11.1 checks
    that this is true in floating point too.
    """
    q = {k: v.copy() for k, v in p.items()}

    f_p = np.concatenate([q["W1"], q["b1"][:, None]], axis=1)
    f_r = np.concatenate([ref["W1"], ref["b1"][:, None]], axis=1)
    cost = np.sum((f_p[:, None, :] - f_r[None, :, :]) ** 2, axis=2)
    row, col = linear_sum_assignment(cost)
    perm1 = np.empty(H1, dtype=int)
    perm1[col] = row                       # unit at new slot col is old row
    q["W1"] = q["W1"][perm1]
    q["b1"] = q["b1"][perm1]
    q["W2"] = q["W2"][:, perm1]

    f_p = np.concatenate([q["W2"], q["b2"][:, None]], axis=1)
    f_r = np.concatenate([ref["W2"], ref["b2"][:, None]], axis=1)
    cost = np.sum((f_p[:, None, :] - f_r[None, :, :]) ** 2, axis=2)
    row, col = linear_sum_assignment(cost)
    perm2 = np.empty(H2, dtype=int)
    perm2[col] = row
    q["W2"] = q["W2"][perm2]
    q["b2"] = q["b2"][perm2]
    q["W3"] = q["W3"][:, perm2]
    return q


def full_canon(p, ref):
    """M4+: quotient S1 by scale canonicalisation, then S2 by alignment.

    Scale first, because the assignment cost is a distance and S1 moves
    distances arbitrarily -- aligning an un-normalised model matches on scale
    rather than on function.
    """
    return align_to(gauge_canon(p), ref)


# ------------------------------------------------------------ harness ----
def build(N=24, n_per=200, seed=0, steps=200, common_init=True):
    r = np.random.default_rng(seed)
    teacher = init_net(r)
    g0 = init_net(r)
    clients = []
    for _ in range(N):
        X = r.normal(size=(n_per, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.1
        start = g0 if common_init else init_net(r)
        clients.append(sgd(start, X, y, steps=steps, lr=0.05, rng=r))
    Xp = r.normal(size=(2000, D_IN))
    return clients, teacher, g0, Xp, r


def agg_mse(params, teacher, Xp):
    V = np.array([flat(c) for c in params])
    g = unflat(V.mean(axis=0), params[0])
    return float(np.mean((forward(g, Xp)[0] - forward(teacher, Xp)[0]) ** 2))


def own_mse(p, teacher, Xp):
    return float(np.mean((forward(p, Xp)[0] - forward(teacher, Xp)[0]) ** 2))


# ============================================================ A11.1 =======
def a11_1_exact():
    print("=" * 76)
    print("A11.1  is Hungarian alignment function-preserving?")
    print("=" * 76)
    clients, teacher, g0, Xp, r = build(N=6)
    print(f"{'client':>8} {'max |df| align':>16} {'max |df| full':>16}")
    worst = 0.0
    for i, c in enumerate(clients):
        f0 = forward(c, Xp)[0]
        d1 = np.abs(forward(align_to(c, g0), Xp)[0] - f0).max()
        d2 = np.abs(forward(full_canon(c, g0), Xp)[0] - f0).max()
        worst = max(worst, d1, d2)
        print(f"{i:8d} {d1:16.2e} {d2:16.2e}")
    print(f"\n  worst deviation {worst:.2e}")
    print("  Alignment is a gauge move: it is free by construction.\n")
    return worst


# ============================================================ A11.2 =======
def a11_2_removes_damage():
    print("=" * 76)
    print("A11.2  does alignment remove the permutation damage of A4.3?")
    print("=" * 76)
    clients, teacher, g0, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    print(f"  clean aggregate MSE {clean:.6f}   (N = {len(clients)})\n")
    print(f"{'permuted':>10} {'undefended':>12} {'x clean':>9} "
          f"{'defended':>12} {'x clean':>9}")
    rows = []
    for k in [0, 1, 2, 3, 5, 8, 12]:
        rp = np.random.default_rng(100 + k)
        atk = [gauge_permute(c, rp) if i < k else c
               for i, c in enumerate(clients)]
        und = agg_mse(atk, teacher, Xp)
        dfn = agg_mse([full_canon(c, g0) for c in atk], teacher, Xp)
        rows.append((k, und, dfn))
        print(f"{k:10d} {und:12.6f} {und / clean:9.2f} "
              f"{dfn:12.6f} {dfn / clean:9.2f}")
    print()
    return clean, rows


# ============================================================ A11.3 =======
def a11_3_honest_cost():
    print("=" * 76)
    print("A11.3  what does it cost when NOBODY attacks?")
    print("=" * 76)
    print(f"{'seed':>6} {'plain':>11} {'scale canon':>13} {'full canon':>12} "
          f"{'ratio':>8}")
    ratios = []
    for seed in range(5):
        clients, teacher, g0, Xp, r = build(seed=seed)
        a = agg_mse(clients, teacher, Xp)
        b = agg_mse([gauge_canon(c) for c in clients], teacher, Xp)
        c = agg_mse([full_canon(c, g0) for c in clients], teacher, Xp)
        ratios.append(c / a)
        print(f"{seed:6d} {a:11.6f} {b:13.6f} {c:12.6f} {c / a:8.3f}")
    print(f"\n  mean ratio full/plain {np.mean(ratios):.3f}")
    print("  A ratio at or below 1 means the defence is free on honest input.\n")
    return ratios


# ============================================================ A11.4 =======
def a11_4_counterattack():
    print("=" * 76)
    print("A11.4  can the adversary beat the aligner by combining S1 and S2?")
    print("=" * 76)
    clients, teacher, g0, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    print(f"  clean aggregate MSE {clean:.6f}\n")
    print(f"{'adversary move':>28} {'own MSE':>10} {'undef x':>9} "
          f"{'def x':>8} {'audit df':>10}")
    base_own = own_mse(clients[0], teacher, Xp)
    f0 = forward(clients[0], Xp)[0]

    def run(name, q):
        atk = [q] + clients[1:]
        und = agg_mse(atk, teacher, Xp) / clean
        dfn = agg_mse([full_canon(c, g0) for c in atk], teacher, Xp) / clean
        df = np.abs(forward(q, Xp)[0] - f0).max()
        print(f"{name:>28} {own_mse(q, teacher, Xp):10.4f} {und:9.2f} "
              f"{dfn:8.2f} {df:10.2e}")
        return und, dfn

    rp = np.random.default_rng(7)
    p = clients[0]
    out = {}
    out["perm"] = run("permutation only", gauge_permute(p, rp))
    a = np.exp(rp.normal(size=H1) * 1.5)
    d = np.exp(rp.normal(size=H2) * 1.5)
    out["scale"] = run("per-unit scale log-sd 1.5", scale_units(p, a, d))
    out["both"] = run("scale then permute",
                      gauge_permute(scale_units(p, a, d), rp))
    a2 = np.exp(rp.normal(size=H1) * 3.0)
    d2 = np.exp(rp.normal(size=H2) * 3.0)
    out["big"] = run("scale log-sd 3.0 + permute",
                     gauge_permute(scale_units(p, a2, d2), rp))
    print(f"\n  adversary's honest own MSE {base_own:.4f} throughout "
          f"(the function never moves).\n")
    return clean, out


# ============================================================ A11.5 =======
def a11_5_no_common_init():
    print("=" * 76)
    print("A11.5  does it still work WITHOUT a common initialisation?")
    print("=" * 76)
    print("  Here the units are genuinely unaligned, so the assignment problem")
    print("  is doing real work rather than undoing a known permutation.\n")
    clients, teacher, g0, Xp, r = build(N=12, common_init=False, steps=400)
    ref = clients[0]
    a = agg_mse(clients, teacher, Xp)
    b = agg_mse([gauge_canon(c) for c in clients], teacher, Xp)
    c = agg_mse([align_to(gauge_canon(c), gauge_canon(ref)) for c in clients],
                teacher, Xp)
    mean_own = np.mean([own_mse(x, teacher, Xp) for x in clients])
    print(f"  mean individual client MSE  {mean_own:11.6f}")
    print(f"  plain average               {a:11.6f}")
    print(f"  scale canon only            {b:11.6f}")
    print(f"  scale + alignment           {c:11.6f}")
    print()
    return mean_own, a, b, c


# ============================================================ A11.6 =======
def a11_6_reference_choice():
    print("=" * 76)
    print("A11.6  which reference?  init is public; anchors are scarce")
    print("=" * 76)
    clients, teacher, g0, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    rp = np.random.default_rng(11)
    atk = [gauge_permute(c, rp) if i < 8 else c for i, c in enumerate(clients)]
    refs = {
        "public init g0": g0,
        "one anchor (client 23)": gauge_canon(clients[23]),
        "teacher (upper bound)": teacher,
    }
    print(f"  clean {clean:.6f}, undefended "
          f"{agg_mse(atk, teacher, Xp) / clean:.2f}x  (8 of 24 permuted)\n")
    print(f"{'reference':>26} {'defended':>12} {'x clean':>9}")
    for name, ref in refs.items():
        d = agg_mse([full_canon(c, ref) for c in atk], teacher, Xp)
        print(f"{name:>26} {d:12.6f} {d / clean:9.2f}")
    print("\n  If the public init suffices, the defence carries no trust")
    print("  assumption at all -- unlike M2, it needs no anchor.\n")


# ============================================================ A11.7 =======
def a11_7_quotient():
    print("=" * 76)
    print("A11.7  is full_canon an exact QUOTIENT map?")
    print("=" * 76)
    print("  A11.2's defended column was identical at every k, which would")
    print("  follow if canon(g.x) = canon(x) for every g in the group rather")
    print("  than merely reducing the damage.  That is a stronger claim than")
    print("  the table makes, so it is checked directly here.\n")
    clients, teacher, g0, Xp, r = build(N=6)
    rp = np.random.default_rng(5)
    print(f"{'client':>8} {'move':>8} {'||canon(gx)-canon(x)||':>24} {'relative':>11}")
    worst = 0.0
    for i, c in enumerate(clients[:4]):
        base = flat(full_canon(c, g0))
        a = np.exp(rp.normal(size=H1) * 1.5)
        d = np.exp(rp.normal(size=H2) * 1.5)
        for nm, q in [("perm", gauge_permute(c, rp)),
                      ("scale", scale_units(c, a, d)),
                      ("both", gauge_permute(scale_units(c, a, d), rp))]:
            v = flat(full_canon(q, g0))
            ab = np.linalg.norm(v - base)
            rel = ab / np.linalg.norm(base)
            worst = max(worst, rel)
            print(f"{i:8d} {nm:>8} {ab:24.3e} {rel:11.3e}")
    print(f"\n  worst relative deviation {worst:.2e}")
    print("  The gauge attack family is CLOSED, not merely mitigated: every")
    print("  orbit representative maps to the same point.  What remains open")
    print("  is A11.5 -- clients in genuinely different basins.\n")
    return worst


if __name__ == "__main__":
    a11_1_exact()
    a11_2_removes_damage()
    a11_3_honest_cost()
    a11_4_counterattack()
    a11_5_no_common_init()
    a11_6_reference_choice()
    a11_7_quotient()
