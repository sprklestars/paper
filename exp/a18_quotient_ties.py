"""A18: the hypothesis the exact-quotient claim needs, and whether it binds.

sec:e6 and sec:m4 state fullcanon(g.p) = fullcanon(p) for every g in Gamma, and
A11.7 measures it at <= 2.0e-16.  R1 pointed out what that measurement cannot
see: minimum-cost assignment has a unique solution only when the cost matrix has
no ties, and on a deliberately tied descriptor the identity fails outright.  That
is correct, and the word "exact" needs the hypothesis attached.

This script establishes three things, in the order that decides how to state it:

  A18.1  the counterexample, so the hypothesis is not hand-waved: two hidden
         units with identical incoming rows and different outgoing columns, the
         permutation swapping them, and the quotient violation that results.
  A18.2  whether ties occur in trained clients -- the margin, measured as the
         minimum descriptor distance and the maximum cosine, before and after
         canon, because canon maps parallel rows to identical ones and so is the
         one step that could manufacture a tie.
  A18.3  whether an adversary can manufacture one within Gamma.  Scaling and
         permuting cannot make two distinct rows equal unless they are already
         parallel, so the tie requires an out-of-Gamma move, and the only one
         available is the dead-unit split already reported in sec:negatives.
"""

import numpy as np

import a11_permutation_canon as a11
from a1_nn_symmetry import flat, forward, init_net
from a4_aggregate_gauge import gauge_canon


def descriptor(p):
    return np.concatenate([p["W1"], p["b1"][:, None]], axis=1)


def swap01(p):
    """The element of S2 that exchanges hidden units 0 and 1 of layer 1."""
    q = {k: v.copy() for k, v in p.items()}
    perm = np.arange(q["W1"].shape[0])
    perm[0], perm[1] = 1, 0
    q["W1"] = q["W1"][perm]
    q["b1"] = q["b1"][perm]
    q["W2"] = q["W2"][:, perm]
    return q


def violation(p, ref):
    a = flat(a11.full_canon(p, ref))
    b = flat(a11.full_canon(swap01(p), ref))
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


# =========================================================== A18.1 ========
def a18_1_counterexample():
    print("=" * 78)
    print("A18.1  the identity fails on an exactly tied descriptor")
    print("=" * 78)
    r = np.random.default_rng(0)
    p = {k: v.copy() for k, v in init_net(r).items()}
    ref = init_net(np.random.default_rng(1))
    X = r.normal(size=(500, p["W1"].shape[1]))

    print(f"\nuntied model, g = swap(0,1):  violation "
          f"{violation(p, ref):.3e}")

    p["W1"][1] = p["W1"][0].copy()          # tie the incoming rows exactly
    p["b1"][1] = p["b1"][0]
    p["W2"][:, 1] = p["W2"][:, 0] * 3.0 + 1.0   # keep the units distinguishable
    q = swap01(p)
    df = float(np.abs(forward(p, X)[0] - forward(q, X)[0]).max())
    v = violation(p, ref)
    print(f"tied model,   g = swap(0,1):  violation {v:.3e}")
    print(f"  g is still function-preserving: max|df| {df:.3e}")
    print(f"  and canon(p), canon(g.p) still compute that same function")

    cp, cq = a11.full_canon(p, ref), a11.full_canon(swap01(p), ref)
    dfc = float(np.abs(forward(cp, X)[0] - forward(cq, X)[0]).max())
    print(f"  max|df| between the two canonical forms: {dfc:.3e}")
    assert v > 0.1, v
    print("\nSo the map is not a quotient without a no-ties hypothesis. Both")
    print("canonical forms compute the same function, so the failure is in the")
    print("REPRESENTATIVE the solver picks, not in function preservation:")
    print("with a tie the arg min is a set and the solver returns one of its")
    print("elements. A18.2 asks whether that set is ever larger than a point.")
    return v


# =========================================================== A18.2 ========
def a18_2_margin():
    print("\n" + "=" * 78)
    print("A18.2  how far trained clients are from a tie")
    print("=" * 78)
    print("\ncanon divides each incoming row by its scale carrier, so two")
    print("PARALLEL rows become identical: cosine 1 after canon is the tie")
    print("condition, and it is the quantity to report, not raw distance.\n")
    clients, teacher, g0, Xp, r = a11.build(seed=0)
    print(f"{'client':>7} {'min dist raw':>13} {'min dist canon':>15} "
          f"{'max cos':>9}")
    print("-" * 48)
    cos_all, gap_all = [], []
    for i, c in enumerate(clients):
        for tag, m in (("raw", c), ("canon", gauge_canon(c))):
            F = descriptor(m)
            D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            if tag == "raw":
                d_raw = D.min()
            else:
                d_can = D.min()
                n = np.linalg.norm(F, axis=1, keepdims=True)
                C = (F / n) @ (F / n).T
                np.fill_diagonal(C, -np.inf)
                cmax = C.max()
        cos_all.append(cmax)
        gap_all.append(d_can)
        if i < 8:
            print(f"{i:>7} {d_raw:>13.4f} {d_can:>15.4f} {cmax:>9.4f}")
    cos_all = np.array(cos_all)
    print(f"\nover {len(clients)} trained clients: max cosine "
          f"{cos_all.max():.4f}, min canon gap {min(gap_all):.4f}")
    print(f"exact ties: {int((cos_all >= 1.0).sum())}")
    assert cos_all.max() < 0.99
    print("A tie needs cosine 1. The margin is not marginal, and it is not an")
    print("accident: distinct units of a trained network solve distinct parts")
    print("of the problem, which is why the descriptor separates them.")
    return cos_all.max()


# =========================================================== A18.3 ========
def a18_3_reachability():
    print("\n" + "=" * 78)
    print("A18.3  can an adversary manufacture a tie inside Gamma?")
    print("=" * 78)
    print("\nGamma is generated by positive per-unit scaling and permutation.")
    print("Scaling multiplies row j by a_j, so it can equalise two rows only if")
    print("they are already parallel; permutation reorders rows without")
    print("changing the multiset. So no element of Gamma creates a tie that")
    print("was not there. Check that, then check the out-of-Gamma route.\n")
    clients, teacher, g0, Xp, r = a11.build(seed=0)
    p = clients[0]
    rr = np.random.default_rng(3)
    H1 = p["W1"].shape[0]
    H2 = p["W2"].shape[0]
    from a4_aggregate_gauge import scale_units

    print(f"{'gauge magnitude':>16} {'max cos after canon':>21}")
    print("-" * 40)
    for s in [0.0, 0.5, 1.5, 3.0]:
        q = scale_units(p, np.exp(rr.normal(size=H1) * s),
                        np.exp(rr.normal(size=H2) * s))
        F = descriptor(gauge_canon(q))
        n = np.linalg.norm(F, axis=1, keepdims=True)
        C = (F / n) @ (F / n).T
        np.fill_diagonal(C, -np.inf)
        print(f"{s:>16.2f} {C.max():>21.4f}")

    print("\nUnchanged, as the argument says: canon undoes the scaling before")
    print("the descriptor is formed. The reachable route is a unit SPLIT, which")
    print("copies one unit's incoming row onto a dead unit and divides the")
    print("outgoing column, preserving the function exactly while creating the")
    print("tie. Splitting is not in Gamma, and it is the same escape")
    print("sec:negatives already reports for dead units.\n")

    clean = a11.agg_mse(clients, teacher, Xp)
    base = {k: v.copy() for k, v in p.items()}
    base["W2"][:, 1] = 0.0                       # a dead unit
    f0 = forward(base, Xp)[0]
    cb = a11.agg_mse([base] + clients[1:], teacher, Xp)
    print(f"{'split t':>9} {'own max|df|':>13} {'undef x':>9} {'M4+ x':>9} "
          f"{'quot viol':>11}")
    print("-" * 56)
    for t in [0.5, 2.0, 10.0, 100.0, 1e4]:
        q = {k: v.copy() for k, v in base.items()}
        cj = q["W2"][:, 0].copy()
        q["W1"][1] = q["W1"][0].copy()
        q["b1"][1] = q["b1"][0]
        q["W2"][:, 0] = t * cj
        q["W2"][:, 1] = (1.0 - t) * cj
        df = float(np.abs(forward(q, Xp)[0] - f0).max())
        und = a11.agg_mse([q] + clients[1:], teacher, Xp) / cb
        dfn = a11.agg_mse([a11.full_canon(c, g0) for c in [q] + clients[1:]],
                          teacher, Xp) / cb
        print(f"{t:>9.1f} {df:>13.3e} {und:>9.4f} {dfn:>9.4f} "
              f"{violation(q, g0):>11.3e}")

    print("\nSo the tie is exploitable, at the same cost of entry as the")
    print("dead-unit case: the adversary needs a unit outside Gamma's reach,")
    print("and having one it can damage the aggregate under M4+. What the")
    print("paper should therefore claim is a quotient of Gamma under a no-ties")
    print("hypothesis on the reference descriptor, with the margin of A18.2")
    print("as the evidence that the hypothesis holds on trained models, and")
    print("the tie listed with the dead unit as the boundary of Gamma.")


if __name__ == "__main__":
    a18_1_counterexample()
    a18_2_margin()
    a18_3_reachability()
    print("\n" + "=" * 78)
    print("A18 conclusion: 'exact quotient of Gamma' is right, and needs the")
    print("hypothesis that the reference descriptor has no exact ties. Trained")
    print("clients clear it with a wide margin, no element of Gamma can create")
    print("a tie, and the moves that can are already outside Gamma and already")
    print("recorded as the boundary. State the hypothesis; do not weaken the")
    print("claim.")
    print("=" * 78)
