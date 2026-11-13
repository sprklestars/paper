"""
B2: published matchers as the alignment step of M4.

Section V justifies M4's second step by observing that hidden-unit permutation
is "the same assignment problem that model fusion solves for accuracy", and
cites FedMA, OT-Fusion and Git Re-Basin.  E6 then measures only OUR matcher.
This script closes the defence-side counterpart of the gap B1 closed on the
attack side: the matchers those papers actually use, dropped into the same
pipeline, measured on the three properties the paper claims for M4.

    free            max |df| over a probe set: is the map in the gauge group?
    exact quotient  ||C(g.x) - C(x)|| / ||C(x)|| over S1, S2 and S1 then S2
    effective       does it return the composed attack to 1.00x clean?

Matchers:

  ours          weight matching on the incoming row [W_j, b_j], Hungarian,
                with gauge_canon applied FIRST.  This is a11's full_canon.
  act           activation matching: describe unit j by its activation vector
                over a probe batch and match those.  This is Git Re-Basin's
                second matcher (Ainsworth et al. 2023, "activation matching")
                and OT-Fusion's activation-based ground metric (Singh &
                Jaggi 2020, Sec. 3).  Scale step first, as in ours.
  act-noscale   the same matcher as the fusion papers actually run it: on the
                submitted weights, with no scale canonicalisation.
  sinkhorn-eps  OT-Fusion in its actual form: an entropically regularised
                transport plan rather than a hard permutation, applied as
                W <- M W.  Measured at two regularisation strengths because
                the answer depends on eps, which is the point of B2.3.
  ours-noscale  our matcher with the scale step applied AFTER alignment
                instead of before.  Section V argues this ordering matters
                because the assignment cost is a distance and S1 moves
                distances arbitrarily.

Two federation configurations are reported throughout, because the answer
differs between them and only one of them tests the matcher:

  common init    every client starts the round from the same global model, as
                 in real FL.  The units are already nearly aligned, so the
                 assignment problem is undoing a known permutation.
  cold init      every client initialises independently: the units are
                 genuinely unaligned and
                 the matcher is doing real work.  This is a11_5's setting.

What was predicted before running, and what happened.  Two of the four
predictions were wrong and are recorded here rather than quietly dropped:

  PREDICTED  act is not S1-invariant, because scaling unit j by a_j multiplies
             its activation vector by a_j.
  HELD, but only for act-noscale.  With gauge_canon applied first the
             sensitivity is gone, which is what the scale step is for; without
             it the quotient property fails at 5.7e-01 even under a common
             init, the one configuration where every other matcher is exact.

  PREDICTED  sinkhorn is not function-preserving at all, because a doubly
             stochastic matrix that is not a permutation does not satisfy
             M^T M = I.
  RIGHT, BUT NOT FOR THE REASON GIVEN.  The property is not lost as a function
             of the algorithm but as a function of eps, and the crossover is
             not a constant of the algorithm either: at eps = 0.01 of the mean
             cost the plan converges to the Hungarian permutation to 3e-18
             under a common init, yet the SAME eps moves the function by 2.9
             under a cold init.  B2.3 reports why -- the normalised runner-up
             gap in the assignment cost is 7.3x smaller when the units are
             genuinely unaligned, so a setting that is safely hard for one
             federation is soft for another.  The server cannot read that gap
             off the protocol.

Run:  python3 -u b2_matcher_baselines.py
"""

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from a1_nn_symmetry import (D_IN, H1, H2, flat, forward, gauge_permute,
                            init_net, sgd)
from a4_aggregate_gauge import gauge_canon, scale_units, unflat
from a11_permutation_canon import agg_mse, align_to, build, full_canon, own_mse

# Sinkhorn iteration budget, and the two regularisation strengths carried into
# the main table.  eps is in units of the mean cost, so it is scale free.
# 0.01 is the value we first chose as "well inside the regime OT-Fusion
# reports"; B2.3 shows it degenerates to hard matching, so 0.3 is carried
# alongside it as a plan that is genuinely soft.
SINK_ITERS = 2000
SINK_EPS_HARD, SINK_EPS_SOFT = 0.01, 0.3
EPS_GRID = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

# The composed attack of E6: per-unit log-sd for the scale part.
ATK_LOGSD = 3.0


# ---------------------------------------------------------------- matchers

def _feat(p, layer, Xp, mode):
    """Descriptor for the units of one hidden layer, under one matching rule."""
    if mode == "weight":
        return np.concatenate([p[f"W{layer}"], p[f"b{layer}"][:, None]], axis=1)
    # activation matching: unit j is described by its post-ReLU output.
    return forward(p, Xp)[1][2 * layer - 1].T


def _cost(fp, fr):
    return np.sum((fp[:, None, :] - fr[None, :, :]) ** 2, axis=2)


def _hard(cost):
    """Hungarian assignment as a permutation matrix M[slot, unit]."""
    row, col = linear_sum_assignment(cost)
    M = np.zeros(cost.shape)
    M[col, row] = 1.0
    return M


def _sinkhorn(cost, eps):
    """Entropic OT plan, in the same M[slot, unit] convention as _hard.

    Uniform marginals on both sides, so n*T is doubly stochastic.  Returned
    scaled by n so its rows sum to one, matching _hard.
    """
    n = cost.shape[0]
    K = np.exp(-cost / (eps * cost.mean() + 1e-30))
    u = np.ones(n) / n
    v = np.ones(n) / n
    for _ in range(SINK_ITERS):
        u = (1.0 / n) / (K @ v + 1e-300)
        v = (1.0 / n) / (K.T @ u + 1e-300)
    T = u[:, None] * K * v[None, :]          # T[unit, slot]
    return (T * n).T                          # M[slot, unit], rows sum to 1


def align_generic(p, ref, Xp, mode, solver):
    """Align p's hidden units onto ref's, layer by layer.

    Layer 2 is matched after layer 1 has been applied, because W2's columns
    are indexed by layer-1 units -- the same ordering align_to uses.  For a
    permutation this is a member of S2 and preserves the function; for a
    general doubly stochastic M it need not, which is what B2.1 measures.
    """
    q = {k: v.copy() for k, v in p.items()}
    M1 = solver(_cost(_feat(q, 1, Xp, mode), _feat(ref, 1, Xp, mode)))
    q["W1"] = M1 @ q["W1"]
    q["b1"] = M1 @ q["b1"]
    q["W2"] = q["W2"] @ M1.T

    M2 = solver(_cost(_feat(q, 2, Xp, mode), _feat(ref, 2, Xp, mode)))
    q["W2"] = M2 @ q["W2"]
    q["b2"] = M2 @ q["b2"]
    q["W3"] = q["W3"] @ M2.T
    return q


def c_ours(p, ref, Xp):
    return full_canon(p, ref)


def c_act(p, ref, Xp):
    """Activation matching with the scale step first, as in ours."""
    return align_generic(gauge_canon(p), gauge_canon(ref), Xp, "act", _hard)


def c_act_noscale(p, ref, Xp):
    """Activation matching on the submitted weights, as the fusion papers run it."""
    return gauge_canon(align_generic(p, gauge_canon(ref), Xp, "act", _hard))


def _c_sink(eps):
    def f(p, ref, Xp):
        return align_generic(gauge_canon(p), gauge_canon(ref), Xp, "weight",
                             lambda c: _sinkhorn(c, eps))
    return f


def c_noscale(p, ref, Xp):
    """Our matcher with the ordering reversed: align first, then canonicalise."""
    return gauge_canon(align_to(p, ref))


MATCHERS = {
    "ours": c_ours,
    "act": c_act,
    "act-noscale": c_act_noscale,
    f"sinkhorn-{SINK_EPS_HARD}": _c_sink(SINK_EPS_HARD),
    f"sinkhorn-{SINK_EPS_SOFT}": _c_sink(SINK_EPS_SOFT),
    "ours-noscale": c_noscale,
}

CONFIGS = [("common init", True), ("cold init", False)]


# ------------------------------------------------------------------- B2.1

def b2_1_free():
    """Is each matcher a gauge move?  Measured, not assumed."""
    print("=" * 78)
    print("B2.1  free?  max |f(C(x)) - f(x)| over a 2000-point probe set")
    print("=" * 78)
    out = {}
    for cname, ci in CONFIGS:
        clients, teacher, g0, Xp, r = build(N=6, common_init=ci)
        print(f"  {cname}")
        print(f"{'matcher':>16} {'max |df|':>12} {'verdict':>24}")
        for name, C in MATCHERS.items():
            worst = 0.0
            for c in clients:
                f0 = forward(c, Xp)[0]
                worst = max(worst, float(np.abs(forward(C(c, g0, Xp), Xp)[0]
                                                - f0).max()))
            out[(cname, name)] = worst
            verdict = "gauge move" if worst < 1e-10 else "CHANGES THE FUNCTION"
            print(f"{name:>16} {worst:12.2e} {verdict:>24}")
        print()
    return out


# ------------------------------------------------------------------- B2.2

def b2_2_quotient():
    """Is each matcher an exact quotient of the group it is supposed to kill?"""
    print("=" * 78)
    print("B2.2  exact quotient?  ||C(g.x) - C(x)|| / ||C(x)||")
    print("=" * 78)
    print("  A quotient map returns the same representative for every member")
    print("  of the orbit, so this ratio must be at machine precision.\n")
    out = {}
    for cname, ci in CONFIGS:
        clients, teacher, g0, Xp, r = build(N=6, common_init=ci)
        rp = np.random.default_rng(5)
        a = np.exp(rp.normal(size=H1) * 1.5)
        d = np.exp(rp.normal(size=H2) * 1.5)
        moves = {
            "S1 scale": lambda x: scale_units(x, a, d),
            "S2 perm": lambda x: gauge_permute(x, np.random.default_rng(5)),
            "S1 then S2": lambda x: gauge_permute(scale_units(x, a, d),
                                                  np.random.default_rng(5)),
        }
        print(f"  {cname}")
        print(f"{'matcher':>16} {'S1 scale':>11} {'S2 perm':>11} "
              f"{'S1 then S2':>11} {'verdict':>16}")
        for name, C in MATCHERS.items():
            cells, worst = [], 0.0
            for mname, mv in moves.items():
                rel = 0.0
                for c in clients:
                    base = flat(C(c, g0, Xp))
                    moved = flat(C(mv(c), g0, Xp))
                    rel = max(rel, float(np.linalg.norm(moved - base)
                                         / (np.linalg.norm(base) + 1e-30)))
                cells.append(rel)
                worst = max(worst, rel)
            out[(cname, name)] = worst
            print(f"{name:>16} {cells[0]:11.2e} {cells[1]:11.2e} "
                  f"{cells[2]:11.2e} "
                  f"{'exact' if worst < 1e-10 else 'NOT a quotient':>16}")
        print()
    return out


# ------------------------------------------------------------------- B2.3

def _runner_up_gap(clients, g0, Xp):
    """Mean normalised gap between the best and second-best slot per unit.

    This is what decides whether a given eps is hard or soft: the Sinkhorn
    kernel exp(-cost/(eps*mean)) is one-hot per row only when the runner-up
    is far behind relative to eps.
    """
    gaps = []
    for c in clients:
        cost = _cost(_feat(gauge_canon(c), 1, Xp, "weight"),
                     _feat(gauge_canon(g0), 1, Xp, "weight"))
        s = np.sort(cost, axis=1)
        gaps.append(float(((s[:, 1] - s[:, 0]) / (cost.mean() + 1e-30)).mean()))
    return float(np.mean(gaps))


def b2_3_sinkhorn_eps():
    """Where does the OT plan stop being a permutation, and what does it cost?

    This replaces the binary claim we first asserted.  The soft plan is a
    gauge move exactly in the limit where it is not soft -- and where that
    limit falls is a property of the federation, not of the algorithm.
    """
    print("=" * 78)
    print("B2.3  OT-Fusion's soft plan as a function of regularisation")
    print("=" * 78)
    print("  A doubly stochastic M with M^T M != I is not a member of S2, so")
    print("  applying it to hidden units is not a gauge move.  At small eps")
    print("  the plan converges to the Hungarian permutation and the")
    print("  objection disappears; the question is where the crossover is.\n")
    clients, teacher, g0, Xp, r = build(N=12)
    clean = agg_mse(clients, teacher, Xp)
    p, ref = gauge_canon(clients[0]), gauge_canon(g0)
    cost = _cost(_feat(p, 1, Xp, "weight"), _feat(ref, 1, Xp, "weight"))
    P = _hard(cost)
    print(f"  layer-1 cost matrix: mean {cost.mean():.3f}, "
          f"mean within-row spread {(cost.max(1) - cost.min(1)).mean():.3f}\n")
    print(f"{'eps':>7} {'max row':>9} {'|M-P|':>10} {'|MtM-I|':>10} "
          f"{'max |df|':>10} {'agg x clean':>12}")
    rows = []
    for eps in EPS_GRID:
        M = _sinkhorn(cost, eps)
        C = _c_sink(eps)
        dfm = 0.0
        for c in clients[:6]:
            f0 = forward(c, Xp)[0]
            dfm = max(dfm, float(np.abs(forward(C(c, g0, Xp), Xp)[0]
                                        - f0).max()))
        agg = agg_mse([C(c, g0, Xp) for c in clients], teacher, Xp) / clean
        rows.append(dict(eps=eps, max_row=float(M.max(1).mean()),
                         dist_perm=float(np.abs(M - P).max()),
                         orth=float(np.abs(M.T @ M - np.eye(H1)).max()),
                         df=dfm, agg=float(agg)))
        print(f"{eps:7.3f} {M.max(1).mean():9.4f} "
              f"{np.abs(M - P).max():10.2e} "
              f"{np.abs(M.T @ M - np.eye(H1)).max():10.2e} "
              f"{dfm:10.2e} {agg:12.2f}")
    print("\n  The plan is a permutation, and therefore safe, only in the")
    print("  low-regularisation limit where it coincides with ours.")

    # Where that limit falls is not a constant of the algorithm.  B2.1 shows
    # eps = 0.01 is hard under a common init and soft under a cold one; the
    # cause is the separation of the assignment cost, measured here.
    print("\n  why the same eps is safe in one federation and not another:")
    gaps = {}
    for cname, ci in CONFIGS:
        cl, _t, gg, Xq, _r = build(N=6, common_init=ci)
        gaps[cname] = _runner_up_gap(cl, gg, Xq)
        print(f"    {cname:>12}  normalised runner-up gap "
              f"{gaps[cname]:.4f}")
    print("  A narrower gap needs a smaller eps to stay hard, and the server")
    print("  cannot read the gap off the protocol.  Our matcher has no such")
    print("  parameter to set.\n")
    return rows, gaps


# ------------------------------------------------------------------- B2.4

def b2_4_effective():
    """Does each matcher actually undo the attack it is deployed against?"""
    print("=" * 78)
    print("B2.4  effective?  the permutation sweep and the composed attack")
    print("=" * 78)
    clients, teacher, g0, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    print(f"  clean aggregate MSE {clean:.6f}  (N = {len(clients)})\n")

    # (a) the permutation sweep of tab:align
    print("  (a) permuting clients, x clean")
    hdr = f"{'permuting':>10} {'undef':>8}"
    for name in MATCHERS:
        hdr += f" {name:>14}"
    print(hdr)
    sweep = {name: [] for name in MATCHERS}
    und_row = []
    for k in [0, 1, 2, 3, 5, 8, 12]:
        rp = np.random.default_rng(100 + k)
        atk = [gauge_permute(c, rp) if i < k else c
               for i, c in enumerate(clients)]
        und = agg_mse(atk, teacher, Xp) / clean
        und_row.append(und)
        line = f"{k:10d} {und:8.2f}"
        for name, C in MATCHERS.items():
            v = agg_mse([C(c, g0, Xp) for c in atk], teacher, Xp) / clean
            sweep[name].append(v)
            line += f" {v:14.2f}"
        print(line)

    # (b) the composed attack.  The draw order matches a11_4_counterattack
    # exactly -- a permutation and one earlier scale pair come off the same
    # generator first -- so that this reproduces fig:defense's 9648x rather
    # than a different draw at the same log-sd.
    print("\n  (b) one client plays S1 at log-sd "
          f"{ATK_LOGSD} then S2, x clean")
    rp = np.random.default_rng(7)
    p = clients[0]
    _ = gauge_permute(p, rp)
    a = np.exp(rp.normal(size=H1) * 1.5)
    d = np.exp(rp.normal(size=H2) * 1.5)
    _ = gauge_permute(scale_units(p, a, d), rp)
    a2 = np.exp(rp.normal(size=H1) * ATK_LOGSD)
    d2 = np.exp(rp.normal(size=H2) * ATK_LOGSD)
    atk = [gauge_permute(scale_units(p, a2, d2), rp)] + clients[1:]
    und = agg_mse(atk, teacher, Xp) / clean
    print(f"    undefended {und:.2f}x   adversary's own MSE "
          f"{own_mse(atk[0], teacher, Xp):.4f} "
          f"(honest {own_mse(clients[0], teacher, Xp):.4f})")
    comb = {}
    for name, C in MATCHERS.items():
        comb[name] = agg_mse([C(c, g0, Xp) for c in atk], teacher, Xp) / clean
        print(f"    {name:>16} {comb[name]:10.2f}x")
    print()
    return clean, und_row, sweep, und, comb


# ------------------------------------------------------------------- B2.5

def b2_5_honest_cost():
    """What does each matcher cost when nobody attacks?"""
    print("=" * 78)
    print("B2.5  honest cost: aggregate MSE ratio on attack-free federations")
    print("=" * 78)
    hdr = f"{'seed':>6} {'plain':>11}"
    for name in MATCHERS:
        hdr += f" {name:>14}"
    print(hdr)
    cost = {name: [] for name in MATCHERS}
    for seed in range(5):
        clients, teacher, g0, Xp, r = build(seed=seed)
        base = agg_mse(clients, teacher, Xp)
        line = f"{seed:6d} {base:11.6f}"
        for name, C in MATCHERS.items():
            v = agg_mse([C(c, g0, Xp) for c in clients], teacher, Xp) / base
            cost[name].append(v)
            line += f" {v:14.3f}"
        print(line)
    print(f"\n{'':6} {'mean':>11}", end="")
    for name in MATCHERS:
        print(f" {np.mean(cost[name]):14.3f}", end="")
    print("\n")
    return {name: [float(v) for v in vs] for name, vs in cost.items()}


# -------------------------------------------------------------------- main

if __name__ == "__main__":
    free = b2_1_free()
    quot = b2_2_quotient()
    eps_rows, gaps = b2_3_sinkhorn_eps()
    clean, und_row, sweep, und_comb, comb = b2_4_effective()
    cost = b2_5_honest_cost()

    print("=" * 78)
    print("B2.6  summary: the three properties section V claims for M4")
    print("=" * 78)
    print(f"{'matcher':>16} {'free':>9} {'quot(warm)':>11} {'quot(cold)':>11} "
          f"{'composed':>10} {'cost':>7}")
    for name in MATCHERS:
        fr = max(free[("common init", name)], free[("cold init", name)])
        print(f"{name:>16} {fr:9.1e} {quot[('common init', name)]:11.1e} "
              f"{quot[('cold init', name)]:11.1e} {comb[name]:9.2f}x "
              f"{np.mean(cost[name]):7.3f}")
    print()
    print("  Reading: the four matchers that end in a hard permutation are")
    print("  free, and the two soft ones are not -- sinkhorn-0.01 included,")
    print("  under the cold init where the cost gap is narrow enough for the")
    print("  plan to stay diffuse.  Among the free ones the quotient property")
    print("  separates them on exactly one axis: whether the scale carrier is")
    print("  canonicalised BEFORE the assignment cost is computed.  Both that")
    print("  do (ours, act) are exact under both initialisations; both that do")
    print("  not (act-noscale, ours-noscale) fail, act-noscale even under a")
    print("  common init.  This is the ordering argument of section V,")
    print("  measured -- and it is a property of the pipeline, not of the")
    print("  assignment rule, since activation matching inherits it intact.\n")

    # ---- assertions pinning what the paper will say
    # ours: gauge move under both configs, exact quotient under both,
    # kills the composed attack, ~free
    for cname, _ in CONFIGS:
        assert free[(cname, "ours")] < 1e-10, (cname, free[(cname, "ours")])
        assert quot[(cname, "ours")] < 1e-10, (cname, quot[(cname, "ours")])
    assert abs(comb["ours"] - 1.0) < 0.02, comb["ours"]
    assert 9600 < und_comb < 9700, und_comb     # fig:defense's 9648x

    # activation matching WITH the scale step is equivalent to ours on all
    # three properties; without it, it is not a quotient even under the
    # favourable configuration
    for cname, _ in CONFIGS:
        assert quot[(cname, "act")] < 1e-10, (cname, quot[(cname, "act")])
        assert quot[(cname, "act-noscale")] > 1e-3, \
            (cname, quot[(cname, "act-noscale")])

    # the ordering claim of section V: aligning before canonicalising is still
    # a gauge move but is not a quotient -- visible only where the assignment
    # does real work, i.e. without a common initialisation
    assert free[("cold init", "ours-noscale")] < 1e-10
    assert quot[("cold init", "ours-noscale")] > 1e-3, \
        quot[("cold init", "ours-noscale")]
    assert quot[("common init", "ours-noscale")] < 1e-10, \
        quot[("common init", "ours-noscale")]

    # the OT plan is a permutation only in the low-regularisation limit
    hard_row = [r for r in eps_rows if r["eps"] == SINK_EPS_HARD][0]
    soft_row = [r for r in eps_rows if r["eps"] == SINK_EPS_SOFT][0]
    assert hard_row["dist_perm"] < 1e-6, hard_row       # degenerates to ours
    assert soft_row["dist_perm"] > 0.1, soft_row        # genuinely soft
    assert soft_row["orth"] > 0.1, soft_row             # not in S2
    assert soft_row["df"] > 1e-6, soft_row              # moves the function
    assert quot[("common init", f"sinkhorn-{SINK_EPS_HARD}")] < 1e-10
    for cname, _ in CONFIGS:
        assert free[(cname, f"sinkhorn-{SINK_EPS_SOFT}")] > 1e-6, cname

    # and where that limit falls is a property of the federation, not of the
    # algorithm: the same eps is a gauge move under a common init and not
    # under a cold one, because the assignment cost is less separated
    assert free[("common init", f"sinkhorn-{SINK_EPS_HARD}")] < 1e-10
    assert free[("cold init", f"sinkhorn-{SINK_EPS_HARD}")] > 1e-6
    assert gaps["common init"] > 5 * gaps["cold init"], gaps

    res = dict(
        free={f"{c}|{n}": v for (c, n), v in free.items()},
        quotient={f"{c}|{n}": v for (c, n), v in quot.items()},
        eps_sweep=eps_rows, runner_up_gap=gaps, clean=clean,
        undefended_sweep=[float(v) for v in und_row],
        sweep={k: [float(x) for x in v] for k, v in sweep.items()},
        composed_undefended=float(und_comb),
        composed={k: float(v) for k, v in comb.items()},
        honest_cost=cost)
    json.dump(res, open("results/b2_matchers.json", "w"), indent=1)
    print("wrote results/b2_matchers.json")
    print("B2 OK")
