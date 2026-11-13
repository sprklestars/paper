"""A2: gauge alignment -- can an adversary hide a POISONED function by moving
along the neural network's symmetry orbit?

A1 showed the symmetry is exact and that distance-based trust scores are
violently sensitive to it.  But A1's adversary made itself LESS trusted, which
is self-harm, not an attack.

The attack is the reverse.  Fix a genuinely poisoned model -- one that a robust
aggregator SHOULD reject -- and then search the gauge orbit for the
representative that minimises the trust score.  The function is invariant along
the orbit, so the poison is fully preserved.  If a rank-1 representative exists,
then:

    the defence is measuring the CHOICE OF COORDINATES, not the model.

and that is a defect with no analogue in the linear / Kalman setting, because a
linear model has no such orbit.

  A2.1  baseline: how visible is the poisoned model with no gauge search?
  A2.2  gauge descent: minimise Krum score over (c1, c2) + permutation
  A2.3  does the poison survive?  (must be checked, not assumed)
  A2.4  sweep poison strength -- where does the defence stop working?
  A2.5  is a norm-clipping / normalisation defence enough?
"""

import numpy as np

from a1_nn_symmetry import (D_IN, H1, H2, cos_to_mean, flat, forward,
                            gauge_permute, gauge_scale, init_net, krum_score,
                            l2_to_mean, make_federation, sgd)


def unflat(v, ref):
    out, o = {}, 0
    for k in sorted(ref):
        sz = ref[k].size
        out[k] = v[o:o + sz].reshape(ref[k].shape)
        o += sz
    return out


def poisoned_client(g0, teacher, rng, strength, n_per=200):
    """A real backdoor: trained by honest SGD on data whose labels are shifted
    on a trigger subspace.  No hand-placed weights, no gradient fabrication."""
    X = rng.normal(size=(n_per, D_IN))
    y = forward(teacher, X)[0] + rng.normal(size=n_per) * 0.1
    trig = X[:, 0] > 1.0                      # trigger region
    y = y + strength * trig                   # backdoor: shift output on trigger
    return sgd(g0, X, y, steps=200, lr=0.05, rng=rng)


def build(seed=3, strength=5.0):
    """Federation of honest clients + one backdoored client at index 0."""
    r = np.random.default_rng(seed)
    teacher = init_net(r)
    g0 = init_net(r)
    honest = []
    for _ in range(29):
        X = r.normal(size=(200, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=200) * 0.1
        honest.append(sgd(g0, X, y, steps=200, lr=0.05, rng=r))
    adv = poisoned_client(g0, teacher, r, strength)
    Xt = r.normal(size=(3000, D_IN))
    return [adv] + honest, teacher, Xt, r


def backdoor_effect(p, teacher, Xt):
    """Mean output shift on the trigger region relative to the teacher."""
    trig = Xt[:, 0] > 1.0
    return float(np.mean(forward(p, Xt[trig])[0] - forward(teacher, Xt[trig])[0]))


def all_ranks(V, i=0):
    N = len(V)
    k = np.array([krum_score(V, j) for j in range(N)])
    l = np.array([l2_to_mean(V, j) for j in range(N)])
    c = np.array([cos_to_mean(V, j) for j in range(N)])
    return (int(np.argsort(np.argsort(k))[i]) + 1,
            int(np.argsort(np.argsort(l))[i]) + 1,
            int(np.argsort(np.argsort(-c))[i]) + 1)


# =========================================================== A2.1 =========
def a2_1_baseline():
    print("=" * 74)
    print("A2.1  visibility of the backdoored client with NO gauge search")
    print("=" * 74)
    print(f"{'poison':>8} {'Krum rank':>11} {'L2 rank':>9} {'cos rank':>10} {'backdoor':>10}")
    for s in [0.0, 1.0, 2.0, 5.0, 10.0]:
        clients, teacher, Xt, _ = build(strength=s)
        V = np.array([flat(c) for c in clients])
        rk, rl, rc = all_ranks(V)
        print(f"{s:8.1f} {rk:11d} {rl:9d} {rc:10d} "
              f"{backdoor_effect(clients[0], teacher, Xt):10.3f}")
    print("\n  (rank 1 = most trusted of 30.)  A strong backdoor is visible:")
    print("  the poisoned client drifts to the tail of the distance ranking.\n")


# =========================================================== A2.2 =========
def gauge_search(adv, others, rng, iters=400):
    """Coordinate descent over the gauge orbit to minimise Krum score.

    Search space is exactly the symmetry group: (c1, c2) rescalings and hidden
    unit permutations.  Nothing outside the orbit is touched, so the function
    is invariant BY CONSTRUCTION -- A2.3 verifies this empirically anyway.
    """
    V = np.array([flat(o) for o in others])

    def score(p):
        W = np.vstack([flat(p)[None, :], V])
        return krum_score(W, 0)

    best_p, best_s = adv, score(adv)
    c1, c2 = 1.0, 1.0
    for it in range(iters):
        step = 0.3 * (0.99 ** it)
        cand = []
        for d1 in (-step, 0.0, step):
            for d2 in (-step, 0.0, step):
                if d1 == 0.0 and d2 == 0.0:
                    continue
                cand.append((c1 * np.exp(d1), c2 * np.exp(d2)))
        improved = False
        for nc1, nc2 in cand:
            p = gauge_scale(adv, nc1, nc2)
            s = score(p)
            if s < best_s:
                best_s, best_p, c1, c2, improved = s, p, nc1, nc2, True
        if it % 40 == 0:                       # occasional permutation probe
            for _ in range(20):
                p = gauge_permute(best_p, rng)
                s = score(p)
                if s < best_s:
                    best_s, best_p = s, p
                    improved = True
        if not improved and step < 1e-4:
            break
    return best_p, best_s, (c1, c2)


def a2_2_gauge_descent():
    print("=" * 74)
    print("A2.2  gauge descent: minimise trust score along the symmetry orbit")
    print("=" * 74)
    print(f"{'poison':>8} {'Krum before':>13} {'Krum after':>12} "
          f"{'rank before':>13} {'rank after':>11}")
    for s in [1.0, 2.0, 5.0, 10.0]:
        clients, teacher, Xt, r = build(strength=s)
        V0 = np.array([flat(c) for c in clients])
        rk0 = all_ranks(V0)[0]
        adv2, _, (c1, c2) = gauge_search(clients[0], clients[1:], r)
        V1 = V0.copy()
        V1[0] = flat(adv2)
        rk1 = all_ranks(V1)[0]
        print(f"{s:8.1f} {krum_score(V0, 0):13.2f} {krum_score(V1, 0):12.2f} "
              f"{rk0:13d} {rk1:11d}")
    print("\n  The orbit is searched, not the model.  The function never changes.\n")


# =========================================================== A2.3 =========
def a2_3_poison_survives():
    print("=" * 74)
    print("A2.3  does the backdoor SURVIVE the gauge move?  (must be checked)")
    print("=" * 74)
    print(f"{'poison':>8} {'backdoor before':>17} {'backdoor after':>16} "
          f"{'max |df|':>11}")
    for s in [1.0, 2.0, 5.0, 10.0]:
        clients, teacher, Xt, r = build(strength=s)
        b0 = backdoor_effect(clients[0], teacher, Xt)
        adv2, _, _ = gauge_search(clients[0], clients[1:], r)
        b1 = backdoor_effect(adv2, teacher, Xt)
        df = np.abs(forward(adv2, Xt)[0] - forward(clients[0], Xt)[0]).max()
        print(f"{s:8.1f} {b0:17.4f} {b1:16.4f} {df:11.2e}")
    print("\n  Identical to floating point.  The gauge move is functionally free.\n")


# =========================================================== A2.5 =========
def a2_5_normalisation():
    print("=" * 74)
    print("A2.5  is the obvious fix -- normalise before scoring -- sufficient?")
    print("=" * 74)
    print("  Canonicalisation candidates:")
    print("    raw       score the parameter vector as submitted")
    print("    l2norm    rescale whole vector to unit norm (norm clipping)")
    print("    layerwise rescale each layer to unit norm (fixes S1)")
    print()
    print(f"{'poison':>8} {'raw':>18} {'l2norm':>18} {'layerwise':>18}")
    print(f"{'':8} {'before/after':>18} {'before/after':>18} {'before/after':>18}")

    def canon(p, mode):
        q = {k: v.copy() for k, v in p.items()}
        if mode == "l2norm":
            n = np.linalg.norm(flat(q))
            for k in q:
                q[k] = q[k] / n
        elif mode == "layerwise":
            for li in ("1", "2", "3"):
                n = np.sqrt(np.linalg.norm(q["W" + li]) ** 2
                            + np.linalg.norm(q["b" + li]) ** 2) + 1e-30
                q["W" + li] = q["W" + li] / n
                q["b" + li] = q["b" + li] / n
        return q

    for s in [2.0, 5.0, 10.0]:
        clients, teacher, Xt, r = build(strength=s)
        adv2, _, _ = gauge_search(clients[0], clients[1:], r)
        row = []
        for mode in ("raw", "l2norm", "layerwise"):
            V0 = np.array([flat(canon(c, mode)) for c in clients])
            V1 = V0.copy()
            V1[0] = flat(canon(adv2, mode))
            row.append(f"{all_ranks(V0)[0]:2d} -> {all_ranks(V1)[0]:2d}")
        print(f"{s:8.1f} {row[0]:>18} {row[1]:>18} {row[2]:>18}")
    print("\n  Layerwise normalisation quotients out S1 but NOT S2 (permutation),")
    print("  which has no cheap canonical form -- matching hidden units across")
    print("  clients is the assignment problem.\n")


if __name__ == "__main__":
    a2_1_baseline()
    a2_2_gauge_descent()
    a2_3_poison_survives()
    a2_5_normalisation()
