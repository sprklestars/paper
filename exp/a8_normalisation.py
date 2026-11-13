"""A8: normalisation does not close the gauge.  It relocates it and adds to it.

The obvious objection to A6 is that modern networks normalise, and normalisation
fixes the scale of each layer, so the gauge freedom the audit gap relies on
should not exist in practice.  The objection is worth taking seriously because
if it holds, the result applies only to a network nobody deploys.

It does not hold, and the reason is that positive homogeneity is a property of
the ACTIVATION, not of the weights.  Normalising the pre-activation removes the
naive move -- rescaling (W, b) genuinely changes the normalised output -- but the
per-unit scale reappears one step later in the affine gain:

    u_j = zhat_j * g_j + beta_j,     relu(a u_j) = a relu(u_j)  for a > 0,

so scaling (g_j, beta_j) by a_j and dividing column j of the next layer is
exactly function preserving.  The group still has one dimension per hidden unit.
Normalisation also ADDS free directions that need no compensation at all:
scaling or shifting the pre-normalisation activations of a normalised layer
leaves the layer's output unchanged by construction.

  A8.1  the group under LayerNorm: which moves are exact, at what magnitude
  A8.2  the audit gap with normalisation vs without, matched architectures
  A8.3  the new free directions, which have no analogue in the plain network
  A8.4  weight decay -- does penalising the gauge suppress it?
  A8.5  canonicalisation under normalisation
"""

import numpy as np

from a7_capacity import agg_mse, build
from mlp import (flat, forward, gauge_canon, gauge_permute, init_mlp,
                 ln_layer_move, mse, rand_units, scale_units, sgd, unflat)

DIMS = [20, 32, 32, 1]


# =========================================================== A8.1 =========
def a8_1_exactness():
    print("=" * 78)
    print("A8.1  is the relocated group EXACT under LayerNorm?")
    print("=" * 78)
    print(f"{'per-unit log-sd':>16} {'plain max|df|':>15} {'LN max|df|':>13} "
          f"{'LN param move':>15}")
    for s in [0.25, 0.5, 1.0, 2.0, 3.0]:
        pl, ln_, mv = [], [], []
        for sd in range(3):
            for use_ln in (False, True):
                clients, teacher, Xp, _ = build(DIMS, N=3, seed=sd, ln=use_ln)
                p = clients[0]
                f0 = forward(p, Xp)[0]
                rr = np.random.default_rng(400 + sd)
                q = scale_units(p, rand_units(p, rr, s))
                df = np.abs(forward(q, Xp)[0] - f0).max()
                if use_ln:
                    ln_.append(df)
                    mv.append(np.linalg.norm(flat(q) - flat(p))
                              / np.linalg.norm(flat(p)))
                else:
                    pl.append(df)
        print(f"{s:16.2f} {np.mean(pl):15.1e} {np.mean(ln_):13.1e} "
              f"{np.mean(mv):15.2f}")
    print("\n  Both columns are machine zero.  The last column is how far the")
    print("  parameter vector travelled while the function did not move.\n")


# =========================================================== A8.2 =========
def a8_2_gap_with_norm():
    print("=" * 78)
    print("A8.2  the audit gap: normalised vs plain, same architecture")
    print("=" * 78)
    print(f"{'architecture':>12} {'log-sd':>7} {'own MSE delta':>14} "
          f"{'agg x clean':>12} {'clean agg':>11}")
    for use_ln in (False, True):
        tag = "LayerNorm" if use_ln else "plain"
        for s in [0.0, 0.5, 1.0, 1.5]:
            dm, rr, cl = [], [], []
            for sd in range(4):
                clients, teacher, Xp, _ = build(DIMS, seed=sd, ln=use_ln)
                ft = forward(teacher, Xp)[0]
                f0 = forward(clients[0], Xp)[0]
                own0 = float(np.mean((f0 - ft) ** 2))
                clean = agg_mse(clients, teacher, Xp)
                r2 = np.random.default_rng(600 + sd)
                q = scale_units(clients[0], rand_units(clients[0], r2, s))
                dm.append(abs(float(np.mean((forward(q, Xp)[0] - ft) ** 2))
                              - own0))
                rr.append(agg_mse(clients, teacher, Xp, q) / clean)
                cl.append(clean)
            print(f"{tag:>12} {s:7.2f} {np.mean(dm):14.2e} "
                  f"{np.mean(rr):12.2f} {np.mean(cl):11.4f}")
    print("\n  If the LayerNorm rows show a gap of the same order, normalisation")
    print("  is not a defence.  The clean-aggregate column is there so that a")
    print("  ratio cannot be inflated by a worse baseline.\n")


# =========================================================== A8.3 =========
def a8_3_extra_directions():
    print("=" * 78)
    print("A8.3  the directions normalisation ADDS (no plain-network analogue)")
    print("=" * 78)
    print("  Scaling or shifting the pre-normalisation activations of a")
    print("  normalised layer changes nothing downstream, and unlike the")
    print("  per-unit scale it requires NO compensating change elsewhere.")
    print()
    print(f"{'move on layer 0':>26} {'max |df|':>11} {'param move':>12} "
          f"{'agg x clean':>12}")
    clients, teacher, Xp, r = build(DIMS, seed=0, ln=True)
    p = clients[0]
    f0 = forward(p, Xp)[0]
    clean = agg_mse(clients, teacher, Xp)
    for name, q in [
        ("scale W,b by 10", ln_layer_move(p, 0, c=10.0)),
        ("scale W,b by 100", ln_layer_move(p, 0, c=100.0)),
        ("shift b by +5", ln_layer_move(p, 0, shift=5.0)),
        ("shift b by +50", ln_layer_move(p, 0, shift=50.0)),
        ("scale 100 then shift 50", ln_layer_move(p, 0, c=100.0, shift=50.0)),
    ]:
        print(f"{name:>26} {np.abs(forward(q, Xp)[0] - f0).max():11.1e} "
              f"{np.linalg.norm(flat(q)-flat(p))/np.linalg.norm(flat(p)):12.2f} "
              f"{agg_mse(clients, teacher, Xp, q)/clean:12.2f}")
    print("\n  The scale move is exact only up to LayerNorm's epsilon, so its")
    print("  df is small-but-nonzero; the shift is exact.  Either way no")
    print("  functional audit at any realistic tolerance sees these.\n")


# =========================================================== A8.4 =========
def a8_4_weight_decay():
    print("=" * 78)
    print("A8.4  does weight decay suppress the gauge?")
    print("=" * 78)
    print("  Weight decay penalises large weights, and the gauge can make")
    print("  weights large, so it plausibly pins the gauge down.  But decay is")
    print("  applied during TRAINING, by the client, and the gauge move is")
    print("  applied to what the client SUBMITS.  Measured both ways.")
    print()

    def train_wd(dims, X, y, g0, wd, steps=250, rng=None):
        p = [{k: v.copy() for k, v in L.items()} for L in g0]
        n = len(y)
        from mlp import grads
        for _ in range(steps):
            idx = rng.integers(0, n, size=32)
            g = grads(p, X[idx], y[idx])
            for L, gi in zip(p, g):
                for k in gi:
                    L[k] -= 0.02 * (gi[k] + wd * L[k])
        return p

    print(f"{'weight decay':>13} {'client own MSE':>15} {'param norm':>11} "
          f"{'agg x clean':>12} {'own MSE delta':>14}")
    for wd in [0.0, 1e-4, 1e-3, 1e-2]:
        rr, dm, nm, om = [], [], [], []
        for sd in range(3):
            r = np.random.default_rng(sd)
            tr = np.random.default_rng(10_000 + sd)
            teacher = init_mlp([20, 24, 24, 1], tr)
            g0 = init_mlp(DIMS, r)
            cl = []
            for _ in range(24):
                X = r.normal(size=(200, 20))
                y = forward(teacher, X)[0] + r.normal(size=200) * 0.1
                cl.append(train_wd(DIMS, X, y, g0, wd, rng=r))
            Xp = r.normal(size=(1500, 20))
            ft = forward(teacher, Xp)[0]
            clean = agg_mse(cl, teacher, Xp)
            own0 = float(np.mean((forward(cl[0], Xp)[0] - ft) ** 2))
            r2 = np.random.default_rng(650 + sd)
            q = scale_units(cl[0], rand_units(cl[0], r2, 1.5))
            rr.append(agg_mse(cl, teacher, Xp, q) / clean)
            dm.append(abs(float(np.mean((forward(q, Xp)[0] - ft) ** 2)) - own0))
            nm.append(np.linalg.norm(flat(cl[0])))
            om.append(own0)
        print(f"{wd:13.0e} {np.mean(om):15.6f} {np.mean(nm):11.2f} "
              f"{np.mean(rr):12.2f} {np.mean(dm):14.2e}")
    print("\n  Decay shrinks the honest weights, which is visible in column 3,")
    print("  and does nothing to the gap: the adversary applies the gauge after")
    print("  training, and the server cannot verify that decay was used.\n")


# =========================================================== A8.5 =========
def a8_5_canon_under_norm():
    print("=" * 78)
    print("A8.5  canonicalisation under normalisation")
    print("=" * 78)
    print(f"{'move':>28} {'raw agg x':>11} {'canon agg x':>13} "
          f"{'canon max|df|':>15}")
    clients, teacher, Xp, r = build(DIMS, seed=0, ln=True)
    gc = [gauge_canon(c) for c in clients]
    braw = agg_mse(clients, teacher, Xp)
    bcan = agg_mse(gc, teacher, Xp)
    print(f"{'(clean baselines, absolute)':>28} {braw:11.4f} {bcan:13.4f} "
          f"{'-':>15}")
    for name, q in [
        ("per-unit sd=1.5", scale_units(clients[0],
                                        rand_units(clients[0], r, 1.5))),
        ("per-unit sd=3.0", scale_units(clients[0],
                                        rand_units(clients[0], r, 3.0))),
        ("permutation", gauge_permute(clients[0], r)),
        ("LN layer scale 100", ln_layer_move(clients[0], 0, c=100.0)),
        ("LN layer shift 50", ln_layer_move(clients[0], 0, shift=50.0)),
    ]:
        df = np.abs(forward(gauge_canon(q), Xp)[0] - forward(q, Xp)[0]).max()
        print(f"{name:>28} {agg_mse(clients, teacher, Xp, q)/braw:11.2f} "
              f"{agg_mse(gc, teacher, Xp, gauge_canon(q))/bcan:13.2f} "
              f"{df:15.1e}")
    print("\n  Canonicalising the gain closes the per-unit group under")
    print("  normalisation too.  It does NOT close the two moves A8.3 found,")
    print("  because those live in (W, b) of a normalised layer, which")
    print("  gain-normalisation never touches.  Closing them needs a second")
    print("  step: normalise each normalised layer's own (W, b) as well, which")
    print("  is free for exactly the same reason.\n")


if __name__ == "__main__":
    a8_1_exactness()
    a8_2_gap_with_norm()
    a8_3_extra_directions()
    a8_4_weight_decay()
    a8_5_canon_under_norm()
