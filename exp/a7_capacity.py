"""A7: the audit gap scales with model capacity.  (fuses with the anchor theorem)

A6 established the gap at one architecture: 20-32-32-1.  A single architecture
cannot distinguish "neural networks have this problem" from "this particular
network has this problem", and it cannot say whether the problem is getting
better or worse as deployed models grow.  Both matter for the paper's claim,
because the existing anchor theorem is stated in terms of the client count $N$
and says nothing about the model.

The continuous gauge group has one dimension per hidden unit.  So its dimension
is a property of the ARCHITECTURE, and it grows with width and depth while the
number of trusted anchors is fixed by what the deployment can afford to attest.
If the gap grows with capacity, then the paper's "anchors are necessary" result
acquires a second axis: how many anchors is not a function of the federation
size alone.

  A7.1  width: hold depth, grow the layer
  A7.2  depth: hold width, stack layers
  A7.3  gap per gauge dimension -- is the growth explained by dimension alone?
  A7.4  the linear control at matched parameter count
  A7.5  does the anchor-style defence degrade with capacity?
"""

import numpy as np

from mlp import (flat, forward, gauge_canon, init_mlp, n_hidden, rand_units,
                 scale_units, sgd, unflat)


def build(dims, N=24, n_per=200, seed=0, steps=250, ln=False):
    """Federation on a shared teacher.  Teacher is fixed across architectures
    so that the LEARNING PROBLEM is constant and only capacity varies."""
    r = np.random.default_rng(seed)
    d_in = dims[0]
    tr = np.random.default_rng(10_000 + seed)
    teacher = init_mlp([d_in, 24, 24, 1], tr, ln=False)
    g0 = init_mlp(dims, r, ln=ln)
    clients = []
    for _ in range(N):
        X = r.normal(size=(n_per, d_in))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.1
        clients.append(sgd(g0, X, y, steps=steps, lr=0.02, rng=r))
    Xp = r.normal(size=(1500, d_in))
    return clients, teacher, Xp, r


def agg_mse(clients, teacher, Xp, adv=None):
    V = np.array([flat(c) for c in clients])
    if adv is not None:
        V[0] = flat(adv)
    g = unflat(V.mean(axis=0), clients[0])
    return float(np.mean((forward(g, Xp)[0] - forward(teacher, Xp)[0]) ** 2))


def gap(dims, seed, sd=1.0, ln=False, reps=3):
    """Returns (own-MSE delta, damage ratio, gauge dim, clean agg, own MSE,
    teacher output variance).

    The last two exist because a ratio alone cannot be read.  If the clean
    aggregate is already as bad as predicting the teacher's mean, the model
    never learned the task at this capacity and the ratio is noise around a
    broken baseline, not a measurement of the gauge.
    """
    clients, teacher, Xp, _ = build(dims, seed=seed, ln=ln)
    ft = forward(teacher, Xp)[0]
    f0 = forward(clients[0], Xp)[0]
    own0 = float(np.mean((f0 - ft) ** 2))
    clean = agg_mse(clients, teacher, Xp)
    dm, ratios = [], []
    for rep in range(reps):
        rr = np.random.default_rng(7000 + 31 * seed + rep)
        q = scale_units(clients[0], rand_units(clients[0], rr, sd))
        dm.append(abs(float(np.mean((forward(q, Xp)[0] - ft) ** 2)) - own0))
        ratios.append(agg_mse(clients, teacher, Xp, q) / clean)
    return (np.mean(dm), np.mean(ratios), n_hidden(clients[0]), clean,
            own0, float(np.var(ft)))


# =========================================================== A7.1 =========
def a7_1_width():
    print("=" * 78)
    print("A7.1  width: the gap as the hidden layers grow")
    print("=" * 78)
    print(f"{'architecture':>22} {'gauge dim':>10} {'own MSE':>9} "
          f"{'clean agg':>10} {'/tvar':>7} {'own MSE delta':>14} "
          f"{'agg x clean':>12} {'sd':>6}")
    for h in [4, 8, 16, 32, 64, 128]:
        dims = [20, h, h, 1]
        dm, rr, cl, ow, tv = [], [], [], [], []
        for sd in range(4):
            a, b, k, c, o, t = gap(dims, sd)
            dm.append(a)
            rr.append(b)
            cl.append(c)
            ow.append(o)
            tv.append(t)
        print(f"{str(dims):>22} {2*h:10d} {np.mean(ow):9.4f} "
              f"{np.mean(cl):10.4f} {np.mean(cl)/np.mean(tv):7.2f} "
              f"{np.mean(dm):14.2e} {np.mean(rr):12.2f} {np.std(rr):6.2f}")
    print("\n  Column 3 is the functional audit and stays at machine zero at")
    print("  every width.  The '/tvar' column is the clean aggregate over the")
    print("  teacher's output variance: at 1.00 the aggregate is no better than")
    print("  predicting the mean, so any ratio against it is meaningless.\n")


# =========================================================== A7.2 =========
def a7_2_depth():
    print("=" * 78)
    print("A7.2  depth: the gap as hidden layers are stacked (width 24)")
    print("=" * 78)
    print(f"{'hidden layers':>14} {'gauge dim':>10} {'clean agg':>10} "
          f"{'/tvar':>7} {'own MSE delta':>14} {'agg x clean':>12} {'sd':>6}")
    for L in [1, 2, 3, 4, 6]:
        dims = [20] + [24] * L + [1]
        dm, rr, cl, tv = [], [], [], []
        for sd in range(4):
            a, b, k, c, o, t = gap(dims, sd)
            dm.append(a)
            rr.append(b)
            cl.append(c)
            tv.append(t)
        print(f"{L:14d} {24*L:10d} {np.mean(cl):10.4f} "
              f"{np.mean(cl)/np.mean(tv):7.2f} {np.mean(dm):14.2e} "
              f"{np.mean(rr):12.2f} {np.std(rr):6.2f}")
    print()


# =========================================================== A7.3 =========
def a7_3_per_dimension():
    print("=" * 78)
    print("A7.3  is the growth explained by gauge DIMENSION alone?")
    print("=" * 78)
    print("  Two architectures with the same number of hidden units but a")
    print("  different shape.  If dimension is the whole story they agree.")
    print()
    print(f"{'architecture':>26} {'gauge dim':>10} {'params':>9} "
          f"{'clean agg':>10} {'/tvar':>7} {'agg x clean':>12}")
    for dims in [[20, 48, 1], [20, 24, 24, 1], [20, 16, 16, 16, 1],
                 [20, 96, 1], [20, 48, 48, 1], [20, 32, 32, 32, 1]]:
        rr, cl, tv = [], [], []
        for sd in range(4):
            _, b, k, c, o, t = gap(dims, sd)
            rr.append(b)
            cl.append(c)
            tv.append(t)
        clients = build(dims, N=2, seed=0, steps=1)[0]
        print(f"{str(dims):>26} {n_hidden(clients[0]):10d} "
              f"{len(flat(clients[0])):9d} {np.mean(cl):10.4f} "
              f"{np.mean(cl)/np.mean(tv):7.2f} {np.mean(rr):12.2f}")
    print("\n  Grouped in pairs by hidden-unit count.  Disagreement within a")
    print("  pair means shape matters beyond dimension.\n")


# =========================================================== A7.4 =========
def a7_4_linear_control():
    print("=" * 78)
    print("A7.4  the control: a linear model at matched parameter count")
    print("=" * 78)
    print("  The network's gap grows with capacity.  A linear model has no")
    print("  orbit at ANY dimension, so growing it must not open a gap.")
    print()
    print(f"{'dimension':>11} {'own MSE':>11} {'own MSE after':>15} "
          f"{'agg x clean':>12}")
    for d in [20, 100, 500, 2000]:
        r = np.random.default_rng(3)
        N, n = 24, max(200, 2 * d)
        wt = r.normal(size=d) / np.sqrt(d)
        W = []
        for _ in range(N):
            X = r.normal(size=(n, d))
            W.append(np.linalg.lstsq(X, X @ wt + r.normal(size=n) * 0.1,
                                     rcond=None)[0])
        W = np.array(W)
        Xp = r.normal(size=(1500, d))
        yp = Xp @ wt

        def m(w):
            return float(np.mean((Xp @ w - yp) ** 2))

        clean = m(W.mean(axis=0))
        V = W.copy()
        V[0] = W[0] * 4.0
        print(f"{d:11d} {m(W[0]):11.6f} {m(W[0]*4.0):15.6f} "
              f"{m(V.mean(axis=0))/clean:12.2f}")
    print("\n  Own MSE moves with the aggregate at every dimension: the audit")
    print("  always sees it.  Capacity is not the mechanism; the ORBIT is.\n")


# =========================================================== A7.5 =========
def a7_5_defence_vs_capacity():
    print("=" * 78)
    print("A7.5  does canonicalisation hold up as capacity grows?")
    print("=" * 78)
    print(f"{'architecture':>22} {'raw agg x':>11} {'canon agg x':>13} "
          f"{'canon max|df|':>15}")
    for h in [8, 32, 128]:
        dims = [20, h, h, 1]
        raws, cans, dfs = [], [], []
        for sd in range(3):
            clients, teacher, Xp, _ = build(dims, seed=sd)
            gc = [gauge_canon(c) for c in clients]
            base_raw = agg_mse(clients, teacher, Xp)
            base_can = agg_mse(gc, teacher, Xp)
            rr = np.random.default_rng(8000 + sd)
            q = scale_units(clients[0], rand_units(clients[0], rr, 1.5))
            raws.append(agg_mse(clients, teacher, Xp, q) / base_raw)
            cans.append(agg_mse(gc, teacher, Xp, gauge_canon(q)) / base_can)
            dfs.append(np.abs(forward(gauge_canon(q), Xp)[0]
                              - forward(q, Xp)[0]).max())
        print(f"{str(dims):>22} {np.mean(raws):11.2f} {np.mean(cans):13.2f} "
              f"{np.mean(dfs):15.1e}")
    print("\n  Ratios are against each pipeline's own clean baseline, so a value")
    print("  of 1.00 in column 3 means the damage is gone rather than that the")
    print("  baseline was already broken -- the mistake A4.5 originally made.\n")


if __name__ == "__main__":
    a7_1_width()
    a7_2_depth()
    a7_3_per_dimension()
    a7_4_linear_control()
    a7_5_defence_vs_capacity()
