"""A4: the aggregate is not gauge invariant -- and the client is auditable.

A2 and A3 both failed, for the same reason, and the reason is worth stating
because it bounds a whole family of attacks:

    Rescaling hidden unit j multiplies W1[j] by a_j and W2[:,j] by 1/a_j.
    Their product is invariant, so by AM-GM the contribution to any
    coordinate-averaged dispersion is floored at 2*sqrt(AB).  Gauge freedom
    cannot manufacture apparent stability.  Measured ceiling: 1.6x (A3).

So the adversary cannot use the orbit to look BETTER.  But there is an
asymmetry it can use, and it is the one that is specific to neural networks:

    the client's FUNCTION is gauge invariant;
    the AGGREGATE of several clients is NOT.

Averaging two networks that compute the identical function in different gauges
does not yield that function.  So an adversary can hold a model that passes
every functional audit exactly -- correct predictions, correct loss, correct
behaviour on any held-out probe the aggregator cares to run, bit for bit --
while contributing a parameter vector that destroys the average.

  "I submitted a correct model" and "I contributed a correct update" come
  apart.  In a linear model they cannot: there is no orbit, so a client that
  submits a correct model necessarily contributes a correct update.

The question this script answers is whether the attack is CHEAP -- whether
damage arrives before detectability does.  That is an empirical tradeoff and it
could easily come out against the attack, as A2 and A3 did.

  A4.1  functional audit vs aggregate damage: does the audit see anything?
  A4.2  the tradeoff curve -- damage per unit of detectability
  A4.3  permutation, which is the discrete part of the group
  A4.4  the linear control: the same attack on a model with no orbit
  A4.5  does layerwise canonicalisation close it?
"""

import numpy as np

from a1_nn_symmetry import (D_IN, H1, H2, flat, forward, gauge_permute,
                            gauge_scale, init_net, krum_score, sgd)


def unflat(v, ref):
    out, o = {}, 0
    for k in sorted(ref):
        sz = ref[k].size
        out[k] = v[o:o + sz].reshape(ref[k].shape)
        o += sz
    return out


def scale_units(p, a, d):
    """Per-hidden-unit positive rescaling: the continuous part of the group.

    a in R_+^H1, d in R_+^H2.  Exactly function preserving for ReLU by positive
    homogeneity, applied unit by unit rather than layer by layer.
    """
    q = {k: v.copy() for k, v in p.items()}
    q["W1"] = q["W1"] * a[:, None]
    q["b1"] = q["b1"] * a
    q["W2"] = q["W2"] / a[None, :]
    q["W2"] = q["W2"] * d[:, None]
    q["b2"] = q["b2"] * d
    q["W3"] = q["W3"] / d[None, :]
    return q


def layer_canon(p):
    """Layerwise unit-norm rescaling.

    NOT a gauge move: it divides W3/b3 as well, which rescales the output and
    changes the function.  Kept because it is the obvious first thing to try
    and A4.5 measures what it costs.
    """
    q = {k: v.copy() for k, v in p.items()}
    for li in ("1", "2", "3"):
        n = np.sqrt(np.linalg.norm(q["W" + li]) ** 2
                    + np.linalg.norm(q["b" + li]) ** 2) + 1e-30
        q["W" + li] = q["W" + li] / n
        q["b" + li] = q["b" + li] / n
    return q


def gauge_canon(p):
    """Canonical representative of the orbit, reached by a gauge move only.

    Normalise each hidden unit's incoming weights to unit norm and push the
    scale forward into the next layer.  This is exactly scale_units with
    a_j = 1/||W1[j], b1[j]||, so the function is preserved, and it fixes the
    per-unit scale rather than one number per layer.
    """
    a = 1.0 / (np.sqrt(np.sum(p["W1"] ** 2, axis=1) + p["b1"] ** 2) + 1e-30)
    d = 1.0 / (np.sqrt(np.sum((p["W2"] / a[None, :]) ** 2, axis=1)
                       + p["b2"] ** 2) + 1e-30)
    return scale_units(p, a, d)


def build(N=30, n_per=200, seed=0, steps=200):
    r = np.random.default_rng(seed)
    teacher = init_net(r)
    g0 = init_net(r)
    clients = []
    for _ in range(N):
        X = r.normal(size=(n_per, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.1
        clients.append(sgd(g0, X, y, steps=steps, lr=0.05, rng=r))
    Xp = r.normal(size=(2000, D_IN))
    return clients, teacher, Xp, r


def agg_mse(clients, teacher, Xp, adv_param=None):
    """Test error of the plain average, optionally with client 0 replaced."""
    V = np.array([flat(c) for c in clients])
    if adv_param is not None:
        V[0] = flat(adv_param)
    g = unflat(V.mean(axis=0), clients[0])
    return float(np.mean((forward(g, Xp)[0] - forward(teacher, Xp)[0]) ** 2))


def zscore(clients, adv_param):
    """Detectability: adversary's distance-to-mean in units of the honest sd."""
    V = np.array([flat(c) for c in clients])
    V[0] = flat(adv_param)
    m = V.mean(axis=0)
    d = np.linalg.norm(V - m, axis=1)
    hon = d[1:]
    return float((d[0] - hon.mean()) / (hon.std() + 1e-30))


# =========================================================== A4.1 =========
def a4_1_audit_blind():
    print("=" * 76)
    print("A4.1  the functional audit vs the aggregate: do they agree?")
    print("=" * 76)
    clients, teacher, Xp, r = build()
    f0 = forward(clients[0], Xp)[0]
    own = float(np.mean((f0 - forward(teacher, Xp)[0]) ** 2))
    clean = agg_mse(clients, teacher, Xp)
    print(f"  adversary's own test MSE (honest model)  {own:.6f}")
    print(f"  clean aggregate test MSE                 {clean:.6f}")
    print()
    print(f"{'gauge move':>28} {'own MSE':>12} {'audit df':>11} "
          f"{'agg MSE':>11} {'x clean':>9}")
    a_big = np.exp(r.normal(size=H1) * 1.5)
    d_big = np.exp(r.normal(size=H2) * 1.5)
    for name, q in [
        ("none", clients[0]),
        ("global c1=2", gauge_scale(clients[0], 2.0, 1.0)),
        ("global c1=8", gauge_scale(clients[0], 8.0, 1.0)),
        ("per-unit sd=0.5", scale_units(clients[0],
                                        np.exp(r.normal(size=H1) * 0.5),
                                        np.exp(r.normal(size=H2) * 0.5))),
        ("per-unit sd=1.5", scale_units(clients[0], a_big, d_big)),
        ("permutation", gauge_permute(clients[0], r)),
    ]:
        m = float(np.mean((forward(q, Xp)[0] - forward(teacher, Xp)[0]) ** 2))
        df = np.abs(forward(q, Xp)[0] - f0).max()
        a = agg_mse(clients, teacher, Xp, q)
        print(f"{name:>28} {m:12.6f} {df:11.2e} {a:11.6f} {a/clean:9.2f}")
    print("\n  Columns 1-2 are what a functional audit sees: unchanged, exactly.")
    print("  Column 3 is what the federation gets.\n")


# =========================================================== A4.2 =========
def a4_2_tradeoff():
    print("=" * 76)
    print("A4.2  is damage CHEAP?  aggregate harm per unit of detectability")
    print("=" * 76)
    clients, teacher, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    V0 = np.array([flat(c) for c in clients])
    N = len(clients)
    print(f"{'per-unit log-sd':>17} {'agg MSE':>11} {'x clean':>9} "
          f"{'z-score':>9} {'Krum rank':>11}")
    for s in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]:
        ms, zs, rk = [], [], []
        for rep in range(5):
            rr = np.random.default_rng(1000 + rep)
            q = scale_units(clients[0], np.exp(rr.normal(size=H1) * s),
                            np.exp(rr.normal(size=H2) * s))
            ms.append(agg_mse(clients, teacher, Xp, q))
            zs.append(zscore(clients, q))
            V = V0.copy()
            V[0] = flat(q)
            rk.append(int(np.argsort(np.argsort(
                np.array([krum_score(V, i) for i in range(N)])))[0]) + 1)
        print(f"{s:17.2f} {np.mean(ms):11.6f} {np.mean(ms)/clean:9.2f} "
              f"{np.mean(zs):9.2f} {np.mean(rk):11.1f}")
    print("\n  If the x-clean column rises while the z-score stays low, damage")
    print("  is cheap.  If detectability outruns damage, the attack is not")
    print("  viable and the honest reading is that distance scoring holds.\n")


# =========================================================== A4.3 =========
def a4_3_permutation():
    print("=" * 76)
    print("A4.3  permutation: the discrete part of the group")
    print("=" * 76)
    clients, teacher, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    V0 = np.array([flat(c) for c in clients])
    N = len(clients)
    print(f"{'permuted clients':>18} {'agg MSE':>11} {'x clean':>9} {'z-score':>9}")
    for k in [0, 1, 2, 3, 5, 8]:
        cs = list(clients)
        for i in range(k):
            cs[i] = gauge_permute(clients[i], r)
        a = agg_mse(cs, teacher, Xp)
        z = zscore(clients, cs[0]) if k else 0.0
        print(f"{k:18d} {a:11.6f} {a/clean:9.2f} {z:9.2f}")
    print("\n  Permutation is free to apply and has no cheap canonical form:")
    print("  aligning hidden units across clients is an assignment problem.\n")


# =========================================================== A4.4 =========
def a4_4_linear_control():
    print("=" * 76)
    print("A4.4  the control: the same attack against a model with NO orbit")
    print("=" * 76)
    r = np.random.default_rng(5)
    d, N, n = 20, 30, 200
    wt = r.normal(size=d)
    W = []
    for _ in range(N):
        X = r.normal(size=(n, d))
        y = X @ wt + r.normal(size=n) * 0.1
        W.append(np.linalg.lstsq(X, y, rcond=None)[0])
    W = np.array(W)
    Xp = r.normal(size=(2000, d))
    yp = Xp @ wt

    def mse(w):
        return float(np.mean((Xp @ w - yp) ** 2))

    clean = mse(W.mean(axis=0))
    print(f"  clean aggregate MSE {clean:.6f}")
    print()
    print(f"{'client 0 transform':>28} {'own MSE':>12} {'agg MSE':>11} {'x clean':>9}")
    print(f"{'none':>28} {mse(W[0]):12.6f} {clean:11.6f} {1.0:9.2f}")
    for c in [2.0, 8.0]:
        V = W.copy()
        V[0] = W[0] * c
        print(f"{'scale by ' + str(c):>28} {mse(W[0]*c):12.6f} "
              f"{mse(V.mean(axis=0)):11.6f} {mse(V.mean(axis=0))/clean:9.2f}")
    V = W.copy()
    V[0] = W[0][r.permutation(d)]
    print(f"{'permute coefficients':>28} {mse(V[0]):12.6f} "
          f"{mse(V.mean(axis=0)):11.6f} {mse(V.mean(axis=0))/clean:9.2f}")
    print("\n  The point of this table is the 'own MSE' column.  Every transform")
    print("  that damages the linear aggregate also damages the client's own")
    print("  model, so a functional audit catches it.  There is no move that is")
    print("  free for the client and costly for the average -- that gap is what")
    print("  the symmetry group opens up, and it exists only for the network.\n")


# =========================================================== A4.5 =========
def a4_5_canonicalisation():
    print("=" * 76)
    print("A4.5  do canonicalisation defences close it?")
    print("=" * 76)
    clients, teacher, Xp, r = build()
    clean = agg_mse(clients, teacher, Xp)
    lc = [layer_canon(c) for c in clients]
    gc = [gauge_canon(c) for c in clients]
    tvar = float(np.var(forward(teacher, Xp)[0]))
    print(f"  clean aggregate MSE                 {clean:.6f}")
    print(f"  layerwise-canon aggregate MSE       {agg_mse(lc, teacher, Xp):.6f}")
    print(f"  gauge-canon aggregate MSE           {agg_mse(gc, teacher, Xp):.6f}")
    print(f"  variance of teacher output          {tvar:.6f}")
    print()
    print("  Layerwise normalisation divides W3 as well, so it rescales the")
    print("  OUTPUT: it is not a gauge move and it destroys the model on its")
    print("  own, before any attack.  Absolute MSE is reported for that reason;")
    print("  a ratio against a broken baseline would look like a defence.")
    print()
    print(f"{'gauge move':>24} {'raw':>10} {'layerwise':>11} {'gauge-canon':>13}")
    moves = [
        ("global c1=8", gauge_scale(clients[0], 8.0, 1.0)),
        ("per-unit sd=0.5", scale_units(clients[0],
                                        np.exp(r.normal(size=H1) * 0.5),
                                        np.exp(r.normal(size=H2) * 0.5))),
        ("per-unit sd=1.5", scale_units(clients[0],
                                        np.exp(r.normal(size=H1) * 1.5),
                                        np.exp(r.normal(size=H2) * 1.5))),
        ("permutation", gauge_permute(clients[0], r)),
    ]
    for name, q in moves:
        print(f"{name:>24} {agg_mse(clients, teacher, Xp, q):10.4f} "
              f"{agg_mse(lc, teacher, Xp, layer_canon(q)):11.4f} "
              f"{agg_mse(gc, teacher, Xp, gauge_canon(q)):13.4f}")
    print()
    print("  Gauge canonicalisation fixes the per-unit scale by a move that")
    print("  stays inside the orbit, so it costs nothing and it does neutralise")
    print("  the continuous part of the group.  It does not touch permutation,")
    print("  for which no cheap canonical form exists: aligning hidden units")
    print("  across clients is an assignment problem.\n")


if __name__ == "__main__":
    a4_1_audit_blind()
    a4_2_tradeoff()
    a4_3_permutation()
    a4_4_linear_control()
    a4_5_canonicalisation()
