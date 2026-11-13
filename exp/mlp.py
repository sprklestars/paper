"""Generic MLP of arbitrary width and depth, with optional LayerNorm.

a1-a6 used a fixed 20-32-32-1 ReLU network.  Two questions need architectures
that net cannot express:

  a7  does the audit gap GROW with width and depth?
  a8  does normalisation remove the gauge group?

so this module provides the network, its analytic gradients (checked against
finite differences in the self-test below, because an unverified backward pass
would silently corrupt every downstream number), and the gauge operations in a
form that works for any depth.

LayerNorm placement is pre-activation, the standard choice:

    z = W a + b,   zhat = (z - mean z) / sqrt(var z + eps),
    u = zhat * g + beta,   a' = relu(u).

The self-test also establishes the three facts a8 depends on.  Stated here
because they are analytic, and measured there:

  (N1) Per-unit rescaling of W, b is NOT a symmetry once the layer is
       normalised: it changes mean and variance across the layer, so zhat
       changes.  This is the usual reason people expect normalisation to kill
       the gauge.
  (N2) But the per-unit scale reappears in the GAIN.  relu is positively
       homogeneous, so scaling (g_j, beta_j) by a_j > 0 scales u_j by a_j and
       hence a'_j by a_j; dividing column j of the next layer restores the
       function exactly.  The group is still one dimension per hidden unit.
  (N3) Normalisation ADDS free directions: scaling (W, b) of a normalised layer
       by c, and shifting b by a constant, both leave zhat unchanged -- with no
       compensation needed anywhere.  These are exact up to the eps in the
       denominator, i.e. to O(eps / var z).
"""

import numpy as np

EPS = 1e-8


# --------------------------------------------------------------- model ----
def init_mlp(dims, rng, ln=False):
    """dims = [d_in, h_1, ..., h_L, d_out].  He init.  LN on hidden layers."""
    p = []
    for i in range(len(dims) - 1):
        layer = {"W": rng.normal(size=(dims[i + 1], dims[i]))
                 * np.sqrt(2.0 / dims[i]),
                 "b": np.zeros(dims[i + 1])}
        if ln and i < len(dims) - 2:
            layer["g"] = np.ones(dims[i + 1])
            layer["beta"] = np.zeros(dims[i + 1])
        p.append(layer)
    return p


def forward(p, X):
    """Returns (output, cache).  Output is raveled; d_out is 1 throughout."""
    a = X
    cache = []
    last = len(p) - 1
    for i, L in enumerate(p):
        z = a @ L["W"].T + L["b"]
        if i == last:
            cache.append({"a_in": a, "z": z})
            return z.ravel(), cache
        if "g" in L:
            mu = z.mean(axis=1, keepdims=True)
            sd = np.sqrt(z.var(axis=1, keepdims=True) + EPS)
            zh = (z - mu) / sd
            u = zh * L["g"] + L["beta"]
        else:
            zh, sd = None, None
            u = z
        cache.append({"a_in": a, "z": z, "zh": zh, "sd": sd, "u": u})
        a = np.maximum(u, 0.0)
    raise AssertionError("unreachable")


def mse(p, X, y):
    return float(np.mean((forward(p, X)[0] - y) ** 2))


def grads(p, X, y):
    """Analytic gradient of mean squared error.  Verified in _selftest."""
    n = len(y)
    out, cache = forward(p, X)
    g = [None] * len(p)
    last = len(p) - 1

    dz = (2.0 * (out - y) / n)[:, None]              # (n, 1)
    for i in range(last, -1, -1):
        L, c = p[i], cache[i]
        if i < last:
            du = dz                                   # dz here means d/d a_out
            du = du * (c["u"] > 0)
            if "g" in L:
                gg = np.sum(du * c["zh"], axis=0)
                gbeta = du.sum(axis=0)
                dzh = du * L["g"]
                m = dzh.mean(axis=1, keepdims=True)
                mz = (dzh * c["zh"]).mean(axis=1, keepdims=True)
                dz = (dzh - m - c["zh"] * mz) / c["sd"]
            else:
                gg = gbeta = None
                dz = du
        else:
            gg = gbeta = None

        gi = {"W": dz.T @ c["a_in"], "b": dz.sum(axis=0)}
        if gg is not None:
            gi["g"], gi["beta"] = gg, gbeta
        g[i] = gi
        if i > 0:
            dz = dz @ L["W"]                          # becomes d/d a_out below
    return g


def sgd(p, X, y, steps=200, lr=0.05, batch=32, rng=None):
    p = [{k: v.copy() for k, v in L.items()} for L in p]
    n = len(y)
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        g = grads(p, X[idx], y[idx])
        for L, gi in zip(p, g):
            for k in gi:
                L[k] -= lr * gi[k]
    return p


# ------------------------------------------------------------- vectors ----
def flat(p):
    return np.concatenate([L[k].ravel() for L in p for k in sorted(L)])


def unflat(v, ref):
    out, o = [], 0
    for L in ref:
        d = {}
        for k in sorted(L):
            d[k] = v[o:o + L[k].size].reshape(L[k].shape)
            o += L[k].size
        out.append(d)
    return out


def copy_net(p):
    return [{k: v.copy() for k, v in L.items()} for L in p]


def n_hidden(p):
    """Dimension of the continuous gauge group: one per hidden unit."""
    return sum(L["b"].size for L in p[:-1])


# ---------------------------------------------------------- gauge group ----
def scale_units(p, factors):
    """Per-hidden-unit positive rescaling.  Exact for any depth.

    factors[i] is the vector of scales for hidden layer i.  On an unnormalised
    layer the scale is applied to (W, b); on a normalised layer it is applied to
    (g, beta), which is where relu homogeneity still reaches -- (N2).
    """
    q = copy_net(p)
    for i, a in enumerate(factors):
        if "g" in q[i]:
            q[i]["g"] = q[i]["g"] * a
            q[i]["beta"] = q[i]["beta"] * a
        else:
            q[i]["W"] = q[i]["W"] * a[:, None]
            q[i]["b"] = q[i]["b"] * a
        q[i + 1]["W"] = q[i + 1]["W"] / a[None, :]
    return q


def rand_units(p, rng, sd):
    return [np.exp(rng.normal(size=L["b"].size) * sd) for L in p[:-1]]


def gauge_canon(p):
    """Orbit representative reached by a gauge move, so it is free.

    Normalise each hidden unit's own scale carrier to unit norm and push the
    scale forward.  The carrier is (g, beta) when the layer is normalised and
    (W row, b) when it is not.
    """
    # Sequential, not all at once: layer i's incoming weights get divided by
    # layer i-1's scale, so its own norm must be recomputed after that lands.
    q = copy_net(p)
    for i in range(len(q) - 1):
        L = q[i]
        if "g" in L:
            nrm = np.sqrt(L["g"] ** 2 + L["beta"] ** 2)
        else:
            nrm = np.sqrt(np.sum(L["W"] ** 2, axis=1) + L["b"] ** 2)
        a = 1.0 / (nrm + 1e-30)
        if "g" in L:
            L["g"] = L["g"] * a
            L["beta"] = L["beta"] * a
        else:
            L["W"] = L["W"] * a[:, None]
            L["b"] = L["b"] * a
        q[i + 1]["W"] = q[i + 1]["W"] / a[None, :]
    return q


def gauge_permute(p, rng):
    """Permute hidden units of every hidden layer.  Exact for any depth."""
    q = copy_net(p)
    for i, L in enumerate(q[:-1]):
        s = rng.permutation(L["b"].size)
        for k in ("W", "b", "g", "beta"):
            if k in L:
                L[k] = L[k][s]
        q[i + 1]["W"] = q[i + 1]["W"][:, s]
    return q


def ln_layer_move(p, i, c=1.0, shift=0.0):
    """(N3): scale and shift the pre-normalisation activations of layer i.

    Only meaningful when layer i is normalised.  No downstream compensation.
    Exact only up to the eps in the LayerNorm denominator.
    """
    q = copy_net(p)
    q[i]["W"] = q[i]["W"] * c
    q[i]["b"] = q[i]["b"] * c + shift
    return q


# ------------------------------------------------------------ self-test ----
def _selftest():
    print("=" * 76)
    print("mlp.py self-test: gradients against finite differences")
    print("=" * 76)
    r = np.random.default_rng(0)
    for ln in (False, True):
        dims = [6, 8, 7, 1]
        p = init_mlp(dims, r, ln=ln)
        # break the symmetric init so g/beta gradients are exercised
        for L in p:
            if "g" in L:
                L["g"] += r.normal(size=L["g"].size) * 0.2
                L["beta"] += r.normal(size=L["beta"].size) * 0.2
        X = r.normal(size=(40, 6))
        y = r.normal(size=40)
        g = grads(p, X, y)
        worst = 0.0
        h = 1e-6
        for i, L in enumerate(p):
            for k in sorted(L):
                flatidx = np.ndindex(L[k].shape)
                for idx in list(flatidx)[:6]:
                    o = L[k][idx]
                    L[k][idx] = o + h
                    f1 = mse(p, X, y)
                    L[k][idx] = o - h
                    f2 = mse(p, X, y)
                    L[k][idx] = o
                    num = (f1 - f2) / (2 * h)
                    worst = max(worst, abs(num - g[i][k][idx])
                                / max(1e-8, abs(num)))
        print(f"  ln={str(ln):<5}  worst relative gradient error {worst:.2e}"
              f"   {'ok' if worst < 2e-4 else 'FAIL'}")
        assert worst < 2e-4

    print()
    print("=" * 76)
    print("mlp.py self-test: which moves preserve the function?")
    print("=" * 76)
    print(f"{'architecture':>16} {'move':>34} {'max |df|':>11} {'symmetry?':>11}")
    for ln in (False, True):
        dims = [10, 16, 16, 1]
        p = init_mlp(dims, r, ln=ln)
        X = r.normal(size=(200, 10))
        y = np.sin(X[:, 0]) + X[:, 1]
        p = sgd(p, X, y, steps=300, lr=0.02, rng=r)
        Xp = r.normal(size=(500, 10))
        f0 = forward(p, Xp)[0]
        tag = "LayerNorm" if ln else "plain ReLU"

        def show(name, q):
            df = np.abs(forward(q, Xp)[0] - f0).max()
            print(f"{tag:>16} {name:>34} {df:11.1e} "
                  f"{'YES' if df < 1e-9 else 'no':>11}")

        # per-unit rescale applied to (W, b) directly: (N1)
        q = copy_net(p)
        a = np.exp(r.normal(size=dims[1]) * 0.5)
        q[0]["W"] = q[0]["W"] * a[:, None]
        q[0]["b"] = q[0]["b"] * a
        q[1]["W"] = q[1]["W"] / a[None, :]
        show("per-unit rescale of (W,b)", q)

        # per-unit rescale through the proper carrier: (N2)
        show("per-unit rescale via scale_units",
             scale_units(p, rand_units(p, r, 0.5)))
        show("permute hidden units", gauge_permute(p, r))
        show("gauge canonicalisation", gauge_canon(p))
        if ln:
            show("layer scale c=5 (no compensation)", ln_layer_move(p, 0, c=5.0))
            show("layer shift +3 (no compensation)",
                 ln_layer_move(p, 0, shift=3.0))
    print()
    print("  Read the LayerNorm block against the plain block.  Normalisation")
    print("  removes the naive per-unit move and leaves the real one, and it")
    print("  adds two directions that need no compensation at all.")
    print()


if __name__ == "__main__":
    _selftest()
