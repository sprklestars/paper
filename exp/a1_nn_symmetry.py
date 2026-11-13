"""A1: does the trust layer inherit a defect that is specific to NEURAL networks?

Motivation.  Every theorem in the current paper survives replacing the model
with a Kalman filter: inverse-variance capture is classical estimation theory,
and translation gauge freedom is classical robust statistics.  Nothing so far
is about AI.

But a neural network has a parameter-space symmetry group that a linear model
does not have, and FL trust scoring operates on PARAMETERS:

  (S1) ReLU positive homogeneity.  relu(cz) = c relu(z) for c>0, so
       W1 -> cW1, b1 -> c b1, W2 -> W2/c leaves the function EXACTLY
       unchanged while moving the parameter vector arbitrarily far.
  (S2) Hidden-unit permutation.  Permuting rows of W1/b1 and columns of W2
       leaves the function exactly unchanged.  h! of these per layer.

Krum, FoolsGold, FLTrust, trimmed mean, and every distance-based scorer read
the parameter vector.  If (S1)-(S2) move that vector without moving the
function, then the trust score is measuring a quantity that is not a property
of the client's model at all.

This script tests, on an actually-trained network (real SGD, no hand-placed
weights):

  A1.1  Is the function preserved exactly?
  A1.2  Is the parameter distance unbounded in the gauge parameter?
  A1.3  Can an adversary dial its Krum / cosine / L2 rank at will?
  A1.4  What does it cost the aggregate?  (the attack is only real if cheap)
"""

import numpy as np

rng_global = np.random.default_rng(0)

D_IN, H1, H2 = 20, 32, 32


# --------------------------------------------------------------- model ----
def init_net(rng):
    """He init, 2 hidden ReLU layers, scalar output."""
    return {
        "W1": rng.normal(size=(H1, D_IN)) * np.sqrt(2.0 / D_IN),
        "b1": np.zeros(H1),
        "W2": rng.normal(size=(H2, H1)) * np.sqrt(2.0 / H1),
        "b2": np.zeros(H2),
        "W3": rng.normal(size=(1, H2)) * np.sqrt(2.0 / H2),
        "b3": np.zeros(1),
    }


def forward(p, X):
    z1 = X @ p["W1"].T + p["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ p["W2"].T + p["b2"]
    a2 = np.maximum(z2, 0.0)
    return (a2 @ p["W3"].T + p["b3"]).ravel(), (z1, a1, z2, a2)


def grads(p, X, y):
    n = len(y)
    out, (z1, a1, z2, a2) = forward(p, X)
    d = (out - y) / n
    g = {}
    g["W3"] = d[None, :] @ a2
    g["b3"] = np.array([d.sum()])
    da2 = np.outer(d, p["W3"].ravel())
    dz2 = da2 * (z2 > 0)
    g["W2"] = dz2.T @ a1
    g["b2"] = dz2.sum(axis=0)
    da1 = dz2 @ p["W2"]
    dz1 = da1 * (z1 > 0)
    g["W1"] = dz1.T @ X
    g["b1"] = dz1.sum(axis=0)
    return g, np.mean((out - y) ** 2)


def sgd(p, X, y, steps=200, lr=0.05, batch=32, rng=None):
    p = {k: v.copy() for k, v in p.items()}
    n = len(y)
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        g, _ = grads(p, X[idx], y[idx])
        for k in p:
            p[k] -= lr * g[k]
    return p


def flat(p):
    return np.concatenate([p[k].ravel() for k in sorted(p)])


# ---------------------------------------------------------- symmetries ----
def gauge_scale(p, c1, c2):
    """(S1) exact function-preserving rescaling of layers 1 and 2."""
    q = {k: v.copy() for k, v in p.items()}
    q["W1"] *= c1
    q["b1"] *= c1
    q["W2"] *= c2 / c1
    q["b2"] *= c2
    q["W3"] /= c2
    return q


def gauge_permute(p, rng):
    """(S2) exact function-preserving permutation of both hidden layers."""
    q = {k: v.copy() for k, v in p.items()}
    s1 = rng.permutation(H1)
    s2 = rng.permutation(H2)
    q["W1"] = q["W1"][s1]
    q["b1"] = q["b1"][s1]
    q["W2"] = q["W2"][s2][:, s1]
    q["b2"] = q["b2"][s2]
    q["W3"] = q["W3"][:, s2]
    return q


# ------------------------------------------------------------- scorers ----
def krum_score(V, i):
    """Sum of squared distances to the n-k-2 nearest neighbours."""
    dist = np.sum((V - V[i]) ** 2, axis=1)
    dist = np.sort(dist)[1:]
    m = max(1, len(V) - 4)
    return dist[:m].sum()


def l2_to_mean(V, i):
    return np.linalg.norm(V[i] - V.mean(axis=0))


def cos_to_mean(V, i):
    m = V.mean(axis=0)
    return float(V[i] @ m / (np.linalg.norm(V[i]) * np.linalg.norm(m) + 1e-30))


# ---------------------------------------------------------------- data ----
def make_federation(N=30, n_per=200, seed=0):
    """Shared teacher, per-client data, shared global init, real local SGD."""
    r = np.random.default_rng(seed)
    teacher = init_net(r)
    g0 = init_net(r)
    clients = []
    for i in range(N):
        X = r.normal(size=(n_per, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.1
        clients.append(sgd(g0, X, y, steps=200, lr=0.05, rng=r))
    Xp = r.normal(size=(500, D_IN))          # probe set for function space
    return clients, g0, Xp


# =========================================================== A1.1 =========
def a1_1_function_preserved():
    print("=" * 74)
    print("A1.1  are the symmetries EXACT on a trained network?")
    print("=" * 74)
    clients, g0, Xp = make_federation()
    p = clients[0]
    f0 = forward(p, Xp)[0]
    r = np.random.default_rng(7)
    print(f"{'transform':>28} {'max |df|':>12} {'rel param move':>16}")
    for name, q in [
        ("scale c1=2, c2=1", gauge_scale(p, 2.0, 1.0)),
        ("scale c1=10, c2=1", gauge_scale(p, 10.0, 1.0)),
        ("scale c1=0.1, c2=1", gauge_scale(p, 0.1, 1.0)),
        ("scale c1=5, c2=5", gauge_scale(p, 5.0, 5.0)),
        ("permute hidden units", gauge_permute(p, r)),
        ("permute + scale", gauge_permute(gauge_scale(p, 3.0, 2.0), r)),
    ]:
        f1 = forward(q, Xp)[0]
        dmax = np.abs(f1 - f0).max()
        move = np.linalg.norm(flat(q) - flat(p)) / np.linalg.norm(flat(p))
        print(f"{name:>28} {dmax:12.2e} {move:16.3f}")
    print("\n  Function identical to floating-point noise; parameter vector moves")
    print("  by up to several times its own norm.  The symmetry is exact.\n")


# =========================================================== A1.2 =========
def a1_2_unbounded():
    print("=" * 74)
    print("A1.2  is parameter distance UNBOUNDED along the gauge orbit?")
    print("=" * 74)
    clients, g0, Xp = make_federation()
    p = clients[0]
    f0 = forward(p, Xp)[0]
    print(f"{'c1':>10} {'||dtheta||':>14} {'max |df|':>12}")
    for c1 in [1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 64.0]:
        q = gauge_scale(p, c1, 1.0)
        print(f"{c1:10.1f} {np.linalg.norm(flat(q) - flat(p)):14.4f} "
              f"{np.abs(forward(q, Xp)[0] - f0).max():12.2e}")
    print("\n  Distance grows without bound at exactly zero functional cost.")
    print("  No distance-based scorer can be robust to this.\n")


# =========================================================== A1.3 =========
def a1_3_rank_dialing():
    print("=" * 74)
    print("A1.3  can the adversary DIAL its trust rank at zero functional cost?")
    print("=" * 74)
    clients, g0, Xp = make_federation()
    N = len(clients)
    base = np.array([flat(c) for c in clients])
    adv = 0

    def ranks(V):
        k = np.array([krum_score(V, i) for i in range(N)])
        l = np.array([l2_to_mean(V, i) for i in range(N)])
        c = np.array([cos_to_mean(V, i) for i in range(N)])
        # rank 1 = most trusted (lowest dist / highest cos)
        return (int(np.argsort(np.argsort(k))[adv]) + 1,
                int(np.argsort(np.argsort(l))[adv]) + 1,
                int(np.argsort(np.argsort(-c))[adv]) + 1)

    print(f"{'c1 applied by adversary':>26} {'Krum':>7} {'L2':>7} {'cos':>7} {'max|df|':>11}")
    f0 = forward(clients[adv], Xp)[0]
    for c1 in [1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 0.95, 0.8]:
        V = base.copy()
        q = gauge_scale(clients[adv], c1, 1.0)
        V[adv] = flat(q)
        rk, rl, rc = ranks(V)
        df = np.abs(forward(q, Xp)[0] - f0).max()
        print(f"{c1:26.2f} {rk:7d} {rl:7d} {rc:7d} {df:11.2e}")
    print(f"\n  (rank 1 = most trusted of {N}; the adversary's FUNCTION is identical")
    print("  in every row.)\n")


# =========================================================== A1.4 =========
def a1_4_cost_to_aggregate():
    print("=" * 74)
    print("A1.4  what does the gauge move cost the AGGREGATE?")
    print("=" * 74)
    clients, g0, Xp = make_federation()
    N = len(clients)
    r = np.random.default_rng(11)
    Xt = r.normal(size=(2000, D_IN))
    teacher_out = None

    def unflat(v, ref):
        out, o = {}, 0
        for k in sorted(ref):
            sz = ref[k].size
            out[k] = v[o:o + sz].reshape(ref[k].shape)
            o += sz
        return out

    base = np.array([flat(c) for c in clients])
    agg_clean = unflat(base.mean(axis=0), clients[0])
    f_clean = forward(agg_clean, Xt)[0]

    print(f"{'c1':>8} {'Krum rank':>11} {'agg MSE vs clean agg':>22} {'rel':>9}")
    for c1 in [1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0]:
        V = base.copy()
        V[0] = flat(gauge_scale(clients[0], c1, 1.0))
        agg = unflat(V.mean(axis=0), clients[0])
        f = forward(agg, Xt)[0]
        err = np.mean((f - f_clean) ** 2)
        rk = int(np.argsort(np.argsort(
            np.array([krum_score(V, i) for i in range(N)])))[0]) + 1
        print(f"{c1:8.2f} {rk:11d} {err:22.6f} {err/np.var(f_clean):9.4f}")
    print("\n  If rank improves faster than aggregate error grows, the attack is")
    print("  cheap and the scorer is measuring the wrong thing.\n")


if __name__ == "__main__":
    a1_1_function_preserved()
    a1_2_unbounded()
    a1_3_rank_dialing()
    a1_4_cost_to_aggregate()
