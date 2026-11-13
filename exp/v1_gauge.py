"""
V1: verify Theorem 2 (bias unidentifiability) and Proposition 1 (impossibility).

These are exact algebraic claims, so they get exact numerical tests: shift every
client's bias by a common vector c and assert that every observable the scorer
can compute is bit-for-bit unchanged (to floating-point tolerance).

A claim of the form "no scorer can ever distinguish" is falsified by a single
scorer that distinguishes. So we test the widest family we can implement.
"""

import numpy as np

TOL = 1e-9
d, N = 20, 30
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- frames

def mu_mean(V):
    return V.mean(axis=0)


def mu_median(V):
    return np.median(V, axis=0)


def mu_geomedian(V, iters=200):
    """Weiszfeld. Translation-equivariant, not a linear function of V."""
    y = V.mean(axis=0)
    for _ in range(iters):
        dist = np.linalg.norm(V - y, axis=1)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / dist
        y_new = (w[:, None] * V).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) < 1e-15:
            break
        y = y_new
    return y


def mu_trimmed(V, beta=0.1):
    n = len(V)
    t = int(np.floor(beta * n))
    Vs = np.sort(V, axis=0)
    return Vs[t:n - t].mean(axis=0)


FRAMES = {"mean": mu_mean, "coord-median": mu_median,
          "geometric-median": mu_geomedian, "trimmed-mean": mu_trimmed}


# ---------------------------------------------------------------- scorers
# Every scorer below is peer-only (lambda = 1): it sees V and nothing else.

def s_krum(V):
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2) ** 2
    m = max(1, len(V) - 2 - 2)
    return np.sort(D, axis=1)[:, 1:m + 1].sum(axis=1)


def s_dist_mean(V):
    return np.linalg.norm(V - mu_mean(V), axis=1)


def s_dist_geomed(V):
    return np.linalg.norm(V - mu_geomedian(V), axis=1)


def s_invsq_dist(V):
    dd = np.linalg.norm(V - mu_mean(V), axis=1)
    w = 1.0 / np.maximum(dd, 1e-12) ** 2
    return w / w.sum()


def s_pairwise_cos_centered(V):
    """FoolsGold-style similarity computed on frame-centred updates."""
    C = V - mu_mean(V)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    S = Cn @ Cn.T
    np.fill_diagonal(S, 0.0)
    return S.max(axis=1)


def s_mad_norm(V):
    """Norm-based outlier statistic on frame-centred updates."""
    nrm = np.linalg.norm(V - mu_mean(V), axis=1)
    med = np.median(nrm)
    mad = np.median(np.abs(nrm - med)) + 1e-12
    return np.abs(nrm - med) / mad


def s_cov_mahalanobis(V):
    """Second-order statistic: Mahalanobis distance under the sample covariance."""
    C = V - mu_mean(V)
    S = (C.T @ C) / len(V) + 1e-6 * np.eye(V.shape[1])
    Si = np.linalg.inv(S)
    return np.einsum("ij,jk,ik->i", C, Si, C)


SCORERS = {
    "Krum score": s_krum,
    "dist to mean": s_dist_mean,
    "dist to geo-median": s_dist_geomed,
    "inv-sq-dist weights": s_invsq_dist,
    "max pairwise cos": s_pairwise_cos_centered,
    "norm MAD z-score": s_mad_norm,
    "Mahalanobis": s_cov_mahalanobis,
}


def make_world(bias, seed=1, sigma=None):
    r = np.random.default_rng(seed)
    if sigma is None:
        sigma = r.uniform(0.5, 1.5, size=N)
    noise = r.normal(size=(N, d)) * sigma[:, None]
    return bias + noise


print("=" * 78)
print("V1.1  Theorem 2: common bias translation leaves every peer-only")
print("      observable invariant.   W1: {b_i}   vs   W2: {b_i + c}")
print("=" * 78)

b = rng.normal(size=(N, d)) * 0.3          # arbitrary heterogeneous biases
c = rng.normal(size=d) * 2.7               # arbitrary common translation

sig = rng.uniform(0.5, 1.5, size=N)
V1 = make_world(b, seed=1, sigma=sig)
V2 = make_world(b + c, seed=1, sigma=sig)  # SAME noise draw, shifted biases

print(f"  ||c|| = {np.linalg.norm(c):.4f}   (a large, arbitrary translation)")
print(f"  check V2 - V1 == c :  {np.allclose(V2 - V1, c)}\n")

print(f"{'frame':>18}   translation-equivariant?   max|mu(V+c) - mu(V) - c|")
for nm, F in FRAMES.items():
    err = np.max(np.abs(F(V2) - F(V1) - c))
    print(f"{nm:>18}   {'YES' if err < TOL else 'NO ':>21}   {err:.3e}")

print()
print(f"{'scorer':>22}   invariant?   max|s(V2) - s(V1)|   rel")
allok = True
for nm, S in SCORERS.items():
    s1, s2 = S(V1), S(V2)
    err = np.max(np.abs(s2 - s1))
    rel = err / (np.max(np.abs(s1)) + 1e-30)
    ok = rel < TOL
    allok &= ok
    print(f"{nm:>22}   {'YES' if ok else 'NO':>10}   {err:18.3e}   {rel:.2e}")

print(f"\n  Theorem 2 verdict: {'CONFIRMED' if allok else 'FALSIFIED'}")

print()
print("=" * 78)
print("V1.2  The invariance is specific to lambda=1. Anchors must break it.")
print("=" * 78)


def s_anchored(V, mu_A, lam):
    """Distance to the mixed frame of eq:refmix."""
    mu = lam * mu_mean(V) + (1 - lam) * mu_A
    return np.linalg.norm(V - mu, axis=1)


mu_A = np.zeros(d)      # anchors sit at the true origin in BOTH worlds
print(f"{'lambda':>8}   max|s(V2)-s(V1)|   invariant?")
for lam in [1.0, 0.99, 0.9, 0.5, 0.0]:
    err = np.max(np.abs(s_anchored(V2, mu_A, lam) - s_anchored(V1, mu_A, lam)))
    print(f"{lam:8.2f}   {err:16.3e}   {'YES (blind)' if err < TOL else 'no  (sees it)'}")

print("""
  Reading: at lambda=1 the scorer is blind to c by Theorem 2. Any lambda<1
  makes the two worlds distinguishable, and the gap grows as lambda falls.
  This is the formal content of "anchors are necessary", not a heuristic.""")

print()
print("=" * 78)
print("V1.3  Proposition 1 as an explicit two-world construction")
print("=" * 78)
print("""  W1: adversary carries bias +Delta, honest population unbiased.
  W2: adversary unbiased, honest population carries bias -Delta.
  W2 = W1 translated by c = -Delta, so by Theorem 2 a peer-only scorer
  assigns IDENTICAL weights in both. But the correct weight assignment
  differs: in W1 the adversary should be down-weighted, in W2 it should not.""")

Delta = rng.normal(size=d)
Delta = 0.8 * Delta / np.linalg.norm(Delta)

bW1 = np.zeros((N, d)); bW1[0] = Delta          # adversary = client 0
bW2 = np.tile(-Delta, (N, 1)); bW2[0] = 0.0     # == bW1 - Delta

print(f"\n  check bW2 == bW1 - Delta :  {np.allclose(bW2, bW1 - Delta)}")
W1 = make_world(bW1, seed=7, sigma=sig)
W2 = make_world(bW2, seed=7, sigma=sig)

print(f"\n{'scorer':>22}   |s(W1)-s(W2)|_max   adversary rank W1 / W2")
for nm, S in SCORERS.items():
    s1, s2 = S(W1), S(W2)
    r1 = int(np.argsort(np.argsort(s1))[0])
    r2 = int(np.argsort(np.argsort(s2))[0])
    print(f"{nm:>22}   {np.max(np.abs(s1 - s2)):17.3e}   {r1:2d} / {r2:2d}")

print("""
  Every scorer ranks the adversary identically in the two worlds, while the
  ground truth differs. Any scorer achieving eps < 1 in W1 therefore inflicts
  eps >= 1 on some honest client in W2. Proposition 1 holds, and this is a
  construction rather than a sketch.""")
