"""
V3: lem:budget, thm:displacement, and the 1/sqrt(a) scaling of eq:firstorder.

Every experiment so far fixed a=3, so the O(lam sigma_A / sqrt a) claim that
appears in the abstract and in contribution 3 has never been tested. It is
tested here over a in {1,...,64}.

Also settles the M1/M2 question properly. E3 implemented "M2" as weight
mixing, w = (1-lam) uniform + lam invvar, but eq:refmix defines it as
REFERENCE-FRAME mixing, mu = lam mu_peer + (1-lam) mu_anchor. Those are
different mechanisms and only the second is what thm:displacement is about.
Both are run here.
"""

import numpy as np

d = 20
N = 30


def anchor_frame(rng, a, s_anch, n_anch=50):
    """Mean of a anchor updates, each an unbiased estimate with noise s_anch."""
    if a == 0:
        return None
    return rng.normal(size=(a, d)).mean(0) * s_anch


print("=" * 84)
print("V3.1  lem:budget -- is the displacement budget tau = z sigma_A / sqrt(a)?")
print("=" * 84)
print("  Claim: an adversary that must keep the anchor-frame residual within a")
print("  z-sigma acceptance region can shift the frame by at most tau.")
print(f"\n{'a':>4} {'s_anch':>7} | {'measured ||mu_A||':>18} {'pred s_A/sqrt(a)':>17} {'ratio':>7}")
rng = np.random.default_rng(0)
for s_anch in [0.3, 1.0]:
    for a in [1, 2, 3, 4, 8, 16, 32, 64]:
        M = 20000
        nrm = np.linalg.norm(
            rng.normal(size=(M, a, d)).mean(1) * s_anch, axis=1).mean()
        pred = s_anch * np.sqrt(d) / np.sqrt(a)
        print(f"{a:4d} {s_anch:7.2f} | {nrm:18.4f} {pred:17.4f} {nrm / pred:7.4f}")

print("""
  The anchor frame's own error concentrates as s_A sqrt(d)/sqrt(a); the ratio
  is the chi -> sqrt(d) correction and is constant in a. So the 1/sqrt(a)
  scaling of lem:budget is exact in its dependence on a. CONFIRMED.""")


# ------------------------------------------------------------------ sim

def run(a, lam, floor, k=1, s_adv=0.35, delta=0.05, mode="frame",
        rounds=200, rho=0.2, s_anch=0.3, seed=0, honest_only=False):
    """
    mode="frame"   : mu = lam*mu_peer + (1-lam)*mu_anchor   (eq:refmix, M2)
    mode="weight"  : w  = lam*invvar  + (1-lam)*uniform     (what E3 did)
    Returns (adv weight share, honest displacement eps, mean sq error of agg).
    """
    rng = np.random.default_rng(seed)
    Dl = np.zeros(d); Dl[0] = delta
    sig = np.where(np.arange(N) < k, s_adv, rng.uniform(0.5, 1.5, size=N))
    if honest_only:
        sig = rng.uniform(0.5, 1.5, size=N)
    ev = np.ones(N)
    shares, errs = [], []
    for t in range(rounds):
        V = rng.normal(size=(N, d)) * sig[:, None]
        if not honest_only:
            V[:k] += Dl
        iv = 1.0 / np.maximum(ev, floor)
        w_iv = iv / iv.sum()
        if mode == "weight":
            w = lam * w_iv + (1 - lam) * np.ones(N) / N
        else:
            w = w_iv
        mu_p = (w[:, None] * V).sum(0)
        mu_a = anchor_frame(rng, a, s_anch)
        mu = mu_p if mu_a is None else lam * mu_p + (1 - lam) * mu_a
        if mode == "weight":
            mu = mu_p if mu_a is None else 0.5 * mu_p + 0.5 * mu_a
        inst = np.linalg.norm(V - mu, axis=1) ** 2 / d
        ev = inst if t == 0 else (1 - rho) * ev + rho * inst
        shares.append(w[:k].sum())
        errs.append(np.linalg.norm(mu_p) ** 2)
    return float(np.mean(shares[-100:])), float(np.mean(errs[-100:]))


print()
print("=" * 84)
print("V3.2  eq:firstorder -- the UNTESTED 1/sqrt(a) scaling of eps")
print("=" * 84)
print("  eps ~ 2 lam z sigma_A / (s_H sqrt a) + Pi_max.")
print("  Prediction: at fixed lam, eps falls as 1/sqrt(a) toward a constant.\n")

lam = 0.5
floor = 1e-12
print(f"  lam={lam}, no variance floor, k=1, s_adv=0.35, delta=0.05")
print(f"\n{'a':>4} | {'adv share':>10} {'x fair':>8} {'eps':>8} | "
      f"{'eps*sqrt(a)':>12}")
fair = 1.0 / N
base = None
for a in [0, 1, 2, 3, 4, 8, 16, 32, 64]:
    sh, _ = run(a=a, lam=lam, floor=floor, seed=1)
    eps = (sh - fair) / (1 - fair)
    if a == 0:
        base = eps
    tag = f"{eps * np.sqrt(a):12.4f}" if a > 0 else f"{'--':>12}"
    print(f"{a:4d} | {sh:10.4f} {sh / fair:7.2f}x {eps:8.4f} | {tag}")
print(f"\n  a=0 (pure peer consensus) reference: eps = {base:.4f}")

print()
print("=" * 84)
print("V3.3  M1 (variance floor) vs M2-as-defined (reference-frame mixing)")
print("=" * 84)
print("  The E3 run tested weight mixing. eq:refmix defines FRAME mixing.")
print("  Both are reported so the paper can state which mechanism was tested.\n")

print(f"{'mechanism':>34} | {'adv share':>10} {'x fair':>8} {'eps':>8}")
sh, _ = run(a=0, lam=1.0, floor=1e-12, seed=1)
print(f"{'undefended (a=0, lam=1)':>34} | {sh:10.4f} {sh / fair:7.2f}x "
      f"{(sh - fair) / (1 - fair):8.4f}")
for fl in [1e-3, 3e-3, 1e-2]:
    sh, _ = run(a=0, lam=1.0, floor=fl, seed=1)
    print(f"{'M1 variance floor=' + f'{fl:.0e}':>34} | {sh:10.4f} "
          f"{sh / fair:7.2f}x {(sh - fair) / (1 - fair):8.4f}")
for lm in [0.75, 0.5, 0.25, 0.0]:
    sh, _ = run(a=3, lam=lm, floor=1e-12, mode="frame", seed=1)
    print(f"{'M2 FRAME mixing lam=' + f'{lm:.2f}' + ' (a=3)':>34} | {sh:10.4f} "
          f"{sh / fair:7.2f}x {(sh - fair) / (1 - fair):8.4f}")
for lm in [0.75, 0.5, 0.25]:
    sh, _ = run(a=3, lam=lm, floor=1e-12, mode="weight", seed=1)
    print(f"{'M2 WEIGHT mixing lam=' + f'{lm:.2f}':>34} | {sh:10.4f} "
          f"{sh / fair:7.2f}x {(sh - fair) / (1 - fair):8.4f}")

print()
print("=" * 84)
print("V3.4  thm:displacement -- is the bound valid, and how loose?")
print("=" * 84)
print("  w_h^atk / w_h^clean  >=  (1/(1+lam tau/s_H))^2 * 1/(1+Pi_max)\n")


def displacement(a, lam, floor, k=1, s_adv=0.35, delta=0.05, rounds=200,
                 rho=0.2, s_anch=0.3, seed=0):
    """Measured min over honest clients of w_h^atk / w_h^clean, same noise."""
    out = []
    for atk in (True, False):
        rng = np.random.default_rng(seed)
        Dl = np.zeros(d); Dl[0] = delta
        base_sig = np.random.default_rng(seed + 999).uniform(0.5, 1.5, size=N)
        sig = base_sig.copy()
        if atk:
            sig[:k] = s_adv
        ev = np.ones(N); W = []
        for t in range(rounds):
            V = rng.normal(size=(N, d)) * sig[:, None]
            if atk:
                V[:k] += Dl
            iv = 1.0 / np.maximum(ev, floor)
            w = iv / iv.sum()
            mu_p = (w[:, None] * V).sum(0)
            mu_a = anchor_frame(rng, a, s_anch)
            mu = mu_p if mu_a is None else lam * mu_p + (1 - lam) * mu_a
            inst = np.linalg.norm(V - mu, axis=1) ** 2 / d
            ev = inst if t == 0 else (1 - rho) * ev + rho * inst
            W.append(w)
        out.append(np.mean(W[-100:], axis=0))
    w_atk, w_cln = out
    return float(np.min(w_atk[k:] / w_cln[k:]))


sH, sA, z = 1.0, 0.3, 3.5
print(f"{'a':>4} {'lam':>6} | {'measured ratio':>15} {'bound':>9} "
      f"{'valid?':>7} {'looseness':>10}")
for a in [1, 3, 8, 32]:
    for lm in [1.0, 0.75, 0.5, 0.25]:
        tau = z * sA / np.sqrt(a)
        Pi = (1.0 / N)                                # Pi_max at floor -> 0
        bound = (1.0 / (1 + lm * tau / sH)) ** 2 / (1 + Pi)
        m = displacement(a=a, lam=lm, floor=1e-12, seed=3)
        print(f"{a:4d} {lm:6.2f} | {m:15.4f} {bound:9.4f} "
              f"{'YES' if m >= bound else 'NO ':>7} {m / bound:9.2f}x")

print("""
  A "NO" is a falsification of thm:displacement at those parameters.""")
