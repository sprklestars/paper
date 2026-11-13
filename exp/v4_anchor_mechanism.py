"""
V4: why anchors do not reduce eps, and what the variance floor must be scaled to.

V3.2 found eps flat in a over a = 1..64, contradicting eq:firstorder. Two
candidate explanations:

  (H1) the test was wrong -- eq:firstorder describes displacement under a
       FRAME-SHIFT adversary, and V3.2 ran a VARIANCE-MASQUERADE adversary.
  (H2) the claim is wrong -- anchors fix the reference frame (Thm 2) but do
       nothing about variance/bias conflation (Thm 1), so eps under TSP-Var
       is genuinely a-independent.

These predict different things, so they are separable. Run both adversaries
against the same anchor sweep.

V3.3 also found M1 inert at floor=1e-2. Here the per-round dispersion is
O(sigma^2) ~ 0.12, whereas in the E3 regression harness updates are scaled by
lr and dispersion is O(1e-3). So an absolute floor is not scale-free. Test
the floor expressed as a fraction of the median dispersion.
"""

import numpy as np

d, N = 20, 30


def sim(adversary, a=0, lam=1.0, floor_frac=0.0, k=1, s_adv=0.35,
        delta=0.05, tau=0.0, rounds=300, rho=0.2, s_anch=0.3, seed=0):
    """
    adversary = "var"   : low apparent variance, small bias  (TSP-Var)
    adversary = "shift" : honest variance, large offset tau  (TSP-Shift)
    adversary = "none"  : all honest
    floor_frac: variance floor as a FRACTION of the running median dispersion.
    """
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.5, 1.5, size=N)
    off = np.zeros((N, d))
    if adversary == "var":
        sig[:k] = s_adv
        off[:k, 0] = delta
    elif adversary == "shift":
        off[:k, 0] = tau
    ev = np.ones(N)
    W = []
    for t in range(rounds):
        V = rng.normal(size=(N, d)) * sig[:, None] + off
        fl = floor_frac * np.median(ev) if floor_frac > 0 else 0.0
        iv = 1.0 / np.maximum(ev, max(fl, 1e-12))
        w = iv / iv.sum()
        mu_p = (w[:, None] * V).sum(0)
        if a > 0:
            mu_a = rng.normal(size=(a, d)).mean(0) * s_anch
            mu = lam * mu_p + (1 - lam) * mu_a
        else:
            mu = mu_p
        inst = np.linalg.norm(V - mu, axis=1) ** 2 / d
        ev = inst if t == 0 else (1 - rho) * ev + rho * inst
        W.append(w)
    W = np.mean(W[-100:], axis=0)
    return W


fair = 1.0 / N


def eps_of(W, k=1):
    return (W[:k].sum() - k * fair) / (1 - k * fair)


print("=" * 80)
print("V4.1  H1 vs H2: does the anchor count matter for EITHER adversary?")
print("=" * 80)
print(f"  lam=0.5, no floor.  fair share = {fair:.4f}\n")
print(f"{'a':>4} | {'TSP-Var eps':>12} {'x fair':>8} | "
      f"{'TSP-Shift eps':>14} {'x fair':>8}")
for a in [0, 1, 2, 4, 8, 16, 32, 64]:
    Wv = sim("var", a=a, lam=0.5, seed=11)
    Ws = sim("shift", a=a, lam=0.5, tau=2.0, seed=11)
    print(f"{a:4d} | {eps_of(Wv):12.4f} {Wv[0] / fair:7.2f}x | "
          f"{eps_of(Ws):14.4f} {Ws[0] / fair:7.2f}x")

print("""
  TSP-Var: eps is flat in a. TSP-Shift: the adversary is already BELOW its
  fair share (self-defeating, as V2 showed), and anchors change little.
  H2 is supported: anchors address Theorem 2, not Theorem 1. eq:firstorder's
  1/sqrt(a) term describes the frame-shift budget, which is not the binding
  constraint under variance masquerading.""")

print()
print("=" * 80)
print("V4.2  What anchors DO fix: the common-mode bias of Theorem 2")
print("=" * 80)
print("  Population-wide bias c injected. Under lam=1 it is invisible (Thm 2).")
print("  Measure: does the aggregate track the TRUE optimum (origin)?\n")


def commonmode(a, lam, c_norm=3.0, rounds=300, rho=0.2, s_anch=0.3, seed=5):
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.5, 1.5, size=N)
    c = np.zeros(d); c[0] = c_norm
    ev = np.ones(N); E = []
    for t in range(rounds):
        V = rng.normal(size=(N, d)) * sig[:, None] + c    # everyone biased by c
        iv = 1.0 / ev
        w = iv / iv.sum()
        mu_p = (w[:, None] * V).sum(0)
        if a > 0:
            mu_a = rng.normal(size=(a, d)).mean(0) * s_anch   # anchors unbiased
            mu = lam * mu_p + (1 - lam) * mu_a
        else:
            mu = mu_p
        inst = np.linalg.norm(V - mu, axis=1) ** 2 / d
        ev = inst if t == 0 else (1 - rho) * ev + rho * inst
        E.append(np.linalg.norm(mu))          # distance of frame from truth
    return float(np.mean(E[-100:]))


print(f"{'a':>4} {'lam':>6} | {'||frame - truth||':>18}  (truth = 0, bias = 3.0)")
for a, lam in [(0, 1.0), (3, 1.0), (3, 0.75), (3, 0.5), (3, 0.25), (3, 0.0),
               (16, 0.5), (64, 0.5)]:
    print(f"{a:4d} {lam:6.2f} | {commonmode(a, lam):18.4f}")

print("""
  At lam=1 the frame sits on the biased consensus and the common-mode bias is
  invisible, exactly as Theorem 2 predicts. Only lam<1 pulls the frame back,
  and the residual error falls with a. THIS is where the anchor budget binds.""")

print()
print("=" * 80)
print("V4.3  M1 must be scaled: floor as a fraction of median dispersion")
print("=" * 80)
print("  An absolute floor is not scale-free -- it depends on the learning")
print("  rate and the update magnitude. A relative floor is.\n")
print(f"{'floor / median disp':>21} | {'adv share':>10} {'x fair':>8} "
      f"{'eps':>8} | {'honest MSE cost':>16}")
for ff in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    Wa = sim("var", a=0, lam=1.0, floor_frac=ff, seed=11)
    Wh = sim("none", a=0, lam=1.0, floor_frac=ff, seed=11)
    # utility: effective variance of the aggregate on an all-honest population
    rng = np.random.default_rng(77)
    sg = np.random.default_rng(11).uniform(0.5, 1.5, size=N)
    mse = float((Wh ** 2 * sg ** 2).sum())
    mse_opt = float(1.0 / (1.0 / sg ** 2).sum())
    lbl = "none" if ff == 0 else f"{ff:.2f}"
    print(f"{lbl:>21} | {Wa[0]:10.4f} {Wa[0] / fair:7.2f}x "
          f"{eps_of(Wa):8.4f} | {mse / mse_opt:15.3f}x")

print("""
  Column 5 is the aggregate's variance relative to the BLUE optimum on an
  all-honest population: the price of disbelieving low variance estimates.
  A relative floor removes the capture and the cost is bounded.""")

print()
print("=" * 80)
print("V4.4  thm:displacement at a=32, lam=0.25 -- real violation or MC noise?")
print("=" * 80)


def disp_min(a, lam, seeds=24, k=1, s_adv=0.35, delta=0.05,
             rounds=300, rho=0.2, s_anch=0.3):
    out = []
    for s in range(seeds):
        Wa = sim("var", a=a, lam=lam, k=k, s_adv=s_adv, delta=delta,
                 rounds=rounds, rho=rho, s_anch=s_anch, seed=100 + s)
        Wc = sim("none", a=a, lam=lam, rounds=rounds, rho=rho,
                 s_anch=s_anch, seed=100 + s)
        out.append(np.min(Wa[k:] / Wc[k:]))
    return np.array(out)


z, sA, sH = 3.5, 0.3, 1.0
print(f"{'a':>4} {'lam':>6} | {'mean ratio':>11} {'min over seeds':>15} "
      f"{'bound':>8} {'violations':>11}")
for a, lm in [(1, 1.0), (3, 0.5), (32, 0.25), (64, 0.25), (64, 0.1)]:
    tau = z * sA / np.sqrt(a)
    bound = (1.0 / (1 + lm * tau / sH)) ** 2 / (1 + 1.0 / N)
    r = disp_min(a, lm)
    print(f"{a:4d} {lm:6.2f} | {r.mean():11.4f} {r.min():15.4f} "
          f"{bound:8.4f} {int((r < bound).sum()):5d}/{len(r)}")

print("""
  Violations here are genuine: the bound's Pi_max term assumes a variance
  floor is in force, and these runs have none. The bound should be stated
  as conditional on M1, not as unconditional.""")
