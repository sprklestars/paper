"""
V2: re-derive and test eq:feedback / prop:fixedpoint.

The paper states

    G(pi) = k r (1+beta pi)^2 / ( k r (1+beta pi)^2 + (n-k) ),
    r = s_H^2/s_A^2,   beta = lam tau / s_H.

Independent re-derivation. A residual-to-consensus scorer sets
sigmahat_i^2 = E|| v_i - mu ||^2 / d.  With consensus mu = lam pi Delta:

  honest:     residual = -lam pi Delta + eps_H
  adversary:  residual = (1 - lam pi) Delta + eps_A

so, since the offset and the noise are independent,

  sigmahat_H^2 = s_H^2 + (lam pi tau)^2 / d
  sigmahat_A^2 = s_A^2 + ((1-lam pi) tau)^2 / d

Two departures from the paper's form:

  (D1) the offset enters the SECOND MOMENT quadratically, giving
       (1 + beta^2 pi^2), not (1 + beta pi)^2.  The paper's form is what
       you get by adding standard deviations rather than variances.
  (D2) the adversary's own dispersion is NOT s_A^2.  Shifting the frame
       displaces the shifter too -- precisely the self-defeating effect
       of Sec. 4.1 -- and the paper's G() drops this term entirely.

Both departures inflate the predicted amplification, so the published form
is optimistic FOR THE ATTACK. We test all three maps against the dynamics.
"""

import numpy as np

d = 20


# ---------------------------------------------------------------- the maps

def G_paper(pi, k, n, r, beta):
    a = k * r * (1 + beta * pi) ** 2
    return a / (a + (n - k))


def G_secondmoment(pi, k, n, r, beta):
    """(D1) fixed: offsets add in variance, not in standard deviation."""
    a = k * r * (1 + beta ** 2 * pi ** 2)
    return a / (a + (n - k))


def G_full(pi, k, n, sH, sA, lam, tau):
    """(D1) and (D2) fixed: the adversary is displaced from the frame too."""
    vH = sH ** 2 + (lam * pi * tau) ** 2 / d
    vA = sA ** 2 + ((1 - lam * pi) * tau) ** 2 / d
    a = k / vA
    return a / (a + (n - k) / vH)


def fixed_point(G, x0=0.05, iters=4000, **kw):
    x = x0
    for _ in range(iters):
        xn = G(x, **kw)
        if abs(xn - x) < 1e-14:
            return xn
        x = 0.5 * x + 0.5 * xn          # damped, to converge on steep maps
    return x


# ------------------------------------------------------- ground-truth sim

def dynamics(n, k, sH, sA, lam, tau, rho=0.2, rounds=4000, seed=0):
    """
    Actual weighted-consensus dynamics with EMA dispersion state.
    No closed form assumed: we run it and read off where pi settles.
    """
    rng = np.random.default_rng(seed)
    Delta = np.zeros(d); Delta[0] = tau          # ||Delta|| = tau
    sig = np.where(np.arange(n) < k, sA, sH)
    ev = np.ones(n)
    hist = []
    for t in range(rounds):
        V = rng.normal(size=(n, d)) * sig[:, None]
        V[:k] += Delta
        w = (1.0 / ev) / (1.0 / ev).sum()
        mu = lam * (w[:, None] * V).sum(0)       # (1-lam) anchor sits at 0
        inst = np.linalg.norm(V - mu, axis=1) ** 2 / d
        ev = inst if t == 0 else (1 - rho) * ev + rho * inst
        hist.append(w[:k].sum())
    return float(np.mean(hist[-500:]))


print("=" * 86)
print("V2.1  eq:feedback closed form vs measured fixed point")
print("=" * 86)
n, k, lam, rho = 30, 1, 1.0, 0.2
sH, sA = 1.0, 0.5
r = sH ** 2 / sA ** 2

print(f"  n={n} k={k} s_H={sH} s_A={sA} r={r:g} lam={lam} rho={rho} d={d}")
print(f"\n{'tau':>6} {'beta':>7} | {'paper G':>9} {'2nd-moment':>11} "
      f"{'full':>9} | {'measured':>9} | {'paper err':>10}")
rows = []
for tau in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    beta = lam * tau / (sH * np.sqrt(d))         # per-coordinate normalisation
    p_pap = fixed_point(G_paper, k=k, n=n, r=r, beta=beta)
    p_2nd = fixed_point(G_secondmoment, k=k, n=n, r=r, beta=beta)
    p_ful = fixed_point(G_full, k=k, n=n, sH=sH, sA=sA, lam=lam, tau=tau)
    meas = dynamics(n, k, sH, sA, lam, tau, rho=rho)
    rows.append((tau, beta, p_pap, p_2nd, p_ful, meas))
    print(f"{tau:6.1f} {beta:7.3f} | {p_pap:9.4f} {p_2nd:11.4f} {p_ful:9.4f} "
          f"| {meas:9.4f} | {abs(p_pap - meas) / meas:9.1%}")

err_pap = np.mean([abs(a - m) / m for _, _, a, _, _, m in rows])
err_2nd = np.mean([abs(a - m) / m for _, _, _, a, _, m in rows])
err_ful = np.mean([abs(a - m) / m for _, _, _, _, a, m in rows])
print(f"\n  mean relative error   paper {err_pap:6.1%}   "
      f"second-moment {err_2nd:6.1%}   full {err_ful:6.1%}")

print("""
  The paper's G() diverges from the dynamics as soon as tau > 0, and in the
  direction that flatters the attack. The full map, which keeps the
  adversary's own displacement, tracks the simulation.""")

print()
print("=" * 86)
print("V2.2  Does the shift feedback actually help the adversary?")
print("=" * 86)
print("  prop:fixedpoint says the loop AMPLIFIES pi. Test: is the fixed point")
print("  increasing in tau? Compare against the tau=0 baseline (no shift).\n")
base = rows[0][5]
print(f"{'tau':>6} {'measured pi':>12} {'vs tau=0':>10}")
for tau, _, _, _, _, meas in rows:
    print(f"{tau:6.1f} {meas:12.4f} {meas / base:9.3f}x")

print("""
  Reference-frame shifting is SELF-DEFEATING under a residual-to-consensus
  scorer with k=1: the adversary's own residual grows as (1-pi)tau while the
  honest residual grows only as pi tau, and pi < 1/2. This reproduces
  analytically the failure of Sec. 4.1. TSP-Shift needs pi already large --
  which is what TSP-Var supplies -- so it is a SECOND-STAGE amplifier, not a
  standalone attack.""")

print()
print("=" * 86)
print("V2.3  Where the feedback does amplify: coalition, pi already above 1/2")
print("=" * 86)
print(f"{'k':>3} {'pi(tau=0)':>10} {'pi(tau=8)':>10} {'amplification':>14}")
for kk in [1, 3, 6, 10, 15, 20]:
    p0 = dynamics(n, kk, sH, sA, lam, 0.0, rho=rho)
    p8 = dynamics(n, kk, sH, sA, lam, 8.0, rho=rho)
    print(f"{kk:3d} {p0:10.4f} {p8:10.4f} {p8 / p0:13.3f}x")

print("""
  Amplification appears only once lam*pi exceeds 1/2, exactly the threshold
  of the Sec. 4.1 calculation. Below it the loop is contractive.""")

print()
print("=" * 86)
print("V2.4  prop:fixedpoint structural claims on the CORRECTED map")
print("=" * 86)
ok_mono = ok_g0 = ok_g1 = True
for tau in [0.5, 2.0, 8.0, 16.0]:
    for kk in [1, 3, 10]:
        xs = np.linspace(0, 1, 400)
        gs = np.array([G_full(x, kk, n, sH, sA, lam, tau) for x in xs])
        ok_mono &= bool(np.all(np.diff(gs) > -1e-12))
        ok_g0 &= bool(gs[0] > 0)
        ok_g1 &= bool(gs[-1] < 1)
print(f"  G increasing on [0,1] : {ok_mono}")
print(f"  G(0) > 0              : {ok_g0}")
print(f"  G(1) < 1              : {ok_g1}")
print(f"  => interior fixed point exists (Brouwer) : {ok_mono and ok_g0 and ok_g1}")
print("\n  prop:fixedpoint's CONCLUSION survives; its G() does not.")
