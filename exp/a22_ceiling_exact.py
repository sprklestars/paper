"""A22: prop:ceiling's proof states an intermediate ratio that contradicts the
formula it then derives, and the formula itself is not the exact one.

The proof (04-attacks.tex) says: with sigma_adv -> 0 the adversary's residual
carries 'the frame's own sampling noise sigma_H^2/N, while an honest client's
carries sigma_H^2(1 - 1/N).  The ratio of estimated variances is therefore
1/(N-1) rather than 0, and inverse-variance weighting gives pi = N/(2N-1).'

Two problems, both checkable in closed form.

(1) Internal inconsistency.  Inverse-variance weighting over one adversary and
    N-1 honest clients gives pi = 1/(1 + (N-1) * ratio).  Substituting the
    proof's own ratio 1/(N-1) gives pi = 1/2 for EVERY N, not N/(2N-1).  The
    stated ratio cannot produce the stated formula.

(2) Both quantities are slightly off.  Writing ubar = (1/N) sum_j u_j with
    u_1 = 0 (the sigma_adv -> 0 limit):

      e_adv = Var(u_1 - ubar) = Var(ubar) = (N-1) sigma_H^2 / N^2,

    which is sigma_H^2/N times (N-1)/N, not sigma_H^2/N; and

      e_H = Var(u_i - ubar) = (1 - 1/N)^2 sigma_H^2 + (N-2) sigma_H^2 / N^2
          = (N^2 - N - 1) sigma_H^2 / N^2,

    because u_i appears inside ubar and the two are correlated.  Hence

      ratio = (N-1) / (N^2 - N - 1)   ~   1/N,   not 1/(N-1),

    and pi = (N^2-N-1) / ((N^2-N-1) + (N-1)^2).

    N/(2N-1) is what ratio = 1/N gives, so the published formula is the
    large-N approximation of the exact one and sits above it at small N.

Everything below is checked against the harness a16_ceiling_residual.py
already runs, whose 'hom' columns are the common-sigma_H case the proof
states.  If the exact form is right it must match those columns more closely
than N/(2N-1) does.
"""

import numpy as np

d = 20


def closed_exact(N):
    """pi from the exact residual variances."""
    a = N * N - N - 1.0
    return a / (a + (N - 1.0) ** 2)


def closed_published(N):
    return N / (2.0 * N - 1.0)


def closed_from_proof_ratio(N):
    """What the proof's own stated ratio 1/(N-1) actually yields."""
    return 1.0 / (1.0 + (N - 1.0) * (1.0 / (N - 1.0)))


print("=" * 78)
print("A22.1  the proof's stated ratio yields 1/2, not N/(2N-1)")
print("=" * 78)
print("  pi = 1/(1 + (N-1) * ratio).  Substituting ratio = 1/(N-1):")
print()
print(f"{'N':>6} {'proof ratio':>12} {'pi from it':>11} {'N/(2N-1)':>10} "
      f"{'agree?':>7}")
for N in [10, 20, 30, 50, 100]:
    pr = 1.0 / (N - 1.0)
    print(f"{N:6d} {pr:12.5f} {closed_from_proof_ratio(N):11.5f} "
          f"{closed_published(N):10.5f} "
          f"{str(abs(closed_from_proof_ratio(N) - closed_published(N)) < 1e-9):>7}")
print("\n  The ratio 1/(N-1) is N-independent in its effect: it gives exactly")
print("  1/2 at every N.  So the sentence's two halves contradict each other.")

print()
print("=" * 78)
print("A22.2  the residual variances, derived and then simulated")
print("=" * 78)
print("  e_adv = Var(ubar) = (N-1)/N^2;  e_H = (1-1/N)^2 + (N-2)/N^2")
print("  = (N^2-N-1)/N^2, both in units of sigma_H^2.  Simulated by direct")
print("  Monte Carlo over the same generator shape the harness uses.")
print()
print(f"{'N':>6} {'e_adv pred':>11} {'e_adv sim':>11} {'e_H pred':>10} "
      f"{'e_H sim':>10} {'max rel err':>12}")
worst = 0.0
for N in [10, 20, 30, 50, 100]:
    r = np.random.default_rng(7000 + N)
    M = 400000
    U = r.normal(size=(M, N))
    U[:, 0] = 0.0                      # sigma_adv -> 0
    ub = U.mean(axis=1)
    ea_s = float(np.var(U[:, 0] - ub))
    eh_s = float(np.mean([np.var(U[:, i] - ub) for i in range(1, N)]))
    ea_p = (N - 1.0) / N ** 2
    eh_p = (N * N - N - 1.0) / N ** 2
    e = max(abs(ea_s / ea_p - 1), abs(eh_s / eh_p - 1))
    worst = max(worst, e)
    print(f"{N:6d} {ea_p:11.6f} {ea_s:11.6f} {eh_p:10.6f} {eh_s:10.6f} "
          f"{e:12.2e}")
print(f"\n  worst relative error {worst:.2e}: the residual variances are as")
print("  derived, and e_adv is NOT sigma_H^2/N -- it is that times (N-1)/N.")
assert worst < 0.02, worst

print()
print("=" * 78)
print("A22.3  the ratio, three ways")
print("=" * 78)
print(f"{'N':>6} {'proof 1/(N-1)':>14} {'exact':>10} {'1/N':>9} "
      f"{'proof err':>10}")
for N in [10, 20, 30, 50, 100]:
    ex = (N - 1.0) / (N * N - N - 1.0)
    print(f"{N:6d} {1.0/(N-1.0):14.5f} {ex:10.5f} {1.0/N:9.5f} "
          f"{(1.0/(N-1.0))/ex - 1.0:10.1%}")
print("\n  The proof's ratio is ~10% high at N=10 and the exact ratio is close")
print("  to 1/N, which is what N/(2N-1) corresponds to.  So the FORMULA is")
print("  the large-N approximation and the stated RATIO is simply wrong.")

print()
print("=" * 78)
print("A22.4  which closed form matches the harness's common-sigma columns?")
print("=" * 78)
print("  a16_ceiling_residual.py's 'hom' columns are the common-sigma_H case")
print("  prop:ceiling states.  Those measurements, transcribed from")
print("  results/a16_ceiling_residual.py.txt, against both closed forms.")
print()
A16_HOM_999 = {10: 0.5235, 20: 0.5120, 30: 0.5082, 50: 0.5047, 100: 0.5027}
print(f"{'N':>6} {'measured':>10} {'exact':>9} {'|err|':>8} "
      f"{'N/(2N-1)':>10} {'|err|':>8}")
we, wp = 0.0, 0.0
for N, m in A16_HOM_999.items():
    ee = abs(closed_exact(N) - m)
    ep = abs(closed_published(N) - m)
    we, wp = max(we, ee), max(wp, ep)
    print(f"{N:6d} {m:10.4f} {closed_exact(N):9.4f} {ee:8.4f} "
          f"{closed_published(N):10.4f} {ep:8.4f}")
print(f"\n  worst deviation: exact {we:.4f} ({we*100:.2f} points), "
      f"published {wp:.4f} ({wp*100:.2f} points)")
print("  The exact form tracks the common-sigma measurement to under a tenth")
print("  of a point at every N; N/(2N-1) is high by up to a quarter point,")
print("  worst at small N where the approximation ratio 1/N is worst.")
assert we < wp, "expected the exact form to beat the published one"
assert we < 0.002, we

print()
print("=" * 78)
print("A22.5  end to end: the fixed point, simulated in the EMA harness")
print("=" * 78)
print("  The closed forms above are static.  This runs the actual estimator")
print("  with a common sigma_H, an unmovable mean frame and sigma_adv -> 0,")
print("  which is the configuration prop:ceiling describes.")
print()


def simulate(N, T=4000, rho=0.99, frac=1e-3, seeds=8):
    pis = []
    for s in range(seeds):
        r = np.random.default_rng(8000 + s)
        sig = np.ones(N)
        sig[0] = frac
        ema = np.ones(N)
        for _ in range(T):
            U = r.normal(size=(N, d)) * sig[:, None]
            ref = U.mean(axis=0)          # unmovable frame: plain mean
            ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
        w = (1.0 / ema) / (1.0 / ema).sum()
        pis.append(w[0])
    return float(np.mean(pis)), float(np.std(pis))


print(f"{'N':>6} {'simulated':>18} {'exact':>9} {'N/(2N-1)':>10} "
      f"{'exact err':>10} {'pub err':>9}")
we2, wp2 = 0.0, 0.0
for N in [10, 20, 30, 50, 100]:
    m, sd = simulate(N)
    ee, ep = abs(closed_exact(N) - m), abs(closed_published(N) - m)
    we2, wp2 = max(we2, ee), max(wp2, ep)
    print(f"{N:6d} {m:11.4f} +-{sd:5.4f} {closed_exact(N):9.4f} "
          f"{closed_published(N):10.4f} {ee:10.4f} {ep:9.4f}")
print(f"\n  worst: exact {we2*100:.2f} points, published {wp2*100:.2f} points")
print("  Both closed forms are within measurement noise of the simulation at")
print("  large N; they separate at small N, where the exact one wins.")

print()
print("=" * 78)
print("A22.6  does eq:ceiling's headline claim survive the correction?")
print("=" * 78)
print("  eq:ceiling asserts pi -> N/(2N-1) ~ 1/2, i.e. a single client takes")
print("  half the aggregate and no more.  The exact form must have the same")
print("  limit for the paper's conclusion to stand.")
print()
print(f"{'N':>8} {'exact pi':>10} {'x fair (pi*N)':>14}")
for N in [10, 30, 100, 1000, 10000]:
    print(f"{N:8d} {closed_exact(N):10.6f} {closed_exact(N)*N:14.2f}")
lim = closed_exact(10 ** 7)
print(f"\n  pi at N = 1e7: {lim:.8f}, so the limit is 1/2 as claimed and the")
print("  'N/2 times its entitlement' reading is unchanged.  The correction")
print("  moves the constant at small N, not the conclusion.")
assert abs(lim - 0.5) < 1e-6, lim

print()
print("=" * 78)
print("A22 conclusion: prop:ceiling's proof must state the ratio as")
print("(N-1)/(N^2-N-1), approximately 1/N and not 1/(N-1), and eq:ceiling")
print("should carry the exact form, whose limit is the 1/2 the paper claims.")
print("Part of the residual sec:e0 attributes to honest heterogeneity is the")
print("published formula's own small-N error, and A22.4 separates the two.")
print("=" * 78)
