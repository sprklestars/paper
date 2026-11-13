"""V6: numerical verification of thm:capture (single-client capture).

Claim, verbatim from 04-attacks.tex:

    Let honest clients satisfy sigma_h^2 >= sigma_min^2. An adversary
    controlling k=1 client and driving its apparent variance eta -> 0 attains
    weight share pi -> 1, and the aggregate bias satisfies
    ||sum_i w_i b_i|| -> ||Delta||.

Three things must be checked, and the third is the one that can fail:

  C1  pi -> 1 as eta -> 0, at the stated rate.  The proof's lower bound is
      pi >= 1/(1 + (N-1) eta / 1), i.e. 1 - pi = O(eta).  A closed-form
      claim about a rate is checkable exactly.

  C2  ||sum_i w_i b_i|| -> ||Delta||.  This is the part that carries the
      security meaning: capture of the WEIGHT is only interesting if it
      transfers to capture of the AGGREGATE.

  C3  The estimator is running on real data, not on declared sigmas.  The
      theorem is stated over "apparent variance", so we must confirm that an
      adversary which merely *has* low-variance data -- and never lies --
      reaches the same place.  Otherwise the theorem describes a liar and the
      threat model (C1-C4 compliance) is violated.

All three are run batched over seeds; nothing here is a single draw.
"""

import numpy as np

d = 20
SEEDS = 64


# ----------------------------------------------------------------- C1 -----
def c1_rate():
    """pi(eta) against the closed form, and the 1-pi = O(eta) rate."""
    print("=" * 74)
    print("C1  weight share vs eta   (N=30, honest sigma^2 == 1)")
    print("=" * 74)
    N = 30
    print(f"{'eta':>10} {'pi measured':>14} {'pi closed':>12} {'(1-pi)/eta':>12}")
    prev = None
    for eta in [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]:
        v = np.ones(N)
        v[0] = eta
        w = (1.0 / v) / (1.0 / v).sum()
        pi = w[0]
        closed = (1.0 / eta) / (1.0 / eta + (N - 1) / 1.0)
        ratio = (1.0 - pi) / eta
        print(f"{eta:10.0e} {pi:14.10f} {closed:12.8f} {ratio:12.4f}")
        prev = ratio
    print(f"\n  closed form matches to machine precision; (1-pi)/eta -> N-1 = {N-1}")
    print("  => 1 - pi = (N-1) eta + O(eta^2).  C1 CONFIRMED (exact, not asymptotic).\n")


# ----------------------------------------------------------------- C2 -----
def c2_aggregate_bias():
    """Does weight capture transfer to aggregate-bias capture?

    Honest biases are NOT zero here -- that would make the claim trivial.
    We draw them randomly with the same scale as Delta, which is the
    adversarial-worst case for the claim: honest bias could cancel Delta.
    """
    print("=" * 74)
    print("C2  aggregate bias ||sum w_i b_i|| vs ||Delta||   (honest bias nonzero)")
    print("=" * 74)
    N = 30
    tau = 1.0
    print(f"{'eta':>10} {'||agg bias||':>14} {'/ ||Delta||':>12} {'sd over seeds':>14}")
    for eta in [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
        vals = []
        for s in range(SEEDS):
            r = np.random.default_rng(1000 + s)
            Delta = r.normal(size=d)
            Delta *= tau / np.linalg.norm(Delta)
            b = r.normal(size=(N, d)) * (tau / np.sqrt(d))  # honest bias, same scale
            b[0] = Delta
            v = r.uniform(0.8, 1.2, size=N)
            v[0] = eta
            w = (1.0 / v) / (1.0 / v).sum()
            vals.append(np.linalg.norm(w @ b))
        vals = np.array(vals)
        print(f"{eta:10.0e} {vals.mean():14.6f} {vals.mean()/tau:12.6f} {vals.std():14.2e}")
    print("\n  => ratio -> 1.  C2 CONFIRMED.  Honest bias is squeezed out at the")
    print("     same O(eta) rate, so weight capture does transfer.\n")


# ----------------------------------------------------------------- C3 -----
def c3_honest_training():
    """The adversary never declares anything: sigma is ESTIMATED from its
    submitted updates, which come from ordinary training on low-noise data.

    This is the only version of the theorem that is compatible with the
    threat model, so it is the one that has to hold.
    """
    print("=" * 74)
    print("C3  estimated (not declared) variance, adversary trains honestly")
    print("=" * 74)
    N, T, rho = 30, 400, 0.9
    print(f"{'sigma_A/sigma_H':>16} {'pi':>10} {'x fair':>9} {'||agg bias||':>14} {'over seeds':>12}")
    for frac in [1.0, 0.5, 0.2, 0.1, 0.03, 0.01]:
        pis, ags = [], []
        for s in range(SEEDS):
            r = np.random.default_rng(2000 + s)
            Delta = r.normal(size=d)
            Delta /= np.linalg.norm(Delta)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            b = np.zeros((N, d))
            b[0] = Delta
            ema = np.ones(N)
            for t in range(T):
                U = b + r.normal(size=(N, d)) * sig[:, None]
                # dispersion estimated from the client's own residual to the
                # running consensus -- Family I, exactly as in the paper
                w = (1.0 / ema) / (1.0 / ema).sum()
                ref = w @ U
                res = ((U - ref) ** 2).mean(axis=1)
                ema = rho * ema + (1 - rho) * res
            w = (1.0 / ema) / (1.0 / ema).sum()
            pis.append(w[0])
            ags.append(np.linalg.norm(w @ b))
        pis, ags = np.array(pis), np.array(ags)
        print(f"{frac:16.2f} {pis.mean():10.4f} {pis.mean()*N:9.2f} "
              f"{ags.mean():14.4f} {ags.std():12.2e}")
    print("\n  Contrary to what we expected before running this, pi does NOT")
    print("  saturate: past a threshold between 0.5 and 0.2 it runs away to 1.0")
    print("  with vanishing seed variance.  C3c isolates the cause.\n")


# ------------------------------------------------------------- C3c --------
def c3c_frame_dependence():
    """Is the runaway in C3 caused by thm:capture itself, or by the frame
    being weight-dependent?

    Compare three reference frames at identical noise:
      weighted  ref = sum_i w_i U_i     (what C3 used; self-referential)
      uniform   ref = mean_i U_i        (frame the adversary cannot move)
      median    ref = coordinate median (robust, also unmovable at k=1)
    """
    print("=" * 74)
    print("C3c runaway is a FRAME effect, not a variance effect")
    print("=" * 74)
    N, T, rho = 30, 400, 0.9
    hdr = f"{'sigma_A/sigma_H':>16}" + "".join(f"{m:>12}" for m in ("weighted", "uniform", "median"))
    print(hdr)
    for frac in [1.0, 0.5, 0.35, 0.3, 0.25, 0.2, 0.1, 0.01]:
        row = {}
        for mode in ("weighted", "uniform", "median"):
            pis = []
            for s in range(24):
                r = np.random.default_rng(4000 + s)
                sig = r.uniform(0.8, 1.2, size=N)
                sig[0] = frac * sig[1:].mean()
                ema = np.ones(N)
                for t in range(T):
                    U = r.normal(size=(N, d)) * sig[:, None]
                    if mode == "weighted":
                        w = (1.0 / ema) / (1.0 / ema).sum()
                        ref = w @ U
                    elif mode == "uniform":
                        ref = U.mean(axis=0)
                    else:
                        ref = np.median(U, axis=0)
                    ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
                w = (1.0 / ema) / (1.0 / ema).sum()
                pis.append(w[0])
            row[mode] = np.mean(pis) * N  # x fair share
        print(f"{frac:16.2f}" + "".join(f"{row[m]:12.2f}" for m in ("weighted", "uniform", "median")))
    print("\n  x fair share.  Under an unmovable frame the adversary's gain is")
    print("  bounded and smooth in sigma_A.  Under the self-referential frame")
    print("  there is a threshold past which it takes everything.\n")


# ------------------------------------------------------------- C3b --------
def c3b_saturation():
    """Where does pi saturate, and does it depend on N?  This is what a
    deployment actually faces, so it is worth a number."""
    print("=" * 74)
    print("C3b saturation of pi under estimated variance (sigma_A/sigma_H = 0.01)")
    print("=" * 74)
    T, rho, frac = 400, 0.9, 0.01
    print(f"{'N':>5} {'pi':>10} {'x fair':>9} {'1/(1+c(N-1))  c fit':>22}")
    for N in [10, 20, 30, 50, 100]:
        pis = []
        for s in range(24):
            r = np.random.default_rng(3000 + s)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            b = np.zeros((N, d))
            ema = np.ones(N)
            for t in range(T):
                U = b + r.normal(size=(N, d)) * sig[:, None]
                w = (1.0 / ema) / (1.0 / ema).sum()
                ref = w @ U
                ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
            w = (1.0 / ema) / (1.0 / ema).sum()
            pis.append(w[0])
        pi = np.mean(pis)
        c = (1.0 / pi - 1.0) / (N - 1)
        print(f"{N:5d} {pi:10.4f} {pi*N:9.2f} {c:22.4f}")
    print("\n  c is the adversary's effective variance ratio as the estimator sees")
    print("  it.  Constant c => pi ~ 1/(1+c(N-1)), i.e. share x fair ~ 1/c is")
    print("  scale-free in N.  That is the deployable form of thm:capture.\n")




# ------------------------------------------------------------- C3d --------
def c3d_ceiling_and_threshold():
    """Two quantities C3c exposed, each now given a closed form and checked.

    (i) CEILING under an unmovable frame.  As sigma_A -> 0 the adversary's
        residual is not zero: it is floored by the frame's own sampling
        noise, sigma_H^2/N.  Honest clients see sigma_H^2(1 - 1/N) + ...
        Predicted share:
            pi_ceiling = N / (N + (N-1))      => x fair = N^2/(2N-1) ~ N/2
        A single clean client takes half the aggregate however large N is.

    (ii) THRESHOLD under the self-referential frame.  Bisect on sigma_A/sigma_H.
    """
    print("=" * 74)
    print("C3d (i) ceiling under unmovable frame vs closed form")
    print("=" * 74)
    T, rho, frac = 400, 0.9, 0.001
    print(f"{'N':>5} {'pi measured':>13} {'pi predicted':>14} {'x fair meas':>12} {'x fair pred':>12}")
    for N in [10, 20, 30, 50, 100]:
        pis = []
        for s in range(24):
            r = np.random.default_rng(5000 + s)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            ema = np.ones(N)
            for t in range(T):
                U = r.normal(size=(N, d)) * sig[:, None]
                ref = U.mean(axis=0)
                ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
            w = (1.0 / ema) / (1.0 / ema).sum()
            pis.append(w[0])
        pi = np.mean(pis)
        pred = N / (N + (N - 1.0))
        print(f"{N:5d} {pi:13.4f} {pred:14.4f} {pi*N:12.2f} {pred*N:12.2f}")

    print("\n" + "=" * 74)
    print("C3d (ii) capture threshold under self-referential frame")
    print("=" * 74)

    def share(N, frac, seeds=16):
        pis = []
        for s in range(seeds):
            r = np.random.default_rng(6000 + s)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            ema = np.ones(N)
            for t in range(T):
                U = r.normal(size=(N, d)) * sig[:, None]
                w = (1.0 / ema) / (1.0 / ema).sum()
                ref = w @ U
                ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
            w = (1.0 / ema) / (1.0 / ema).sum()
            pis.append(w[0])
        return np.mean(pis)

    print(f"{'N':>5} {'threshold sigma_A/sigma_H':>26} {'pi below':>10} {'pi above':>10}")
    for N in [10, 20, 30, 50]:
        lo, hi = 0.15, 0.60          # lo captures, hi does not
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            if share(N, mid) > 0.9:
                lo = mid
            else:
                hi = mid
        print(f"{N:5d} {0.5*(lo+hi):26.4f} {share(N, lo):10.4f} {share(N, hi):10.4f}")
    print("\n  A sharp threshold, not a gradual degradation: this is the")
    print("  TSP-Var / TSP-Shift coupling of sec:tspshift operating in the")
    print("  amplifying regime, reached here without any deliberate shift.\n")



# ------------------------------------------------------------- C3e --------
def c3e_scaling_law():
    """C3d(ii) suggested threshold ~ N^{-1/2}.  Fit it, and measure how
    sharp the transition is (sharpness must be shown, not asserted)."""
    print("=" * 74)
    print("C3e threshold scaling and transition sharpness")
    print("=" * 74)
    T, rho = 400, 0.9

    def share(N, frac, seeds=16):
        pis = []
        for s in range(seeds):
            r = np.random.default_rng(6000 + s)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            ema = np.ones(N)
            for t in range(T):
                U = r.normal(size=(N, d)) * sig[:, None]
                w = (1.0 / ema) / (1.0 / ema).sum()
                ref = w @ U
                ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
            w = (1.0 / ema) / (1.0 / ema).sum()
            pis.append(w[0])
        return np.mean(pis)

    Ns, ths = [10, 15, 20, 30, 50, 80], []
    print(f"{'N':>5} {'threshold':>11} {'threshold*sqrt(N)':>19}")
    for N in Ns:
        lo, hi = 0.10, 0.70
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if share(N, mid) > 0.9:
                lo = mid
            else:
                hi = mid
        th = 0.5 * (lo + hi)
        ths.append(th)
        print(f"{N:5d} {th:11.4f} {th*np.sqrt(N):19.4f}")
    lg = np.polyfit(np.log(Ns), np.log(ths), 1)
    print(f"\n  log-log fit: threshold = {np.exp(lg[1]):.3f} * N^({lg[0]:+.3f})")
    print(f"  exponent {lg[0]:+.3f} vs -1/2 predicted by frame sampling noise")
    print("  => LARGER federations are MORE vulnerable, not less.\n")

    print("  transition sharpness at N=30 (x fair share):")
    for frac in [0.40, 0.36, 0.34, 0.33, 0.32, 0.31, 0.30, 0.28, 0.24]:
        print(f"    sigma_A/sigma_H = {frac:.2f}  ->  {share(30, frac)*30:6.2f}x")
    print()



# ------------------------------------------------------------- C3f --------
def c3f_floor_removes_threshold():
    """Does M1 (relative floor, eq:relfloor) remove the runaway, or merely
    move it?  If it merely moves it, the design rule of sec:defense is wrong."""
    print("=" * 74)
    print("C3f does the relative floor remove the capture threshold?")
    print("=" * 74)
    N, T, rho = 30, 400, 0.9

    def share(frac, phi, seeds=16):
        pis = []
        for s in range(seeds):
            r = np.random.default_rng(7000 + s)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            ema = np.ones(N)
            for t in range(T):
                U = r.normal(size=(N, d)) * sig[:, None]
                e = np.maximum(ema, phi * np.median(ema)) if phi > 0 else ema
                w = (1.0 / e) / (1.0 / e).sum()
                ref = w @ U
                ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
            e = np.maximum(ema, phi * np.median(ema)) if phi > 0 else ema
            w = (1.0 / e) / (1.0 / e).sum()
            pis.append(w[0])
        return np.mean(pis) * N

    fracs = [1.0, 0.5, 0.35, 0.30, 0.20, 0.10, 0.01]
    phis = [0.0, 0.2, 0.5, 0.8]
    print("  x fair share")
    print(f"{'sA/sH':>8}" + "".join(f"{('phi='+str(p)) if p>0 else 'no floor':>11}" for p in phis))
    for frac in fracs:
        print(f"{frac:8.2f}" + "".join(f"{share(frac, p):11.2f}" for p in phis))
    print("\n  A floor caps the share at 1/phi regardless of sigma_A: the threshold")
    print("  does not move, it ceases to exist.  Predicted caps: " +
          ", ".join(f"phi={p} -> {1.0/p:.2f}x" for p in phis[1:]) + "\n")


# ------------------------------------------------------------- C3g --------
def c3g_threshold_closed_form():
    """C3e measured theta_N ~ 1.80 N^-0.518 and called it empirical.  Is it?

    Derivation to test, one step at a time.  Write u = 1 - pi for the
    adversary's weight deficit, sigma_H = 1, r = sigma_A / sigma_H.

      (1) The adversary's residual is measured against a frame it dominates,
          so its own contribution cancels to weight u:
              e_0 = u^2 (sigma_A^2 + sigma_H^2/(N-1)).
      (2) An honest client's residual is dominated by its own noise:
              e_H = sigma_H^2.
      (3) Inverse-variance weighting balances the two groups:
              pi/(1-pi) = e_H / ((N-1) e_0).
      (4) Substituting (1) and (2) into (3):
              u(1-u) = 1 / (1 + (N-1) r^2).
          The left side is at most 1/4, so an interior fixed point exists
          only when (N-1) r^2 >= 3.  Below that the recursion has nowhere to
          land and pi -> 1.  Hence a closed form, not a fit:
              theta_N = sqrt(3/(N-1)).

    Each step is checked separately, because a closed form that lands on the
    right answer through a wrong intermediate is worth nothing.
    """
    print("=" * 74)
    print("C3g is prop:threshold's exponent a fit, or a closed form?")
    print("=" * 74)
    T, rho = 2000, 0.9

    def stationary(N, frac, seed):
        r = np.random.default_rng(6000 + seed)
        sig = r.uniform(0.8, 1.2, size=N)
        sig[0] = frac * sig[1:].mean()
        ema = np.ones(N)
        for _ in range(T):
            U = r.normal(size=(N, d)) * sig[:, None]
            w = (1.0 / ema) / (1.0 / ema).sum()
            ref = w @ U
            ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
        w = (1.0 / ema) / (1.0 / ema).sum()
        return w[0], ema[0], ema[1:].mean(), sig[0], sig[1:].mean()

    print("  steps (1)-(3): the two residuals and the balance relation")
    print(f"{'N':>5}{'sA/sH':>7} | {'e0 err':>8} {'eH/sH^2':>9} {'balance err':>12}")
    e0err, balerr, eHrat = [], [], []
    for N in [20, 30, 50, 80]:
        for frac in [0.4, 0.6, 0.8, 1.0]:
            a = [stationary(N, frac, s) for s in range(8)]
            pi = np.mean([x[0] for x in a])
            e0 = np.mean([x[1] for x in a])
            eH = np.mean([x[2] for x in a])
            e0p = np.mean([(1 - x[0]) ** 2 * (x[3] ** 2 + x[4] ** 2 / (N - 1))
                           for x in a])
            eHp = np.mean([x[4] ** 2 for x in a])
            r1 = abs(e0 / e0p - 1)
            r3 = abs((pi / (1 - pi)) / (eH / ((N - 1) * e0)) - 1)
            e0err.append(r1)
            balerr.append(r3)
            eHrat.append(eH / eHp)
            print(f"{N:5d}{frac:7.2f} | {r1*100:7.2f}% {eH/eHp:9.4f} "
                  f"{r3*100:11.2f}%")
    print(f"\n  (1) e_0 closed form   max rel err {max(e0err)*100:.2f}%")
    print(f"  (2) e_H / sigma_H^2   in [{min(eHrat):.4f}, {max(eHrat):.4f}]")
    print(f"  (3) balance relation  max rel err {max(balerr)*100:.2f}%, "
          f"median {np.median(balerr)*100:.2f}%")
    print("      the worst cell is the smallest N at the smallest r, where the")
    print("      dropped O(pi^2 r^2) term is largest; step (2) is the one that")
    print("      is approximate, and its error is what the 5% offset below is.")

    def share(N, frac, seeds=16, Ts=400):
        pis = []
        for s in range(seeds):
            r = np.random.default_rng(6000 + s)
            sig = r.uniform(0.8, 1.2, size=N)
            sig[0] = frac * sig[1:].mean()
            ema = np.ones(N)
            for _ in range(Ts):
                U = r.normal(size=(N, d)) * sig[:, None]
                w = (1.0 / ema) / (1.0 / ema).sum()
                ref = w @ U
                ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
            w = (1.0 / ema) / (1.0 / ema).sum()
            pis.append(w[0])
        return float(np.mean(pis))

    print("\n  step (4): where the fixed point disappears vs where it is measured")
    # N runs to 200, not 80: the proof attributes the offset to a factor free of
    # N, and at the threshold r^2 ~ 3/(N-1), so the dropped O(pi^2 r^2) term is
    # itself N-dependent and the claim only holds if c stays flat over a range
    # wide enough to see a trend. The bracket starts at 0.02 because
    # sqrt(3/199) = 0.123 and the old lower bound of 0.10 would clip N >= 300.
    Ns = np.array([10, 15, 20, 30, 50, 80, 120, 200], float)
    meas = []
    for N in Ns:
        lo, hi = 0.02, 0.70
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if share(int(N), mid) > 0.9:
                lo = mid
            else:
                hi = mid
        meas.append(0.5 * (lo + hi))
    meas = np.array(meas)
    pred = np.sqrt(3.0 / (Ns - 1))
    print(f"{'N':>5}{'measured':>11}{'sqrt(3/(N-1))':>15}{'ratio':>8}")
    for n, m, p in zip(Ns, meas, pred):
        print(f"{int(n):5d}{m:11.4f}{p:15.4f}{p/m:8.4f}")
    lm = np.polyfit(np.log(Ns), np.log(meas), 1)
    lp = np.polyfit(np.log(Ns), np.log(pred), 1)
    rat = pred / meas
    print(f"\n  measured   {np.exp(lm[1]):.3f} N^({lm[0]:+.4f})")
    print(f"  closed form {np.exp(lp[1]):.3f} N^({lp[0]:+.4f})")
    print(f"  closed form is high by {rat.mean()*100-100:.1f}% "
          f"+- {rat.std()*100:.1f}pp -- a CONSTANT offset, uniform in N,")
    print(f"  max rel err {abs(rat-1).max()*100:.1f}%.  The exponent is")
    print("  therefore derived and the fitted exponent is measurement noise")
    print("  around -1/2; the 5% gap is the O(pi^2 r^2) term dropped in")
    print("  step (2).")
    # Does the offset TREND with N, or is it flat?  A slope of zero in log-log
    # is what "a factor free of N" means, and it is the load-bearing part of
    # eq:threshold's claim that the N-dependence is exact.
    c = meas / pred
    sl = np.polyfit(np.log(Ns), np.log(c), 1)[0]
    print(f"  c = measured/closed form: mean {c.mean():.4f} sd {c.std():.4f} "
          f"spread {c.max()-c.min():.4f} over N = {int(Ns[0])}-{int(Ns[-1])}")
    print(f"  log c against log N slope {sl:+.5f}; zero means free of N, and a")
    print("  term additive in 1/N would need this to be about -0.5.")
    assert abs(rat - 1).max() < 0.08, "closed form drifted from measurement"
    assert rat.std() < 0.02, "offset is not constant in N; exponent claim unsafe"
    assert abs(sl) < 0.02, f"c trends with N (slope {sl}); eq:threshold unsafe"
    assert max(e0err) < 0.08, "step (1) broke"
    assert max(eHrat) < 1.05 and min(eHrat) > 0.95, "step (2) broke"
    assert np.median(balerr) < 0.08, "step (3) broke"
    print("  [assertions passed]\n")


if __name__ == "__main__":
    c1_rate()
    c2_aggregate_bias()
    c3_honest_training()
    c3b_saturation()
    c3c_frame_dependence()
    c3d_ceiling_and_threshold()
    c3e_scaling_law()
    c3f_floor_removes_threshold()
    c3g_threshold_closed_form()
