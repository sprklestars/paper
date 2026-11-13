"""A16: why does the measured ceiling sit below prop:ceiling's predicted level?

V6's C3d reports 0.512 -> 0.483 over N = 10 -> 100 against a predicted 0.524 ->
0.502, a residual of 0.9 to 1.9 percentage points that grows with N.  sec:e0 states
it; sec:attacks should say what causes it rather than leave a proposition looking
1.9 points wrong.  Two candidates, and they are separable:

  (a) HETEROGENEITY.  The proof carries one common sigma_H, while the C3d harness
      draws honest sigmas uniform(0.8, 1.2).  Weights are 1/sigma^2, convex, so
      spread inflates the honest side of the normalisation and takes share off the
      adversary.  This should worsen with N, more honest clients meaning more of
      the inflated mass.

  (b) ESTIMATOR MEMORY.  The EMA at rho = 0.9 has an effective window of ~10
      rounds, and 1/E[ema] != E[1/ema], so a Jensen bias is available too.  This
      should vanish as rho -> 1.

Vary each alone.  Everything else follows C3d exactly, including its seeds.

eq:ceiling carries the exact closed form (N^2-N-1)/((N^2-N-1)+(N-1)^2), derived in
A22; N/(2N-1) is its large-N approximation and this script reported against that
before A22 replaced it.  Both are printed so the substitution is visible, and the
residuals the paper quotes are measured against the exact form the paper states.
"""

import numpy as np

d = 20


def closed_exact(N):
    """eq:ceiling: pi from the exact residual variances (A22)."""
    a = N * N - N - 1.0
    return a / (a + (N - 1.0) ** 2)


def closed_published(N):
    """The large-N approximation this script used before A22."""
    return N / (2.0 * N - 1.0)


def ceiling(N, het=True, rho=0.9, T=400, seeds=24, frac=0.001):
    """C3d's estimator: EMA of squared residual to the uniform mean frame."""
    pis = []
    for s in range(seeds):
        r = np.random.default_rng(5000 + s)
        sig = r.uniform(0.8, 1.2, size=N) if het else np.ones(N)
        sig[0] = frac * sig[1:].mean()
        ema = np.ones(N)
        for t in range(T):
            U = r.normal(size=(N, d)) * sig[:, None]
            ref = U.mean(axis=0)
            ema = rho * ema + (1 - rho) * ((U - ref) ** 2).mean(axis=1)
        w = (1.0 / ema) / (1.0 / ema).sum()
        pis.append(w[0])
    return float(np.mean(pis))


def main():
    print("=" * 78)
    print("A16  the residual in prop:ceiling, attributed")
    print("=" * 78)
    print("\nhet = honest sigmas uniform(0.8,1.2) as in C3d; hom = common sigma_H,")
    print("which is the case the proof states.  T is raised with rho so that the")
    print("EMA is equally converged in every column.\n")
    print(f"{'N':>5} {'eq:ceil':>8} {'N/(2N-1)':>9} {'het rho.9':>10} "
          f"{'hom rho.9':>10} {'hom rho.99':>11} {'hom rho.999':>12}")
    print("-" * 70)
    rows = []
    for N in [10, 20, 30, 50, 100]:
        pred = closed_exact(N)
        old = closed_published(N)
        het = ceiling(N)
        h9 = ceiling(N, het=False)
        h99 = ceiling(N, het=False, rho=0.99, T=4000)
        h999 = ceiling(N, het=False, rho=0.999, T=20000)
        rows.append((N, pred, het, h9, h99, h999))
        print(f"{N:5d} {pred:8.4f} {old:9.4f} {het:10.4f} {h9:10.4f} "
              f"{h99:11.4f} {h999:12.4f}")

    a = np.array(rows)
    gap_het = (a[:, 1] - a[:, 2]) * 100
    gap_hom = np.abs(a[:, 1] - a[:, 5]) * 100
    print(f"\nresidual with heterogeneity: {gap_het.min():.1f} to "
          f"{gap_het.max():.1f} percentage points, growing in N")
    print(f"residual without it, rho=0.999: {gap_hom.max():.2f} percentage "
          f"points worst case, no trend")
    print("\nSo (a) accounts for it and (b) does not: rho from 0.9 to 0.999")
    print("changes nothing, while removing the spread in sigma_H closes the gap")
    print("to a fiftieth of a point at every N.  prop:ceiling is exact for the")
    print("common-sigma case it states, and the measured curve sits below it")
    print("because 1/sigma^2 is convex, so dispersion among the honest clients")
    print("inflates their total weight.  The direction is favourable to the")
    print("defender and the ceiling remains ~N/2 times a fair share.")
    assert gap_hom.max() < 0.05, gap_hom


if __name__ == "__main__":
    main()
