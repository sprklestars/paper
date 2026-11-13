"""A21: prop:threshold's constant c depends on the criterion used to locate
the threshold, and the paper's c is measured at a different one from the
criterion its proof derives.

The proof locates the threshold where the interior fixed point of
u(1-u) = 1/(1+(N-1)r^2) disappears.  The left side peaks at 1/4 at u = 1/2, so
the fixed point vanishes exactly when pi = 1 - u = 1/2.

v6_capture.py bisects on `share(N, mid) > 0.9`, i.e. it locates where pi
exceeds 0.9, which is past the vanishing point.  If those give different c the
offset the paper attributes to the dropped O(pi^2 r^2) term is partly an
artefact of the criterion, and the text must say which criterion c belongs to.

Same harness as v6 C3g step (4): identical generator, rho, seed offsets and
bracket.  Two things vary, one at a time: the criterion, and the horizon T.
T matters because the capture is a runaway, so a longer horizon lets a weaker
adversary finish arriving -- v6 runs T = 400 and that is the column the
paper's 0.955 must be read against.
"""

import numpy as np

d = 20


def share(N, frac, seeds=16, Ts=400, rho=0.9):
    """v6_capture.py C3g step (4) generator, unchanged; Ts = 400 as there."""
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


def threshold(N, crit, Ts=400, lo=0.02, hi=0.70, steps=14):
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if share(N, mid, Ts=Ts) > crit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


Ns = np.array([10, 20, 30, 50, 80], float)
pred = np.sqrt(3.0 / (Ns - 1))

print("=" * 78)
print("A21.1  c at the criterion the SCRIPT uses (pi > 0.9) and the one the")
print("       PROOF derives (pi > 0.5, where the interior fixed point dies)")
print("=" * 78)
print("  u(1-u) peaks at 1/4 at u = 1/2, so the fixed point vanishes at")
print("  pi = 1/2.  All columns use the identical generator and bracket; the")
print("  T = 400 pair is v6's own horizon, the T = 2000 pair shows what a")
print("  longer horizon does to the same measurement.")
print()
print(f"{'N':>5} {'sqrt(3/(N-1))':>14} {'c @.9 T400':>11} {'c @.5 T400':>11} "
      f"{'c @.9 T2k':>10} {'c @.5 T2k':>10}")
c9, c5, c9L, c5L = [], [], [], []
for N, p in zip(Ns, pred):
    n = int(N)
    a = threshold(n, 0.9, Ts=400) / p
    b = threshold(n, 0.5, Ts=400) / p
    e = threshold(n, 0.9, Ts=2000) / p
    f = threshold(n, 0.5, Ts=2000) / p
    c9.append(a); c5.append(b); c9L.append(e); c5L.append(f)
    print(f"{n:5d} {p:14.4f} {a:11.4f} {b:11.4f} {e:10.4f} {f:10.4f}")
c9, c5 = np.array(c9), np.array(c5)
c9L, c5L = np.array(c9L), np.array(c5L)
print()
print(f"  T=400  c at pi>0.9 : mean {c9.mean():.4f} sd {c9.std():.4f}   "
      f"(the paper quotes 0.955)")
print(f"  T=400  c at pi>0.5 : mean {c5.mean():.4f} sd {c5.std():.4f}")
print(f"  T=2000 c at pi>0.9 : mean {c9L.mean():.4f} sd {c9L.std():.4f}")
print(f"  T=2000 c at pi>0.5 : mean {c5L.mean():.4f} sd {c5L.std():.4f}")
print()
print(f"  criterion effect at fixed T=400 : {c5.mean()-c9.mean():+.4f}")
print(f"  horizon   effect at fixed crit  : {c9L.mean()-c9.mean():+.4f}")
off = 1.0 - c9.mean()
print(f"  the offset the paper attributes to the dropped O(pi^2 r^2) term is")
print(f"  1 - c = {off:.4f}; the criterion alone accounts for "
      f"{(c5.mean()-c9.mean())/off*100:.0f}% of it and the horizon for "
      f"{(c9L.mean()-c9.mean())/off*100:.0f}%.")
print("  Neither is a property of the dropped term, so the offset is not")
print("  wholly attributable to it.")

sl9 = np.polyfit(np.log(Ns), np.log(c9), 1)[0]
sl5 = np.polyfit(np.log(Ns), np.log(c5), 1)[0]
print(f"\n  log c vs log N slope: {sl9:+.5f} at pi>0.9, {sl5:+.5f} at pi>0.5")
print("  Both flat, so 'the N-dependence is exact' survives either way: the")
print("  criterion moves the CONSTANT, not the exponent.")
assert abs(sl9) < 0.03 and abs(sl5) < 0.03, (sl9, sl5)
assert c5.mean() > c9.mean(), "expected the proof's criterion to sit higher"
assert abs(c9.mean() - 0.955) < 0.02, ("T=400 pi>0.9 must reproduce v6's c",
                                       c9.mean())

print()
print("=" * 78)
print("A21.2  how sharp is the transition between the two criteria?")
print("=" * 78)
print("  If pi jumps from 0.5 to 0.9 over a narrow band of r the criterion")
print("  hardly matters; if the band is wide, c is criterion-dependent and the")
print("  paper must name its criterion.")
print()
N = 30
print(f"  N = {N}, sqrt(3/(N-1)) = {np.sqrt(3/(N-1)):.4f}, T = 400")
print(f"{'r = sigma_A/sigma_H':>21} {'pi':>8} {'x fair':>9}")
band = []
for frac in [0.45, 0.40, 0.36, 0.34, 0.33, 0.32, 0.31, 0.30, 0.28, 0.24]:
    pi = share(N, frac)
    band.append((frac, pi))
    print(f"{frac:21.2f} {pi:8.4f} {pi*N:9.2f}")
lo = max((f for f, p in band if p < 0.5), default=float("nan"))
hi = min((f for f, p in band if p > 0.9), default=float("nan"))
print(f"\n  pi crosses 0.5 below r = {lo:.2f} and 0.9 below r = {hi:.2f}, a band")
print(f"  of {lo-hi:.2f} in r, i.e. {(lo-hi)/np.sqrt(3/(N-1))*100:.0f}% of the")
print("  threshold itself.  Not a knife edge, so the criterion belongs in the")
print("  text alongside c.")

print()
print("=" * 78)
print("A21.3  the OTHER claim: does harm at FIXED adversary capability rise")
print("       with N, as 'larger federations are more vulnerable' suggests?")
print("=" * 78)
print("  The threshold in r falls like N^-1/2.  That is a statement about the")
print("  BAR, not about the harm at a fixed r.  Sweeping N at fixed r:")
print()
print(f"{'r':>6}" + "".join(f"{'N=' + str(int(n)):>10}" for n in Ns))
for frac in [0.5, 0.35, 0.2, 0.1]:
    row = [share(int(n), frac) * n for n in Ns]
    print(f"{frac:6.2f}" + "".join(f"{v:10.2f}" for v in row))
print("\n  Read as multiples of fair share.  A row that rises with N means a")
print("  larger federation is worse at fixed capability; a row that falls")
print("  means dilution wins.  Both appear, depending on which side of")
print("  theta_N the fixed r sits on, which is the point: the correct claim")
print("  is about the THRESHOLD, and 'more vulnerable' must be said of the")
print("  bar rather than of the damage.")

print()
print("=" * 78)
print("A21.4  is the offset unit in tab:e0 right?")
print("=" * 78)
rel = (1.0 / c9.mean() - 1.0) * 100
sd = (1.0 / c9).std() * 100
print("  v6 prints 'closed form is high by X% +- Ypp' where X and Y are the")
print("  mean and sd of pred/meas expressed in per cent.")
print(f"  With c = {c9.mean():.4f} that is {rel:.1f}% +- {sd:.1f}, and the")
print("  MEAN is a relative error: the closed form exceeds the measured")
print("  threshold by that fraction of the threshold.")
print("  tab:e0 reports '4.7 +- 0.5 pp', attaching a percentage-point unit to")
print("  both.  The spread of a set of percentages is fairly quoted in pp, but")
print("  the 4.7 is not a difference of two probabilities: it is 4.7% OF the")
print("  threshold, so the entry should read '4.7% (spread 0.5 pp)'.")

print()
print("=" * 78)
print("A21 conclusion: the exponent -1/2 is derived and holds at both")
print("criteria; c is criterion- and horizon-dependent, so prop:threshold must")
print("state that its c is measured where pi exceeds 0.9 at T = 400 and not at")
print("the fixed point's disappearance, and tab:e0's '4.7 pp' should read")
print("'4.7%'.")
print("=" * 78)
