"""A20: prop:impossible as stated is false, and what replaces it is an exact
identity rather than an impossibility.

The claim under test is 'With a = 0, no scorer satisfies def:resistance with
epsilon < 1.'  def:resistance bounds the honest weight ratio
w_h^atk / w_h^clean >= 1 - epsilon.

Two things are wrong with it.

(1) A scorer whose weights ignore the updates has that ratio identically 1, so
    it attains epsilon = 0 with a = 0.  Uniform FedAvg is such a scorer and
    tab:recent already reports it as a control.  So the quantifier 'no scorer'
    is false, and the proposition needs a hypothesis.

(2) The proof's final step, 'any w suppressing it suppresses the one client
    tracking updstar ... giving eps_2 >= 1', argues about model quality.
    def:resistance measures a weight ratio.  Suppressing a client to a
    positive fraction s of its clean weight gives 1 - s, which is below 1
    whenever s > 0.  Only a rule that zeroes the client reaches 1.

What IS true is stronger than a bound, because thm:unidentifiable makes the
two worlds one object: the weight vector is literally identical in both, so
whatever factor the rule applies to the suspected client in W1 it applies to
the origin-tracking client in W2.  Suppression in W1 EQUALS displacement in
W2, as an identity.  That is measured here across bias magnitudes and seeds.
"""

import numpy as np

d, N = 20, 30
NSEED = 8


def sigmas(seed):
    return np.random.default_rng(1000 + seed).uniform(0.5, 1.5, size=N)


def world(bias, seed, sig):
    r = np.random.default_rng(seed)
    return bias + r.normal(size=(N, d)) * sig[:, None]


def three_worlds(dmag, seed):
    """W1: client 0 carries +Delta, peers unbiased.
       W2: client 0 unbiased, peers carry -Delta.  W2 = W1 - Delta.
       clean: nobody biased -- the counterfactual both are scored against."""
    sig = sigmas(seed)
    r = np.random.default_rng(2000 + seed)
    Delta = r.normal(size=d)
    Delta = dmag * Delta / np.linalg.norm(Delta)
    b1 = np.zeros((N, d)); b1[0] = Delta
    b2 = np.tile(-Delta, (N, 1)); b2[0] = 0.0
    return (world(b1, seed, sig), world(b2, seed, sig),
            world(np.zeros((N, d)), seed, sig))


# --------------------------------------------------------------- weight maps
# Every rule is peer-only (lambda = 1) and returns weights, not bare scores.

def w_fedavg(V):
    return np.full(len(V), 1.0 / len(V))


def w_invsq(V):
    dd = np.linalg.norm(V - V.mean(0), axis=1)
    w = 1.0 / np.maximum(dd, 1e-12) ** 2
    return w / w.sum()


def w_softmin(V, tau=1.0):
    e = np.exp(-np.linalg.norm(V - V.mean(0), axis=1) / tau)
    return e / e.sum()


def w_geomed(V):
    y = V.mean(0)
    for _ in range(200):
        dist = np.maximum(np.linalg.norm(V - y, axis=1), 1e-12)
        wq = 1.0 / dist
        y2 = (wq[:, None] * V).sum(0) / wq.sum()
        if np.linalg.norm(y2 - y) < 1e-15:
            break
        y = y2
    w = 1.0 / np.maximum(np.linalg.norm(V - y, axis=1), 1e-12) ** 2
    return w / w.sum()


def _krum_scores(V, f=2):
    D = np.linalg.norm(V[:, None] - V[None], axis=2) ** 2
    m = max(1, len(V) - f - 2)
    return np.sort(D, axis=1)[:, 1:m + 1].sum(1)


def w_multikrum(V, f=2):
    keep = np.argsort(_krum_scores(V, f))[:len(V) - f]
    w = np.zeros(len(V)); w[keep] = 1.0 / len(keep)
    return w


def w_krum1(V, f=2):
    w = np.zeros(len(V)); w[int(np.argmin(_krum_scores(V, f)))] = 1.0
    return w


def w_normtrim(V, beta=0.1):
    nrm = np.linalg.norm(V - V.mean(0), axis=1)
    t = int(np.floor(beta * len(V)))
    keep = np.argsort(nrm)[:len(V) - t]
    w = np.zeros(len(V)); w[keep] = 1.0 / len(keep)
    return w


RULES = {"FedAvg (uniform)": w_fedavg, "inverse sq distance": w_invsq,
         "distance softmin": w_softmin, "geo-median inv sq": w_geomed,
         "Multi-Krum (f=2)": w_multikrum, "Krum (select 1)": w_krum1,
         "norm-trimmed mean": w_normtrim}
HARD = {"Multi-Krum (f=2)", "Krum (select 1)", "norm-trimmed mean"}
MAGS = [0.0, 0.8, 2.0, 5.0, 10.0, 30.0]

print("=" * 78)
print("A20.1  the two worlds are ONE object: w(W1) = w(W2) exactly")
print("=" * 78)
print("  thm:unidentifiable says a peer-only scorer cannot tell the worlds")
print("  apart.  Checked as float equality, not as a tolerance.")
print()
print(f"{'rule':>21} {'max|w(W1)-w(W2)|':>18} {'bitwise equal':>14}")
worst_gap, all_bit = 0.0, True
for nm, f in RULES.items():
    g, bits = 0.0, True
    for s in range(NSEED):
        for dm in MAGS:
            V1, V2, _ = three_worlds(dm, s)
            w1, w2 = f(V1), f(V2)
            g = max(g, float(np.max(np.abs(w1 - w2))))
            bits &= bool(np.array_equal(w1, w2))
    worst_gap = max(worst_gap, g)
    all_bit &= bits
    print(f"{nm:>21} {g:18.3e} {str(bits):>14}")
print(f"\n  worst deviation over {NSEED} seeds x {len(MAGS)} magnitudes: "
      f"{worst_gap:.2e}")
assert worst_gap < 1e-15, worst_gap

print()
print("=" * 78)
print("A20.2  suppression in W1 IS displacement in W2, as an identity")
print("=" * 78)
print("  supp = w_0(W1)/w_0(clean): what the rule does to the biased client.")
print("  eps_2 = 1 - w_0(W2)/w_0(clean): what def:resistance charges the rule")
print("  for doing it to the honest origin-tracking client of the translated")
print("  world.  If these are the same number the trade-off is exact.")
print()
print(f"{'rule':>21} {'|Delta|':>8} {'supp':>9} {'eps_2':>9} {'1-supp-eps_2':>13}")
worst_id = 0.0
for nm, f in RULES.items():
    for dm in MAGS:
        sp, e2 = [], []
        for s in range(NSEED):
            V1, V2, VC = three_worlds(dm, s)
            w1, w2, wc = f(V1), f(V2), f(VC)
            if wc[0] <= 0:            # clean-world exclusion: ratio undefined
                continue
            sp.append(w1[0] / wc[0])
            e2.append(1.0 - w2[0] / wc[0])
        if not sp:
            print(f"{nm:>21} {dm:8.1f} {'excl':>9} {'excl':>9} {'--':>13}")
            continue
        sp_m, e2_m = float(np.mean(sp)), float(np.mean(e2))
        gap = max(abs((1 - a) - b) for a, b in zip(sp, e2))
        worst_id = max(worst_id, gap)
        print(f"{nm:>21} {dm:8.1f} {sp_m:9.4f} {e2_m:9.4f} {gap:13.1e}")
print(f"\n  worst discrepancy {worst_id:.2e}: eps_2 = 1 - supp exactly.")
print("  So eps_2 >= 1 requires supp <= 0, that is outright exclusion.  A rule")
print("  that suppresses to a positive fraction has eps_2 < 1, and the")
print("  proposition's conclusion does not follow for it.")
assert worst_id < 1e-12, worst_id

print()
print("=" * 78)
print("A20.3  epsilon at a = 0 for each rule, at the largest bias")
print("=" * 78)
print("  eps_1 charges displacement of the honest peers in W1; eps_2 charges")
print("  displacement of client 0 in W2.  The proposition claims max >= 1 for")
print("  every scorer.")
print()
print(f"{'rule':>21} {'eps_1':>9} {'eps_2':>9} {'eps':>9} {'eps < 1':>9}")
viol = []
for nm, f in RULES.items():
    e1s, e2s = [], []
    for s in range(NSEED):
        V1, V2, VC = three_worlds(30.0, s)
        w1, w2, wc = f(V1), f(V2), f(VC)
        m = wc[1:] > 0
        e1s.append(1.0 - float(np.min(w1[1:][m] / wc[1:][m])))
        e2s.append(1.0 if wc[0] <= 0 else 1.0 - w2[0] / wc[0])
    e1, e2 = float(np.mean(e1s)), float(np.mean(e2s))
    eps = max(e1, e2)
    if eps < 1.0:
        viol.append((nm, eps))
    print(f"{nm:>21} {e1:9.4f} {e2:9.4f} {eps:9.4f} {str(eps < 1.0):>9}")
print()
print(f"  {len(viol)} of {len(RULES)} rules attain eps < 1 with a = 0:")
for nm, e in viol:
    print(f"    {nm:>21}  eps = {e:.4f}")
print("  The proposition as written is therefore FALSE.  Every rule attaining")
print("  eps < 1 does so by declining to exclude: the three hard-exclusion")
print("  rules are the only ones that reach 1.")
assert viol, "expected at least the FedAvg counterexample"

print()
print("=" * 78)
print("A20.4  what eps = 0 costs: the rule can no longer rank")
print("=" * 78)
print("  Spearman correlation of each rule's weights against true per-client")
print("  bias magnitude on an all-honest heterogeneous population, alongside")
print("  its eps at a = 0.  A rule that cannot rank cannot be captured and")
print("  cannot decide either -- the E1b landscape, here as a theorem's cost.")
print()


from tools.rank_metrics import spearman, format_correlation


print(f"{'rule':>21} {'rho(w, |bias|)':>15} {'eps at a=0':>11}")
for nm, f in RULES.items():
    rs = []
    for s in range(NSEED):
        r = np.random.default_rng(3000 + s)
        bh = r.normal(size=(N, d)) * 3.0
        VH = world(bh, s, sigmas(s))
        rs.append(spearman(f(VH), np.linalg.norm(bh, axis=1)))
    e1s, e2s = [], []
    for s in range(NSEED):
        V1, V2, VC = three_worlds(30.0, s)
        w1, w2, wc = f(V1), f(V2), f(VC)
        m = wc[1:] > 0
        e1s.append(1.0 - float(np.min(w1[1:][m] / wc[1:][m])))
        e2s.append(1.0 if wc[0] <= 0 else 1.0 - w2[0] / wc[0])
    print(f"{nm:>21} {format_correlation(float(np.mean(rs))):>15} "
          f"{max(float(np.mean(e1s)), float(np.mean(e2s))):11.4f}")
print()
print("  FedAvg 的权重恒定，Spearman 未定义（N/A），不是相关系数零。")
print("  eps = 0 的恒等反例不受这项事后秩统计修正影响。")

print()
print("=" * 78)
print("A20.5  the repaired statement, as a predicate")
print("=" * 78)
print("  Claim: with a = 0, for every translation-equivariant scorer, the")
print("  suppression applied to a common-mode-biased client in W1 equals the")
print("  displacement inflicted on the origin-tracking honest client in W2;")
print("  hence a scorer that suppresses at all (supp < 1) has eps > 0, and one")
print("  that excludes (supp = 0) has eps = 1.  No scorer both suppresses and")
print("  keeps eps = 0.")
print()
print(f"{'rule':>21} {'supp(|D|=30)':>13} {'eps':>9} {'suppresses':>11} "
      f"{'eps > 0':>8}")
bad = 0
for nm, f in RULES.items():
    sp, ep = [], []
    for s in range(NSEED):
        V1, V2, VC = three_worlds(30.0, s)
        w1, w2, wc = f(V1), f(V2), f(VC)
        m = wc[1:] > 0
        e1 = 1.0 - float(np.min(w1[1:][m] / wc[1:][m]))
        if wc[0] <= 0:
            sp.append(0.0); ep.append(1.0)
        else:
            sp.append(w1[0] / wc[0]); ep.append(max(e1, 1.0 - w2[0] / wc[0]))
    s_m, e_m = float(np.mean(sp)), float(np.mean(ep))
    supps, epspos = s_m < 1.0 - 1e-9, e_m > 1e-9
    if supps and not epspos:
        bad += 1
    print(f"{nm:>21} {s_m:13.4f} {e_m:9.4f} {str(supps):>11} "
          f"{str(epspos):>8}")
print(f"\n  rules that suppress at zero cost: {bad}")
assert bad == 0, "a scorer suppressed the biased client for free"
print("  The repaired proposition holds on every rule tested, and it is an")
print("  equality rather than an impossibility: anchors are what buy")
print("  suppression without displacement, which is the design claim sec:m2")
print("  actually needs.")

print()
print("=" * 78)
print("A20 conclusion: prop:impossible must acquire the hypothesis that the")
print("scorer discriminates, and its content is the exact identity of A20.2,")
print("not the unquantified 'eps >= 1' its proof asserts.")
print("=" * 78)
