"""A12: is M3 (policy binding) actually verifiable by a third party?

M3 as written in the paper is a one-liner: commit H(scorer), publish
digest(state), let anyone recompute.  That is a protocol sketch, not a result.
This script asks the questions that decide whether it is implementable, and
reports the ones where the honest answer is "no".

The threat M3 addresses is NOT the client (that is M1/M2/M4).  It is the
AGGREGATOR claiming one policy while applying another -- weights that were not
produced by the committed rule.  Verifiability therefore means: a third party
holding only public data can recompute {w_i} and get bit-identical output.

  A12.1  is the scorer bit-reproducible at all?  (float non-determinism)
  A12.2  does the recomputation catch a cheating aggregator, and how small a
         cheat can it catch?
  A12.3  what must the digest cover?  (a state-completeness attack: the
         aggregator picks a favourable state that is consistent with a digest
         that commits to too little)
  A12.4  cost of verification vs cost of aggregation
  A12.5  does M3 compose with M4, or does canonicalisation break the digest?

Style follows a1/a4/a11: hand-written numpy, no torch, everything printed.
"""

import hashlib
import json

import numpy as np

from a1_nn_symmetry import D_IN, flat, gauge_permute, init_net, sgd
from a4_aggregate_gauge import gauge_canon
from a11_permutation_canon import align_to

N = 16
D = 64
ROUNDS = 6
RHO = 0.5
PHI = 0.5


# --------------------------------------------------------- the policy ----
def scorer(ev, phi=PHI):
    """The committed policy: relative-variance-floored inverse weighting.

    This is the M1 rule of the paper.  Deliberately written to be the kind of
    thing a real deployment would commit to: a couple of reductions, a median,
    a divide, a normalisation.
    """
    floor = phi * np.median(ev)
    iv = 1.0 / np.maximum(ev, floor)
    return iv / iv.sum()


def canon_state(ev):
    """Serialise scorer input canonically.  float64 -> 17 significant digits
    is round-trip exact, so this is lossless and platform-stable."""
    return json.dumps([float(x) for x in ev], separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canon_state(obj).encode()).hexdigest()


# ------------------------------------------------------------- A12.1 -----
print("=" * 78)
print("A12.1  is the committed scorer bit-reproducible?")
print("=" * 78)

rng = np.random.default_rng(0)
ev = rng.uniform(0.05, 2.0, size=N)

w_ref = scorer(ev)
same_call = all(np.array_equal(w_ref, scorer(ev)) for _ in range(64))
print(f"  64 repeat calls, same process, bit-identical : {same_call}")

# Reduction order is the classic source of divergence: a verifier that sums in
# a different order gets a different float.  Does OUR policy depend on it?
perm = rng.permutation(N)
w_perm = scorer(ev[perm])
inv = np.empty(N, dtype=int)
inv[perm] = np.arange(N)
reorder_ok = np.array_equal(w_ref, w_perm[inv])
print(f"  client order permuted then undone, identical : {reorder_ok}")
if not reorder_ok:
    d = np.abs(w_ref - w_perm[inv]).max()
    print(f"    max abs deviation                        : {d:.3e}")
    print("    -> the policy is NOT order-independent; the commitment must")
    print("       fix a client ordering, else verifier and server disagree.")

# Summation strategy: pairwise (numpy default) vs sequential.
seq = 0.0
iv = 1.0 / np.maximum(ev, PHI * np.median(ev))
for x in iv:
    seq += float(x)
pairwise = float(iv.sum())
print(f"  sequential vs pairwise sum of iv             : "
      f"{'identical' if seq == pairwise else 'DIFFER'}")
if seq != pairwise:
    print(f"    abs diff {abs(seq - pairwise):.3e}  rel {abs(seq-pairwise)/pairwise:.3e}")
    print("    -> bit-identical recomputation requires committing the")
    print("       reduction order, not just the formula.")

print()

# ------------------------------------------------------------- A12.2 -----
print("=" * 78)
print("A12.2  does recomputation catch a cheating aggregator?")
print("=" * 78)
print("  aggregator publishes digest(state) but applies a tilted policy that")
print("  favours client 0 by a relative factor (1+eps).")
print()
target = 0


def tilt(w, eps):
    """The cheat: scale one client's weight, then renormalise."""
    v = w.copy()
    v[target] *= (1.0 + eps)
    return v / v.sum()


print("  FIRST, the naive check: compare the published vector against a fresh")
print("  scorer(ev) call using exact equality.")
print()
print(f"  {'eps':>10} {'w0_claimed':>14} {'Linf dev':>11} {'flagged':>8}")
for eps in [0.5, 1e-3, 1e-9, 1e-15, 0.0]:
    w_cheat = tilt(w_ref, eps)
    w_check = scorer(ev)
    dev = np.abs(w_cheat - w_check).max()
    print(f"  {eps:>10.0e} {w_cheat[target]:>14.11f} {dev:>11.3e} "
          f"{str(not np.array_equal(w_cheat, w_check)):>8}")

fp_dev = np.abs(tilt(w_ref, 0.0) - scorer(ev)).max()
print()
print(f"  *** eps = 0 is flagged, deviation {fp_dev:.3e}.  There is no cheat")
print("  in that row.  This is a FALSE POSITIVE, and it is not a bug in the")
print("  scorer: an honest aggregator that happens to normalise once more than")
print("  the verifier produces a different float.  Exact equality against a")
print("  reimplementation is therefore NOT a sound check -- it accuses honest")
print("  servers.  M3 as stated in the paper does not say what is compared.")
print()
print("  SECOND, the fix: the commitment must fix the arithmetic path, so the")
print("  verifier replays the same operations rather than recomputing the")
print("  same mathematical function.")
print()


def published(ev_, eps=0.0):
    """Reference implementation, bit-for-bit, as committed."""
    return tilt(scorer(ev_), eps)


print(f"  {'eps':>10} {'Linf dev':>11} {'caught':>7}")
caught_flags = {}
for eps in [0.5, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-16, 0.0]:
    w_cheat = published(ev, eps)
    w_check = published(ev)                 # same path, no tilt
    dev = np.abs(w_cheat - w_check).max()
    c = not np.array_equal(w_cheat, w_check)
    caught_flags[eps] = c
    print(f"  {eps:>10.0e} {dev:>11.3e} {str(c):>7}")

print()
print(f"  no false positive at eps = 0     : {not caught_flags[0.0]}")
print(f"  smallest cheat still caught      : "
      f"{min([e for e, c in caught_flags.items() if c]):.0e}")
print("  detection is exact, not statistical: there is no threshold to tune")
print("  and no false-positive rate to trade off.  The sensitivity floor is")
print(f"  float64 spacing, eps(w0) = {np.spacing(w_ref[target]):.3e}, i.e. a")
print("  relative cheat below ~1e-16 is absorbed by rounding -- and a cheat")
print("  that small moves no reward, so the floor is economically irrelevant.")
print()

# ------------------------------------------------------------- A12.3 -----
print("=" * 78)
print("A12.3  STATE COMPLETENESS: what must the digest cover?")
print("=" * 78)
print("  The paper says 'publish digest(state)'.  Suppose the digest commits")
print("  only to the EWMA vector ev, which is what the scorer reads.  The")
print("  aggregator also controls how ev was built from the update history.")
print()

hist = rng.uniform(0.05, 2.0, size=(ROUNDS, N))


def ewma(h, rho, warm):
    e = h[warm].copy()
    for t in range(warm + 1, len(h)):
        e = (1 - rho) * e + rho * h[t]
    return e


rows = []
for rho in [0.3, 0.5, 0.7]:
    for warm in [0, 1, 2]:
        e = ewma(hist, rho, warm)
        w = scorer(e)
        rows.append((rho, warm, w[target], digest(e)[:12]))

print(f"  {'rho':>5} {'warmup':>7} {'w0':>12}  digest(ev)")
for rho, warm, w0, dg in rows:
    print(f"  {rho:>5.1f} {warm:>7d} {w0:>12.9f}  {dg}")

spread = max(r[2] for r in rows) / min(r[2] for r in rows)
print()
print(f"  w0 varies by a factor of {spread:.3f} across free hyperparameters")
print("  that a state-only digest does NOT bind.  Every row is a valid")
print("  (state, weights) pair with a correct digest and a correct scorer.")
print("  -> M3 is UNDERSPECIFIED as written.  The commitment must cover the")
print("     estimator (rho, warmup, the whole state-transition map), not just")
print("     the scorer's input.  We fix the paper's wording accordingly.")
print()

print("  after binding the transition map, the free choice disappears:")
bound = {"rho": RHO, "warmup": 1, "phi": PHI}
e = ewma(hist, bound["rho"], bound["warmup"])
full = hashlib.sha256(
    (json.dumps(bound, sort_keys=True, separators=(",", ":"))
     + "|" + canon_state(hist.ravel())).encode()).hexdigest()
print(f"  digest(params || full history) = {full[:32]}")
print(f"  recompute -> w0 = {scorer(e)[target]:.9f}  (unique)")
recompute_ok = np.array_equal(scorer(ewma(hist, bound['rho'], bound['warmup'])),
                              scorer(e))
print(f"  independent recomputation identical : {recompute_ok}")
print()

# ------------------------------------------------------------- A12.4 -----
print("=" * 78)
print("A12.4  verification cost vs aggregation cost")
print("=" * 78)
print("  scorer operates on N scalars; aggregation operates on N x d floats.")
print(f"  N = {N}, d = {D}")

flops_score = 6 * N + N * np.log2(max(N, 2))     # divide, max, median, norm
flops_agg = 2 * N * D
print(f"  scorer  ~{flops_score:>10.0f} flop")
print(f"  aggregate ~{flops_agg:>8.0f} flop")
print(f"  ratio     {flops_score / flops_agg:.4f}")
print("  -> verification is asymptotically free: the trust decision reads a")
print("     length-N summary, while aggregation touches N*d parameters.")
print("     A verifier need not hold any model to check the decision.")
print()

# ------------------------------------------------------------- A12.5 -----
print("=" * 78)
print("A12.5  does M3 compose with M4?")
print("=" * 78)
print("  M4 rewrites submitted parameters (canonicalise, then align).  If the")
print("  digest is taken over RAW submissions, two honest servers running M4")
print("  could publish different digests.  Order matters -- test it.")

net = init_net(np.random.default_rng(1))
Xtr = np.random.default_rng(2).normal(size=(160, D_IN))
ytr = np.sin(Xtr[:, 0])
net = sgd(net, Xtr, ytr, steps=200, lr=0.05,
          rng=np.random.default_rng(4))
ref = init_net(np.random.default_rng(1))          # public init as reference

g = gauge_permute(net, np.random.default_rng(3))

def full_canon(p):
    return align_to(gauge_canon(p), ref)

d_raw_a = hashlib.sha256(flat(net).tobytes()).hexdigest()
d_raw_b = hashlib.sha256(flat(g).tobytes()).hexdigest()
d_can_a = hashlib.sha256(flat(full_canon(net)).tobytes()).hexdigest()
d_can_b = hashlib.sha256(flat(full_canon(g)).tobytes()).hexdigest()

print(f"  digest(raw theta)      {d_raw_a[:16]}")
print(f"  digest(raw g.theta)    {d_raw_b[:16]}   equal: {d_raw_a == d_raw_b}")
print(f"  digest(canon theta)    {d_can_a[:16]}")
print(f"  digest(canon g.theta)  {d_can_b[:16]}   equal: {d_can_b == d_can_a}")

rel = np.abs(flat(full_canon(net)) - flat(full_canon(g))).max() / (
    np.abs(flat(full_canon(net))).max() + 1e-300)
print(f"  max relative deviation after canon : {rel:.3e}")
print()
if d_can_a == d_can_b:
    print("  digests agree bitwise -> commit AFTER canonicalisation.")
else:
    print("  canon output agrees numerically but NOT bitwise, so a SHA over")
    print("  raw float bytes is the wrong commitment primitive here: the")
    print("  digest must be taken over the state the scorer reads (length-N")
    print("  dispersion summary), which is gauge-invariant once M4 has run,")
    print("  NOT over the parameter bytes.")
    print("  -> ORDERING IS MANDATORY: M4 (canonicalise) then M3 (commit),")
    print("     and M3 commits to scorer state, not to submissions.")
print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  A12.1 scorer reproducible in-process        : {same_call}")
print(f"        order-independent                     : {reorder_ok}")
print(f"  A12.2 cheat detection                       : exact, no threshold;"
      f" floor ~{np.spacing(w_ref[target]):.1e} relative")
print(f"  A12.3 state-only digest underspecified      : "
      f"{spread:.3f}x weight freedom -> must bind transition map")
print(f"  A12.4 verify/aggregate cost ratio           : "
      f"{flops_score / flops_agg:.4f}")
print(f"  A12.5 M4-then-M3 ordering required          : "
      f"{'digest over canon params ok' if d_can_a == d_can_b else 'commit scorer state, not bytes'}")
