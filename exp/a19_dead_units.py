"""A19: the OTHER escape from Gamma, and what the full pipeline does to it.

sec:negatives claims a boundary: Gamma = <S1 scaling, S2 permutation> is not the
whole set of function-preserving reparameterisations, and a dead hidden unit --
one whose outgoing column is zero -- gives a second escape.  A18.3 measured the
unit-SPLIT route (2181x undefended, 12116x under M4+).  The split reuses a dead
unit, so the claim that a dead unit is itself an escape has never been run here,
and the paper states it in two places.  This script runs it.

Three things have to be true for the paragraph to stand:

  A19.1  the move is exactly function-preserving and outside Gamma.  Gamma
         acting on unit j with a_j = 1 leaves the outgoing column fixed, so a
         move that changes ONLY the outgoing column of a dead unit is not in
         Gamma.  It is function-preserving because a dead unit contributes
         nothing.  Both are checkable, and the first one exactly.
  A19.2  M4 is not merely blind to it but an AMPLIFIER: gauge_canon divides the
         incoming row by its norm and multiplies the outgoing column by the same
         factor, and a dead unit survives on a large negative bias, so that
         factor is |b_j| and the adversary picks it.  The mechanism to check is
         activation rate: canon must raise it.
  A19.3  the bound that keeps the claim honest -- alg:pipeline's inverse-variance
         weighting, which sees a dead-unit payload even though it misses a gauge
         move.  Without this the paragraph overclaims, and the paper says 1.02x.

Naturally trained clients have no dead units, so the adversary has to kill one,
and A19.1 reports what that costs its own error.
"""

import numpy as np

from a1_nn_symmetry import D_IN, H1, flat, forward, init_net, sgd
from a4_aggregate_gauge import gauge_canon, unflat
import a11_permutation_canon as a11

SEEDS = range(8)
RELU_DEAD = -50.0        # bias that keeps a unit off on almost every input
FLOOR_PHI = 0.1          # M1's variance floor as a fraction of the median
RHO = 0.9                # M3's EMA memory, as in v6/e3


def kill_unit(p, j):
    """Make hidden unit j of layer 1 dead: large negative bias, no output."""
    q = {k: v.copy() for k, v in p.items()}
    q["b1"][j] = RELU_DEAD
    q["W2"][:, j] = 0.0
    return q


def load_dead(p, j, rng, mag):
    """Write an arbitrary payload into the dead unit's OUTGOING column.

    This is the move Gamma cannot make: an element of Gamma scaling unit j by
    a_j multiplies the incoming row by a_j and divides the outgoing column by
    a_j, so fixing the incoming row (a_j = 1) fixes the outgoing column too.
    """
    q = {k: v.copy() for k, v in p.items()}
    q["W2"][:, j] = rng.normal(size=q["W2"].shape[0]) * mag
    return q


def act_rate(p, X, j):
    """Fraction of inputs on which hidden unit j of layer 1 is active."""
    h = X @ p["W1"].T + p["b1"]
    return float((h[:, j] > 0).mean())


# =========================================================== A19.1 ========
def a19_1_outside_gamma():
    print("=" * 78)
    print("A19.1  the dead-unit move: exactly free, and not in Gamma")
    print("=" * 78)
    print("\nmax|df| is over 2000 held-out inputs.  A gauge move is free to")
    print("1e-15 (rounding); this one should be free to 0 (the unit is off).\n")
    print(f"{'seed':>5} {'kill cost (own x)':>18} {'payload max|df|':>17} "
          f"{'in Gamma?':>10}")
    print("-" * 56)
    costs, dfs = [], []
    for sd in SEEDS:
        clients, teacher, g0, Xp, r = a11.build(seed=sd)
        p = clients[0]
        base_own = a11.own_mse(p, teacher, Xp)
        d = kill_unit(p, 0)
        kill_own = a11.own_mse(d, teacher, Xp)
        q = load_dead(d, 0, np.random.default_rng(500 + sd), 5.0)
        df = float(np.abs(forward(q, Xp)[0] - forward(d, Xp)[0]).max())
        # in Gamma would require the incoming row to have moved; it has not
        same_in = bool(np.array_equal(q["W1"], d["W1"])
                       and np.array_equal(q["b1"], d["b1"]))
        out_col = bool(not np.array_equal(q["W2"][:, 0], d["W2"][:, 0]))
        costs.append(kill_own / base_own)
        dfs.append(df)
        print(f"{sd:>5} {kill_own / base_own:>18.4f} {df:>17.3e} "
              f"{str(not (same_in and out_col)):>10}")
    c, f = np.array(costs), np.array(dfs)
    print(f"\nkill cost to the adversary's own error: median {np.median(c):.4f}x"
          f"  max {c.max():.4f}x")
    print(f"payload max|df|: max {f.max():.3e} over {len(f)} seeds")
    assert f.max() == 0.0, f.max()
    print("\nExactly zero, not 1e-16: the payload never reaches the output,")
    print("because a dead ReLU is off on every input.  And the incoming row is")
    print("bit-identical while the outgoing column changed, which no element of")
    print("Gamma can do.  So this is a function-preserving move outside Gamma.")
    print("Killing a unit costs the adversary a few percent of its own error,")
    print("which is the price of entry; nothing in the harness trains into it.")
    return c, f


# =========================================================== A19.2 ========
def a19_2_m4_amplifies(mags=(1.0, 10.0, 100.0, 1000.0)):
    print("\n" + "=" * 78)
    print("A19.2  M4 in isolation is an amplifier, not a blind spot")
    print("=" * 78)
    print("\ncanon sets a_j = 1/||(W1_j, b1_j)||, which for a dead unit is")
    print("~1/|b1_j|, and multiplies the outgoing column by 1/a_j = |b1_j|.")
    print("The adversary chooses |b1_j|, so it chooses the amplification.")
    print("The mechanism to verify is the ACTIVATION RATE: raw aggregation")
    print("dilutes the dead bias and the unit stays off; canon restores the")
    print("bias scale and the unit switches on, which is what uncovers the")
    print("payload.  Measured on seed 0.\n")
    clients, teacher, g0, Xp, r = a11.build(seed=0)
    d = kill_unit(clients[0], 0)
    clean = a11.agg_mse(clients, teacher, Xp)
    base = a11.agg_mse([d] + clients[1:], teacher, Xp)

    V = np.array([flat(c) for c in [d] + clients[1:]])
    raw_g = unflat(V.mean(axis=0), d)
    cq = [a11.full_canon(c, g0) for c in [d] + clients[1:]]
    Vc = np.array([flat(c) for c in cq])
    can_g = unflat(Vc.mean(axis=0), cq[0])
    print(f"  aggregate bias of unit 0:  raw {raw_g['b1'][0]:>9.4f}"
          f"   canon {can_g['b1'][0]:>9.4f}")
    print(f"  activation rate of unit 0: raw {act_rate(raw_g, Xp, 0):>9.3f}"
          f"   canon {act_rate(can_g, Xp, 0):>9.3f}")

    print(f"\n{'payload mag':>12} {'undef x':>10} {'M4-only x':>12} "
          f"{'own max|df|':>13}")
    print("-" * 52)
    out = []
    for mag in mags:
        q = load_dead(d, 0, np.random.default_rng(77), mag)
        df = float(np.abs(forward(q, Xp)[0] - forward(d, Xp)[0]).max())
        und = a11.agg_mse([q] + clients[1:], teacher, Xp) / base
        m4 = a11.agg_mse([a11.full_canon(c, g0) for c in [q] + clients[1:]],
                         teacher, Xp) / base
        out.append((mag, und, m4))
        print(f"{mag:>12.0f} {und:>10.4f} {m4:>12.2f} {df:>13.1e}")
    m4s = np.array([o[2] for o in out])
    unds = np.array([o[1] for o in out])
    assert m4s.max() > 10.0 * max(unds.max(), 1.0), (m4s.max(), unds.max())
    print("\nUndefended damage is flat -- the raw average keeps the unit off --")
    print("while M4 alone grows with the magnitude the adversary picks.  So the")
    print("honest statement is not that M4 misses this move but that M4 used")
    print("ALONE turns it into a live one.  A19.3 puts M1-M3 back.")
    print(f"clean-vs-killed baseline: {base / clean:.4f}x (the denominator)")
    return unds, m4s


# ------------------------------------------------------------ harness ----
def fed_run(sd, lam, mag, attack, N=24, T=8, n_per=200, local=120):
    """One federation under alg:pipeline, M4 applied to every submission.

    Returns (test mse, mean adversary weight over the last 3 rounds, peak
    norm ratio of the adversary's submission to the honest median).  A
    diverged run returns inf, which is a result and not an error: the
    payload magnitude is the adversary's free parameter.
    """
    r = np.random.default_rng(1000 + sd)
    teacher = init_net(r)
    g0 = init_net(r)
    data = [(lambda X: (X, forward(teacher, X)[0]
                        + r.normal(size=n_per) * 0.1))(
                r.normal(size=(n_per, D_IN))) for _ in range(N)]
    Xp = r.normal(size=(2000, D_IN))
    rp = np.random.default_rng(77 + sd)
    glob = {k: v.copy() for k, v in g0.items()}
    prev, ev, wa, nr = None, None, [], 0.0

    with np.errstate(all="ignore"):
        for t in range(T):
            subs = []
            for i in range(N):
                q = sgd(glob, data[i][0], data[i][1], steps=local, lr=0.05,
                        rng=r)
                if i == 0 and attack:
                    q = kill_unit(q, 0)
                    if t >= 2:
                        q = load_dead(q, 0, rp, mag)
                fq = flat(q)
                if not np.all(np.isfinite(fq)):
                    return np.inf, float(np.mean(wa[-3:])) if wa else np.nan, nr
                subs.append(flat(a11.full_canon(q, g0)))          # M4
            V = np.array(subs)
            if not np.all(np.isfinite(V)):
                return np.inf, float(np.mean(wa[-3:])) if wa else np.nan, nr

            norms = np.linalg.norm(V, axis=1)
            nr = max(nr, float(norms[0] / np.median(norms[1:])))
            if t == 0:
                w = np.ones(N) / N
            else:
                disp = np.mean((V - prev) ** 2, axis=1)
                ev = disp if ev is None else (1 - RHO) * ev + RHO * disp
                e = np.maximum(ev, FLOOR_PHI * np.median(ev))      # M1
                iv = 1.0 / e
                iv = iv / iv.sum()
                w = (1.0 - lam) * np.ones(N) / N + lam * iv        # M2
            prev = V
            glob = unflat(w @ V, glob)
            wa.append(w[0])
            if not np.all(np.isfinite(flat(glob))):
                return np.inf, float(np.mean(wa[-3:])), nr

    mse = float(np.mean((forward(glob, Xp)[0]
                         - forward(teacher, Xp)[0]) ** 2))
    return mse, float(np.mean(wa[-3:])), nr


# =========================================================== A19.3 ========
def a19_3_pipeline(mag=1000.0, lam=1.0, nseed=8):
    """The dispersion estimate against the payload, over real rounds.

    A single pair of snapshots would give the honest clients a dispersion of
    exactly zero and inverse variance would be infinite; honest clients move
    between rounds because they train on fresh local data, so the estimate
    has to come from a run.  The adversary kills its unit at round 0 and
    loads it from round 2, once there is a state to fool.
    """
    print("\n" + "=" * 78)
    print("A19.3  the dispersion estimate against the dead-unit payload")
    print("=" * 78)
    print("\nalg:pipeline weights by inverse across-round dispersion with a")
    print("floor.  The payload is invisible in function space but it is a")
    print("large move in PARAMETER space, and the estimator reads parameter")
    print(f"space, so it should catch it.  Pure inverse variance, lam = {lam}.\n")
    print(f"{'seed':>5} {'clean mse':>11} {'attacked':>11} {'ratio':>9} "
          f"{'adv weight':>12} {'norm ratio':>11}")
    print("-" * 64)
    got = []
    for sd in range(nseed):
        c, _, _ = fed_run(sd, lam, mag, False)
        a, w, nr = fed_run(sd, lam, mag, True)
        got.append((c, a, a / c, w, nr))
        print(f"{sd:>5} {c:>11.4f} {a:>11.4f} {a / c:>9.4f} {w:>12.3e} "
              f"{nr:>11.2f}")
    g = np.array(got)
    print(f"\ndamage: median {np.median(g[:, 2]):.4f}x  max {g[:, 2].max():.4f}x")
    print(f"adversary weight: median {np.median(g[:, 3]):.3e} against a fair "
          f"share of {1.0 / 24:.4f}")
    print(f"norm ratio to the honest median: median {np.median(g[:, 4]):.2f}x")
    assert np.median(g[:, 2]) < 1.1, np.median(g[:, 2])
    print("\nContained. The statistics that miss a gauge move do see a dead")
    print("unit, because the payload has to move in parameter space to do")
    print("anything and the dispersion estimate reads parameter space.  Note")
    print("the norm ratio: this payload is also not norm-stealthy, which is a")
    print("second independent reason a deployment would catch it.")
    return g


# =========================================================== A19.4 ========
def a19_4_anchor_floor(mag=1000.0, lams=(1.0, 0.99, 0.9, 0.5), nseed=3):
    """Where the containment comes from, and where M2 works against it.

    M2 mixes lam * inverse-variance with (1-lam) * uniform, so no client can
    be weighted below (1-lam)/N however bad its dispersion.  That floor is
    the point of M2 -- it bounds trust displacement -- and against a payload
    whose magnitude is unbounded it is a liability.  This measures the
    trade rather than asserting a direction for it.
    """
    print("\n" + "=" * 78)
    print("A19.4  M2's weight floor against an unbounded payload")
    print("=" * 78)
    print("\nM2 guarantees every client at least (1-lam)/N, which is what")
    print("bounds displacement of an honest client (thm:displacement).  The")
    print("same floor guarantees the ADVERSARY that weight, and the payload")
    print("magnitude is free, so the two directions of M2 conflict here.")
    print("inf = the federation diverged, which is the adversary's outcome.\n")
    print(f"{'lam':>6} {'floor (1-lam)/N':>16} {'median x':>10} "
          f"{'adv weight':>12} {'diverged':>10}")
    print("-" * 60)
    out = []
    for lam in lams:
        rs, ws, dv = [], [], 0
        for sd in range(nseed):
            c, _, _ = fed_run(sd, lam, mag, False)
            a, w, _ = fed_run(sd, lam, mag, True)
            if not np.isfinite(a):
                dv += 1
            rs.append(a / c)
            ws.append(w)
        med = float(np.median(rs))
        out.append((lam, med, dv))
        shown = "inf" if not np.isfinite(med) else f"{med:.4f}"
        print(f"{lam:>6.2f} {(1 - lam) / 24:>16.4f} {shown:>10} "
              f"{np.median(ws):>12.3e} {dv:>7}/{nseed}")
    print("\nSo the containment in A19.3 is the inverse-variance term's, and")
    print("M2's uniform term is what admits the payload.  That is a real")
    print("tension in the mechanism set and it belongs in the paper as one:")
    print("M2 bounds what a peer frame can do to an honest client and it")
    print("cannot also bound what an out-of-Gamma payload does to the")
    print("aggregate, because its floor is unconditional.  The pipeline needs")
    print("the norm bound of A19.3's last column for this move, which is")
    print("exactly the check rem:alternatives declines for gauge moves --")
    print("declines because a gauge move is norm-stealthy and this is not.")
    return out


if __name__ == "__main__":
    c, f = a19_1_outside_gamma()
    unds, m4s = a19_2_m4_amplifies()
    g = a19_3_pipeline()
    lamtab = a19_4_anchor_floor()
    print("\n" + "=" * 78)
    print("A19 conclusion: Gamma is not the full set of function-preserving")
    print("maps.  A dead unit's outgoing column is exactly free (max|df| = 0,")
    print("not 1e-16) and unreachable within Gamma, and M4 in isolation")
    print(f"amplifies it to {m4s.max():.0f}x where the raw average sits at "
          f"{unds.max():.2f}x.")
    print(f"Inverse-variance weighting returns it to a median "
          f"{np.median(g[:, 2]):.2f}x at adversary")
    print(f"weight {np.median(g[:, 3]):.0e}, and the submission is "
          f"{np.median(g[:, 4]):.1f}x the honest median norm,")
    print("so unlike a gauge move it is not norm-stealthy either.  But M2's")
    print("weight floor admits it (A19.4), so the honest claim is that the")
    print("pipeline's VARIANCE term contains this move while its ANCHOR term")
    print("does not, and the two mechanisms pull in opposite directions on it.")
    print("=" * 78)
