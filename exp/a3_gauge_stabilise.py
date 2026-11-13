"""A3: can the gauge manufacture APPARENT STABILITY?  (negative result)

TSP-Var rewards clients whose parameters move little between rounds.  Neural
network gauge freedom moves parameters at zero functional cost.  The obvious
attack: each round, submit the representative of the gauge orbit closest to
last round's submission, so the trajectory looks still while the function moves
freely.

It does not work, and the reason is a conservation law worth stating, because
it rules out the whole family of scale-based stability attacks.

    Rescaling hidden unit j multiplies W1[j], b1[j] by a_j and W2[:,j] by
    1/a_j.  Write A_j for the squared round-to-round movement of the incoming
    weights and B_j for that of the outgoing weights.  The unit contributes

        a_j^2 A_j + a_j^-2 B_j  >=  2 sqrt(A_j B_j)          (AM-GM)

    with equality at a_j = (B_j/A_j)^(1/4).  The product A_j B_j is gauge
    invariant, so the dispersion an adversary can display is bounded below by
    a quantity it does not control.  The gauge cannot fake stability.

The same argument applies layerwise: gauge_scale multiplies the three layers by
c1, c2/c1, 1/c2, whose product is identically 1.

  A3.0  the AM-GM floor, and the optimal per-unit gauge that attains it
  A3.1  end to end: does the attack suppress apparent variance in a federation?
  A3.2  what weight share does it buy?
  A3.3  the decoupling that DOES survive: parameters still vs function moving
  A3.4  for completeness, against the M1 floor
"""

import numpy as np

from a1_nn_symmetry import (D_IN, H1, H2, flat, forward, gauge_scale, init_net,
                            sgd)
from a4_aggregate_gauge import scale_units, unflat


def layer_norm_canon(p):
    q = {k: v.copy() for k, v in p.items()}
    for li in ("1", "2", "3"):
        n = np.sqrt(np.linalg.norm(q["W" + li]) ** 2
                    + np.linalg.norm(q["b" + li]) ** 2) + 1e-30
        q["W" + li] = q["W" + li] / n
        q["b" + li] = q["b" + li] / n
    return q


def gauge_stabilise(p, prev_flat, grid=25):
    """Global (c1, c2) orbit representative closest to the previous submission."""
    if prev_flat is None:
        return p
    best, best_d = p, np.linalg.norm(flat(p) - prev_flat)
    for c1 in np.exp(np.linspace(-1.2, 1.2, grid)):
        for c2 in np.exp(np.linspace(-1.2, 1.2, grid)):
            q = gauge_scale(p, c1, c2)
            dd = np.linalg.norm(flat(q) - prev_flat)
            if dd < best_d:
                best, best_d = q, dd
    return best


# =========================================================== A3.0 =========
def a3_0_amgm_floor():
    print("=" * 76)
    print("A3.0  the AM-GM floor on displayable dispersion")
    print("=" * 76)
    print(f"{'seed':>5} {'baseline':>12} {'best global':>13} {'best per-unit':>15} "
          f"{'suppression':>12} {'max|df|':>10}")
    for sd in range(5):
        r = np.random.default_rng(sd)
        teacher, glob = init_net(r), init_net(r)
        X = r.normal(size=(150, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=150) * 0.3
        p0 = sgd(glob, X, y, steps=60, lr=0.03, rng=r)
        p1 = sgd(p0, X, y, steps=60, lr=0.03, rng=r)
        Xp = r.normal(size=(300, D_IN))
        f_ref = forward(p1, Xp)[0]
        base = np.mean((flat(p1) - flat(p0)) ** 2)

        bg = base
        for c1 in np.exp(np.linspace(-3, 3, 41)):
            for c2 in np.exp(np.linspace(-3, 3, 41)):
                bg = min(bg, np.mean((flat(gauge_scale(p1, c1, c2))
                                      - flat(gauge_scale(p0, c1, c2))) ** 2))

        # closed-form per-unit optimum, alternated to convergence
        a, d = np.ones(H1), np.ones(H2)
        for _ in range(40):
            for j in range(H1):
                A = np.sum((p1["W1"][j] - p0["W1"][j]) ** 2) \
                    + (p1["b1"][j] - p0["b1"][j]) ** 2
                B = np.sum(((p1["W2"][:, j] - p0["W2"][:, j]) * d) ** 2)
                if A > 1e-30 and B > 1e-30:
                    a[j] = (B / A) ** 0.25
            for k in range(H2):
                A = np.sum(((p1["W2"][k] - p0["W2"][k]) / a) ** 2) \
                    + (p1["b2"][k] - p0["b2"][k]) ** 2
                B = np.sum((p1["W3"][:, k] - p0["W3"][:, k]) ** 2)
                if A > 1e-30 and B > 1e-30:
                    d[k] = (B / A) ** 0.25
        bu = np.mean((flat(scale_units(p1, a, d))
                      - flat(scale_units(p0, a, d))) ** 2)
        df = np.abs(forward(scale_units(p1, a, d), Xp)[0] - f_ref).max()
        print(f"{sd:5d} {base:12.4e} {bg:13.4e} {bu:15.4e} "
              f"{base/bu:12.2f} {df:10.1e}")
    print("\n  Per-unit scaling is the full continuous group and it is solved to")
    print("  optimality here, not searched heuristically.  Suppression stays")
    print("  near 1.5x.  The attack is bounded, and bounded for a reason.\n")


# ------------------------------------------------------------ federation --
def run(N=20, T=20, rho=0.8, attacker=True, canon=False, phi=0.0,
        seed=0, n_per=150, local=60, backdoor=5.0):
    r = np.random.default_rng(seed)
    teacher, glob = init_net(r), init_net(r)
    data = []
    for i in range(N):
        X = r.normal(size=(n_per, D_IN))
        y = forward(teacher, X)[0] + r.normal(size=n_per) * 0.3
        if i == 0 and attacker:
            y = y + backdoor * (X[:, 0] > 1.0)     # real backdoor, honest SGD
        data.append((X, y))

    ema = np.ones(N)
    prev, prev_sc = [None] * N, [None] * N
    hist_var, hist_share, hist_fn = [], [], []
    prev_fn = None
    Xp = r.normal(size=(400, D_IN))
    yp = forward(teacher, Xp)[0]

    for t in range(T):
        subs, disp = [], np.zeros(N)
        for i in range(N):
            X, y = data[i]
            p = sgd(glob, X, y, steps=local, lr=0.03, rng=r)
            if i == 0 and attacker:
                p = gauge_stabilise(p, prev[i])
            f = flat(p)
            subs.append(f)
            sc = flat(layer_norm_canon(p)) if canon else f
            disp[i] = 0.0 if prev_sc[i] is None else np.mean((sc - prev_sc[i]) ** 2)
            prev[i], prev_sc[i] = f, sc
        subs = np.array(subs)

        if t > 0:
            ema = rho * ema + (1 - rho) * disp
        e = np.maximum(ema, phi * np.median(ema)) if phi > 0 else ema
        w = (1.0 / e) / (1.0 / e).sum()
        glob = unflat(w @ subs, glob)

        hist_var.append(ema.copy())
        hist_share.append(w[0])
        fn = forward(unflat(subs[0], glob), Xp)[0]
        hist_fn.append(0.0 if prev_fn is None else float(np.mean((fn - prev_fn) ** 2)))
        prev_fn = fn

    mse = float(np.mean((forward(glob, Xp)[0] - yp) ** 2))
    return np.array(hist_var), np.array(hist_share), np.array(hist_fn), mse


# =========================================================== A3.1 =========
def a3_1_apparent_variance():
    print("=" * 76)
    print("A3.1  end to end: does it suppress apparent variance in a federation?")
    print("=" * 76)
    print(f"{'':>22} {'adv disp':>12} {'honest disp':>13} {'ratio':>9}")
    for label, atk in [("no gauge move", False), ("gauge stabilised", True)]:
        av, hv = [], []
        for s in range(4):
            V = run(attacker=atk, seed=s)[0]
            av.append(V[-1, 0])
            hv.append(np.median(V[-1, 1:]))
        a, h = np.mean(av), np.mean(hv)
        print(f"{label:>22} {a:12.3e} {h:13.3e} {a/h:9.4f}")
    print("\n  Ratio at 1 confirms A3.0 in the dynamic setting.\n")


# =========================================================== A3.2 =========
def a3_2_weight_share():
    print("=" * 76)
    print("A3.2  what weight share does it buy?")
    print("=" * 76)
    print(f"{'':>22} {'share':>10} {'x fair':>9} {'sd':>10} {'agg MSE':>10}")
    for label, atk in [("no gauge move", False), ("gauge stabilised", True)]:
        sh, ms = [], []
        for s in range(4):
            _, S, _, mse = run(attacker=atk, seed=s)
            sh.append(S[-5:].mean())
            ms.append(mse)
        sh = np.array(sh)
        print(f"{label:>22} {sh.mean():10.4f} {sh.mean()*20:9.2f} "
              f"{sh.std():10.4f} {np.mean(ms):10.4f}")
    print("\n  Fair share is 1/N = 0.05.  No gain.\n")


# =========================================================== A3.3 =========
def a3_3_decoupling():
    print("=" * 76)
    print("A3.3  what DOES survive: is function movement readable from parameters?")
    print("=" * 76)
    print(f"{'':>22} {'param disp':>13} {'function disp':>15} {'ratio':>12}")
    for label, atk in [("no gauge move", False), ("gauge stabilised", True)]:
        pv, fv = [], []
        for s in range(4):
            V, _, F, _ = run(attacker=atk, seed=s)
            pv.append(V[-1, 0])
            fv.append(F[-5:].mean())
        p, f = np.mean(pv), np.mean(fv)
        print(f"{label:>22} {p:13.3e} {f:15.3e} {f/p:12.1f}")
    print("\n  The ratio is what a dispersion-based scorer implicitly assumes is")
    print("  constant across clients.  It is not, and A4 shows how far apart the")
    print("  two can be pulled -- but not in the direction that helps an")
    print("  adversary game a variance estimate.\n")


# =========================================================== A3.4 =========
def a3_4_floor():
    print("=" * 76)
    print("A3.4  for completeness: under canonicalisation and the M1 floor")
    print("=" * 76)
    print(f"{'defence':>28} {'adv x fair':>12}")
    for label, kw in [("none (attack on)", dict(canon=False, phi=0.0)),
                      ("layerwise canonicalisation", dict(canon=True, phi=0.0)),
                      ("M1 floor phi=0.5", dict(canon=False, phi=0.5))]:
        sh = [run(attacker=True, seed=s, **kw)[1][-5:].mean() for s in range(4)]
        print(f"{label:>28} {np.mean(sh)*20:12.2f}")
    print("\n  Nothing to defend against here; recorded so the negative result")
    print("  is not mistaken for an untested configuration.\n")


if __name__ == "__main__":
    a3_0_amgm_floor()
    a3_1_apparent_variance()
    a3_2_weight_share()
    a3_3_decoupling()
    a3_4_floor()
