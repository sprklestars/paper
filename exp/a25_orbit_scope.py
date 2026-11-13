"""A25: which hypothesis classes carry a function-preserving orbit?

sec:gauge said "for a linear model there is none" while sec:tspgauge named the
Kalman filter as the orbit-free control.  Those two are inconsistent: a linear
state-space model is linear and carries an exact orbit.  This script separates
the two claims and fixes the scope to redundancy of the parameterisation.

  A25.1  plain linear regression y = w^T x.  The parameter vector IS the
         function, so the only available move changes the client's own
         predictions.  The claim holds here.
  A25.2  linear state-space model, the class a Kalman filter carries.  The
         similarity transform (A,B,C) -> (T A T^-1, T B, C T^-1) preserves the
         input-output map exactly for any invertible T and moves the parameters
         arbitrarily far, so averaging breaks.  The claim fails here.
  A25.3  where the boundary runs: multiplicative redundancy breaks averaging,
         additive redundancy does not.
"""

import numpy as np

rng = np.random.default_rng(0)
N = 10


def ssm_forward(A, B, C, us):
    """x_{t+1} = A x_t + B u_t ; y_t = C x_t, zero initial state."""
    x = np.zeros(A.shape[0])
    return np.array([(C @ x, x := A @ x + B @ u)[0] for u in us])


def avg(ps):
    return tuple(np.mean([p[k] for p in ps], axis=0) for k in range(len(ps[0])))


def main():
    print("=" * 78)
    print("A25.1  plain linear regression y = w^T x")
    print("=" * 78)
    d = 3
    w = rng.normal(size=d)
    X = rng.normal(size=(200, d))
    f = X @ w
    for t in (2.0, 5.0):
        print(f"  scale w by {t}: max|df| = {np.abs(X @ (t * w) - f).max():.3e}"
              "  -> the client's own function moved")
    print("  no nontrivial move preserves it: the orbit is trivial.")

    print()
    print("=" * 78)
    print("A25.2  linear state-space model (the Kalman filter's class)")
    print("=" * 78)
    n_s, n_u = 3, 2
    us = rng.normal(size=(60, n_u))
    clients = [(rng.normal(size=(n_s, n_s)) * 0.3,
                rng.normal(size=(n_s, n_u)) * 0.5,
                rng.normal(size=(2, n_s)) * 0.5) for _ in range(N)]
    clean = ssm_forward(*avg(clients), us)
    print(f"{'T':>20} {'own max|df|':>13} {'aggregate damage':>18}")
    worst_own, best_dmg = 0.0, 0.0
    for desc, T in [("diag(2,1,1)", np.diag([2.0, 1.0, 1.0])),
                    ("diag(5,1,1)", np.diag([5.0, 1.0, 1.0])),
                    ("diag(20,1,1)", np.diag([20.0, 1.0, 1.0])),
                    ("random invertible", rng.normal(size=(n_s, n_s)))]:
        Ti = np.linalg.inv(T)
        A, B, C = clients[0]
        moved = (T @ A @ Ti, T @ B, C @ Ti)
        own = np.abs(ssm_forward(*moved, us) - ssm_forward(A, B, C, us)).max()
        atk = ssm_forward(*avg([moved] + clients[1:]), us)
        dmg = ((atk - clean) ** 2).mean() / (clean ** 2).mean()
        worst_own, best_dmg = max(worst_own, own), max(best_dmg, dmg)
        print(f"{desc:>20} {own:13.3e} {dmg:18.4g}")
    print(f"\n  own function moves by at most {worst_own:.1e}, float round-off,")
    print(f"  while the aggregate moves by {best_dmg:.3g} of its own scale.")
    assert worst_own < 1e-12, worst_own
    assert best_dmg > 0.1, best_dmg

    print()
    print("=" * 78)
    print("A25.3  which redundancy breaks averaging")
    print("=" * 78)
    # multiplicative: f(x) = a * (w^T x), orbit (a,w) -> (t a, w/t)
    Xm = rng.normal(size=(200, 4))
    cl = [(rng.normal() + 2.0, rng.normal(size=4)) for _ in range(N)]
    fm = lambda p: p[0] * (Xm @ p[1])
    am = lambda ps: (np.mean([p[0] for p in ps]),
                     np.mean([p[1] for p in ps], axis=0))
    clean_m = fm(am(cl))
    t = 5.0
    mv = (t * cl[0][0], cl[0][1] / t)
    own_m = np.abs(fm(mv) - fm(cl[0])).max()
    dmg_m = np.abs(fm(am([mv] + cl[1:])) - clean_m).max() / np.abs(clean_m).max()
    print(f"  multiplicative (a,w)->(ta,w/t):  own {own_m:.2e}  "
          f"aggregate shift {dmg_m:.4g}")
    # additive: f(x) = (u+v)^T x, orbit (u,v) -> (u+c, v-c)
    cla = [(rng.normal(size=4), rng.normal(size=4)) for _ in range(N)]
    fa = lambda p: Xm @ (p[0] + p[1])
    aa = lambda ps: (np.mean([p[0] for p in ps], axis=0),
                     np.mean([p[1] for p in ps], axis=0))
    clean_a = fa(aa(cla))
    c = np.full(4, 9.0)
    av = (cla[0][0] + c, cla[0][1] - c)
    own_a = np.abs(fa(av) - fa(cla[0])).max()
    dmg_a = np.abs(fa(aa([av] + cla[1:])) - clean_a).max() / np.abs(clean_a).max()
    print(f"  additive     (u,v)->(u+c,v-c):  own {own_a:.2e}  "
          f"aggregate shift {dmg_a:.4g}")
    assert own_m < 1e-12 and dmg_m > 0.1
    assert own_a < 1e-12 and dmg_a < 1e-12
    print("\n  Both are exact orbits; only the multiplicative one survives")
    print("  averaging, because averaging commutes with a common translation")
    print("  and not with a common rescaling.  eq:homog is multiplicative.")

    print()
    print("=" * 78)
    print("A25 conclusion: the orbit-free property is minimal parameterisation")
    print("and not linearity.  sec:gauge must say 'linear regressor', and the")
    print("Kalman filter cannot serve as the orbit-free control it was used as.")
    print("=" * 78)


if __name__ == "__main__":
    main()
