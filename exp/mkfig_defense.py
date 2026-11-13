"""Generates fig/defense.pdf -- what the defence closes, and what it costs.

Left:   the combined attack. Four adversary moves inside the gauge group,
        undefended vs canonicalised, log scale. The undefended bar reaches
        9648x while the audit sees nothing; every defended bar returns to 1.00.
Right:  the cost, on two axes that must be shown together --
        honest-federation cost (the 0.1% we charge every round) against the
        damage removed. A defence that removed the damage by destroying the
        aggregate would show up here.

All numbers are recomputed by calling a11's own build/agg/canon functions.
Nothing is transcribed.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a1_nn_symmetry import H1, H2, forward, gauge_permute
from a4_aggregate_gauge import scale_units
from a11_permutation_canon import agg_mse, build, full_canon, own_mse

C_UND = "#c0392b"
C_DEF = "#27ae60"
C_AUD = "#2c3e50"

# ------------------------------------------------- left: the attack table --
print("recomputing the combined attack (same calls as a11.4)...")
clients, teacher, g0, Xp, _ = build()
clean = agg_mse(clients, teacher, Xp)
f0 = forward(clients[0], Xp)[0]
p = clients[0]
rp = np.random.default_rng(7)


def run(q):
    atk = [q] + clients[1:]
    und = agg_mse(atk, teacher, Xp) / clean
    dfn = agg_mse([full_canon(c, g0) for c in atk], teacher, Xp) / clean
    aud = float(np.abs(forward(q, Xp)[0] - f0).max())
    return und, dfn, aud


moves = []
moves.append(("permute\nonly", gauge_permute(p, rp)))
a = np.exp(rp.normal(size=H1) * 1.5)
d = np.exp(rp.normal(size=H2) * 1.5)
moves.append(("scale\n$s{=}1.5$", scale_units(p, a, d)))
moves.append(("scale then\npermute", gauge_permute(scale_units(p, a, d), rp)))
a2 = np.exp(rp.normal(size=H1) * 3.0)
d2 = np.exp(rp.normal(size=H2) * 3.0)
moves.append(("scale $s{=}3$\n+ permute",
              gauge_permute(scale_units(p, a2, d2), rp)))

labels, und, dfn, aud = [], [], [], []
for name, q in moves:
    u, f, ad = run(q)
    labels.append(name)
    und.append(u)
    dfn.append(f)
    aud.append(ad)
    print(f"  {name.replace(chr(10), ' '):26} undef {u:10.2f}  def {f:5.2f}  "
          f"audit {ad:.2e}")

own_const = own_mse(p, teacher, Xp)
print(f"  adversary own MSE constant at {own_const:.4f} throughout")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))

a_ = ax[0]
xi = np.arange(len(labels))
w = 0.38
a_.bar(xi - w / 2, und, w, color=C_UND, label="Undefended", zorder=3)
a_.bar(xi + w / 2, dfn, w, color=C_DEF, label="With M4", zorder=3)
a_.axhline(1.0, color="#7f8c8d", lw=0.9, ls=":", zorder=2)
a_.set_yscale("log")
a_.set_xticks(xi)
a_.set_xticklabels(labels, fontsize=7.4)
a_.set_ylabel(r"$\mathcal{L}_{\rm atk}/\mathcal{L}_{\rm clean}$")
a_.set_title("(a) The gauge group, closed", fontsize=9.5, pad=6)
a_.legend(frameon=False, fontsize=7.6, loc="upper left")

for i, (u, f) in enumerate(zip(und, dfn)):
    a_.annotate(f"{u:,.0f}$\\times$" if u > 10 else f"{u:.2f}$\\times$",
                xy=(i - w / 2, u), xytext=(0, 2.5),
                textcoords="offset points", ha="center", fontsize=7.0,
                color=C_UND, weight="bold")
    a_.annotate(f"{f:.2f}", xy=(i + w / 2, f), xytext=(0, 2.5),
                textcoords="offset points", ha="center", fontsize=7.0,
                color=C_DEF, weight="bold")

a_.annotate(f"functional audit $\\leq${max(aud):.0e} on every bar",
            xy=(0.5, 0.03), xycoords="axes fraction", fontsize=6.9,
            color=C_AUD, ha="center")

# --------------------------------------- right: honest cost of the defence --
print("\nmeasuring the honest cost across seeds (no attacker present)...")
seeds = [0, 1, 2, 3, 4]
base, cano = [], []
for sd in seeds:
    cl, te, gg, XX, _ = build(seed=sd)
    b = agg_mse(cl, te, XX)
    c = agg_mse([full_canon(x, gg) for x in cl], te, XX)
    base.append(b)
    cano.append(c)
    print(f"  seed {sd}: {b:.6f} -> {c:.6f}   ratio {c / b:.4f}")

base = np.array(base)
cano = np.array(cano)
ratio = cano / base

b_ = ax[1]
xs = np.arange(len(seeds))
b_.bar(xs - 0.19, base, 0.38, color="#95a5a6", label="No canonicalisation",
       zorder=3)
b_.bar(xs + 0.19, cano, 0.38, color=C_DEF, label="With M4", zorder=3)
b_.set_xticks(xs)
b_.set_xticklabels([f"{s}" for s in seeds], fontsize=8)
b_.set_xlabel("Seed (attack-free federation)")
b_.set_ylabel("Aggregate MSE")
b_.set_title(f"(b) Honest cost: ${ratio.mean():.3f}\\times$",
             fontsize=9.5, pad=6)
b_.legend(frameon=False, fontsize=7.6, loc="upper left")
b_.set_ylim(0, max(base.max(), cano.max()) * 1.32)

for i, r in enumerate(ratio):
    b_.annotate(f"{r:.3f}", xy=(i, max(base[i], cano[i])),
                xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=6.9, color="#555555")

for a2_ in ax:
    a2_.spines[["top", "right"]].set_visible(False)
    a2_.tick_params(labelsize=8)
    a2_.grid(axis="y", lw=0.4, color="#dfe4e6", zorder=0)
    a2_.set_axisbelow(True)

fig.tight_layout(w_pad=2.0)
os.makedirs("../paper/fig", exist_ok=True)
fig.savefig("../paper/fig/defense.pdf", bbox_inches="tight")
fig.savefig("/tmp/defense.png", dpi=200, bbox_inches="tight")
print("\nwrote ../paper/fig/defense.pdf")
print(f"FIGURE CLAIMS: worst undefended={max(und):.2f}x  "
      f"defended max={max(dfn):.4f}x  audit<={max(aud):.1e}  "
      f"honest cost={ratio.mean():.4f}x (per-seed max {ratio.max():.4f})")
