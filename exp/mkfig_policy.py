"""Figure for E7 (sec:e7): the two defects in M3 as written.

Like mkfig_maglaw / mkfig_defense, this recomputes rather than transcribes.  It
imports a12_policy_binding itself, so the numbers in the figure come from
exactly the code that produced the numbers in the text -- there is no second
implementation that could drift.  Importing that module runs it (it is a script
that prints), so we capture its stdout and keep only our own output; the run is
sub-second.

Panel (a) is A12.2: exact equality against a reimplementation flags an HONEST
aggregator, and binding the arithmetic path removes the false positive without
introducing a threshold.
Panel (b) is A12.3: a digest over scorer state alone leaves the aggregator a
factor of 1.236 of freedom in one client's weight, every value of which has a
valid digest.
"""

import contextlib
import io

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import a12_policy_binding as a12

print("imported a12_policy_binding (its own output suppressed; "
      f"{len(_buf.getvalue().splitlines())} lines)")

# house style, matching mkfig_silence
C_A = "#c0392b"          # the attack / the broken check
C_C = "#2c3e50"          # the fix / the honest baseline
C_MUTE = "#95a5a6"

# ---------------------------------------------------------------- panel a ----
# Reuse a12's own state and policy.  ev and w_ref are the module's, so this is
# the same instance the text reports.
ev, w_ref = a12.ev, a12.w_ref
EPS = [0.5, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 0.0]

naive, bound = [], []
for eps in EPS:
    # naive: published vector vs a fresh recomputation of the same MATH
    naive.append(float(np.abs(a12.tilt(w_ref, eps) - a12.scorer(ev)).max()))
    # bound: replay of the same arithmetic PATH
    bound.append(float(np.abs(a12.published(ev, eps)
                              - a12.published(ev)).max()))

FLOOR = float(np.spacing(w_ref[a12.target]))
print(f"panel a: float64 spacing at w0 = {FLOOR:.3e}")
print(f"  naive check on the honest row (eps=0) : {naive[-1]:.3e}"
      f"  -> {'FALSE POSITIVE' if naive[-1] > 0 else 'clean'}")
print(f"  bound check on the honest row (eps=0) : {bound[-1]:.3e}"
      f"  -> {'flagged' if bound[-1] > 0 else 'clean'}")
smallest = min(e for e, d in zip(EPS, bound) if e > 0 and d > 0)
print(f"  smallest cheat the bound check catches: {smallest:.0e}")

# ---------------------------------------------------------------- panel b ----
RHOS, WARMS = [0.3, 0.5, 0.7], [0, 1, 2]
w0 = np.array([[a12.scorer(a12.ewma(a12.hist, r, wu))[a12.target]
                for wu in WARMS] for r in RHOS])
spread = w0.max() / w0.min()
w0_bound = a12.scorer(a12.ewma(a12.hist, a12.RHO, 1))[a12.target]
print(f"panel b: w0 spans {w0.min():.4f}-{w0.max():.4f} = {spread:.3f}x")
print(f"  value under the bound transition map  : {w0_bound:.4f}")

# guards: the figure must not disagree with sec:e7
assert abs(spread - 1.236) < 0.002, f"spread {spread:.3f} != 1.236 in sec:e7"
assert naive[-1] > 0, "A12.2's false positive vanished; sec:e7 is now wrong"
assert bound[-1] == 0.0, "bound path now has a false positive"

# ------------------------------------------------------------------ draw ----
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7))

x = np.arange(len(EPS))
a.plot(x, np.maximum(naive, FLOOR * 1e-2), "o-", color=C_A, lw=1.4, ms=4.6,
       mec="white", mew=0.7, label="exact equality vs reimplementation")
# a zero cannot be drawn on a log axis; park the clean rows on the axis floor
YFLOOR = FLOOR * 1e-2
b_draw = [d if d > 0 else YFLOOR for d in bound]
a.plot(x, b_draw, "s-", color=C_C, lw=1.4, ms=4.2, mec="white", mew=0.7,
       label="replay of committed arithmetic path")
a.axhline(FLOOR, color=C_MUTE, lw=0.9, ls=":")
a.text(0.04, FLOOR * 1.5, "float64 spacing at $w_0$", fontsize=7,
       color=C_MUTE, transform=a.get_yaxis_transform())

# mark the row that decides the argument
a.annotate("honest server,\nflagged", xy=(x[-1], naive[-1]),
           xytext=(x[-1] - 2.5, naive[-1] * 90), fontsize=7, color=C_A,
           arrowprops=dict(arrowstyle="->", color=C_A, lw=0.8))
a.scatter([x[-1]], [naive[-1]], s=90, facecolors="none", edgecolors=C_A,
          lw=1.2, zorder=5)

a.set_yscale("log")
a.set_xticks(x)
a.set_xticklabels([("0" if e == 0 else f"$10^{{{int(np.log10(e))}}}$"
                    if e != 0.5 else "0.5") for e in EPS], fontsize=7)
a.set_xlabel("relative cheat $\\epsilon$ applied to $w_0$", fontsize=8)
a.set_ylabel("$L_\\infty$ deviation from published", fontsize=8)
a.set_title("(a) what the verifier compares decides soundness", fontsize=8.5)
a.legend(fontsize=6.6, loc="lower right", frameon=False)

for i, r in enumerate(RHOS):
    b.plot(WARMS, w0[i], "o-", lw=1.4, ms=4.6, mec="white", mew=0.7,
           label=f"$\\rho = {r}$")
b.axhspan(w0.min(), w0.max(), color=C_A, alpha=0.08, zorder=0)
b.axhline(w0_bound, color=C_C, lw=1.1, ls="--")
b.text(0.98, w0_bound, " bound map: unique", fontsize=7, color=C_C,
       ha="right", va="bottom", transform=b.get_yaxis_transform())
b.annotate(f"{spread:.3f}$\\times$ freedom,\nevery point a valid digest",
           xy=(0.5, 0.06), xycoords="axes fraction", fontsize=7,
           color=C_A, ha="center")
b.set_xticks(WARMS)
b.set_xlabel("warm-up rounds discarded", fontsize=8)
b.set_ylabel("$w_0$ under a state-only digest", fontsize=8)
b.set_title("(b) a digest over state binds too little", fontsize=8.5)
b.legend(fontsize=6.6, loc="upper right", frameon=False, ncol=3,
         columnspacing=0.8, handlelength=1.2)

for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", lw=0.4, color="#dfe4e6")
    ax.set_axisbelow(True)

fig.tight_layout(w_pad=2.0)
fig.savefig("../paper/fig/policy.pdf", bbox_inches="tight")
fig.savefig("/tmp/policy.png", dpi=200, bbox_inches="tight")
print("\nwrote ../paper/fig/policy.pdf")
print(f"FIGURE CLAIMS: false positive={naive[-1]:.3e}  "
      f"smallest caught={smallest:.0e}  spread={spread:.3f}x")
