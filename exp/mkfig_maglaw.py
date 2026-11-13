"""Generates fig/maglaw.pdf -- the AI-specific result, in one figure.

Left:   damage vs gauge magnitude, with the functional audit on the same axes.
        The audit line sits at machine zero across four orders of magnitude of
        damage.  This is prop:auditgap made visual: there is no magnitude at
        which the audit starts to respond.
Right:  the magnitude law itself, log-log, damage-1 against |q-p|^2/N^2, with
        the fitted slope.  Slope < 1 is the ReLU-clipping deviation we report
        rather than hide.

Every number is recomputed here by calling the same sweep() used by
a10_magnitude_law.py -- nothing is transcribed from the results text.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a10_magnitude_law import DIMS, SEEDS, sweep

N_CLIENTS = 24
SDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# The law is fitted on exactly the window a10_2_quadratic() uses.  Below
# s = 0.5 the damage is at the level of seed noise (the s = 0.25 row is
# 1.00 to two decimals), so including those points fits the law to noise and
# moves the slope -- 0.971 instead of 0.940 when we tried it.  The figure must
# report the same law as the text, on the same points.
FIT_FROM = 0.5

C_DMG = "#c0392b"
C_AUD = "#2c3e50"
C_FIT = "#2980b9"

print("recomputing the magnitude sweep (this is the same call a10 makes)...")
rows = []
for s in SDS:
    nrm, dmg, aud, ratios = sweep(DIMS, s, seeds=SEEDS, N=N_CLIENTS)
    rows.append((s, nrm, dmg, aud, np.median(ratios), ratios.min(),
                 ratios.max()))
    print(f"  log-sd {s:4.2f}  |q-p| {nrm:10.2f}  damage {dmg:12.2f}  "
          f"audit {aud:.2e}")

sd = np.array([r[0] for r in rows])
nrm = np.array([r[1] for r in rows])
dmg = np.array([r[2] for r in rows])
aud = np.array([r[3] for r in rows])
med = np.array([r[4] for r in rows])
lo = np.array([r[5] for r in rows])
hi = np.array([r[6] for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.75))

# ---------------------------------------------------------------- left ----
a = ax[0]
a.fill_between(sd, lo, hi, color=C_DMG, alpha=0.13, lw=0, zorder=2)
a.plot(sd, dmg, color=C_DMG, lw=1.7, marker="o", ms=3.4,
       label="Aggregate damage (mean)", zorder=4)
a.plot(sd, med, color=C_DMG, lw=1.1, ls="--", alpha=0.75,
       label="Aggregate damage (median)", zorder=3)

# the audit, floored so it is visible on a log axis
FLOOR = 1e-17
a.plot(sd, np.maximum(aud, FLOOR) + 1.0, color=C_AUD, lw=1.7,
       marker="s", ms=3.2, label="Functional audit response", zorder=5)
a.axhline(1.0, color="#7f8c8d", lw=0.8, ls=":", zorder=1)

a.set_yscale("log")
a.set_xlabel("Gauge magnitude $s$ (log-sd)")
a.set_ylabel(r"$\mathcal{L}_{\rm atk}/\mathcal{L}_{\rm clean}$")
a.set_title("(a) Damage grows; the audit does not", fontsize=9.5, pad=6)
a.legend(frameon=False, fontsize=7.2, loc="upper left")

a.annotate(f"{dmg[-1]:.0f}$\\times$", xy=(sd[-1], dmg[-1]),
           xytext=(-26, -2), textcoords="offset points",
           fontsize=8.5, color=C_DMG, weight="bold")
a.annotate("machine zero at every point", xy=(sd[4], 1.0),
           xytext=(-4, -22), textcoords="offset points",
           fontsize=7.2, color=C_AUD)

# --------------------------------------------------------------- right ----
b = ax[1]
disp = nrm ** 2 / N_CLIENTS ** 2
m = dmg - 1.0
ok = (sd >= FIT_FROM) & (m > 1e-12) & (disp > 0)
x, y = disp[ok], m[ok]

slope, intc = np.polyfit(np.log10(x), np.log10(y), 1)
r = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
print(f"\nfitted on s >= {FIT_FROM} ({ok.sum()} points, matching a10.2)")
print(f"log-log slope {slope:.3f}   r = {r:.5f}")

# guard: the figure must not disagree with the number in the paper
assert abs(slope - 0.940) < 0.005, (
    f"slope {slope:.3f} disagrees with the 0.940 reported in sec:e5; "
    "figure and text must be regenerated together")

xf = np.logspace(np.log10(x.min()) - 0.3, np.log10(x.max()) + 0.3, 64)
b.plot(xf, 10 ** intc * xf ** slope, color=C_FIT, lw=1.3, ls="-",
       label=f"fit: slope {slope:.3f}, $r$ = {r:.4f}", zorder=3)
ref = y[0] / x[0] * xf
b.plot(xf, ref, color="#7f8c8d", lw=1.0, ls="--",
       label="exactly quadratic (slope 1)", zorder=2)
b.plot(x, y, "o", color=C_DMG, ms=4.6, mec="white", mew=0.7,
       label=f"measured ($s \\geq {FIT_FROM}$)", zorder=4)
# the excluded low-magnitude points, shown hollow so the reader can see
# exactly which points the law is and is not fitted on
exl = (sd < FIT_FROM) & (m > 0)
if exl.any():
    b.plot(disp[exl], m[exl], "o", mfc="none", mec="#95a5a6", mew=0.9,
           ms=4.2, label="below noise floor (excluded)", zorder=3)

b.set_xscale("log")
b.set_yscale("log")
b.set_xlabel(r"$\|q-\theta\|^2/N^2$")
b.set_ylabel(r"$\mathcal{L}_{\rm atk}/\mathcal{L}_{\rm clean}-1$")
b.set_title("(b) The magnitude law", fontsize=9.5, pad=6)
b.legend(frameon=False, fontsize=7.2, loc="upper left")

b.annotate("sub-quadratic:\nReLU clipping", xy=(x[-1], y[-1]),
           xytext=(-64, 6), textcoords="offset points", fontsize=7.2,
           color="#555555", linespacing=1.2)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
    a_.tick_params(labelsize=8)
    a_.grid(axis="y", lw=0.4, color="#dfe4e6", zorder=0)
    a_.set_axisbelow(True)

fig.tight_layout(w_pad=2.0)
os.makedirs("../paper/fig", exist_ok=True)
fig.savefig("../paper/fig/maglaw.pdf", bbox_inches="tight")
fig.savefig("/tmp/maglaw.png", dpi=200, bbox_inches="tight")
print("\nwrote ../paper/fig/maglaw.pdf")
print(f"max damage {dmg.max():.2f}x   max audit {aud.max():.3e}")
print(f"FIGURE CLAIMS: slope={slope:.3f}  r={r:.5f}  "
      f"maxdamage={dmg[-1]:.2f}  audit<={aud.max():.1e}")
