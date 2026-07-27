#!/usr/bin/env python3
"""The two-run identification, as a chart — the half of the bench rig story
that cannot be filmed.

WHY THIS EXISTS RATHER THAN A SECOND CLIP
------------------------------------------
The rig's headline is a *ratio*: the same held current spins a bare rotor, then
the same rotor carrying a disc of known inertia, and the ratio of the two
acceleration slopes yields kt and the rotor's own inertia. Showing that as two
films fails twice over:

  1. The claim is quantitative. Two side-by-side clips show "one is faster",
     which is the vibe of the result, not the result.

  2. The bare rotor cannot honestly be filmed at all. `strip_flywheel_geom`
     removes the disc but not the `index` mark painted on it, so the bare model
     renders a white bar rotating in mid-air. Giving it a marker somewhere else
     would mean inventing geometry the plant does not contain.

So the contrast ships as a chart, drawn from the same committed tracks the film
is drawn from. Sim Replay, and Sim Replay may carry engineering numbers.

    ~/projects/overboard/.venv/bin/python viz/src/bench_plot.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# The page is dark-only (--bg #0F1922). These two are the brand amber and mint
# stepped down until they pass the categorical palette checks against that
# surface — lightness band, chroma floor, CVD separation, normal-vision floor
# and contrast. Validated, not eyeballed: the brand values themselves sit above
# the dark-mode lightness band and fail it.
BG = "#0F1922"
AMBER = "#C88231"   # flywheel-loaded — the same amber the disc is rendered in
MINT = "#229683"    # bare rotor — the colour of its hub
INK_HI = "#E8EEF0"
INK_MID = "#94A6AE"
INK_LOW = "#3A4A55"


def ratio_str(fit: dict) -> str:
    return f"{fit['alpha_bare_rad_s2'] / fit['alpha_loaded_rad_s2']:.1f}×"


def _load(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    npz = np.load(ROOT / f"viz/scenes/replay/{name}.otrk.npz", allow_pickle=False)
    m = json.loads(str(npz["manifest"]))
    return npz["t"], npz["ch/shaft_rad_s"], m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out/bench_identify_plot.png")
    args = ap.parse_args()

    t_b, w_b, m_b = _load("bench_identify_bare")
    t_l, w_l, m_l = _load("bench_identify_loaded")
    fit = m_l["fit"]
    window_s = 0.005

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(11.5, 4.9), dpi=170,
        gridspec_kw=dict(width_ratios=[1.65, 1.0], wspace=0.26))
    fig.patch.set_facecolor(BG)

    for a in (ax, axz):
        a.set_facecolor(BG)
        # Recessive: the data is the only thing that should read at a glance.
        a.grid(True, color=INK_LOW, linewidth=0.6, alpha=0.55)
        a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_color(INK_LOW)
        a.tick_params(colors=INK_MID, labelsize=9)

    # --- left: the whole captured run -------------------------------------
    ax.plot(t_b * 1e3, w_b, color=MINT, linewidth=2.0, label="Bare rotor")
    ax.plot(t_l * 1e3, w_l, color=AMBER, linewidth=2.0, label="+ flywheel")
    # Direct labels as well as the legend, so identity never rests on colour.
    ax.annotate("Bare rotor", (t_b[-1] * 1e3, w_b[-1]), xytext=(-6, 10),
                textcoords="offset points", color=INK_HI, fontsize=10,
                ha="right", fontweight="bold")
    ax.annotate("+ flywheel", (t_l[-1] * 1e3, w_l[-1]), xytext=(-6, 12),
                textcoords="offset points", color=INK_HI, fontsize=10,
                ha="right", fontweight="bold")
    ax.axvspan(0, window_s * 1e3, color=INK_MID, alpha=0.10, linewidth=0)
    ax.set_xlabel("time (ms)", color=INK_MID, fontsize=10)
    ax.set_ylabel("shaft rate (rad/s)", color=INK_MID, fontsize=10)
    ax.set_title("Same current, same rotor, one disc added",
                 color=INK_HI, fontsize=12.5, fontweight="bold", pad=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(INK_MID)

    # --- right: the window the fit is actually taken over ------------------
    mb, ml = t_b <= window_s, t_l <= window_s
    axz.plot(t_b[mb] * 1e3, w_b[mb], color=MINT, linewidth=2.0)
    axz.plot(t_l[ml] * 1e3, w_l[ml], color=AMBER, linewidth=2.0)
    axz.set_xlabel("time (ms)", color=INK_MID, fontsize=10)
    axz.set_title(f"The {window_s*1e3:.0f} ms fit window",
                  color=INK_HI, fontsize=12.5, fontweight="bold", pad=11, loc="left")
    # Anchored in axes fraction, not data coordinates: pinned to the last data
    # point these ran off the right spine, because both curves end at the very
    # edge of the window they are fitted over.
    axz.text(0.04, 0.93, f"α = {fit['alpha_bare_rad_s2']:,.0f} rad/s²",
             transform=axz.transAxes, color=MINT, fontsize=10.5,
             fontweight="bold", va="top")
    axz.text(0.04, 0.81, f"α = {fit['alpha_loaded_rad_s2']:,.0f} rad/s²",
             transform=axz.transAxes, color=AMBER, fontsize=10.5,
             fontweight="bold", va="top")
    axz.text(0.04, 0.69, f"{ratio_str(fit)} steeper", transform=axz.transAxes,
             color=INK_MID, fontsize=9.5, va="top")

    ratio = fit["alpha_bare_rad_s2"] / fit["alpha_loaded_rad_s2"]
    fig.text(0.012, 0.070,
             f"Two slopes, two unknowns:  kt = {fit['kt_fit_nm_per_a']:.4f} N·m/A"
             f"   ·   J_rotor = {fit['j_bare_fit_kg_m2']:.3e} kg·m²"
             f"   ·   J_disc = {fit['j_disc_known_kg_m2']:.3e} kg·m² (known)"
             f"   ·   slope ratio {ratio:.1f}×",
             color=INK_MID, fontsize=9.2)
    fig.text(0.012, 0.022,
             f"Sim Replay · replay of {m_l['source']['model_file']} · "
             f"{m_l['source']['commanded_current_a']:.0f} A held · "
             f"profile {m_l['source']['imperfection_profile']} · "
             f"rates are the sensed channel, as fitted",
             color=INK_LOW, fontsize=8.2)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.235)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=BG)
    print(f"wrote {args.out.relative_to(ROOT)}  (ratio {ratio:.2f}×)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
