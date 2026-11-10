# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:14:36 2026

@author: francesca.boschi
"""

# -*- coding: utf-8 -*-
"""
DMOs report - Capacity vs Performance (single day).

Compares the patient's gait in daily life (walking bouts, "performance")
with a standardised test entered manually ("capacity").

Figures
-------
1. bout_sequence     : every walking bout of the day, in order (time of day
                       if available, otherwise WB order), one row per gait
                       parameter, with the test value as a reference band
                       and a rolling median to show changes during the day
2. speed_vs_duration : gait velocity vs walking-bout duration (log scale),
                       with the test value: how long must a bout be before
                       the patient walks as in the test?
3. spider            : median of short / medium / long WBs as % of the test,
                       one axis per gait parameter

Input: WB-level CSV (one row per walking bout), same file as the main report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

plt.close("all")


# =====================================================================
# USER INPUTS
# =====================================================================
PATIENT_ID = "PAT401"
DAY = "2023-07-14"                      # single day to analyse

WB_LEVEL_PATH = Path(r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit\PAT401\outputs_\PAT401_multiple_dates_wb_parameters_average.csv")

OUTPUT_FOLDER = Path(r"report_output")

# ---- Test values (capacity), typed manually ----
TEST_NAME = "4 m walk test"
TEST_VALUES = {
    "gait velocity [m/s]": 1.10,
    "stride length [m]": 1.25,
    "cadence_[steps_per_min]": 105,
    "stride time [s]": 1.14,
}
# Reference band around the test value (± %), e.g. measurement error.
# 0 = draw only the line.
TEST_BAND_PCT = 5

# ---- Time of day on the x axis (figure 1) ----
# Option A: the WB file has a column with the start time of each WB
TIME_COLUMN = "start_time"              # used only if the column exists
# Option B: clock time when the recording started (that day / session);
# WB start time = RECORDING_START + start / SAMPLING_RATE_HZ
RECORDING_START = None                  # e.g. "08:15:00", or None
# Neither available -> x axis = WB order in the day

SAMPLING_RATE_HZ = 102.4

# Very short WBs (< 10 s) give unreliable gait parameters
# (excluded by Mobilise-D for speed / stride length)
INCLUDE_VERY_SHORT = False

# Figures to produce (flag on/off)
FIGURES = {
    "bout_sequence": True,
    "speed_vs_duration": True,
    "spider": True,
}

ROLLING_WINDOW = 5          # WBs in the rolling median (figure 1)
SPIDER_MIN_WB = 3           # min WBs of a category to draw its polygon

SHOW_VALUES = True
SHOW_PLOTS = True
SAVE_PLOTS = False


# =====================================================================
# STYLE (same palette as the Amount & Pattern report)
# =====================================================================
COLORS = {
    "ink": "#0F172A",
    "ink_muted": "#64748B",
    "grid": "#E2E8F0",
    "surface": "#FFFFFF",

    # WB categories (one step darker than the count bars so dots read well)
    "very_short": "#C3D0E0",
    "short": "#8DA5C4",
    "medium": "#6683A8",
    "long": "#2F4B6E",

    # test / capacity reference
    "test": "#D97706",          # amber line
    "test_band": "#FDE8C8",     # light amber band
    "trend": "#0F172A",         # rolling median
}

WB_LABEL_TEXT = {
    "very_short": "Very short < 10 s",
    "short": "Short 10-30 s",
    "medium": "Medium 30-60 s",
    "long": "Long > 60 s",
}

PARAMS = [
    # (column, short name, unit, decimals, inverted for "% of test")
    ("gait velocity [m/s]",     "Gait velocity", "m/s",       2, False),
    ("stride length [m]",       "Stride length", "m",         2, False),
    ("cadence_[steps_per_min]", "Cadence",       "steps/min", 0, False),
    ("stride time [s]",         "Stride time",   "s",         2, True),
]


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.facecolor": COLORS["surface"],
        "figure.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["ink_muted"],
        "axes.labelsize": 9,
        "axes.titlecolor": COLORS["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": COLORS["ink_muted"],
        "ytick.color": COLORS["ink_muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# =====================================================================
# DATA
# =====================================================================
def load_day_wbs(path, patient_id, day, include_very_short=False):
    """WBs of one patient on one day, ordered in time."""
    wb = pd.read_csv(path)
    wb = wb[wb["patient_id"].astype(str) == str(patient_id)].copy()
    wb["recording_date"] = pd.to_datetime(wb["recording_date"])
    wb = wb[wb["recording_date"] == pd.to_datetime(day)]

    if wb.empty:
        raise ValueError(f"No walking bouts for {patient_id} on {day}.")

    if not include_very_short:
        wb = wb[wb["WB_label"] != "very_short"]
        if wb.empty:
            raise ValueError(f"Only very short WBs on {day}; nothing to compare.")

    sort_cols = [c for c in ("session_id", "start") if c in wb.columns]
    wb = wb.sort_values(sort_cols).reset_index(drop=True)
    wb["wb_order"] = np.arange(1, len(wb) + 1)
    return wb


def check_test_values(test_values):
    missing = [col for col, *_ in PARAMS if col not in test_values]
    if missing:
        raise ValueError(f"TEST_VALUES is missing: {missing}")


def add_time_axis_column(wb, day, time_column=None, recording_start=None,
                         fs=SAMPLING_RATE_HZ):
    """
    Adds 'x_time' (datetime of WB start) when possible.
    Returns (wb, mode) with mode = "time" or "order".
    """
    if time_column and time_column in wb.columns and wb[time_column].notna().any():
        t = pd.to_datetime(wb[time_column], errors="coerce")
        # a time-only column ("14:32:10") is put on the analysed day
        if t.notna().any() and (t.dt.year == 1900).all():
            t = pd.to_datetime(day) + (t - t.dt.normalize())
        wb["x_time"] = t
        return wb, "time"

    if recording_start:
        t0 = pd.to_datetime(f"{pd.to_datetime(day).date()} {recording_start}")
        wb["x_time"] = t0 + pd.to_timedelta(wb["start"] / fs, unit="s")
        return wb, "time"

    return wb, "order"


def pct_of_test(value, test, inverted=False):
    """% of test; inverted parameters (stride time) use test / value."""
    if value is None or pd.isna(value) or value == 0:
        return np.nan
    return 100 * (test / value if inverted else value / test)


# =====================================================================
# SHARED HELPERS
# =====================================================================
def section_header(fig, ax_ref, title, subtitle):
    fig_h = fig.get_figheight()
    x0 = ax_ref.get_position().x0
    fig.text(x0, 1 - 0.18 / fig_h, title, ha="left", va="top",
             fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.text(x0, 1 - 0.47 / fig_h, subtitle, ha="left", va="top",
             fontsize=9.5, color=COLORS["ink_muted"])


def draw_test_reference(ax, value, band_pct, orientation="h", label=True):
    """Test value as a line (+ optional ± band)."""
    if orientation == "h":
        if band_pct:
            ax.axhspan(value * (1 - band_pct / 100), value * (1 + band_pct / 100),
                       color=COLORS["test_band"], zorder=1, lw=0)
        ax.axhline(value, color=COLORS["test"], linewidth=1.6, zorder=2)


def wb_marker_sizes(duration_s, s_min=18, s_max=180):
    """Dot area grows with WB duration (sqrt scaling, capped)."""
    d = np.sqrt(np.clip(duration_s, 1, 600))
    d = (d - np.sqrt(1)) / (np.sqrt(600) - np.sqrt(1))
    return s_min + d * (s_max - s_min)


def wb_legend_handles(labels_present):
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7,
                      markerfacecolor=COLORS[lab], markeredgecolor=COLORS["surface"])
               for lab in labels_present]
    texts = [WB_LABEL_TEXT[lab] for lab in labels_present]
    return handles, texts


# =====================================================================
# FIGURE 1 - BOUT SEQUENCE vs TEST
# =====================================================================
def _segments(x, x_mode, max_gap_min):
    """
    Index ranges of consecutive WBs not separated by more than max_gap_min
    (time mode only), so the trend line is not drawn across long gaps.
    """
    idx = np.arange(len(x))
    if x_mode != "time" or not max_gap_min:
        return [idx]
    gaps = x.diff().dt.total_seconds().fillna(0).to_numpy() / 60
    breaks = np.where(gaps > max_gap_min)[0]
    return [seg for seg in np.split(idx, breaks) if len(seg)]


def plot_bout_sequence(wb, test_values, x_mode="order", band_pct=5,
                       rolling_window=5, params=PARAMS, row_height=1.9,
                       width=8.3, max_gap_min=60):
    """
    One row per gait parameter, x = time of day (or WB order).
    Dots = WBs (colour = WB category, size = duration), dark line =
    rolling median, amber line/band = test value.

    max_gap_min : in time mode, the rolling median is interrupted where two
                  WBs are more than this many minutes apart.
    """
    n = len(params)
    fig, axes = plt.subplots(n, 1, sharex=True,
                             figsize=(width, row_height * n + 1.6),
                             gridspec_kw={"hspace": 0.35})
    axes = np.atleast_1d(axes)

    x = wb["x_time"] if x_mode == "time" else wb["wb_order"]
    sizes = wb_marker_sizes(wb["duration_s"].to_numpy())
    colors = [COLORS[lab] for lab in wb["WB_label"]]

    for ax, (col, name, unit, dec, _inv) in zip(axes, params):
        y = pd.to_numeric(wb[col], errors="coerce")
        test = test_values[col]

        draw_test_reference(ax, test, band_pct)

        ax.scatter(x, y, s=sizes, c=colors, edgecolor=COLORS["surface"],
                   linewidth=0.8, zorder=3)

        if rolling_window and len(y) >= rolling_window:
            trend = y.rolling(rolling_window, center=True, min_periods=2).median()
            for seg in _segments(x, x_mode, max_gap_min):
                ax.plot(x.iloc[seg], trend.iloc[seg], color=COLORS["trend"],
                        linewidth=1.4, zorder=4)

        # y range: always include the test value
        lo = np.nanmin([y.min(), test * (1 - band_pct / 100)])
        hi = np.nanmax([y.max(), test * (1 + band_pct / 100)])
        pad = 0.1 * (hi - lo if hi > lo else abs(hi) or 1)
        ax.set_ylim(lo - pad, hi + pad)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(name, pad=4, fontsize=10)

        if SHOW_VALUES:
            ax.annotate(f"test {test:.{dec}f}", xy=(1, test),
                        xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, fontweight="bold", color=COLORS["test"])

    # x axis
    last = axes[-1]
    if x_mode == "time":
        last.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        last.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        last.set_xlabel("Time of day")
    else:
        last.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        last.set_xlabel("Walking bout (order in the day)")

    # legend
    labels_present = [lab for lab in ("very_short", "short", "medium", "long")
                      if lab in set(wb["WB_label"])]
    h, t = wb_legend_handles(labels_present)
    h += [Line2D([], [], color=COLORS["trend"], linewidth=1.4),
          Line2D([], [], color=COLORS["test"], linewidth=1.6)]
    t += [f"Rolling median ({rolling_window} WBs)", f"Test (± {band_pct} %)" if band_pct else "Test"]

    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.6 / fig_h, bottom=0.55 / fig_h,
                        left=0.1, right=0.9)
    fig.legend(h, t, loc="upper left",
               bbox_to_anchor=(axes[0].get_position().x0, 1 - 0.8 / fig_h),
               ncol=len(t), frameon=False, fontsize=8.5,
               labelcolor=COLORS["ink_muted"], handletextpad=0.3,
               columnspacing=1.1)
    fig.text(axes[0].get_position().x0, 1 - 1.12 / fig_h,
             "dot size = WB duration", ha="left", va="top", fontsize=8,
             style="italic", color=COLORS["ink_muted"])
    return fig, axes


# =====================================================================
# FIGURE 2 - SPEED vs WB DURATION
# =====================================================================
def plot_speed_vs_duration(wb, test_values, band_pct=5,
                           column="gait velocity [m/s]", width=8.3, height=4.2,
                           n_bins=6):
    """
    Gait velocity of each WB against its duration (log scale). A binned
    median line shows how speed changes with bout length; the amber line is
    the test value.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    test = test_values[column]

    draw_test_reference(ax, test, band_pct)

    dur = wb["duration_s"].to_numpy(dtype=float)
    spd = pd.to_numeric(wb[column], errors="coerce").to_numpy()
    colors = [COLORS[lab] for lab in wb["WB_label"]]

    ax.scatter(dur, spd, s=55, c=colors, edgecolor=COLORS["surface"],
               linewidth=0.8, zorder=3)
    ax.set_xscale("log")

    # binned median (equal-width bins on log scale, >= 3 WBs per bin)
    ok = np.isfinite(dur) & np.isfinite(spd) & (dur > 0)
    if ok.sum() >= 6:
        edges = np.logspace(np.log10(dur[ok].min()), np.log10(dur[ok].max()), n_bins + 1)
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (dur >= lo) & (dur <= hi)
            if m.sum() >= 3:
                mids.append(np.sqrt(lo * hi))
                meds.append(np.median(spd[m]))
        if len(mids) >= 2:
            ax.plot(mids, meds, color=COLORS["trend"], linewidth=1.6,
                    marker="o", markersize=4, zorder=4)

    # % of test on the right axis (same scale, relabelled)
    lo = np.nanmin([np.nanmin(spd), test * (1 - band_pct / 100)])
    hi = np.nanmax([np.nanmax(spd), test * (1 + band_pct / 100)])
    pad = 0.1 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax_r = ax.twinx()
    ax_r.set_ylim(np.array(ax.get_ylim()) / test * 100)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:.0f} %"))
    ax_r.tick_params(axis="y", colors=COLORS["test"])
    ax_r.set_ylabel("% of test", color=COLORS["test"])
    for sp in ax_r.spines.values():
        sp.set_visible(False)

    ticks = [t for t in (5, 10, 15, 30, 60, 120, 300, 600, 1800)
             if dur[ok].min() * 0.8 <= t <= dur[ok].max() * 1.2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t} s" if t < 60 else f"{t // 60} min" for t in ticks])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Walking-bout duration (log scale)")
    ax.set_ylabel("Gait velocity (m/s)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    labels_present = [lab for lab in ("very_short", "short", "medium", "long")
                      if lab in set(wb["WB_label"])]
    h, t = wb_legend_handles(labels_present)
    h += [Line2D([], [], color=COLORS["trend"], linewidth=1.6, marker="o", markersize=4),
          Line2D([], [], color=COLORS["test"], linewidth=1.6)]
    t += ["Median by duration", f"Test {test:.2f} m/s"]
    ax.legend(h, t, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=3, labelcolor=COLORS["ink_muted"], handletextpad=0.3,
              columnspacing=1.1)
    ax.set_title("Gait velocity vs bout duration", pad=34)

    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.75 / fig_h, bottom=0.65 / fig_h,
                        left=0.1, right=0.9)
    return fig, ax


# =====================================================================
# FIGURE 3 - SPIDER: WB CATEGORIES AS % OF TEST
# =====================================================================
def plot_spider(wb, test_values, labels=("short", "medium", "long"),
                min_wb=3, params=PARAMS, rings=(50, 80), r_max=115,
                width=8.3, height=5.0):
    """
    One axis per gait parameter; each WB category is a polygon of its
    median expressed as % of the test (test = 100 %).
    Inverted parameters (stride time) use test / home so that "outside"
    always means closer to the test.
    """
    order = [params[0], params[1], params[3], params[2]]   # velocity, SL, stride time, cadence
    n = len(order)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    ang_c = np.r_[ang, ang[0]]

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.04, 0.06, 0.55, 0.74], polar=True)
    ax.set_ylim(0, r_max)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(ang)
    ax.set_xticklabels([f"{p[1]}{'*' if p[4] else ''}" for p in order],
                       fontsize=9.5, color=COLORS["ink"])
    ax.tick_params(axis="x", pad=8)

    circle = np.linspace(0, 2 * np.pi, 240)
    ax.plot(circle, [100] * len(circle), color=COLORS["test"], lw=1.8, zorder=3)
    for r in rings:
        ax.plot(circle, [r] * len(circle), color=COLORS["ink_muted"], lw=0.7,
                ls=(0, (2, 2)), zorder=2)
        ax.text(np.pi / 2 + 0.14, r, f"{r} %", fontsize=7.5,
                color=COLORS["ink_muted"], ha="left", va="center")

    table = {}
    for lab in labels:
        sub = wb[wb["WB_label"] == lab]
        n_wb = len(sub)
        vals = [pct_of_test(pd.to_numeric(sub[c], errors="coerce").median(),
                            test_values[c], inv) for c, *_rest, inv in order]
        table[lab] = (n_wb, vals)
        if n_wb < min_wb or not np.isfinite(vals).all():
            continue
        col = COLORS[lab]
        ax.fill(ang_c, np.r_[vals, vals[0]], color=col, alpha=0.10, zorder=3)
        ax.plot(ang_c, np.r_[vals, vals[0]], color=col, lw=2, marker="o", ms=5,
                markeredgecolor=COLORS["surface"], zorder=4)

    # legend
    h = [Line2D([], [], color=COLORS["test"], lw=1.8)]
    t = ["Test = 100 %"]
    if rings:
        h.append(Line2D([], [], color=COLORS["ink_muted"], lw=0.7, ls=(0, (2, 2))))
        t.append(" / ".join(f"{r} %" for r in rings) + " of test")
    for lab in labels:
        n_wb = table[lab][0]
        drawn = n_wb >= min_wb
        h.append(Line2D([], [], color=COLORS[lab] if drawn else COLORS["grid"],
                        lw=2, marker="o", ms=5))
        t.append(f"{WB_LABEL_TEXT[lab]} (n={n_wb})" + ("" if drawn else " – not drawn"))
    fig.legend(h, t, loc="upper left", bbox_to_anchor=(0.62, 0.8),
               frameon=False, fontsize=8.5, labelcolor=COLORS["ink_muted"])

    # value table
    y = 0.42
    head = f"{'':14s}" + "".join(f"{lab.capitalize():>8s}" for lab in labels) + f"{'Test':>10s}"
    fig.text(0.62, y, head, family="monospace", fontsize=8, color=COLORS["ink_muted"])
    for i, (c, name, unit, dec, inv) in enumerate(order):
        y -= 0.045
        cells = ""
        for lab in labels:
            n_wb, vals = table[lab]
            v = vals[i]
            cells += f"{v:7.0f}%" if (n_wb >= min_wb and np.isfinite(v)) else f"{'–':>8s}"
        line = f"{name + ('*' if inv else ''):14s}{cells}{test_values[c]:>7.{dec}f} {unit}"
        fig.text(0.62, y, line, family="monospace", fontsize=8, color=COLORS["ink"])
    fig.text(0.62, y - 0.065,
             "* stride time inverted (test ÷ home):\n  outside = closer to the test",
             fontsize=7.5, color=COLORS["ink_muted"], style="italic")
    return fig, ax, table


# =====================================================================
# MAIN
# =====================================================================
def main():
    apply_style()
    check_test_values(TEST_VALUES)

    wb = load_day_wbs(WB_LEVEL_PATH, PATIENT_ID, DAY,
                      include_very_short=INCLUDE_VERY_SHORT)
    wb, x_mode = add_time_axis_column(wb, DAY, TIME_COLUMN, RECORDING_START,
                                      SAMPLING_RATE_HZ)

    day_txt = pd.to_datetime(DAY).strftime("%a %d %b %Y")
    wb_txt = "WBs ≥ 10 s" if not INCLUDE_VERY_SHORT else "all WBs"
    subtitle = f"{PATIENT_ID}  ·  {day_txt}  ·  {len(wb)} {wb_txt}  ·  test: {TEST_NAME}"
    print(f"{PATIENT_ID} {DAY}: {len(wb)} WBs, x axis = {x_mode}")

    figures = {}

    if FIGURES.get("bout_sequence"):
        fig, axes = plot_bout_sequence(wb, TEST_VALUES, x_mode=x_mode,
                                       band_pct=TEST_BAND_PCT,
                                       rolling_window=ROLLING_WINDOW)
        section_header(fig, axes[0], "Capacity vs Performance – walking bouts", subtitle)
        figures["bout_sequence"] = fig

    if FIGURES.get("speed_vs_duration"):
        fig, ax = plot_speed_vs_duration(wb, TEST_VALUES, band_pct=TEST_BAND_PCT)
        section_header(fig, ax, "Capacity vs Performance – bout duration", subtitle)
        figures["speed_vs_duration"] = fig

    if FIGURES.get("spider"):
        fig, ax, table = plot_spider(wb, TEST_VALUES, min_wb=SPIDER_MIN_WB)
        fig.text(0.04, 0.965, "Capacity vs Performance – summary", ha="left",
                 va="top", fontsize=15, fontweight="bold", color=COLORS["ink"])
        fig.text(0.04, 0.905, subtitle + "  ·  median as % of test",
                 ha="left", va="top", fontsize=9.5, color=COLORS["ink_muted"])
        figures["spider"] = fig

    if SAVE_PLOTS:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = OUTPUT_FOLDER / f"{PATIENT_ID}_{DAY}_{name}.png"
            fig.savefig(path)
            print(f"Saved: {path}")

    if SHOW_PLOTS:
        plt.show()

    return figures


if __name__ == "__main__":
    main()