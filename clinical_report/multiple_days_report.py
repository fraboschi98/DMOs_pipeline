
# -*- coding: utf-8 -*-
"""
DMOs report - Section 1: Amount & Pattern.

One figure, three panels sharing the same x axis (one tick per day,
1 to 7 days):

    Strip   - Wear        : thin strip, one horizontal bar per day showing
                            wearing time out of 24 h
    Panel 1 - Walking time: stacked bar of time in LPA + MVPA per day, with
                            the longest walking bout as a dot (y axis = time)
    Panel 2 - WB counts   : stacked bar of very short / short / medium / long
                            walking bouts per day
    Panel 3 - Strides &   : total strides per day (left y axis, bars) and
              distance      estimated distance walked (right y axis, km, line)

Input: day-level DMOs CSV (one row per patient and recording_date).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.close("all")


# =====================================================================
# USER INPUTS
# =====================================================================
PATIENT_ID = "PAT401"
WB_LEVEL_PATH = Path(r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit\PAT401\outputs_\PAT401_multiple_dates_wb_parameters_average.csv")
DAY_DMOS_PATH = Path(r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit\PAT401\outputs_\PAT401_day_dmos.csv")

OUTPUT_FOLDER = Path(r"report_output")

# Period (1 to 7 days) - choose ONE option
DATES = "all"                                   # all available days
# DATES = ["2023-07-10", "2023-07-14"]          # selected days
# DATES = ("2023-07-10", "2023-07-17")          # from day -> to day (inclusive)

MAX_DAYS = 7

# Panels of the Amount & Pattern section (flag on/off)
PANELS = {
    "wear_strip": True,         # wearing time out of 24 h (thin strip)
    "time": True,               # LPA + MVPA stacked bar, longest WB dot
    "wb_counts": True,          # stacked bar of WB counts
    "strides_distance": True,   # strides (y1) and distance (y2)
}

SHOW_VALUES = True     # value labels on dots / bars / lines

# Optional marker on the wear strip for a valid-day threshold (e.g. 12 h).
# None = no marker.
WEAR_VALID_HOURS = None
# Sections of the report (flag on/off)
SECTIONS = {
    "amount_pattern": True,
    "variability": True,
    "kde": True,
    "pauses": True,
}

# Pauses section
PAUSE_PANELS = {
    "counts": True,     # stacked bar of pause categories per day
    "kde": True,        # distribution of pause duration
}
PAUSE_KDE_BY_DAY = False   # True = one KDE curve per day

# KDE section: minimum number of WBs of a category to draw its curve
KDE_MIN_WB = 5
SHOW_PLOTS = True
SAVE_PLOTS = True


# =====================================================================
# STYLE - modern clinical palette (slate neutrals, indigo ramp, teal, amber)
# =====================================================================
COLORS = {
    # neutrals
    "ink": "#0F172A",          # main text
    "ink_muted": "#64748B",    # secondary text, axes
    "grid": "#E2E8F0",         # grid lines
    "surface": "#FFFFFF",

    # Walking-bout counts: pastel blue-grey ramp, light (shortest) -> dark (longest)
    "very_short": "#DCE4EE",
    "short": "#AEC0D6",
    "medium": "#7C95B6",
    "long": "#4A6588",

    # Wear strip
    "wear": "#64748B",         # slate (wearing-time fill)
    "wear_track": "#EEF2F6",   # empty part of the 24 h track

    # Walking time: soft sage greens
    "lpa": "#C5DCC8",          # light sage
    "mvpa": "#5E8C6E",         # deep sage
    "max_wb": "#4A6588",       # SAME as long WBs

    # Strides & distance
    "strides": "#CBD5E1",      # light slate bars
    "strides_text": "#475569",
    "distance": "#D97706",     # amber line
    
    # Bout-to-bout variability: pace = blues, rhythm = ambers
    "cov_speed": "#2F5D8A",          # deep blue
    "cov_stride_length": "#8FB3D9",  # light blue
    "cov_cadence": "#B45309",        # deep amber
    "cov_stride_time": "#F0B46A",    # light amber
    
    # KDE curves per WB category (same blue-grey family as the WB counts,
    # one step darker so lines read well)
    "kde_short": "#8DA5C4",
    "kde_medium": "#6683A8",
    "kde_long": "#2F4B6E",
    
    # Pauses: warm stone ramp, light (shortest) -> dark (longest),
    # clearly apart from the blue-grey walking-bout ramp
    "pause_1": "#EFE9DF",
    "pause_2": "#DCCFBB",
    "pause_3": "#BBA88A",
    "pause_4": "#8F7C60",
    "pause_5": "#5F5241",
    "pause_kde": "#8F7C60",
}


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
# DATA HELPERS
# =====================================================================
def load_day_dmos(path: Path, patient_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["patient_id"] == patient_id].copy()
    if df.empty:
        raise ValueError(f"No rows for patient {patient_id} in {path}")
    df["recording_date"] = pd.to_datetime(df["recording_date"])
    return df.sort_values("recording_date").reset_index(drop=True)


def select_dates(df: pd.DataFrame, dates, max_days: int = MAX_DAYS) -> pd.DataFrame:
    """
    dates = "all"                        -> every day
    dates = ["2023-07-10", ...]          -> listed days
    dates = ("2023-07-10", "2023-07-14") -> range, inclusive
    """
    if isinstance(dates, str) and dates == "all":
        out = df
    elif isinstance(dates, tuple) and len(dates) == 2:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        out = df[(df["recording_date"] >= start) & (df["recording_date"] <= end)]
    else:
        if isinstance(dates, str):
            dates = [dates]
        out = df[df["recording_date"].isin(pd.to_datetime(dates))]

    if out.empty:
        raise ValueError(f"No data for the selected dates: {dates}")

    if len(out) > max_days:
        raise ValueError(
            f"{len(out)} days selected; this section shows at most {max_days}. "
            "Narrow DATES."
        )

    return out.reset_index(drop=True)


def _num(day_df, col):
    return pd.to_numeric(day_df[col], errors="coerce")


# =====================================================================
# FORMATTING
# =====================================================================
def format_duration(value, _pos=None) -> str:
    """Axis label (minutes): 0 -> '0', 30 -> '30 min', 60 -> '1 h', 90 -> '1 h 30'"""
    total = int(round(value))
    if total == 0:
        return "0"
    h, m = divmod(total, 60)
    if h == 0:
        return f"{m} min"
    if m == 0:
        return f"{h} h"
    return f"{h} h {m:02d}"


def format_value_time(minutes) -> str:
    """Value label (minutes): 0.5 -> '31 s', 1.6 -> '1 min 36 s', 642 -> '10 h 42 min'"""
    seconds = float(minutes) * 60
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 600:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m} min {s:02d} s"
    h, m = divmod(int(round(seconds / 60)), 60)
    if h == 0:
        return f"{m} min"
    return f"{h} h {m:02d} min"


def nice_minute_step(max_minutes: float) -> int:
    """Tick step (minutes) giving about 4-6 ticks."""
    for step in (5, 10, 15, 30, 60, 120, 180):
        if max_minutes / step <= 6:
            return step
    return 240


# =====================================================================
# SHARED HELPERS
# =====================================================================
def style_day_axis(ax, day_df, show_labels=True):
    """One x tick per day: 'Mon\\n10 Jul'."""
    x = np.arange(len(day_df))
    ax.set_xticks(x)
    if show_labels:
        ax.set_xticklabels(day_df["recording_date"].dt.strftime("%a\n%d %b"))
    ax.set_xlim(-0.6, len(day_df) - 0.4)
    return x


def panel_header(ax, title, handles=None, labels=None, ncol=4):
    """Panel title on the left, legend on the same line on the right."""
    ax.set_title(title, pad=22)
    if handles:
        ax.legend(
            handles, labels,
            loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol,
            handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2,
            labelcolor=COLORS["ink_muted"],
        )


def place_labels_right(ax, items, x_offset=0.09, min_gap_frac=0.11):
    """
    Value labels to the right of their markers; labels of the same day are
    pushed apart vertically so they never overlap.

    items : list of (x, y, text, color)
    """
    y0, y1 = ax.get_ylim()
    min_gap = (y1 - y0) * min_gap_frac

    by_day = {}
    for item in items:
        by_day.setdefault(item[0], []).append(item)

    for xi, day_items in by_day.items():
        day_items = sorted(day_items, key=lambda t: t[1])   # bottom -> top
        placed = []
        for _, y, _, _ in day_items:
            placed.append(y if not placed else max(y, placed[-1] + min_gap))

        for (_, _, text, color), y_new in zip(day_items, placed):
            ax.text(
                xi + x_offset, y_new, text,
                ha="left", va="center", fontsize=8.5, fontweight="bold",
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc=COLORS["surface"],
                          ec="none", alpha=0.9),
            )


# =====================================================================
# STRIP - WEARING TIME OUT OF 24 h
# =====================================================================
def plot_wear_strip(ax, day_df, show_values=True, show_xlabels=False,
                    bar_width=0.55, track_height=0.4, valid_hours=None):
    """
    One small horizontal bar per day: the full track is 24 h, the filled
    part is the wearing time. Value written above the track.

    valid_hours : optional threshold (e.g. 12) drawn as a small tick on
                  each track, to spot days with too little wear.
    """
    x = style_day_axis(ax, day_df, show_labels=show_xlabels)
    wear_h = _num(day_df, "wearing_time_hours")

    ax.set_ylim(0, 1)
    y_track = 0.25

    for xi, h in zip(x, wear_h):
        left = xi - bar_width / 2
        # empty 24 h track
        ax.barh(y_track, bar_width, left=left, height=track_height,
                color=COLORS["wear_track"], zorder=2)
        if pd.notna(h):
            frac = min(max(h / 24.0, 0.0), 1.0)
            ax.barh(y_track, bar_width * frac, left=left, height=track_height,
                    color=COLORS["wear"], zorder=3)
            if show_values:
                ax.text(xi, y_track + track_height / 2 + 0.08,
                        f"{format_value_time(h * 60)}  / 24 h",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color=COLORS["wear"])
        if valid_hours is not None:
            xv = left + bar_width * valid_hours / 24.0
            ax.plot([xv, xv],
                    [y_track - track_height / 2 - 0.05, y_track + track_height / 2 + 0.05],
                    color=COLORS["ink_muted"], linewidth=1, zorder=4)

    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = "Wearing time"
    if valid_hours is not None:
        title += f"  (tick = {valid_hours:g} h)"
    ax.set_title(title, pad=4)
    return ax


# =====================================================================
# PANEL 1 - WALKING TIME: LPA + MVPA stacked, longest WB as a dot
# =====================================================================
def plot_time_panel(ax, day_df, show_values=True, show_xlabels=True,
                    bar_width=0.55, headroom=1.15, min_segment_frac=0.08,
                    connect_max_wb=True):
    """
    Stacked bar per day: time in LPA (bottom) + time in MVPA (top).
    The longest walking bout of the day is a diamond in the middle of the
    bar, connected day to day with a line to show the trend.

    Segment values are written on the left half of the bar, the longest-WB
    value on the right of its diamond, so they never overlap.

    Uses day-level columns:
        time_spent_LPA_s, time_spent_MVPA_s, max_WB_duration_s
    """
    x = style_day_axis(ax, day_df, show_labels=show_xlabels)

    lpa = (_num(day_df, "time_spent_LPA_s") / 60).fillna(0).to_numpy()
    mvpa = (_num(day_df, "time_spent_MVPA_s") / 60).fillna(0).to_numpy()
    max_wb = (_num(day_df, "max_WB_duration_s") / 60).to_numpy()
    total = lpa + mvpa

    y_max = max(float(np.nanmax(np.r_[total, max_wb])), 1.0)
    y_top = y_max * headroom
    ax.set_ylim(0, y_top)

    # ---- stacked bars ----
    ax.bar(x, lpa, width=bar_width, color=COLORS["lpa"], label="LPA",
           edgecolor=COLORS["surface"], linewidth=1.5, zorder=3)
    ax.bar(x, mvpa, bottom=lpa, width=bar_width, color=COLORS["mvpa"],
           label="MVPA", edgecolor=COLORS["surface"], linewidth=1.5, zorder=3)

    if show_values:
        x_text = x - bar_width / 2 + 0.03          # left half of the bar
        segments = [
            (lpa, np.zeros_like(lpa), COLORS["ink"]),
            (mvpa, lpa, COLORS["surface"]),
        ]
        for values, bottoms, txt_color in segments:
            for xi, v, b in zip(x_text, values, bottoms):
                if v > 0 and v >= y_top * min_segment_frac:
                    ax.text(xi, b + v / 2, format_value_time(v),
                            ha="left", va="center", fontsize=8.5,
                            fontweight="bold", color=txt_color, zorder=4)

    # ---- longest WB: diamond in the middle of the bar, line across days ----
    ax.plot(x, max_wb,
            linestyle="-" if connect_max_wb else "none",
            linewidth=1.8, color=COLORS["max_wb"],
            marker="D", markersize=8.5,
            markeredgecolor=COLORS["surface"], markeredgewidth=2,  # white ring
            label="Longest WB", zorder=5, clip_on=False)

    if show_values:
        for xi, v in zip(x, max_wb):
            if pd.notna(v):
                ax.annotate(format_value_time(v), (xi, v), xytext=(8, 7),
                            textcoords="offset points", ha="left", va="bottom",
                            fontsize=8.5, fontweight="bold",
                            color=COLORS["max_wb"], zorder=6,
                            bbox=dict(boxstyle="round,pad=0.12",
                                      fc=COLORS["surface"], ec="none", alpha=0.9))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(nice_minute_step(y_max)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_duration))
    ax.set_ylabel("Time")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    h, l = ax.get_legend_handles_labels()
    order = [l.index(name) for name in ("LPA", "MVPA", "Longest WB")]
    panel_header(ax, "Walking time & longest bout",
                 [h[i] for i in order], [l[i] for i in order], ncol=3)
    return ax

# =====================================================================
# PANEL 2 - WALKING-BOUT COUNTS (stacked bar)
# =====================================================================
WB_LABELS = [
    # (key, day-level column, legend label) - bottom -> top of the stack
    ("long",       "n_long_WB",       "Long > 60 s"),
    ("medium",     "n_medium_WB",     "Medium 30-60 s"),
    ("short",      "n_short_WB",      "Short 10-30 s"),
    ("very_short", "n_very_short_WB", "Very short < 10 s"),
]
DARK_SEGMENTS = ("long", "medium")   # white text inside these


def plot_wb_counts_panel(ax, day_df, show_values=True, show_xlabels=True,
                         bar_width=0.55, min_segment_frac=0.06, headroom=1.12):
    x = style_day_axis(ax, day_df, show_labels=show_xlabels)

    counts = {key: _num(day_df, col).fillna(0).to_numpy(dtype=float)
              for key, col, _ in WB_LABELS}
    totals = sum(counts.values())
    y_top = max(float(totals.max()), 1.0) * headroom
    ax.set_ylim(0, y_top)

    bottoms = np.zeros(len(x))
    for key, _, label in WB_LABELS:
        v = counts[key]
        ax.bar(
            x, v, bottom=bottoms, width=bar_width,
            color=COLORS[key], label=label,
            edgecolor=COLORS["surface"], linewidth=1.5, zorder=3,
        )
        if show_values:
            txt_color = COLORS["surface"] if key in DARK_SEGMENTS else COLORS["ink"]
            for xi, val, b in zip(x, v, bottoms):
                if val > 0 and val >= y_top * min_segment_frac:
                    ax.text(xi, b + val / 2, f"{int(val)}",
                            ha="center", va="center", fontsize=8.5,
                            fontweight="bold", color=txt_color, zorder=4)
        bottoms = bottoms + v

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.set_ylabel("Walking bouts (n)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    # legend in reading order: shortest -> longest
    h, l = ax.get_legend_handles_labels()
    panel_header(ax, "Walking-bout pattern", h[::-1], l[::-1], ncol=4)
    return ax


# =====================================================================
# PANEL 3 - STRIDES (y1) AND DISTANCE (y2)
# =====================================================================
def plot_strides_distance_panel(ax, day_df, show_values=True, show_xlabels=True,
                                bar_width=0.55, bars_headroom=1.45,
                                line_headroom=1.12):
    """
    bars_headroom > line_headroom keeps the distance line above the bar tops
    (strides and distance are strongly related, so with equal scaling the
    line markers would sit exactly on the bar tops and hide the labels).
    """
    x = style_day_axis(ax, day_df, show_labels=show_xlabels)

    strides = _num(day_df, "n_strides_total_day")
    distance_km = _num(day_df, "estimated_distance_walked_m") / 1000.0

    # ---- y1: strides (bars) ----
    s_max = max(float(np.nanmax(strides.to_numpy())), 1.0)
    ax.set_ylim(0, s_max * bars_headroom)
    ax.bar(x, strides.fillna(0), width=bar_width, color=COLORS["strides"],
           label="Strides", zorder=3)

    if show_values:
        # inside the top of the bar; above it when the bar is too short
        for xi, v in zip(x, strides):
            if pd.notna(v):
                inside = v >= s_max * bars_headroom * 0.15
                ax.annotate(
                    f"{int(v):,}", (xi, v),
                    xytext=(0, -5 if inside else 4), textcoords="offset points",
                    ha="center", va="top" if inside else "bottom",
                    fontsize=8.5, fontweight="bold",
                    color=COLORS["strides_text"], zorder=4,
                )

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{int(v):,}"))
    ax.tick_params(axis="y", colors=COLORS["strides_text"])
    ax.set_ylabel("Strides (n)", color=COLORS["strides_text"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    # ---- y2: distance (line) ----
    ax2 = ax.twinx()
    d_max = max(float(np.nanmax(distance_km.to_numpy())), 0.1)
    ax2.set_ylim(0, d_max * line_headroom)
    ax2.plot(
        x, distance_km, color=COLORS["distance"], linewidth=2,
        marker="o", markersize=7,
        markeredgecolor=COLORS["surface"], markeredgewidth=1.4,
        label="Estimated distance", zorder=5, clip_on=False,
    )

    if show_values:
        for xi, v in zip(x, distance_km):
            if pd.notna(v):
                ax2.annotate(
                    f"{v:.1f} km", (xi, v), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color=COLORS["distance"],
                    bbox=dict(boxstyle="round,pad=0.12", fc=COLORS["surface"],
                              ec="none", alpha=0.9),
                )

    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g} km"))
    ax2.tick_params(axis="y", colors=COLORS["distance"], length=0)
    ax2.set_ylabel("Distance (km)", color=COLORS["distance"])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    panel_header(ax, "Strides & distance", h1 + h2, l1 + l2, ncol=2)
    return ax, ax2


# =====================================================================
# SECTION - AMOUNT & PATTERN (the 3 panels, shared x axis)
# =====================================================================
PANEL_FUNCTIONS = {
    "wear_strip": plot_wear_strip,
    "time": plot_time_panel,
    "wb_counts": plot_wb_counts_panel,
    "strides_distance": plot_strides_distance_panel,
}


# relative height of each panel (the wear strip is thin)
PANEL_HEIGHTS = {
    "wear_strip": 0.3,
    "time": 1.0,
    "wb_counts": 1.0,
    "strides_distance": 1.0,
}


def build_amount_pattern_section(day_df, patient_id, panels=PANELS,
                                 show_values=True, width=8.3, panel_height=2.6,
                                 wear_valid_hours=None):
    """
    Stack the selected panels in one figure sharing the x axis (days).
    Day labels are written only under the last panel.
    """
    selected = [name for name, on in panels.items() if on]
    if not selected:
        raise ValueError("No panel selected in PANELS.")

    ratios = [PANEL_HEIGHTS.get(name, 1.0) for name in selected]

    fig, axes = plt.subplots(
        len(selected), 1, sharex=True,
        figsize=(width, panel_height * sum(ratios) + 1.0),
        gridspec_kw={"hspace": 0.55, "height_ratios": ratios},
    )
    axes = np.atleast_1d(axes)

    for i, (name, ax) in enumerate(zip(selected, axes)):
        is_last = i == len(selected) - 1
        kwargs = dict(show_values=show_values, show_xlabels=is_last)
        if name == "wear_strip":
            kwargs["valid_hours"] = wear_valid_hours
        PANEL_FUNCTIONS[name](ax, day_df, **kwargs)
        if not is_last:
            ax.tick_params(axis="x", labelbottom=False)

    # Section header (title + patient / period), above the first panel
    start = day_df["recording_date"].min().strftime("%d %b %Y")
    end = day_df["recording_date"].max().strftime("%d %b %Y")
    period = start if start == end else f"{start} – {end}"
    n_days = len(day_df)

    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.15 / fig_h, bottom=0.6 / fig_h,
                        left=0.1, right=0.9)
    x0 = axes[0].get_position().x0

    fig.text(x0, 1 - 0.18 / fig_h, "Amount & Pattern", ha="left", va="top",
             fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.text(x0, 1 - 0.47 / fig_h,
             f"{patient_id}  ·  {period}  ·  {n_days} day{'s' if n_days > 1 else ''}",
             ha="left", va="top", fontsize=9.5, color=COLORS["ink_muted"])

    return fig, axes

# =====================================================================
# SECTION 2 - BOUT-TO-BOUT VARIABILITY
# =====================================================================
# Which WB category the CoV is computed on: "long", "medium" or "short"
VARIABILITY_WB = "long"

WB_CATEGORY_TEXT = {
    "short": "short WBs (10-30 s)",
    "medium": "medium WBs (30-60 s)",
    "long": "long WBs (> 60 s)",
}

VARIABILITY_PARAMS = [
    # (key in COLORS, column suffix in day file, legend label, marker)
    ("cov_speed",         "gait velocity [m/s]",      "Gait velocity", "o"),
    ("cov_stride_length", "stride length [m]",        "Stride length", "s"),
    ("cov_cadence",       "cadence_[steps_per_min]",  "Cadence",       "o"),
    ("cov_stride_time",   "stride time [s]",          "Stride time",   "s"),
]


def plot_variability_panel(ax, day_df, wb_category="long", show_values=True,
                           show_xlabels=True, headroom=1.2, connect_days=True,
                           show_n_wb=True, low_n=3):
    """
    Bout-to-bout coefficient of variation (%) per day for gait velocity,
    stride length, cadence and stride time, computed on one WB category.

    Uses day-level columns:
        cov_<category>_WB_gait velocity [m/s]
        cov_<category>_WB_stride length [m]
        cov_<category>_WB_cadence_[steps_per_min]
        cov_<category>_WB_stride time [s]
    Days without WBs of that category (NaN) are marked "no data".

    show_n_wb : write the number of WBs of that category under each day
                (column n_<category>_WB); counts below `low_n` are written
                in italic as a reminder that the CoV rests on few bouts.
    """
    x = style_day_axis(ax, day_df, show_labels=show_xlabels)

    values = {}
    for key, suffix, _, _ in VARIABILITY_PARAMS:
        col = f"cov_{wb_category}_WB_{suffix}"
        if col not in day_df.columns:
            raise KeyError(f"Column not found in day file: '{col}'")
        values[key] = _num(day_df, col)

    all_vals = np.concatenate([v.to_numpy(dtype=float) for v in values.values()])
    y_max = np.nanmax(all_vals) if np.isfinite(all_vals).any() else 10.0
    ax.set_ylim(0, max(y_max, 1.0) * headroom)

    label_items = []
    for key, _, label, marker in VARIABILITY_PARAMS:
        v = values[key]
        ax.plot(
            x, v,
            linestyle="-" if connect_days else "none",
            linewidth=1.8, color=COLORS[key],
            marker=marker, markersize=8,
            markeredgecolor=COLORS["surface"], markeredgewidth=1.5,
            label=label, zorder=4, clip_on=False,
        )
        for xi, val in zip(x, v):
            if pd.notna(val):
                label_items.append((xi, float(val), f"{val:.0f} %", COLORS[key]))

    if show_values:
        place_labels_right(ax, label_items, min_gap_frac=0.07)

    # days without data for this WB category
    no_data = pd.concat(values, axis=1).isna().all(axis=1).to_numpy()
    for xi, missing in zip(x, no_data):
        if missing:
            ax.text(xi, ax.get_ylim()[1] * 0.08, "no data",
                    ha="center", va="bottom", fontsize=8.5, style="italic",
                    color=COLORS["ink_muted"])

    # number of WBs the CoV is based on, under each day label
    n_col = f"n_{wb_category}_WB"
    if show_n_wb and n_col in day_df.columns:
        n_wb = _num(day_df, n_col).fillna(0).astype(int)
        for xi, n in zip(x, n_wb):
            ax.annotate(
                f"{n} {wb_category} WB{'s' if n != 1 else ''}",
                xy=(xi, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -38), textcoords="offset points",
                ha="center", va="top", fontsize=8.5,
                fontweight="bold" if n >= low_n else "normal",
                style="normal" if n >= low_n else "italic",
                color=COLORS["ink"] if n >= low_n else COLORS["ink_muted"],
            )

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g} %"))
    ax.set_ylabel("CoV (%)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    h, l = ax.get_legend_handles_labels()
    panel_header(ax, "Coefficient of variation between walking bouts", h, l, ncol=4)
    return ax


def build_variability_section(day_df, patient_id, wb_category="long",
                              show_values=True, width=8.3, height=3.9):
    """Section 2 figure: one panel, same day axis as Amount & Pattern."""
    fig, ax = plt.subplots(figsize=(width, height))
    plot_variability_panel(ax, day_df, wb_category=wb_category,
                           show_values=show_values, show_xlabels=True)

    start = day_df["recording_date"].min().strftime("%d %b %Y")
    end = day_df["recording_date"].max().strftime("%d %b %Y")
    period = start if start == end else f"{start} – {end}"

    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.15 / fig_h, bottom=0.85 / fig_h,
                        left=0.1, right=0.9)
    x0 = ax.get_position().x0

    fig.text(x0, 1 - 0.18 / fig_h, "Bout-to-Bout Variability", ha="left",
             va="top", fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.text(x0, 1 - 0.47 / fig_h,
             f"{patient_id}  ·  {period}  ·  {WB_CATEGORY_TEXT.get(wb_category, wb_category)}",
             ha="left", va="top", fontsize=9.5, color=COLORS["ink_muted"])

    return fig, ax

# =====================================================================
# SECTION 3 - GAIT DISTRIBUTIONS (KDE per WB category)
# =====================================================================
# Logic from KDEPlotter (dmos_pipeline/kde_plotter.py): gaussian KDE of the
# WB-level values, pooled over the selected days, one curve per WB_label,
# labels with fewer than `min_wb` bouts (or no variability) are skipped.
# Vertical lines: median (solid) and P95 (dashed) of each WB category.

from scipy.stats import gaussian_kde

KDE_WB_LABELS = ["short", "medium", "long"]      # very short excluded (as in KDEPlotter)

KDE_FEATURES = [
    # (column in WB file, panel title, unit for labels, decimals)
    ("gait velocity [m/s]",     "Gait velocity", "m/s",       2),
    ("stride length [m]",       "Stride length", "m",         2),
    ("cadence_[steps_per_min]", "Cadence",       "steps/min", 0),
    ("stride time [s]",         "Stride time",   "s",         2),
]

WB_LABEL_TEXT = {
    "very_short": "Very short < 10 s",
    "short": "Short 10-30 s",
    "medium": "Medium 30-60 s",
    "long": "Long > 60 s",
}


def load_wb_level(path: Path, patient_id: str, dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    WB-level file (one row per walking bout), filtered to the patient and to
    the days already selected for the report (dates_df = selected day_df).
    """
    wb = pd.read_csv(path)
    wb = wb[wb["patient_id"].astype(str) == str(patient_id)].copy()
    wb["recording_date"] = pd.to_datetime(wb["recording_date"])
    wb = wb[wb["recording_date"].isin(dates_df["recording_date"])]
    return wb.reset_index(drop=True)


def plot_kde_panel(ax, wb_df, feature, title, unit, decimals,
                   wb_labels=KDE_WB_LABELS, min_wb=5, n_grid=300,
                   show_values=True):
    """
    One gait parameter: KDE per WB category, with median (solid) and P95
    (dashed) drawn from the baseline up to the curve.

    Returns a dict {label: {"n": .., "median": .., "p95": .., "plotted": bool}}.
    """
    stats = {}
    all_values = pd.to_numeric(
        wb_df.loc[wb_df["WB_label"].isin(wb_labels), feature], errors="coerce"
    ).dropna()

    if all_values.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", style="italic", color=COLORS["ink_muted"])
        ax.set_title(title, pad=6)
        ax.set_yticks([])
        return stats

    span = all_values.max() - all_values.min()
    pad = 0.15 * span if span > 0 else 0.1 * abs(all_values.mean())
    x_grid = np.linspace(all_values.min() - pad, all_values.max() + pad, n_grid)

    y_peak = 0.0
    curves = []
    for label in wb_labels:
        values = pd.to_numeric(
            wb_df.loc[wb_df["WB_label"] == label, feature], errors="coerce"
        ).dropna()
        n = int(len(values))
        entry = {"n": n, "median": None, "p95": None, "plotted": False}

        if n >= 1:
            entry["median"] = float(values.median())
            entry["p95"] = float(values.quantile(0.95))

        if n < min_wb or values.nunique() <= 1:
            stats[label] = entry
            continue

        kde = gaussian_kde(values)
        y = kde(x_grid)
        y_peak = max(y_peak, float(y.max()))
        curves.append((label, kde, y, entry))
        entry["plotted"] = True
        stats[label] = entry

    for label, kde, y, entry in curves:
        color = COLORS[f"kde_{label}"]
        ax.fill_between(x_grid, 0, y, color=color, alpha=0.12, zorder=2)
        ax.plot(x_grid, y, color=color, linewidth=2, zorder=3)

        # median and P95: from baseline up to the curve
        for value, style in ((entry["median"], "-"), (entry["p95"], (0, (3, 2)))):
            ax.plot([value, value], [0, float(kde(value)[0])],
                    color=color, linewidth=1.6, linestyle=style, zorder=4)

    # headroom on top keeps the value table clear of the curves
    ax.set_ylim(0, y_peak * (1.55 if show_values else 1.1) if y_peak > 0 else 1)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_yticks([])                       # density values are not clinically meaningful
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.set_xlabel(unit)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=6)

    # small value table in the top-right corner: n, median, P95 per category
    if show_values:
        lines_y = 0.97
        for label in wb_labels:
            entry = stats.get(label)
            if entry is None:
                continue
            if entry["plotted"]:
                text = (f"med {entry['median']:.{decimals}f}  ·  "
                        f"P95 {entry['p95']:.{decimals}f}  (n={entry['n']})")
                color, weight = COLORS[f"kde_{label}"], "bold"
            else:
                text = f"n={entry['n']} – not plotted"
                color, weight = COLORS["ink_muted"], "normal"
            ax.text(0.99, lines_y, text, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, fontweight=weight,
                    color=color)
            lines_y -= 0.1

    return stats


def build_kde_section(wb_df, day_df, patient_id, features=KDE_FEATURES,
                      wb_labels=KDE_WB_LABELS, min_wb=5, show_values=True,
                      width=8.3, panel_height=2.7):
    """
    Section 3 figure: 2 x 2 grid of KDEs (pace on top, rhythm below),
    one shared legend for WB categories and line styles.
    """
    n = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width, panel_height * n_rows + 1.3),
                             gridspec_kw={"hspace": 0.65, "wspace": 0.12})
    axes = np.atleast_1d(axes).ravel()

    all_stats = {}
    for ax, (feature, title, unit, decimals) in zip(axes, features):
        if feature not in wb_df.columns:
            raise KeyError(f"Column not found in WB file: '{feature}'")
        all_stats[feature] = plot_kde_panel(
            ax, wb_df, feature, title, unit, decimals,
            wb_labels=wb_labels, min_wb=min_wb, show_values=show_values,
        )
    for ax in axes[n:]:
        ax.set_visible(False)

    # header
    start = day_df["recording_date"].min().strftime("%d %b %Y")
    end = day_df["recording_date"].max().strftime("%d %b %Y")
    period = start if start == end else f"{start} – {end}"

    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.35 / fig_h, bottom=0.55 / fig_h,
                        left=0.06, right=0.97)
    x0 = axes[0].get_position().x0

    fig.text(x0, 1 - 0.18 / fig_h, "Gait Distributions", ha="left", va="top",
             fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.text(x0, 1 - 0.47 / fig_h,
             f"{patient_id}  ·  {period}  ·  WB-level values pooled over the period",
             ha="left", va="top", fontsize=9.5, color=COLORS["ink_muted"])

    # shared legend: WB categories + line styles
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[f"kde_{lab}"], linewidth=2.5)
               for lab in wb_labels]
    labels = [WB_LABEL_TEXT[lab] for lab in wb_labels]
    handles += [Line2D([], [], color=COLORS["ink_muted"], linewidth=1.6),
                Line2D([], [], color=COLORS["ink_muted"], linewidth=1.6,
                       linestyle=(0, (3, 2)))]
    labels += ["Median", "P95"]
    fig.legend(handles, labels, loc="upper left",
               bbox_to_anchor=(x0, 1 - 0.72 / fig_h), ncol=len(labels),
               frameon=False, fontsize=8.5, handlelength=1.8,
               columnspacing=1.4, labelcolor=COLORS["ink_muted"])

    return fig, axes, all_stats

# =====================================================================
# SECTION 4 - PAUSES BETWEEN WALKING BOUTS (inter-bout time, IBT)
# =====================================================================
# IBT = time from the end of one WB to the start of the next WB, within the
# same recording session (Mobilise-D D6.4 definition). The last WB of a
# session has no following pause and is skipped.

SAMPLING_RATE_HZ = 102.4

# Pause categories (upper limits in seconds). 1, 5 and 10 min are the IBT
# thresholds used in Mobilise-D D6.4 to define walking clusters.
PAUSE_BINS = [
    # (key in COLORS, upper limit s, legend label)
    ("pause_1", 10,      "< 10 s"),
    ("pause_2", 60,      "10 s – 1 min"),
    ("pause_3", 300,     "1 – 5 min"),
    ("pause_4", 600,     "5 – 10 min"),
    ("pause_5", np.inf,  "> 10 min"),
]

# reference lines drawn on the KDE (Mobilise-D cluster IBT thresholds)
PAUSE_REFERENCE_S = [60, 300, 600]


def compute_pauses(wb_df: pd.DataFrame, fs: float = SAMPLING_RATE_HZ) -> pd.DataFrame:
    """
    One row per pause between consecutive WBs of the same day and session.

    Uses WB-level columns: recording_date, session_id, start, end (samples).
    Returns columns: recording_date, session_id, pause_s.
    """
    group_cols = ["recording_date"]
    if "session_id" in wb_df.columns:
        group_cols.append("session_id")

    df = wb_df.sort_values(group_cols + ["start"]).copy()
    next_start = df.groupby(group_cols)["start"].shift(-1)
    df["pause_s"] = (next_start - df["end"]) / fs

    out = df.loc[df["pause_s"].notna() & (df["pause_s"] >= 0),
                 group_cols + ["pause_s"]]
    return out.reset_index(drop=True)


def format_seconds_axis(seconds) -> str:
    """Axis label for durations in seconds: 5 s, 30 s, 1 min, 10 min, 1 h, 2 h."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:g} s"
    if seconds < 3600:
        return f"{seconds / 60:g} min"
    return f"{seconds / 3600:g} h"


def plot_pause_counts_panel(ax, day_df, pauses_df, show_values=True,
                            show_xlabels=True, bar_width=0.55,
                            min_segment_frac=0.06, headroom=1.12):
    """
    Stacked bar per day: number of pauses in each duration category
    (shortest at the bottom).
    """
    x = style_day_axis(ax, day_df, show_labels=show_xlabels)

    edges = [0] + [upper for _, upper, _ in PAUSE_BINS]
    counts = {key: np.zeros(len(day_df)) for key, _, _ in PAUSE_BINS}
    for i, day in enumerate(day_df["recording_date"]):
        p = pauses_df.loc[pauses_df["recording_date"] == day, "pause_s"]
        for (key, _, _), lo, hi in zip(PAUSE_BINS, edges[:-1], edges[1:]):
            counts[key][i] = ((p >= lo) & (p < hi)).sum()

    totals = sum(counts.values())
    y_top = max(float(totals.max()), 1.0) * headroom
    ax.set_ylim(0, y_top)

    bottoms = np.zeros(len(x))
    for key, _, label in PAUSE_BINS:
        v = counts[key]
        ax.bar(x, v, bottom=bottoms, width=bar_width, color=COLORS[key],
               label=label, edgecolor=COLORS["surface"], linewidth=1.5, zorder=3)
        if show_values:
            txt = COLORS["surface"] if key in ("pause_4", "pause_5") else COLORS["ink"]
            for xi, val, b in zip(x, v, bottoms):
                if val > 0 and val >= y_top * min_segment_frac:
                    ax.text(xi, b + val / 2, f"{int(val)}", ha="center",
                            va="center", fontsize=8.5, fontweight="bold",
                            color=txt, zorder=4)
        bottoms = bottoms + v

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.set_ylabel("Pauses (n)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    h, l = ax.get_legend_handles_labels()
    panel_header(ax, "Pauses between walking bouts", h, l, ncol=5)
    return counts


def plot_pause_kde_panel(ax, pauses_df, day_df=None, by_day=False,
                         show_values=True, min_pauses=5, n_grid=400,
                         show_reference=True):
    """
    KDE of pause duration on a log time axis (pause durations are strongly
    right-skewed: seconds to hours). Median (solid) and P90 (dashed).

    by_day=False : one curve, all pauses of the period pooled
    by_day=True  : one curve per day (days need >= min_pauses pauses)
    """
    p_all = pauses_df["pause_s"].clip(lower=0.5)
    if p_all.empty:
        ax.text(0.5, 0.5, "no pauses", transform=ax.transAxes, ha="center",
                va="center", style="italic", color=COLORS["ink_muted"])
        return {}

    log_all = np.log10(p_all)
    pad = 0.15 * (log_all.max() - log_all.min() or 1)
    grid = np.linspace(log_all.min() - pad, log_all.max() + pad, n_grid)

    if by_day and day_df is not None:
        days = list(day_df["recording_date"])
        # stone ramp, light -> dark in day order (same family as the bars)
        from matplotlib.colors import LinearSegmentedColormap
        stone = LinearSegmentedColormap.from_list(
            "stone", [COLORS["pause_3"], COLORS["pause_5"], "#2E271F"])
        ramp = stone(np.linspace(0.0, 1.0, max(len(days), 1)))
        groups = [(d.strftime("%a %d %b"),
                   pauses_df.loc[pauses_df["recording_date"] == d, "pause_s"],
                   ramp[i]) for i, d in enumerate(days)]
    else:
        groups = [("All days", pauses_df["pause_s"], COLORS["pause_kde"])]

    stats, y_peak, curves = {}, 0.0, []
    for name, values, color in groups:
        values = values.clip(lower=0.5)
        n = int(len(values))
        entry = {"n": n,
                 "median_s": float(values.median()) if n else None,
                 "p90_s": float(values.quantile(0.90)) if n else None,
                 "plotted": False}
        if n >= min_pauses and values.nunique() > 1:
            kde = gaussian_kde(np.log10(values))
            y = kde(grid)
            y_peak = max(y_peak, float(y.max()))
            curves.append((name, kde, y, color, entry))
            entry["plotted"] = True
        stats[name] = entry

    for name, kde, y, color, entry in curves:
        ax.fill_between(grid, 0, y, color=color, alpha=0.12, zorder=2)
        ax.plot(grid, y, color=color, linewidth=2, zorder=3,
                label=name if by_day else None)
        for value, style in ((entry["median_s"], "-"), (entry["p90_s"], (0, (3, 2)))):
            lv = np.log10(value)
            ax.plot([lv, lv], [0, float(kde(lv)[0])], color=color,
                    linewidth=1.6, linestyle=style, zorder=4)

    ax.set_ylim(0, y_peak * (1.5 if show_values else 1.1) if y_peak else 1)
    ax.set_xlim(grid[0], grid[-1])

    # Mobilise-D cluster thresholds (1, 5, 10 min)
    if show_reference:
        for ref in PAUSE_REFERENCE_S:
            lr = np.log10(ref)
            if grid[0] < lr < grid[-1]:
                ax.axvline(lr, color=COLORS["ink_muted"], linewidth=0.8,
                           linestyle=(0, (1, 2)), zorder=1)
                ax.text(lr, ax.get_ylim()[1] * 0.02, f" {format_seconds_axis(ref)}",
                        rotation=90, ha="left", va="bottom", fontsize=7.5,
                        color=COLORS["ink_muted"])

    # log axis with readable time labels
    tick_s = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 14400]
    ticks = [np.log10(t) for t in tick_s if grid[0] <= np.log10(t) <= grid[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_seconds_axis(10 ** t) for t in ticks])
    ax.set_xlabel("Pause duration (log scale)")
    ax.set_yticks([])
    ax.grid(False)          # the 1 / 5 / 10 min reference lines act as guides
    ax.spines["bottom"].set_color(COLORS["grid"])

    if show_values:
        y_txt = 0.97
        for name, entry in stats.items():
            if entry["plotted"]:
                text = (f"{name + ':  ' if by_day else ''}"
                        f"median {format_value_time(entry['median_s'] / 60)}  ·  "
                        f"P90 {format_value_time(entry['p90_s'] / 60)}  (n={entry['n']})")
                color = next(c for nm, _, _, c, _ in curves if nm == name)
                weight = "bold"
            else:
                text = f"{name}: n={entry['n']} – not plotted"
                color, weight = COLORS["ink_muted"], "normal"
            ax.text(0.99, y_txt, text, transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, fontweight=weight, color=color)
            y_txt -= 0.09

    from matplotlib.lines import Line2D
    style_handles = [Line2D([], [], color=COLORS["ink_muted"], linewidth=1.6),
                     Line2D([], [], color=COLORS["ink_muted"], linewidth=1.6,
                            linestyle=(0, (3, 2)))]
    h, l = ax.get_legend_handles_labels()
    panel_header(ax, "Pause duration distribution",
                 h + style_handles, l + ["Median", "P90"],
                 ncol=len(h) + 2)
    return stats


def build_pauses_section(wb_df, day_df, patient_id, panels=None,
                         kde_by_day=False, show_values=True,
                         fs=SAMPLING_RATE_HZ, width=8.3, panel_height=2.8):
    """
    Section 4 figure: pause counts per day (stacked bar) and/or the KDE of
    pause duration. panels = {"counts": bool, "kde": bool}.
    """
    panels = panels or {"counts": True, "kde": True}
    selected = [name for name, on in panels.items() if on]
    if not selected:
        raise ValueError("No pause panel selected.")

    pauses_df = compute_pauses(wb_df, fs=fs)

    fig, axes = plt.subplots(len(selected), 1,
                             figsize=(width, panel_height * len(selected) + 1.0),
                             gridspec_kw={"hspace": 0.7})
    axes = np.atleast_1d(axes)

    stats = {}
    for name, ax in zip(selected, axes):
        if name == "counts":
            stats["counts"] = plot_pause_counts_panel(ax, day_df, pauses_df,
                                                      show_values=show_values)
        elif name == "kde":
            stats["kde"] = plot_pause_kde_panel(ax, pauses_df, day_df=day_df,
                                                by_day=kde_by_day,
                                                show_values=show_values)

    start = day_df["recording_date"].min().strftime("%d %b %Y")
    end = day_df["recording_date"].max().strftime("%d %b %Y")
    period = start if start == end else f"{start} – {end}"

    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.15 / fig_h, bottom=0.6 / fig_h,
                        left=0.1, right=0.95)
    x0 = axes[0].get_position().x0
    fig.text(x0, 1 - 0.18 / fig_h, "Pauses", ha="left", va="top",
             fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.text(x0, 1 - 0.47 / fig_h,
             f"{patient_id}  ·  {period}  ·  time between consecutive walking bouts",
             ha="left", va="top", fontsize=9.5, color=COLORS["ink_muted"])

    return fig, axes, pauses_df, stats
# =====================================================================
# MAIN
# =====================================================================
def main():
    apply_style()

    day_df = load_day_dmos(DAY_DMOS_PATH, PATIENT_ID)
    day_df = select_dates(day_df, DATES)

    print(f"{PATIENT_ID}: {len(day_df)} day(s) -> "
          f"{', '.join(day_df['recording_date'].dt.strftime('%Y-%m-%d'))}")

    figures = {}

    if SECTIONS.get("amount_pattern"):
        fig, _ = build_amount_pattern_section(day_df, PATIENT_ID,
                                              panels=PANELS, show_values=SHOW_VALUES,
                                              wear_valid_hours=WEAR_VALID_HOURS)
        figures["amount_pattern"] = fig

    if SECTIONS.get("variability"):
        fig, _ = build_variability_section(day_df, PATIENT_ID,
                                           wb_category=VARIABILITY_WB,
                                           show_values=SHOW_VALUES)
        figures[f"variability_{VARIABILITY_WB}"] = fig
    if SECTIONS.get("kde"):
        wb_df = load_wb_level(WB_LEVEL_PATH, PATIENT_ID, day_df)
        fig, _, kde_stats = build_kde_section(wb_df, day_df, PATIENT_ID,
                                              min_wb=KDE_MIN_WB,
                                              show_values=SHOW_VALUES)
    figures["gait_distributions"] = fig
    if SECTIONS.get("pauses"):
        wb_df = load_wb_level(WB_LEVEL_PATH, PATIENT_ID, day_df)
        fig, _, pauses_df, pause_stats = build_pauses_section(
            wb_df, day_df, PATIENT_ID, panels=PAUSE_PANELS,
            kde_by_day=PAUSE_KDE_BY_DAY, show_values=SHOW_VALUES,
        )
        figures["pauses"] = fig
    
    
    
    if SAVE_PLOTS:
        OUTPUT_FOLDER = Path(r'C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit\prova_clinical_report')
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)  # create it if missing
        for name, fig in figures.items():
            path = OUTPUT_FOLDER / f"{PATIENT_ID}_{name}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")

    if SHOW_PLOTS:
        plt.show()

    return figures


if __name__ == "__main__":
    main()