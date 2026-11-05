# -*- coding: utf-8 -*-
"""
DMOs report - monolithic prototype.

Plots
-----
1. time_variables : wearing time, time in LPA and MVPA per day (dots)
2. wb_pattern     : number of walking bouts per day by WB label
                    (very short / short / medium / long), drawn either as
                    connected lines or as a stacked bar, with time spent
                    walking (LPA + MVPA) per day on a second y axis (right,
                    same 0 as the left axis) and a daily summary table under
                    each day (wearing time, MVPA, LPA, total strides)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.close('all')
# =====================================================================
# USER INPUTS
# =====================================================================
PATIENT_ID = "PAT401"

DAY_DMOS_PATH = Path(r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit\PAT401\outputs_\PAT401_day_dmos.csv")
WB_LEVEL_PATH = Path(r"PAT401_multiple_dates_wb_parameters_average.csv")  # used in next steps

OUTPUT_FOLDER = Path(r"report_output")

# Date selection - choose ONE option
DATES = "all"                                   # all available days
# DATES = ["2023-07-10", "2023-07-14"]          # selected days
# DATES = ("2023-07-10", "2023-07-14")          # range: from day -> to day (inclusive)

# Plots to produce (flag on/off)
PLOTS = {
    "time_variables": False,
    "wb_pattern": True,
    "distance_max_wb": True,
}
# Walking-bout pattern options
WB_PATTERN_STYLE = "stacked"        # "lines" or "stacked"
WB_PATTERN_SHOW_VALUES = True       # numbers on dots / bar segments
WB_PATTERN_SHOW_DAY_SUMMARY = True  # text table under each day
WB_PATTERN_SHOW_WALKING_TIME = True # time walking (LPA + MVPA) on a second (right) y axis

SHOW_PLOTS = True
SAVE_PLOTS = False


# =====================================================================
# STYLE (clinical dashboard: blues / greys, minimal chrome)
# =====================================================================
COLORS = {
    "wear": "#1F3A5F",        # deep navy
    "lpa": "#8DB3DD",         # light blue
    "mvpa": "#2F6DB5",        # mid blue
    "ink": "#1E2329",         # main text
    "ink_muted": "#6B7480",   # secondary text
    "grid": "#E3E7EC",        # grid lines
    "stem": "#C9D1DA",        # connector between dots
    "surface": "#FFFFFF",
    # Walking-bout labels: one blue ramp, light (shortest) -> dark (longest)
    "very_short": "#A9C4E2",
    "short": "#6E9CCF",
    "medium": "#2F6DB5",
    "long": "#1F3A5F",
    # Time spent walking (right axis): warm orange, high contrast with the blues
    "walking": "#D9731F",
    # Plot 3: longest WB (left axis) and estimated distance (right axis)
    "max_wb": "#1F3A5F",      # navy
    "distance": "#178F82",    # teal, high contrast with navy
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.facecolor": COLORS["surface"],
        "figure.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["ink_muted"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": COLORS["ink_muted"],
        "ytick.color": COLORS["ink_muted"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# =====================================================================
# DATA HELPERS
# =====================================================================
def load_day_dmos(path: Path, patient_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["patient_id"] == patient_id].copy()
    df["recording_date"] = pd.to_datetime(df["recording_date"])
    return df.sort_values("recording_date").reset_index(drop=True)


def select_dates(df: pd.DataFrame, dates) -> pd.DataFrame:
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

    return out.reset_index(drop=True)


def format_duration(value, _pos=None) -> str:
    """Axis label: 0 -> '0', 30 -> '30 min', 60 -> '1 h', 90 -> '1 h 30'"""
    total = int(round(value))
    if total == 0:
        return "0"
    h, m = divmod(total, 60)
    if h == 0:
        return f"{m} min"
    if m == 0:
        return f"{h} h"
    return f"{h} h {m:02d}"


def format_duration_long(value) -> str:
    """Value label: 642 -> '10 h 42 min', 25 -> '25 min'"""
    h, m = divmod(int(round(value)), 60)
    if h == 0:
        return f"{m} min"
    return f"{h} h {m:02d} min"


# =====================================================================
# SHARED AXIS HELPERS (reusable on any plot with days on the x axis)
# =====================================================================
def set_day_axis(ax, day_df):
    """Days on the x axis: 'Mon\\n10 Jul 2023'."""
    x = np.arange(len(day_df))
    ax.set_xticks(x)
    ax.set_xticklabels(day_df["recording_date"].dt.strftime("%a\n%d %b %Y"))
    ax.set_xlim(-0.6, len(day_df) - 0.4)
    return x


def set_title_and_legend(ax, title, patient_id="", ncol=4, extra_axes=()):
    """Title on the left + one legend collecting handles of all given axes."""
    ax.set_title(f"{title}  ·  {patient_id}" if patient_id else title, pad=28)

    handles, labels = ax.get_legend_handles_labels()
    for other in extra_axes:
        h, l = other.get_legend_handles_labels()
        handles += h
        labels += l

    ax.legend(
        handles, labels,
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol,
        fontsize=9, handletextpad=0.4, columnspacing=1.4,
        labelcolor=COLORS["ink_muted"],
    )


DAY_SUMMARY_ROWS = [
    # (row label, function day_row -> text, color key)
    ("Wearing time",
     lambda r: format_duration_long(r["wearing_time_hours"] * 60), "wear"),
    ("MVPA",
     lambda r: format_duration_long(r["time_spent_MVPA_s"] / 60), "mvpa"),
    ("LPA",
     lambda r: format_duration_long(r["time_spent_LPA_s"] / 60), "lpa"),
    ("Total strides",
     lambda r: f"{int(r['n_strides_total_day']):,}", "ink"),
]


def add_day_summary(ax, day_df, rows=DAY_SUMMARY_ROWS,
                    first_row_offset=42, row_height=16):
    """
    Write a small text table under the x tick labels: one row per variable,
    one value under each day. Row names are written on the left, next to
    the y axis.

    rows : list of (row label, function(day_row) -> str, color key)
    first_row_offset : distance (points) between the x axis and the first row
    row_height : distance (points) between rows
    """
    x = np.arange(len(day_df))

    for i, (row_label, fmt, color_key) in enumerate(rows):
        y_offset = -(first_row_offset + i * row_height)

        # Row name, left of the plot area
        ax.annotate(
            row_label,
            xy=(0, 0), xycoords="axes fraction",
            xytext=(-8, y_offset), textcoords="offset points",
            ha="right", va="center", fontsize=9,
            color=COLORS["ink_muted"],
        )

        # One value under each day
        for xi, (_, day_row) in zip(x, day_df.iterrows()):
            try:
                text = fmt(day_row)
            except (KeyError, TypeError, ValueError):
                text = "-"
            if "nan" in text.lower():
                text = "-"

            ax.annotate(
                text,
                xy=(xi, 0), xycoords=("data", "axes fraction"),
                xytext=(0, y_offset), textcoords="offset points",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=COLORS[color_key],
            )


# =====================================================================
# TIME SPENT WALKING ON A RIGHT AXIS (reusable on any plot with days on x)
# =====================================================================
def _nice_minute_step(max_minutes: float) -> int:
    """Tick step (minutes) giving about 4-6 ticks."""
    for step in (5, 10, 15, 30, 60, 120, 180):
        if max_minutes / step <= 6:
            return step
    return 240


def add_walking_time_axis(
    ax,
    x,
    day_df,
    show_values=True,
    lpa_column="time_spent_LPA_s",
    mvpa_column="time_spent_MVPA_s",
    headroom=1.15,
):
    """
    Time spent walking per day (LPA + MVPA, in minutes) on a second y axis
    (right side), drawn as dots connected day to day. Both axes start at 0
    and share the full plot height; the right axis, its ticks and its label
    use the walking color.

    headroom : top of the right axis = max walking time * headroom
    """
    lpa = pd.to_numeric(day_df[lpa_column], errors="coerce").fillna(0)
    mvpa = pd.to_numeric(day_df[mvpa_column], errors="coerce").fillna(0)
    walking_min = (lpa + mvpa) / 60.0          # s -> min

    w_max = max(float(walking_min.max()), 1.0)

    ax2 = ax.twinx()
    ax2.set_ylim(0, w_max * headroom)

    ax2.plot(
        x, walking_min,
        color=COLORS["walking"], linewidth=2, linestyle=(0, (4, 2)),
        marker="D", markersize=7,
        markeredgecolor=COLORS["surface"], markeredgewidth=1.5,
        label="Time walking (LPA + MVPA)", zorder=5, clip_on=False,
    )

    if show_values:
        for xi, v in zip(x, walking_min):
            if pd.notna(v):
                ax2.annotate(
                    format_duration_long(v),
                    (xi, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=COLORS["walking"],
                    bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["surface"],
                              ec="none", alpha=0.9),
                )

    ax2.yaxis.set_major_locator(mticker.MultipleLocator(_nice_minute_step(w_max)))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(format_duration))
    ax2.tick_params(axis="y", colors=COLORS["walking"])
    ax2.set_ylabel("Time spent walking", color=COLORS["walking"])
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Walking line drawn above the bars/lines of the main axis,
    # main axis background kept visible
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)

    return ax2


# =====================================================================
# PLOT 1 - TIME VARIABLES
# =====================================================================
def plot_time_variables(
    day_df: pd.DataFrame,
    patient_id: str = "",
    tick_minutes: int = 30,
    ax=None,
):
    """
    One column of dots per day: wearing time, time in LPA, time in MVPA.

    Uses day-level columns:
        wearing_time_hours, time_spent_LPA_s, time_spent_MVPA_s
    All values are converted to minutes; y ticks every `tick_minutes`.
    """
    data = pd.DataFrame({
        "wear": day_df["wearing_time_hours"] * 60.0,   # h -> min
        "lpa": day_df["time_spent_LPA_s"] / 60.0,      # s -> min
        "mvpa": day_df["time_spent_MVPA_s"] / 60.0,    # s -> min
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(data) + 3), 5))
    else:
        fig = ax.figure

    x = set_day_axis(ax, day_df)

    # Light connector from lowest to highest value of each day
    lows = data[["wear", "lpa", "mvpa"]].min(axis=1)
    highs = data[["wear", "lpa", "mvpa"]].max(axis=1)
    ax.vlines(x, lows, highs, color=COLORS["stem"], linewidth=2, zorder=1)

    series = [
        ("wear", "Wearing time", 110),
        ("mvpa", "Time in MVPA", 90),
        ("lpa", "Time in LPA", 90),
    ]

    for key, label, size in series:
        ax.scatter(
            x, data[key],
            s=size, color=COLORS[key],
            edgecolor=COLORS["surface"], linewidth=1.5,
            label=label, zorder=3,
        )

    # Value labels
    #   wearing time -> above the dot
    #   MVPA         -> right of the dot
    #   LPA          -> left of the dot
    label_specs = [
        ("wear", (0, 9),   "center", "bottom"),
        ("mvpa", (10, 0),  "left",   "center"),
        ("lpa",  (-10, 0), "right",  "center"),
    ]

    for key, offset, ha, va in label_specs:
        for xi, value in zip(x, data[key]):
            if pd.notna(value):
                ax.annotate(
                    format_duration_long(value),
                    (xi, value), xytext=offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=9, fontweight="bold",
                    color=COLORS[key],
                )

    y_max = np.nanmax(data[["wear", "lpa", "mvpa"]].to_numpy())
    y_top = (np.ceil(y_max / tick_minutes) + 1) * tick_minutes
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_minutes))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_duration))
    ax.set_ylabel("Time")

    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    set_title_and_legend(ax, "Daily wearing and activity time", patient_id, ncol=3)

    fig.tight_layout()
    return fig, ax


# =====================================================================
# PLOT 2 - WALKING-BOUT PATTERN (lines or stacked bar)
# =====================================================================
WB_LABELS = [
    # (key in COLORS, day-level column, legend label)
    # order = longest -> shortest (bottom -> top in the stacked bar)
    ("long", "n_long_WB", "Long (> 60 s)"),
    ("medium", "n_medium_WB", "Medium (30-60 s)"),
    ("short", "n_short_WB", "Short (10-30 s)"),
    ("very_short", "n_very_short_WB", "Very short (< 10 s)"),
]


def _place_value_labels(ax, labels, x_offset=0.07, min_gap_frac=0.05):
    """
    Write value labels to the right of their dots, nudging labels of the
    same day apart vertically so they never overlap.

    labels : list of (x, y, text, color)
    min_gap_frac : minimum vertical distance between labels, as a fraction
                   of the y-axis range
    """
    y0, y1 = ax.get_ylim()
    min_gap = (y1 - y0) * min_gap_frac

    by_day = {}
    for item in labels:
        by_day.setdefault(item[0], []).append(item)

    for xi, items in by_day.items():
        items = sorted(items, key=lambda t: t[1])      # bottom -> top
        placed_y = []
        for _, y, _, _ in items:
            y_new = y if not placed_y else max(y, placed_y[-1] + min_gap)
            placed_y.append(y_new)

        for (_, _, text, color), y_new in zip(items, placed_y):
            ax.text(
                xi + x_offset, y_new, text,
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["surface"],
                          ec="none", alpha=0.9),
            )


def _draw_wb_lines(ax, x, counts, show_values):
    """One line per WB label, each dot connected to the next day."""
    value_labels = []

    for key, _, label in WB_LABELS:
        if key not in counts:
            continue
        values = counts[key]

        ax.plot(
            x, values,
            color=COLORS[key], linewidth=2,
            marker="o", markersize=9,
            markeredgecolor=COLORS["surface"], markeredgewidth=1.5,
            label=label, zorder=3,
            clip_on=False,   # keep dots at 0 fully visible
        )

        for xi, v in zip(x, values):
            value_labels.append((xi, float(v), f"{int(v)}", COLORS[key]))

    y_max = max(v.max() for v in counts.values())
    ax.set_ylim(0, y_max * 1.15 + 1)

    if show_values:
        _place_value_labels(ax, value_labels)


def _draw_wb_stacked(ax, x, counts, show_values, bar_width=0.5,
                     min_segment_frac=0.04):
    """
    Stacked bar per day: long at the bottom, very short on top.
    Segment values are written inside the segment when it is tall enough.
    """
    bottoms = np.zeros(len(x))
    totals = sum(counts.values())
    y_max = totals.max()
    y_top = y_max * 1.15 + 1
    ax.set_ylim(0, y_top)
    min_height = y_top * min_segment_frac

    for key, _, label in WB_LABELS:
        if key not in counts:
            continue
        values = counts[key].to_numpy(dtype=float)

        ax.bar(
            x, values, bottom=bottoms, width=bar_width,
            color=COLORS[key], label=label,
            edgecolor=COLORS["surface"], linewidth=1.5,   # gap between segments
            zorder=3,
        )

        if show_values:
            # dark segments -> white text, light segments -> dark text
            text_color = (COLORS["surface"] if key in ("long", "medium", "short")
                          else COLORS["ink"])
            for xi, v, b in zip(x, values, bottoms):
                if v >= min_height and v > 0:
                    ax.text(
                        xi, b + v / 2, f"{int(v)}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=text_color, zorder=4,
                    )

        bottoms = bottoms + values


def plot_wb_pattern(
    day_df: pd.DataFrame,
    patient_id: str = "",
    style: str = "lines",
    labels_to_show=("very_short", "short", "medium", "long"),
    show_values: bool = True,
    show_day_summary: bool = True,
    show_walking_time: bool = True,
    ax=None,
):
    """
    Number of walking bouts per day by WB duration label.

    style = "lines"   -> one line per label, dots connected day to day
    style = "stacked" -> one stacked bar per day (long at the bottom)

    show_walking_time -> time spent walking (LPA + MVPA) per day on a second
                         y axis (right), both axes starting from the same 0

    Uses day-level columns:
        n_very_short_WB, n_short_WB, n_medium_WB, n_long_WB
    for the walking-time axis:
        time_spent_LPA_s, time_spent_MVPA_s
    and, for the summary table:
        wearing_time_hours, time_spent_MVPA_s, time_spent_LPA_s,
        n_strides_total_day
    """
    if style not in ("lines", "stacked"):
        raise ValueError("style must be 'lines' or 'stacked'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(9.5, 1.6 * len(day_df) + 4), 6))
    else:
        fig = ax.figure

    x = set_day_axis(ax, day_df)

    counts = {
        key: pd.to_numeric(day_df[col], errors="coerce").fillna(0)
        for key, col, _ in WB_LABELS
        if key in labels_to_show
    }

    if style == "lines":
        _draw_wb_lines(ax, x, counts, show_values)
    else:
        _draw_wb_stacked(ax, x, counts, show_values)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.set_ylabel("Number of walking bouts")

    extra_axes = []
    if show_walking_time:
        ax2 = add_walking_time_axis(ax, x, day_df, show_values=show_values)
        extra_axes.append(ax2)

    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    if show_day_summary:
        add_day_summary(ax, day_df)

    set_title_and_legend(ax, "Mobility Amount & Pattern", patient_id, ncol=5,
                         extra_axes=extra_axes)

    fig.tight_layout()
    return fig, ax

def format_min_sec(seconds) -> str:
    """96 -> '1 min 36 s', 31 -> '31 s', 1918 -> '32 min', 4000 -> '1 h 07 min'"""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 600:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m} min {s:02d} s"
    return format_duration_long(seconds / 60)


# =====================================================================
# PLOT 3 - DISTANCE AND MAXIMUM WALKING-BOUT DURATION
# =====================================================================
def plot_distance_max_wb(
    day_df: pd.DataFrame,
    patient_id: str = "",
    show_values: bool = True,
    show_day_summary: bool = False,
    headroom: float = 1.2,
    dodge: float = 0.00,
    ax=None,
):
    """
    Two lines over the days, both axes starting from 0:
        left y  (minutes) : longest walking bout of the day
        right y (km)      : estimated distance walked

    dodge : small horizontal shift (in day units) between the two markers
            of the same day, so they never hide each other

    Uses day-level columns:
        max_WB_duration_s, estimated_distance_walked_m
    """
    max_wb_min = pd.to_numeric(day_df["max_WB_duration_s"], errors="coerce") / 60.0
    distance_km = pd.to_numeric(day_df["estimated_distance_walked_m"], errors="coerce") / 1000.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(9.5, 1.6 * len(day_df) + 4), 5.5))
    else:
        fig = ax.figure

    x = set_day_axis(ax, day_df)
    x_left = x - dodge     # max WB duration
    x_right = x + dodge    # distance

    # ---- Left axis: max WB duration (minutes) ----
    m_max = max(float(np.nanmax(max_wb_min.to_numpy())), 1.0)
    ax.set_ylim(0, m_max * headroom)

    ax.plot(
        x_left, max_wb_min,
        color=COLORS["max_wb"], linewidth=2,
        marker="o", markersize=9,
        markeredgecolor=COLORS["surface"], markeredgewidth=1.5,
        label="Longest walking bout", zorder=3, clip_on=False,
    )

    ax.yaxis.set_major_locator(mticker.MultipleLocator(_nice_minute_step(m_max)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_duration))
    ax.tick_params(axis="y", colors=COLORS["max_wb"])
    ax.set_ylabel("Longest walking bout (min)", color=COLORS["max_wb"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    # ---- Right axis: estimated distance (km) ----
    ax2 = ax.twinx()
    d_max = max(float(np.nanmax(distance_km.to_numpy())), 0.1)
    ax2.set_ylim(0, d_max * headroom)

    ax2.plot(
        x_right, distance_km,
        color=COLORS["distance"], linewidth=2, linestyle=(0, (4, 2)),
        marker="D", markersize=7,
        markeredgecolor=COLORS["surface"], markeredgewidth=1.5,
        label="Estimated distance walked", zorder=5, clip_on=False,
    )

    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g} km"))
    ax2.tick_params(axis="y", colors=COLORS["distance"])
    ax2.set_ylabel("Estimated distance (km)", color=COLORS["distance"])
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)

    # ---- Value labels: duration above its dot, distance below its dot ----
    if show_values:
        for xi, v in zip(x_left, max_wb_min):
            if pd.notna(v):
                ax.annotate(
                    format_min_sec(v * 60),
                    (xi, v), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=COLORS["max_wb"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["surface"],
                              ec="none", alpha=0.9),
                )
        for xi, v in zip(x_right, distance_km):
            if pd.notna(v):
                ax2.annotate(
                    f"{v:.2f} km",
                    (xi, v), xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=9, fontweight="bold",
                    color=COLORS["distance"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["surface"],
                              ec="none", alpha=0.9),
                )

    if show_day_summary:
        add_day_summary(ax, day_df)

    set_title_and_legend(ax, "Walking distance & longest bout", patient_id,
                         ncol=2, extra_axes=[ax2])

    fig.tight_layout()
    return fig, ax
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

    if PLOTS.get("time_variables"):
        fig, _ = plot_time_variables(day_df, patient_id=PATIENT_ID)
        figures["time_variables"] = fig

    if PLOTS.get("wb_pattern"):
        fig, _ = plot_wb_pattern(
            day_df,
            patient_id=PATIENT_ID,
            style=WB_PATTERN_STYLE,
            show_values=WB_PATTERN_SHOW_VALUES,
            show_day_summary=WB_PATTERN_SHOW_DAY_SUMMARY,
            show_walking_time=WB_PATTERN_SHOW_WALKING_TIME,
        )
        figures[f"wb_pattern_{WB_PATTERN_STYLE}"] = fig
    
    if PLOTS.get("distance_max_wb"):
        fig, _ = plot_distance_max_wb(day_df, patient_id=PATIENT_ID)
        figures["distance_max_wb"] = fig
    
    if SAVE_PLOTS:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = OUTPUT_FOLDER / f"{PATIENT_ID}_{name}.png"
            fig.savefig(path)
            print(f"Saved: {path}")

    if SHOW_PLOTS:
        plt.show()

    return figures


if __name__ == "__main__":
    main()