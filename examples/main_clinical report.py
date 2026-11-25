
# -*- coding: utf-8 -*-
"""
DMOs clinical report - one PDF (2 A4 pages) per patient and period.

    Page 1 - Amount & Pattern     : wearing time, walking time (LPA/MVPA +
                                    longest WB), walking-bout counts,
                                    strides & distance
           - Bout-to-bout variability : CoV of gait parameters per day
    Page 2 - Gait distributions   : KDE per WB category (2 x 2)
           - Pauses               : pauses per day + pause-duration KDE

Usage: edit the settings in the MAIN block at the bottom and run the file.

Inputs
    day-level DMOs CSV : one row per patient and recording_date
    WB-level CSV       : one row per walking bout
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


# =====================================================================
# STYLE
# =====================================================================
COLORS = {
    # neutrals
    "ink": "#0F172A",          # main text
    "ink_muted": "#64748B",    # secondary text, axes
    "grid": "#E2E8F0",         # grid lines
    "rule": "#CBD5E1",
    "surface": "#FFFFFF",

    # Walking-bout counts: blue ramp, light (shortest) -> dark (longest)
    "very_short": "#D3E3F7",
    "short": "#7FAEE3",
    "medium": "#2F67B1",
    "long": "#0E2A57",

    # Wear strip: slate
    "wear": "#475569",
    "wear_track": "#E5E9F0",

    # Walking time: greens
    "lpa": "#A7D7B8",
    "mvpa": "#1F7A4D",
    "max_wb": "#0E2A57",       # SAME as long WBs

    # Strides & distance
    "strides": "#FBE3A1",      # light warm yellow bars
    "strides_text": "#78350F", # dark brown: bar values, left axis
    "distance": "#B45309",     # deep amber line (= cov_cadence)

    # Bout-to-bout variability: pace = blues, rhythm = ambers
    "cov_speed": "#0E2A57",
    "cov_stride_length": "#5B93D6",
    "cov_cadence": "#B45309",
    "cov_stride_time": "#E9A23B",

    # KDE curves per WB category
    "kde_short": "#5B93D6",
    "kde_medium": "#2F67B1",
    "kde_long": "#0E2A57",

    # Pauses: warm brown ramp, light (shortest) -> dark (longest)
    "pause_1": "#F1EADF",
    "pause_2": "#D2BE9E",
    "pause_3": "#A68660",
    "pause_4": "#6E5538",
    "pause_5": "#3A2B1A",
    "pause_kde": "#6E5538",
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.facecolor": COLORS["surface"],
    "figure.facecolor": COLORS["surface"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["ink_muted"],
    "axes.labelsize": 8.5,
    "axes.titlecolor": COLORS["ink"],
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.color": COLORS["ink_muted"],
    "ytick.color": COLORS["ink_muted"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,        # embed fonts as text (selectable, editable)
}


# =====================================================================
# CONSTANTS (categories, columns, labels)
# =====================================================================
WB_LABELS = [
    # (key, day-level column, legend label) - bottom -> top of the stack
    ("long",       "n_long_WB",       "Long > 60 s"),
    ("medium",     "n_medium_WB",     "Medium 30-60 s"),
    ("short",      "n_short_WB",      "Short 10-30 s"),
    ("very_short", "n_very_short_WB", "Very short < 10 s"),
]
DARK_WB_SEGMENTS = ("long", "medium")          # white text inside these

WB_LABEL_TEXT = {
    "very_short": "Very short < 10 s",
    "short": "Short 10-30 s",
    "medium": "Medium 30-60 s",
    "long": "Long > 60 s",
}

WB_CATEGORY_TEXT = {
    "short": "short WBs (10-30 s)",
    "medium": "medium WBs (30-60 s)",
    "long": "long WBs (> 60 s)",
}

VARIABILITY_PARAMS = [
    # (key in COLORS, column suffix in day file, legend label, marker, line)
    ("cov_speed",         "gait velocity [m/s]",     "Gait velocity", "o", "-"),
    ("cov_stride_length", "stride length [m]",       "Stride length", "s", "-"),
    ("cov_cadence",       "cadence_[steps_per_min]", "Cadence",       "o", "--"),
    ("cov_stride_time",   "stride time [s]",         "Stride time",   "s", "--"),
]

KDE_WB_LABELS = ["short", "medium", "long"]    # very short excluded

KDE_FEATURES = [
    # (column in WB file, panel title, unit, decimals)
    ("gait velocity [m/s]",     "Gait velocity", "m/s",       2),
    ("stride length [m]",       "Stride length", "m",         2),
    ("cadence_[steps_per_min]", "Cadence",       "steps/min", 0),
    ("stride time [s]",         "Stride time",   "s",         2),
]

# Pause categories (upper limits in seconds). 1, 5 and 10 min are the
# inter-bout thresholds used in Mobilise-D D6.4 to define walking clusters.
PAUSE_BINS = [
    # (key in COLORS, upper limit s, legend label)
    ("pause_1", 10,     "< 10 s"),
    ("pause_2", 60,     "10 s – 1 min"),
    ("pause_3", 300,    "1 – 5 min"),
    ("pause_4", 600,    "5 – 10 min"),
    ("pause_5", np.inf, "> 10 min"),
]
DARK_PAUSE_SEGMENTS = ("pause_4", "pause_5")
PAUSE_REFERENCE_S = [60, 300, 600]

# A4 page and margins (inches)
PAGE_SIZE = (8.27, 11.69)
MARGIN_LEFT = 0.85
MARGIN_RIGHT = 0.75
MARGIN_TOP = 0.45
MARGIN_BOTTOM = 0.55
COLUMN_GAP = 0.35


# =====================================================================
# FORMATTERS (module level, used by matplotlib tick formatters)
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
    """Value label (minutes): 0.5 -> '30 s', 1.6 -> '1 min 36 s', 642 -> '10 h 42 min'"""
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


def format_seconds_axis(seconds) -> str:
    """Durations in seconds: 5 s, 30 s, 1 min, 10 min, 1 h."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:g} s"
    if seconds < 3600:
        return f"{seconds / 60:g} min"
    return f"{seconds / 3600:g} h"


def format_thousands(value, _pos=None) -> str:
    return f"{int(value):,}"


def format_km(value, _pos=None) -> str:
    return f"{value:g} km"


def format_percent(value, _pos=None) -> str:
    return f"{value:g} %"


def nice_minute_step(max_minutes: float) -> int:
    """Tick step (minutes) giving about 4-6 ticks."""
    for step in (5, 10, 15, 30, 60, 120, 180):
        if max_minutes / step <= 6:
            return step
    return 240


# =====================================================================
# REPORT CLASS
# =====================================================================
class DMOsReport:
    """
    One patient, one period (1 to `max_days` days).

    dates = "all"                        -> every available day
    dates = ["2023-07-10", "2023-07-14"] -> listed days
    dates = ("2023-07-10", "2023-07-16") -> from day -> to day (inclusive)
    """

    def __init__(self, patient_id, day_dmos_path, wb_level_path, dates="all",
                 max_days=7, sampling_rate_hz=102.4, sections=None,
                 amount_panels=None, pause_panels=None, show_values=True,
                 wear_valid_hours=None, variability_wb="long", kde_min_wb=5,
                 pause_kde_by_day=False, pause_reference_lines=False,
                 report_title="Mobility Report"):
        self.patient_id = str(patient_id)
        self.day_dmos_path = Path(day_dmos_path)
        self.wb_level_path = Path(wb_level_path)
        self.dates = dates
        self.max_days = max_days
        self.fs = sampling_rate_hz
        self.show_values = show_values
        self.wear_valid_hours = wear_valid_hours
        self.variability_wb = variability_wb
        self.kde_min_wb = kde_min_wb
        self.pause_kde_by_day = pause_kde_by_day
        self.pause_reference_lines = pause_reference_lines
        self.report_title = report_title

        self.sections = {"amount_pattern": True, "variability": True,
                         "kde": True, "pauses": True}
        self.sections.update(sections or {})
        self.amount_panels = {"wear_strip": True, "time": True,
                              "wb_counts": True, "strides_distance": True}
        self.amount_panels.update(amount_panels or {})
        self.pause_panels = {"counts": True, "kde": True}
        self.pause_panels.update(pause_panels or {})

        self.day_df = self.load_day_dmos()
        self.day_df = self.select_dates(self.day_df)
        self.wb_df = self.load_wb_level()
        self.pauses_df = self.compute_pauses()
        self.figures = []
        self.stats = {}

        print(f"{self.patient_id}: {len(self.day_df)} day(s) -> "
              f"{', '.join(self.day_df['recording_date'].dt.strftime('%Y-%m-%d'))}")

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------
    def load_day_dmos(self) -> pd.DataFrame:
        df = pd.read_csv(self.day_dmos_path)
        df = df[df["patient_id"].astype(str) == self.patient_id].copy()
        if df.empty:
            raise ValueError(f"No rows for patient {self.patient_id} in {self.day_dmos_path}")
        df["recording_date"] = pd.to_datetime(df["recording_date"])
        return df.sort_values("recording_date").reset_index(drop=True)

    def select_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = self.dates
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
            raise ValueError(f"No data for the selected dates: {self.dates}")
        if len(out) > self.max_days:
            raise ValueError(f"{len(out)} days selected; the report shows at most "
                             f"{self.max_days}. Narrow the dates.")
        return out.reset_index(drop=True)

    def load_wb_level(self) -> pd.DataFrame:
        """WB-level file filtered to the patient and the selected days."""
        wb = pd.read_csv(self.wb_level_path)
        wb = wb[wb["patient_id"].astype(str) == self.patient_id].copy()
        wb["recording_date"] = pd.to_datetime(wb["recording_date"])
        wb = wb[wb["recording_date"].isin(self.day_df["recording_date"])]
        return wb.reset_index(drop=True)

    def compute_pauses(self) -> pd.DataFrame:
        """
        Inter-bout time (IBT): end of one WB -> start of the next WB, within
        the same day and session (Mobilise-D D6.4). Columns start / end are
        in samples. Returns: recording_date, [session_id], pause_s.
        """
        group_cols = ["recording_date"]
        if "session_id" in self.wb_df.columns:
            group_cols.append("session_id")

        df = self.wb_df.sort_values(group_cols + ["start"]).copy()
        next_start = df.groupby(group_cols)["start"].shift(-1)
        df["pause_s"] = (next_start - df["end"]) / self.fs
        out = df.loc[df["pause_s"].notna() & (df["pause_s"] >= 0), group_cols + ["pause_s"]]
        return out.reset_index(drop=True)

    def col(self, name) -> pd.Series:
        return pd.to_numeric(self.day_df[name], errors="coerce")

    @property
    def period_text(self) -> str:
        start = self.day_df["recording_date"].min().strftime("%d %b %Y")
        end = self.day_df["recording_date"].max().strftime("%d %b %Y")
        n = len(self.day_df)
        period = start if start == end else f"{start} – {end}"
        return f"{period}  ·  {n} day{'s' if n > 1 else ''}"

    # -----------------------------------------------------------------
    # SHARED AXIS HELPERS
    # -----------------------------------------------------------------
    def style_day_axis(self, ax, show_labels=True):
        """One x tick per day: 'Mon\\n10 Jul'."""
        x = np.arange(len(self.day_df))
        ax.set_xticks(x)
        if show_labels:
            ax.set_xticklabels(self.day_df["recording_date"].dt.strftime("%a\n%d %b"))
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(self.day_df) - 0.4)
        return x

    @staticmethod
    def panel_header(ax, title, handles=None, labels=None, ncol=4):
        """Panel title on the left, legend on the same line on the right."""
        ax.set_title(title, pad=8)
        if handles:
            ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                      ncol=ncol, handletextpad=0.4, columnspacing=1.0,
                      borderaxespad=0.15, labelcolor=COLORS["ink_muted"])

    @staticmethod
    def place_labels_right(ax, items, x_offset=0.09, min_gap_frac=0.095):
        """Value labels right of their markers, pushed apart within a day."""
        y0, y1 = ax.get_ylim()
        min_gap = (y1 - y0) * min_gap_frac
        by_day = {}
        for item in items:
            by_day.setdefault(item[0], []).append(item)

        for xi, day_items in by_day.items():
            day_items = sorted(day_items, key=lambda t: t[1])
            placed = []
            for item in day_items:
                placed.append(item[1] if not placed else max(item[1], placed[-1] + min_gap))
            for (_, _, text, color), y_new in zip(day_items, placed):
                ax.text(xi + x_offset, y_new, text, ha="left", va="center",
                        fontsize=7.5, fontweight="bold", color=color, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc=COLORS["surface"],
                                  ec="none", alpha=0.9))

    # -----------------------------------------------------------------
    # PANELS - AMOUNT & PATTERN
    # -----------------------------------------------------------------
    def plot_wear_strip(self, ax, show_xlabels=False, bar_width=0.55, track_height=0.4):
        """Wearing time per day as a filled 24 h track."""
        x = self.style_day_axis(ax, show_labels=show_xlabels)
        wear_h = self.col("wearing_time_hours")
        ax.set_ylim(0, 1)
        y_track = 0.25

        for xi, h in zip(x, wear_h):
            left = xi - bar_width / 2
            ax.barh(y_track, bar_width, left=left, height=track_height,
                    color=COLORS["wear_track"], zorder=2)
            if pd.notna(h):
                frac = min(max(h / 24.0, 0.0), 1.0)
                ax.barh(y_track, bar_width * frac, left=left, height=track_height,
                        color=COLORS["wear"], zorder=3)
                if self.show_values:
                    ax.text(xi, y_track + track_height / 2 + 0.08,
                            f"{format_duration(h * 60)}",
                            ha="center", va="bottom", fontsize=7.5,
                            fontweight="bold", color=COLORS["wear"])
            if self.wear_valid_hours is not None:
                xv = left + bar_width * self.wear_valid_hours / 24.0
                ax.plot([xv, xv], [y_track - track_height / 2 - 0.05,
                                   y_track + track_height / 2 + 0.05],
                        color=COLORS["ink_muted"], linewidth=1, zorder=4)

        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        title = "Wearing time (of 24 h)"
        if self.wear_valid_hours is not None:
            title += f"  (tick = {self.wear_valid_hours:g} h)"
        ax.set_title(title, pad=2)

    def plot_time(self, ax, show_xlabels=True, bar_width=0.55, headroom=1.15,
                  min_segment_frac=0.08):
        """LPA + MVPA stacked bars, longest WB as a connected diamond."""
        x = self.style_day_axis(ax, show_labels=show_xlabels)
        lpa = (self.col("time_spent_LPA_s") / 60).fillna(0).to_numpy()
        mvpa = (self.col("time_spent_MVPA_s") / 60).fillna(0).to_numpy()
        max_wb = (self.col("max_WB_duration_s") / 60).to_numpy()

        y_max = max(float(np.nanmax(np.r_[lpa + mvpa, max_wb])), 1.0)
        y_top = y_max * headroom
        ax.set_ylim(0, y_top)

        ax.bar(x, lpa, width=bar_width, color=COLORS["lpa"], label="LPA",
               edgecolor=COLORS["surface"], linewidth=1.2, zorder=3)
        ax.bar(x, mvpa, bottom=lpa, width=bar_width, color=COLORS["mvpa"],
               label="MVPA", edgecolor=COLORS["surface"], linewidth=1.2, zorder=3)

        if self.show_values:
            x_text = x - bar_width / 2 + 0.03
            segments = [(lpa, np.zeros_like(lpa), COLORS["ink"]),
                        (mvpa, lpa, COLORS["surface"])]
            for values, bottoms, txt_color in segments:
                for xi, v, b in zip(x_text, values, bottoms):
                    if v > 0 and v >= y_top * min_segment_frac:
                        ax.text(xi, b + v / 2, format_value_time(v), ha="left",
                                va="center", fontsize=7.5, fontweight="bold",
                                color=txt_color, zorder=4)

        ax.plot(x, max_wb, linestyle="-", linewidth=1.6, color=COLORS["max_wb"],
                marker="D", markersize=7, markeredgecolor=COLORS["surface"],
                markeredgewidth=1.8, label="Longest WB", zorder=5, clip_on=False)

        if self.show_values:
            for xi, v in zip(x, max_wb):
                if pd.notna(v):
                    ax.annotate(format_value_time(v), (xi, v), xytext=(7, 6),
                                textcoords="offset points", ha="left", va="bottom",
                                fontsize=7.5, fontweight="bold", color=COLORS["max_wb"],
                                zorder=6, bbox=dict(boxstyle="round,pad=0.1",
                                                    fc=COLORS["surface"], ec="none",
                                                    alpha=0.9))

        ax.yaxis.set_major_locator(mticker.MultipleLocator(nice_minute_step(y_max)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_duration))
        ax.set_ylabel("Duration")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

        h, l = ax.get_legend_handles_labels()
        order = [l.index(name) for name in ("LPA", "MVPA", "Longest WB")]
        self.panel_header(ax, "Time Spent Walking & Longest Bout",
                          [h[i] for i in order], [l[i] for i in order], ncol=3)

    def plot_wb_counts(self, ax, show_xlabels=True, bar_width=0.55,
                       min_segment_frac=0.06, headroom=1.08):
        """Stacked bar of walking bouts per duration category."""
        x = self.style_day_axis(ax, show_labels=show_xlabels)
        counts = {key: self.col(c).fillna(0).to_numpy(dtype=float) for key, c, _ in WB_LABELS}
        y_top = max(float(sum(counts.values()).max()), 1.0) * headroom
        ax.set_ylim(0, y_top)

        bottoms = np.zeros(len(x))
        for key, _, label in WB_LABELS:
            v = counts[key]
            ax.bar(x, v, bottom=bottoms, width=bar_width, color=COLORS[key],
                   label=label, edgecolor=COLORS["surface"], linewidth=1.2, zorder=3)
            if self.show_values:
                txt_color = COLORS["surface"] if key in DARK_WB_SEGMENTS else COLORS["ink"]
                for xi, val, b in zip(x, v, bottoms):
                    if val > 0 and val >= y_top * min_segment_frac:
                        ax.text(xi, b + val / 2, f"{int(val)}", ha="center",
                                va="center", fontsize=7.5, fontweight="bold",
                                color=txt_color, zorder=4)
            bottoms = bottoms + v

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        ax.set_ylabel("Walking bouts (n)")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        h, l = ax.get_legend_handles_labels()
        self.panel_header(ax, "Walking-bout pattern", h[::-1], l[::-1], ncol=4)

    def plot_strides_distance(self, ax, show_xlabels=True, bar_width=0.55,
                              bars_headroom=1.45, line_headroom=1.12):
        """Strides (bars, left axis) and estimated distance (line, right axis)."""
        x = self.style_day_axis(ax, show_labels=show_xlabels)
        strides = self.col("n_strides_total_day")
        distance_km = self.col("estimated_distance_walked_m") / 1000.0

        s_max = max(float(np.nanmax(strides.to_numpy())), 1.0)
        ax.set_ylim(0, s_max * bars_headroom)
        ax.bar(x, strides.fillna(0), width=bar_width, color=COLORS["strides"],
               label="Strides", zorder=3)

        if self.show_values:
            for xi, v in zip(x, strides):
                if pd.notna(v):
                    inside = v >= s_max * bars_headroom * 0.15
                    ax.annotate(f"{int(v):,}", (xi, v), xytext=(0, -4 if inside else 3),
                                textcoords="offset points", ha="center",
                                va="top" if inside else "bottom", fontsize=7.5,
                                fontweight="bold", color=COLORS["strides_text"], zorder=4)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_thousands))
        ax.tick_params(axis="y", colors=COLORS["strides_text"])
        ax.set_ylabel("Strides (n)", color=COLORS["strides_text"])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

        ax2 = ax.twinx()
        d_max = max(float(np.nanmax(distance_km.to_numpy())), 0.1)
        ax2.set_ylim(0, d_max * line_headroom)
        ax2.plot(x, distance_km, color=COLORS["distance"], linewidth=1.8,
                 marker="o", markersize=6, markeredgecolor=COLORS["surface"],
                 markeredgewidth=1.2, label="Estimated distance", zorder=5, clip_on=False)

        if self.show_values:
            for xi, v in zip(x, distance_km):
                if pd.notna(v):
                    ax2.annotate(f"{v:.1f} km", (xi, v), xytext=(8, 0),
                                 textcoords="offset points", ha="left", va="center",
                                 fontsize=7.5, fontweight="bold", color=COLORS["distance"],
                                 bbox=dict(boxstyle="round,pad=0.1", fc=COLORS["surface"],
                                           ec="none", alpha=0.9))

        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(format_km))
        ax2.tick_params(axis="y", colors=COLORS["distance"], length=0)
        ax2.set_ylabel("Distance (km)", color=COLORS["distance"])
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_visible(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        self.panel_header(ax, "Total Strides Count & Estimated Distance Walked", h1 + h2, l1 + l2, ncol=2)

    # -----------------------------------------------------------------
    # PANEL - BOUT-TO-BOUT VARIABILITY
    # -----------------------------------------------------------------
    def plot_variability(self, ax, headroom=1.3, low_n=3):
        """CoV (%) per day of 4 gait parameters, on one WB category."""
        cat = self.variability_wb
        x = self.style_day_axis(ax, show_labels=True)

        values = {}
        for key, suffix, _, _, _ in VARIABILITY_PARAMS:
            name = f"cov_{cat}_WB_{suffix}"
            if name not in self.day_df.columns:
                raise KeyError(f"Column not found in day file: '{name}'")
            values[key] = self.col(name)

        all_vals = np.concatenate([v.to_numpy(dtype=float) for v in values.values()])
        y_max = np.nanmax(all_vals) if np.isfinite(all_vals).any() else 10.0
        ax.set_ylim(0, max(y_max, 1.0) * headroom)

        label_items = []
        for key, _, label, marker, line in VARIABILITY_PARAMS:
            v = values[key]
            ax.plot(x, v, linestyle=line, linewidth=1.6, color=COLORS[key],
                    marker=marker, markersize=6.5, markeredgecolor=COLORS["surface"],
                    markeredgewidth=1.2, label=label, zorder=4, clip_on=False)
            for xi, val in zip(x, v):
                if pd.notna(val):
                    label_items.append((xi, float(val), f"{val:.0f} %", COLORS[key]))

        if self.show_values:
            self.place_labels_right(ax, label_items)

        no_data = pd.concat(values, axis=1).isna().all(axis=1).to_numpy()
        for xi, missing in zip(x, no_data):
            if missing:
                ax.text(xi, ax.get_ylim()[1] * 0.08, "no data", ha="center",
                        va="bottom", fontsize=7.5, style="italic",
                        color=COLORS["ink_muted"])

        n_col = f"n_{cat}_WB"
        if n_col in self.day_df.columns:
            n_wb = self.col(n_col).fillna(0).astype(int)
            for xi, n in zip(x, n_wb):
                ax.annotate(f"{n} {cat} WB{'s' if n != 1 else ''}",
                            xy=(xi, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -30), textcoords="offset points",
                            ha="center", va="top", fontsize=7.5,
                            fontweight="bold" if n >= low_n else "normal",
                            style="normal" if n >= low_n else "italic",
                            color=COLORS["ink"] if n >= low_n else COLORS["ink_muted"])

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_percent))
        ax.set_ylabel("CoV (%)")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        h, l = ax.get_legend_handles_labels()
        self.panel_header(ax, f"CoV between {cat} walking bouts", h, l, ncol=4)

    # -----------------------------------------------------------------
    # PANEL - GAIT DISTRIBUTION (one KDE panel per gait parameter)
    # -----------------------------------------------------------------
    def plot_kde(self, ax, feature, title, unit, decimals, n_grid=300):
        """KDE per WB category with median (solid) and P95 (dashed)."""
        wb = self.wb_df
        stats = {}
        if feature not in wb.columns:
            raise KeyError(f"Column not found in WB file: '{feature}'")
        all_values = pd.to_numeric(wb.loc[wb["WB_label"].isin(KDE_WB_LABELS), feature],
                                   errors="coerce").dropna()

        ax.set_title(title, pad=4)
        ax.set_yticks([])
        if all_values.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", style="italic", color=COLORS["ink_muted"])
            return stats

        span = all_values.max() - all_values.min()
        pad = 0.15 * span if span > 0 else 0.1 * abs(all_values.mean())
        x_grid = np.linspace(all_values.min() - pad, all_values.max() + pad, n_grid)

        y_peak = 0.0
        curves = []
        for label in KDE_WB_LABELS:
            values = pd.to_numeric(wb.loc[wb["WB_label"] == label, feature],
                                   errors="coerce").dropna()
            n = int(len(values))
            entry = {"n": n, "median": None, "p95": None, "plotted": False}
            if n >= 1:
                entry["median"] = float(values.median())
                entry["p95"] = float(values.quantile(0.95))
            if n >= self.kde_min_wb and values.nunique() > 1:
                kde = gaussian_kde(values)
                y = kde(x_grid)
                y_peak = max(y_peak, float(y.max()))
                curves.append((label, kde, y, entry))
                entry["plotted"] = True
            stats[label] = entry

        for label, kde, y, entry in curves:
            color = COLORS[f"kde_{label}"]
            ax.fill_between(x_grid, 0, y, color=color, alpha=0.12, zorder=2)
            ax.plot(x_grid, y, color=color, linewidth=1.8, zorder=3)
            for value, style in ((entry["median"], "-"), (entry["p95"], (0, (3, 2)))):
                ax.plot([value, value], [0, float(kde(value)[0])], color=color,
                        linewidth=1.4, linestyle=style, zorder=4)

        ax.set_ylim(0, y_peak * (1.6 if self.show_values else 1.1) if y_peak > 0 else 1)
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.set_xlabel(unit)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

        if self.show_values:
            y_txt = 0.97
            for label in KDE_WB_LABELS:
                entry = stats[label]
                if entry["plotted"]:
                    text = (f"med {entry['median']:.{decimals}f}  ·  "
                            f"P95 {entry['p95']:.{decimals}f}  (n={entry['n']})")
                    color, weight = COLORS[f"kde_{label}"], "bold"
                else:
                    text = f"{label}: n={entry['n']} – not plotted"
                    color, weight = COLORS["ink_muted"], "normal"
                ax.text(0.99, y_txt, text, transform=ax.transAxes, ha="right",
                        va="top", fontsize=7, fontweight=weight, color=color)
                y_txt -= 0.11
        return stats

    # -----------------------------------------------------------------
    # PANELS - PAUSES
    # -----------------------------------------------------------------
    def plot_pause_counts(self, ax, bar_width=0.55, min_segment_frac=0.06, headroom=1.08):
        """Stacked bar per day: number of pauses per duration category."""
        x = self.style_day_axis(ax, show_labels=True)
        edges = [0] + [upper for _, upper, _ in PAUSE_BINS]
        counts = {key: np.zeros(len(self.day_df)) for key, _, _ in PAUSE_BINS}
        for i, day in enumerate(self.day_df["recording_date"]):
            p = self.pauses_df.loc[self.pauses_df["recording_date"] == day, "pause_s"]
            for (key, _, _), lo, hi in zip(PAUSE_BINS, edges[:-1], edges[1:]):
                counts[key][i] = ((p >= lo) & (p < hi)).sum()

        y_top = max(float(sum(counts.values()).max()), 1.0) * headroom
        ax.set_ylim(0, y_top)

        bottoms = np.zeros(len(x))
        for key, _, label in PAUSE_BINS:
            v = counts[key]
            ax.bar(x, v, bottom=bottoms, width=bar_width, color=COLORS[key],
                   label=label, edgecolor=COLORS["surface"], linewidth=1.2, zorder=3)
            if self.show_values:
                txt = COLORS["surface"] if key in DARK_PAUSE_SEGMENTS else COLORS["ink"]
                for xi, val, b in zip(x, v, bottoms):
                    if val > 0 and val >= y_top * min_segment_frac:
                        ax.text(xi, b + val / 2, f"{int(val)}", ha="center",
                                va="center", fontsize=7.5, fontweight="bold",
                                color=txt, zorder=4)
            bottoms = bottoms + v

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        ax.set_ylabel("Pauses (n)")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        h, l = ax.get_legend_handles_labels()
        self.panel_header(ax, "Pauses per day", h, l, ncol=5)
        return counts

    def plot_pause_kde(self, ax, min_pauses=5, n_grid=400):
        """Pause-duration KDE on a log time axis, median (solid) and P90 (dashed)."""
        pauses = self.pauses_df
        stats = {}
        if pauses.empty:
            ax.text(0.5, 0.5, "no pauses", transform=ax.transAxes, ha="center",
                    va="center", style="italic", color=COLORS["ink_muted"])
            ax.set_yticks([])
            self.panel_header(ax, "Pause duration distribution")
            return stats

        log_all = np.log10(pauses["pause_s"].clip(lower=0.5))
        pad = 0.15 * ((log_all.max() - log_all.min()) or 1)
        grid = np.linspace(log_all.min() - pad, log_all.max() + pad, n_grid)

        if self.pause_kde_by_day:
            days = list(self.day_df["recording_date"])
            stone = LinearSegmentedColormap.from_list(
                "stone", [COLORS["pause_3"], COLORS["pause_5"], "#2E271F"])
            ramp = stone(np.linspace(0.0, 1.0, max(len(days), 1)))
            groups = [(d.strftime("%a %d %b"),
                       pauses.loc[pauses["recording_date"] == d, "pause_s"], ramp[i])
                      for i, d in enumerate(days)]
        else:
            groups = [("All days", pauses["pause_s"], COLORS["pause_kde"])]

        y_peak, curves = 0.0, []
        for name, values, color in groups:
            values = values.clip(lower=0.5)
            n = int(len(values))
            entry = {"n": n,
                     "median_s": float(values.median()) if n else None,
                     "p90_s": float(values.quantile(0.90)) if n else None,
                     "plotted": False, "color": color}
            if n >= min_pauses and values.nunique() > 1:
                kde = gaussian_kde(np.log10(values))
                y = kde(grid)
                y_peak = max(y_peak, float(y.max()))
                curves.append((name, kde, y, color, entry))
                entry["plotted"] = True
            stats[name] = entry

        for name, kde, y, color, entry in curves:
            ax.fill_between(grid, 0, y, color=color, alpha=0.12, zorder=2)
            ax.plot(grid, y, color=color, linewidth=1.8, zorder=3,
                    label=name if self.pause_kde_by_day else None)
            for value, style in ((entry["median_s"], "-"), (entry["p90_s"], (0, (3, 2)))):
                lv = np.log10(value)
                ax.plot([lv, lv], [0, float(kde(lv)[0])], color=color,
                        linewidth=1.4, linestyle=style, zorder=4)

        ax.set_ylim(0, y_peak * (1.5 if self.show_values else 1.1) if y_peak else 1)
        ax.set_xlim(grid[0], grid[-1])

        if self.pause_reference_lines:
            for ref in PAUSE_REFERENCE_S:
                lr = np.log10(ref)
                if grid[0] < lr < grid[-1]:
                    ax.axvline(lr, color=COLORS["ink_muted"], linewidth=0.8,
                               linestyle=(0, (1, 2)), zorder=1)

        tick_s = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 14400]
        ticks = [np.log10(t) for t in tick_s if grid[0] <= np.log10(t) <= grid[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_seconds_axis(10 ** t) for t in ticks])
        ax.set_xlabel("Pause duration (log scale)")
        ax.set_yticks([])
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_color(COLORS["grid"])

        if self.show_values:
            y_txt = 0.97
            for name, entry in stats.items():
                prefix = f"{name}:  " if self.pause_kde_by_day else ""
                if entry["plotted"]:
                    text = (f"{prefix}median {format_value_time(entry['median_s'] / 60)}  ·  "
                            f"P90 {format_value_time(entry['p90_s'] / 60)}  (n={entry['n']})")
                    color, weight = entry["color"], "bold"
                else:
                    text = f"{name}: n={entry['n']} – not plotted"
                    color, weight = COLORS["ink_muted"], "normal"
                ax.text(0.99, y_txt, text, transform=ax.transAxes, ha="right",
                        va="top", fontsize=7, fontweight=weight, color=color)
                y_txt -= 0.09

        style_handles = [Line2D([], [], color=COLORS["ink_muted"], linewidth=1.4),
                         Line2D([], [], color=COLORS["ink_muted"], linewidth=1.4,
                                linestyle=(0, (3, 2)))]
        h, l = ax.get_legend_handles_labels()
        self.panel_header(ax, "Pause duration distribution", h + style_handles,
                          l + ["Median", "P90"], ncol=min(len(h) + 2, 5))
        return stats

    # -----------------------------------------------------------------
    # PAGE LAYOUT
    # -----------------------------------------------------------------
    # A page is a list of rows, top to bottom:
    #   ("space",  inches, [anchor])          empty space (titles, labels)
    #   ("header", inches, title, subtitle)   section title
    #   ("fixed",  inches, [keys])            axes row of fixed height
    #   ("flex",   weight, [keys])            axes row sharing the free height
    # Axes in the same row are placed side by side.
    def layout_page(self, fig, rows):
        page_w, page_h = fig.get_size_inches()
        used = sum(r[1] for r in rows if r[0] != "flex")
        weights = sum(r[1] for r in rows if r[0] == "flex")
        free = page_h - MARGIN_TOP - MARGIN_BOTTOM - used
        unit = free / weights if weights else 0
        if weights and unit < 0.6:
            print(f"Warning: page is crowded (flex unit = {unit:.2f} in)")

        axes, anchors = {}, {}
        y = page_h - MARGIN_TOP
        row_w = page_w - MARGIN_LEFT - MARGIN_RIGHT
        for row in rows:
            kind = row[0]
            if kind == "space":
                if len(row) > 2:
                    anchors[row[2]] = y
                y -= row[1]
            elif kind == "header":
                self.draw_section_header(fig, y, row[2], row[3])
                y -= row[1]
            else:
                height = row[1] if kind == "fixed" else row[1] * unit
                keys = row[2]
                width = (row_w - COLUMN_GAP * (len(keys) - 1)) / len(keys)
                for i, key in enumerate(keys):
                    left = MARGIN_LEFT + i * (width + COLUMN_GAP)
                    axes[key] = fig.add_axes([left / page_w, (y - height) / page_h,
                                              width / page_w, height / page_h])
                y -= height
        return axes, anchors

    def draw_page_header(self, fig, page, n_pages):
        page_w, page_h = fig.get_size_inches()
        x0, x1 = MARGIN_LEFT / page_w, 1 - MARGIN_RIGHT / page_w
        y = 1 - MARGIN_TOP / page_h
        fig.text(x0, y, self.report_title, ha="left", va="top", fontsize=15,
                 fontweight="bold", color=COLORS["ink"])
        fig.text(x0, y - 0.3 / page_h, f"Patient {self.patient_id}  ·  {self.period_text}",
                 ha="left", va="top", fontsize=9.5, color=COLORS["ink_muted"])
        fig.text(x1, y, f"Page {page} of {n_pages}", ha="right", va="top",
                 fontsize=8.5, color=COLORS["ink_muted"])

        fig.text(x0, 0.3 / page_h, f"Generated {date.today():%d %b %Y}  ·  "
                 "dmos_kit", ha="left", va="bottom",
                 fontsize=7.5, color=COLORS["ink_muted"])
    @staticmethod
    def draw_section_header(fig, y_inches, title, subtitle, rule_gap=0.14):
        """Horizontal rule across the page, then section title and subtitle."""
        page_w, page_h = fig.get_size_inches()
        x0, x1 = MARGIN_LEFT / page_w, 1 - MARGIN_RIGHT / page_w
        y_rule = y_inches / page_h
        fig.add_artist(Line2D([x0, x1], [y_rule, y_rule], transform=fig.transFigure,
                              color=COLORS["rule"], linewidth=1))
        y_title = y_inches - rule_gap
        fig.text(x0, y_title / page_h, title, ha="left", va="top", fontsize=12.5,
                 fontweight="bold", color=COLORS["ink"])
        if subtitle:
            fig.text(x0, (y_title - 0.24) / page_h, subtitle, ha="left", va="top",
                     fontsize=8.5, color=COLORS["ink_muted"])
    @staticmethod
    def draw_section_header(fig, y_inches, title, subtitle):
        page_w, page_h = fig.get_size_inches()
        x0 = MARGIN_LEFT / page_w
        fig.text(x0, y_inches / page_h, title, ha="left", va="top", fontsize=12.5,
                 fontweight="bold", color=COLORS["ink"])
        if subtitle:
            fig.text(x0, (y_inches - 0.24) / page_h, subtitle, ha="left", va="top",
                     fontsize=8.5, color=COLORS["ink_muted"])

    def rows_page_1(self):
        """Amount & Pattern + Variability."""
        rows = []
        panels = [k for k in ("wear_strip", "time", "wb_counts", "strides_distance")
                  if self.sections["amount_pattern"] and self.amount_panels.get(k)]
        if panels:
            rows.append(("header", 0.62, "Amount & Pattern", "Daily DMOs: wearing time, time spent walking, bout pattern, strides count & estimated distance walked"))
            for key in panels:
                if key == "wear_strip":
                    rows += [("space", 0.22), ("fixed", 0.32, [key])]  # title: "Wearing time (of 24 h)"
                else:
                    rows += [("space", 0.42), ("flex", 1.0, [key])]
            rows.append(("space", 0.55))          # day labels under the last panel
        if self.sections["variability"]:
            subtitle = (f"Coefficient of variation between "
                        f"{WB_CATEGORY_TEXT.get(self.variability_wb, self.variability_wb)}"
                        "  ·  dashed = rhythm parameters")
            rows += [("header", 0.62, "Bout-to-bout variability", subtitle),
                     ("space", 0.38), ("flex", 1.15, ["variability"]),
                     ("space", 0.75)]             # day labels + number of WBs
        return rows, panels

    def rows_page_2(self):
        """Gait distributions + Pauses."""
        rows = []
        if self.sections["kde"]:
            rows += [("header", 0.62, "Gait distributions",
                      "Walking-bout values pooled over the period, by bout duration"),
                     ("space", 0.25, "kde_legend"),
                     ("space", 0.25), ("flex", 0.9, ["kde_0", "kde_1"]),
                     ("space", 0.65), ("flex", 0.9, ["kde_2", "kde_3"]),
                     ("space", 0.55)]
        pause_keys = [k for k in ("counts", "kde")
                      if self.sections["pauses"] and self.pause_panels.get(k)]
        if pause_keys:
            rows.append(("header", 0.62, "Pauses",
                         "Time between consecutive walking bouts within a recording"))
            if "counts" in pause_keys:
                rows += [("space", 0.42), ("flex", 1.0, ["pause_counts"]), ("space", 0.55)]
            if "kde" in pause_keys:
                rows += [("space", 0.42), ("flex", 1.0, ["pause_kde"]), ("space", 0.5)]
        return rows

    # -----------------------------------------------------------------
    # PAGES
    # -----------------------------------------------------------------
    def build_page_1(self, page, n_pages):
        rows, panels = self.rows_page_1()
        fig = plt.figure(figsize=PAGE_SIZE)
        self.draw_page_header(fig, page, n_pages)
        axes, _ = self.layout_page(fig, [("space", 0.85)] + rows)

        plot_methods = {"wear_strip": self.plot_wear_strip, "time": self.plot_time,
                        "wb_counts": self.plot_wb_counts,
                        "strides_distance": self.plot_strides_distance}
        for i, key in enumerate(panels):
            plot_methods[key](axes[key], show_xlabels=(i == len(panels) - 1))
        if "variability" in axes:
            self.plot_variability(axes["variability"])
        return fig

    def build_page_2(self, page, n_pages):
        rows = self.rows_page_2()
        fig = plt.figure(figsize=PAGE_SIZE)
        self.draw_page_header(fig, page, n_pages)
        axes, anchors = self.layout_page(fig, [("space", 0.85)] + rows)

        if "kde_0" in axes:
            self.stats["kde"] = {}
            for i, (feature, title, unit, decimals) in enumerate(KDE_FEATURES[:4]):
                self.stats["kde"][feature] = self.plot_kde(axes[f"kde_{i}"], feature,
                                                           title, unit, decimals)
            self.draw_kde_legend(fig, anchors["kde_legend"])
        if "pause_counts" in axes:
            self.stats["pause_counts"] = self.plot_pause_counts(axes["pause_counts"])
        if "pause_kde" in axes:
            self.stats["pause_kde"] = self.plot_pause_kde(axes["pause_kde"])
        return fig

    @staticmethod
    def draw_kde_legend(fig, y_inches):
        page_w, page_h = fig.get_size_inches()
        handles = [Line2D([], [], color=COLORS[f"kde_{lab}"], linewidth=2.2)
                   for lab in KDE_WB_LABELS]
        labels = [WB_LABEL_TEXT[lab] for lab in KDE_WB_LABELS]
        handles += [Line2D([], [], color=COLORS["ink_muted"], linewidth=1.4),
                    Line2D([], [], color=COLORS["ink_muted"], linewidth=1.4,
                           linestyle=(0, (3, 2)))]
        labels += ["Median", "P95"]
        fig.legend(handles, labels, loc="upper left",
                   bbox_to_anchor=(MARGIN_LEFT / page_w - 0.008, y_inches / page_h),
                   ncol=len(labels), frameon=False, fontsize=8, handlelength=1.8,
                   columnspacing=1.3, labelcolor=COLORS["ink_muted"])

    def build(self):
        """Create the page figures (1 or 2 depending on the sections switched on)."""
        plt.rcParams.update(STYLE)
        page_builders = []
        if self.rows_page_1()[0]:
            page_builders.append(self.build_page_1)
        if self.rows_page_2():
            page_builders.append(self.build_page_2)
        if not page_builders:
            raise ValueError("All sections are switched off.")

        self.figures = [builder(i + 1, len(page_builders))
                        for i, builder in enumerate(page_builders)]
        return self.figures

    def save_pdf(self, path):
        """Write all pages to one PDF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.figures:
            self.build()
        with PdfPages(path) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            info = pdf.infodict()
            info["Title"] = f"{self.report_title} - {self.patient_id}"
            info["Subject"] = self.period_text
        print(f"Saved: {path}")
        return path

    def save_pngs(self, folder, dpi=200):
        """Optional: one PNG per page."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if not self.figures:
            self.build()
        paths = []
        for i, fig in enumerate(self.figures, start=1):
            p = folder / f"{self.patient_id}_report_page{i}.png"
            fig.savefig(p, dpi=dpi)
            paths.append(p)
            print(f"Saved: {p}")
        return paths


# =====================================================================
# MAIN - edit the settings below and run
# =====================================================================
if __name__ == "__main__":

    PATIENT_ID = "PAT401"

    DATA_FOLDER = Path(r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)"
                       r"\MobilityAPP_Pipeline\Prova_dmos_kit")
    DAY_DMOS_PATH = DATA_FOLDER / PATIENT_ID / "outputs_" / f"{PATIENT_ID}_day_dmos.csv"
    WB_LEVEL_PATH = (DATA_FOLDER / PATIENT_ID / "outputs_"
                     / f"{PATIENT_ID}_multiple_dates_wb_parameters_average.csv")
    OUTPUT_FOLDER = DATA_FOLDER / "prova_clinical_report"

    # Period (max 7 days) - choose ONE option
    # DATES = "all"                                   # all available days
    # DATES = ["2023-07-10", "2023-07-14"]          # selected days
    DATES = ("2023-07-10", "2023-07-16")          # from day -> to day (inclusive)

    SECTIONS = {"amount_pattern": True, "variability": True, "kde": True, "pauses": True}
    AMOUNT_PANELS = {"wear_strip": True, "time": True, "wb_counts": True,
                     "strides_distance": True}
    PAUSE_PANELS = {"counts": True, "kde": True}

    SHOW_VALUES = True             # value labels on bars / dots / lines
    WEAR_VALID_HOURS = None        # e.g. 12 -> tick on the wear strip
    VARIABILITY_WB = "long"        # "short", "medium" or "long"
    KDE_MIN_WB = 5                 # min WBs of a category to draw its curve
    PAUSE_KDE_BY_DAY = False       # True = one pause curve per day
    PAUSE_REFERENCE_LINES = False  # True = 1 / 5 / 10 min lines on pause KDE

    SAVE_PNG = False               # also save one PNG per page
    SHOW_PLOTS = True              # open the pages in a window

    plt.close("all")

    report = DMOsReport(
        patient_id=PATIENT_ID,
        day_dmos_path=DAY_DMOS_PATH,
        wb_level_path=WB_LEVEL_PATH,
        dates=DATES,
        sections=SECTIONS,
        amount_panels=AMOUNT_PANELS,
        pause_panels=PAUSE_PANELS,
        show_values=SHOW_VALUES,
        wear_valid_hours=WEAR_VALID_HOURS,
        variability_wb=VARIABILITY_WB,
        kde_min_wb=KDE_MIN_WB,
        pause_kde_by_day=PAUSE_KDE_BY_DAY,
        pause_reference_lines=PAUSE_REFERENCE_LINES,
    )

    report.build()
    report.save_pdf(OUTPUT_FOLDER / f"{PATIENT_ID}_DMOs_report.pdf")

    if SAVE_PNG:
        report.save_pngs(OUTPUT_FOLDER)

    if SHOW_PLOTS:
        plt.show()