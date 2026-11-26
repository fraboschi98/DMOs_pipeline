# -*- coding: utf-8 -*-
"""
DMOs report - Capacity vs Performance (single day), one PDF of 2 A4 pages.

Compares gait in daily life (walking bouts = "performance") with a
standardised test typed in by the user ("capacity").

    Page 1 - Walking bouts across the day : every WB of the day, one row per
                                            gait parameter, test value as a
                                            reference line + band, rolling median
    Page 2 - Test reached in daily life   : % of short / medium / long WBs
                                            that reach the test value, per
                                            gait parameter
           - Gait velocity vs bout duration

"Test reached": the WB value is within the band around the test value or
better (higher velocity / stride length / cadence, shorter stride time).

Usage
    from dmos_pipeline import CapacityPerformanceReport

    report = CapacityPerformanceReport(patient_id, day, wb_level_path,
                                       test_name, test_values)
    report.build()
    report.save_pdf("PAT401_2023-07-11_capacity_vs_performance.pdf")

    See examples/main_clinical_report_single_day.py for a full example.

Input: WB-level CSV (one row per walking bout), same file as the multi-day
report (WalkingBouts_analyzer.save_wb_parameters_average).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =====================================================================
# STYLE (same palette as the DMOs report)
# =====================================================================
COLORS = {
    "ink": "#0F172A",
    "ink_muted": "#64748B",
    "grid": "#E2E8F0",
    "surface": "#FFFFFF",

    # WB categories (blue ramp, light -> dark)
    "very_short": "#D3E3F7",
    "short": "#7FAEE3",
    "medium": "#2F67B1",
    "long": "#0E2A57",

    # test / capacity reference
    "test": "#B45309",          # deep amber line
    "test_band": "#FBE3A1",     # light warm yellow band
    "trend": "#0F172A",         # rolling median
    "not_enough": "#CBD5E1",    # categories with too few WBs
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
    "pdf.fonttype": 42,
}


# =====================================================================
# CONSTANTS
# =====================================================================
WB_LABEL_TEXT = {
    "very_short": "Very short < 10 s",
    "short": "Short 10-30 s",
    "medium": "Medium 30-60 s",
    "long": "Long > 60 s",
}
WB_ORDER = ["very_short", "short", "medium", "long"]

PARAMS = [
    # (column, name, unit, decimals, lower_is_better)
    ("gait velocity [m/s]",     "Gait velocity", "m/s",       2, False),
    ("stride length [m]",       "Stride length", "m",         2, False),
    ("cadence_[steps_per_min]", "Cadence",       "steps/min", 0, False),
    ("stride time [s]",         "Stride time",   "s",         2, True),
]
SPEED_COLUMN = "gait velocity [m/s]"

# A4 page and margins (inches)
PAGE_SIZE = (8.27, 11.69)
MARGIN_LEFT = 0.85
MARGIN_RIGHT = 0.85
MARGIN_TOP = 0.45
MARGIN_BOTTOM = 0.55
COLUMN_GAP = 0.35


# =====================================================================
# FORMATTERS
# =====================================================================
def format_percent(value, _pos=None) -> str:
    return f"{value:.0f} %"


def format_hours(value, _pos=None) -> str:
    return f"{value:g} h"


def format_duration_tick(seconds) -> str:
    return f"{seconds:g} s" if seconds < 60 else f"{seconds / 60:g} min"


# =====================================================================
# REPORT CLASS
# =====================================================================
class CapacityPerformanceReport:
    """
    One patient, one day.

    test_values : {column: value}; a value can be None if that parameter
                  was not measured in the test (its comparison is skipped).
    """

    def __init__(self, patient_id, day, wb_level_path, test_name, test_values,
                 test_band_pct=5, time_column="start_time", recording_start=None,
                 sampling_rate_hz=102.4, include_very_short=False,
                 rolling_window=5, max_gap_min=60, reached_min_wb=3,
                 sections=None, show_values=True,
                 report_title="Capacity vs Performance"):
        self.patient_id = str(patient_id)
        self.day = pd.to_datetime(day)
        self.wb_level_path = Path(wb_level_path)
        self.test_name = test_name
        self.test_values = {col: test_values.get(col) for col, *_ in PARAMS}
        self.band_pct = test_band_pct or 0
        self.time_column = time_column
        self.recording_start = recording_start
        self.fs = sampling_rate_hz
        self.include_very_short = include_very_short
        self.rolling_window = rolling_window
        self.max_gap_min = max_gap_min
        self.reached_min_wb = reached_min_wb
        self.show_values = show_values
        self.report_title = report_title

        self.sections = {"bout_sequence": True, "test_reached": True,
                         "speed_vs_duration": True}
        self.sections.update(sections or {})

        if all(v is None for v in self.test_values.values()):
            raise ValueError("No test value given: fill in TEST_VALUES.")

        self.wb = self.load_day_wbs()
        self.x_mode = self.add_time_column()
        self.categories = [lab for lab in WB_ORDER if lab in set(self.wb["WB_label"])
                           and (self.include_very_short or lab != "very_short")]
        self.reached = self.compute_test_reached()
        self.figures = []

        print(f"{self.patient_id} {self.day:%Y-%m-%d}: {len(self.wb)} WBs, "
              f"x axis = {self.x_mode}")

    # -----------------------------------------------------------------
    # TEST VALUES FROM THE CONSOLE (optional)
    # -----------------------------------------------------------------
    @staticmethod
    def ask_test_values(defaults=None):
        """Ask each test value in the console; Enter keeps the default, '-' = not measured."""
        defaults = defaults or {}
        values = {}
        print("Test values (Enter = keep default, '-' = not measured)")
        for col, name, unit, _, _ in PARAMS:
            default = defaults.get(col)
            answer = input(f"  {name} [{unit}] ({default}): ").strip().replace(",", ".")
            if answer == "":
                values[col] = default
            elif answer == "-":
                values[col] = None
            else:
                values[col] = float(answer)
        return values

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------
    def load_day_wbs(self) -> pd.DataFrame:
        """WBs of the patient on the selected day, in time order."""
        wb = pd.read_csv(self.wb_level_path)
        wb = wb[wb["patient_id"].astype(str) == self.patient_id].copy()
        wb["recording_date"] = pd.to_datetime(wb["recording_date"])
        wb = wb[wb["recording_date"] == self.day]
        if wb.empty:
            raise ValueError(f"No walking bouts for {self.patient_id} on {self.day:%Y-%m-%d}.")

        if not self.include_very_short:
            wb = wb[wb["WB_label"] != "very_short"]
            if wb.empty:
                raise ValueError("Only very short WBs on this day; nothing to compare.")

        if "duration_s" not in wb.columns:
            wb["duration_s"] = (wb["end"] - wb["start"]) / self.fs

        sort_cols = [c for c in ("session_id", "start") if c in wb.columns]
        wb = wb.sort_values(sort_cols).reset_index(drop=True)
        wb["wb_order"] = np.arange(1, len(wb) + 1)
        return wb

    def add_time_column(self) -> str:
        """
        x axis of page 1, first option available:
          "time"    : clock time - start-time column, or recording_start + start / fs
          "elapsed" : hours since the recording started (start / fs)
          "order"   : WB order in the day
        """
        wb = self.wb
        if self.time_column and self.time_column in wb.columns and wb[self.time_column].notna().any():
            t = pd.to_datetime(wb[self.time_column], errors="coerce")
            if t.notna().any() and (t.dt.year == 1900).all():     # time only ("14:32:10")
                t = self.day + (t - t.dt.normalize())
            wb["x_time"] = t
            return "time"

        if self.recording_start:
            t0 = pd.to_datetime(f"{self.day.date()} {self.recording_start}")
            wb["x_time"] = t0 + pd.to_timedelta(wb["start"] / self.fs, unit="s")
            return "time"

        # no clock time: hours since the recording started (start is in samples).
        # Only with one session that day - sessions restart at sample 0.
        n_sessions = wb["session_id"].nunique() if "session_id" in wb.columns else 1
        if "start" in wb.columns and n_sessions == 1:
            wb["x_elapsed_h"] = wb["start"] / self.fs / 3600
            return "elapsed"

        if n_sessions > 1:
            print(f"{n_sessions} sessions on this day and no clock time: x axis = WB order")
        return "order"

    def reached_mask(self, values, test, lower_is_better):
        """True where the WB reaches the test (within the band or better)."""
        if lower_is_better:
            return values <= test * (1 + self.band_pct / 100)
        return values >= test * (1 - self.band_pct / 100)

    def compute_test_reached(self) -> pd.DataFrame:
        """One row per parameter and WB category: n WBs, n reached, % reached."""
        rows = []
        for col, name, _, _, lower_better in PARAMS:
            test = self.test_values[col]
            for lab in self.categories:
                values = pd.to_numeric(self.wb.loc[self.wb["WB_label"] == lab, col],
                                       errors="coerce").dropna()
                n = int(len(values))
                if test is None or n == 0:
                    n_reached, pct = None, np.nan
                else:
                    n_reached = int(self.reached_mask(values, test, lower_better).sum())
                    pct = 100 * n_reached / n
                rows.append({"parameter": name, "column": col, "WB_label": lab,
                             "n_wb": n, "n_reached": n_reached, "pct_reached": pct,
                             "test_value": test})
        return pd.DataFrame(rows)

    def x_values(self):
        if self.x_mode == "time":
            return self.wb["x_time"]
        if self.x_mode == "elapsed":
            return self.wb["x_elapsed_h"]
        return self.wb["wb_order"]

    def trend_segments(self):
        """Index ranges not separated by more than max_gap_min (time mode only)."""
        idx = np.arange(len(self.wb))
        if self.x_mode == "order" or not self.max_gap_min:
            return [idx]
        if self.x_mode == "time":
            gaps = self.wb["x_time"].diff().dt.total_seconds().fillna(0).to_numpy() / 60
        else:
            gaps = self.wb["x_elapsed_h"].diff().fillna(0).to_numpy() * 60
        breaks = np.where(gaps > self.max_gap_min)[0]
        return [seg for seg in np.split(idx, breaks) if len(seg)]

    @staticmethod
    def marker_sizes(duration_s, s_min=12, s_max=110):
        """Dot area grows with WB duration (sqrt scaling, capped at 10 min)."""
        d = np.sqrt(np.clip(duration_s, 1, 600))
        d = (d - 1) / (np.sqrt(600) - 1)
        return s_min + d * (s_max - s_min)

    @property
    def wb_text(self):
        return "all WBs" if self.include_very_short else "WBs ≥ 10 s"

    @property
    def test_text(self):
        parts = []
        for col, name, unit, dec, _ in PARAMS:
            v = self.test_values[col]
            parts.append(f"{name.lower()} {v:.{dec}f} {unit}" if v is not None
                         else f"{name.lower()} –")
        band = f"  ·  band ± {self.band_pct:g} %" if self.band_pct else ""
        return f"Test: {self.test_name}  ·  " + "  ·  ".join(parts) + band

    # -----------------------------------------------------------------
    # DRAWING HELPERS
    # -----------------------------------------------------------------
    def draw_test_reference(self, ax, test):
        if test is None:
            return
        if self.band_pct:
            ax.axhspan(test * (1 - self.band_pct / 100), test * (1 + self.band_pct / 100),
                       color=COLORS["test_band"], zorder=1, lw=0)
        ax.axhline(test, color=COLORS["test"], linewidth=1.5, zorder=2)

    def wb_legend_handles(self):
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=6.5,
                          markerfacecolor=COLORS[lab], markeredgecolor=COLORS["surface"])
                   for lab in self.categories]
        return handles, [WB_LABEL_TEXT[lab] for lab in self.categories]

    @staticmethod
    def panel_header(ax, title, handles=None, labels=None, ncol=4):
        """Title left + legend right; with no title the legend starts on the left."""
        if title:
            ax.set_title(title, pad=8)
        if handles:
            loc, anchor = ("lower right", (1.0, 1.0)) if title else ("lower left", (-0.01, 1.0))
            ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor,
                      ncol=ncol, handletextpad=0.4, columnspacing=1.0,
                      borderaxespad=0.15, labelcolor=COLORS["ink_muted"])

    # -----------------------------------------------------------------
    # PANEL - ONE GAIT PARAMETER ACROSS THE DAY
    # -----------------------------------------------------------------
    def plot_bout_row(self, ax, column, name, unit, decimals, show_xlabels):
        x = self.x_values()
        y = pd.to_numeric(self.wb[column], errors="coerce")
        test = self.test_values[column]

        self.draw_test_reference(ax, test)
        ax.scatter(x, y, s=self.marker_sizes(self.wb["duration_s"].to_numpy()),
                   c=[COLORS[lab] for lab in self.wb["WB_label"]],
                   edgecolor=COLORS["surface"], linewidth=0.7, zorder=3)

        if self.rolling_window and y.notna().sum() >= self.rolling_window:
            trend = y.rolling(self.rolling_window, center=True, min_periods=2).median()
            for seg in self.trend_segments():
                ax.plot(x.iloc[seg], trend.iloc[seg], color=COLORS["trend"],
                        linewidth=1.3, zorder=4)

        candidates = [y.min(), y.max()]
        if test is not None:
            candidates += [test * (1 - self.band_pct / 100), test * (1 + self.band_pct / 100)]
        lo, hi = np.nanmin(candidates), np.nanmax(candidates)
        pad = 0.1 * (hi - lo if hi > lo else abs(hi) or 1)
        ax.set_ylim(lo - pad, hi + pad)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(name, pad=4)

        if self.show_values and test is not None:
            ax.annotate(f"test\n{test:.{decimals}f}", xy=(1, test),
                        xycoords=("axes fraction", "data"), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7.5, fontweight="bold", color=COLORS["test"])

        if self.x_mode == "time":
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.set_xlim(x.min() - pd.Timedelta(minutes=20), x.max() + pd.Timedelta(minutes=20))
        elif self.x_mode == "elapsed":
            x0, x1 = float(x.min()), float(x.max())
            span = max(x1 - x0, 0.5)
            step = 0.5 if span <= 3 else (1 if span <= 8 else 2)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_hours))
            ax.set_xlim(x0 - 0.04 * span, x1 + 0.04 * span)
        else:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
            ax.set_xlim(0, len(x) + 1)
        if not show_xlabels:
            ax.tick_params(axis="x", labelbottom=False)
        elif self.x_mode == "time":
            ax.set_xlabel("Time of day")
        elif self.x_mode == "elapsed":
            ax.set_xlabel("Time since start of recording")
        else:
            ax.set_xlabel("Walking bout (order in the day)")

    def draw_bout_legend(self, fig, y_inches):
        page_w, page_h = fig.get_size_inches()
        handles, labels = self.wb_legend_handles()
        handles += [Line2D([], [], color=COLORS["trend"], linewidth=1.3),
                    Line2D([], [], color=COLORS["test"], linewidth=1.5)]
        labels += [f"Rolling median ({self.rolling_window} WBs)",
                   f"Test ± {self.band_pct:g} %" if self.band_pct else "Test"]
        fig.legend(handles, labels, loc="upper left",
                   bbox_to_anchor=(MARGIN_LEFT / page_w - 0.008, y_inches / page_h),
                   ncol=len(labels), frameon=False, fontsize=8, handletextpad=0.3,
                   columnspacing=1.1, labelcolor=COLORS["ink_muted"])
        fig.text(MARGIN_LEFT / page_w, (y_inches - 0.27) / page_h,
                 "dot size = walking-bout duration", ha="left", va="top",
                 fontsize=7.5, style="italic", color=COLORS["ink_muted"])

    # -----------------------------------------------------------------
    # PANEL - TEST REACHED PER WB CATEGORY
    # -----------------------------------------------------------------
    def plot_test_reached(self, ax, group_width=0.78):
        """Grouped bars: % of WBs of each category that reach the test."""
        n_cat = len(self.categories)
        bar_w = group_width / max(n_cat, 1)
        ax.set_ylim(0, 110)

        for gi, (col, name, _, _, _) in enumerate(PARAMS):
            sub = self.reached[self.reached["column"] == col].set_index("WB_label")
            if self.test_values[col] is None:
                ax.text(gi, 50, "no test\nvalue", ha="center", va="center",
                        fontsize=8, style="italic", color=COLORS["ink_muted"])
                continue
            for ci, lab in enumerate(self.categories):
                xb = gi - group_width / 2 + bar_w * (ci + 0.5)
                n = int(sub.loc[lab, "n_wb"])
                if n < self.reached_min_wb:
                    ax.bar(xb, 100, width=bar_w * 0.88, color="none",
                           edgecolor=COLORS["not_enough"], linewidth=1, hatch="////",
                           zorder=2)
                    ax.text(xb, 102, f"n={n}", ha="center", va="bottom", fontsize=7,
                            style="italic", color=COLORS["ink_muted"])
                    continue
                pct = float(sub.loc[lab, "pct_reached"])
                k = int(sub.loc[lab, "n_reached"])
                ax.bar(xb, pct, width=bar_w * 0.88, color=COLORS[lab], zorder=3)
                if self.show_values:
                    ax.text(xb, pct + 2, f"{pct:.0f} %", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold", color=COLORS["ink"])
                    ax.text(xb, -4, f"{k}/{n}", ha="center", va="top", fontsize=7,
                            color=COLORS["ink_muted"])

        ax.set_xticks(range(len(PARAMS)))
        ax.set_xticklabels([p[1] for p in PARAMS], fontsize=9, color=COLORS["ink"])
        ax.tick_params(axis="x", pad=16 if self.show_values else 4)
        ax.set_xlim(-0.6, len(PARAMS) - 0.4)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_percent))
        ax.set_ylabel("WBs reaching the test")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

        handles = [Patch(color=COLORS[lab]) for lab in self.categories]
        labels = [WB_LABEL_TEXT[lab] for lab in self.categories]
        handles.append(Patch(facecolor="none", edgecolor=COLORS["not_enough"], hatch="////"))
        labels.append(f"< {self.reached_min_wb} WBs")
        self.panel_header(ax, "", handles,
                          labels, ncol=len(labels))

    # -----------------------------------------------------------------
    # PANEL - SPEED vs WB DURATION
    # -----------------------------------------------------------------
    def plot_speed_vs_duration(self, ax, n_bins=6):
        test = self.test_values[SPEED_COLUMN]
        dur = self.wb["duration_s"].to_numpy(dtype=float)
        spd = pd.to_numeric(self.wb[SPEED_COLUMN], errors="coerce").to_numpy()
        ok = np.isfinite(dur) & np.isfinite(spd) & (dur > 0)

        self.draw_test_reference(ax, test)
        ax.scatter(dur[ok], spd[ok], s=34,
                   c=[COLORS[lab] for lab, keep in zip(self.wb["WB_label"], ok) if keep],
                   edgecolor=COLORS["surface"], linewidth=0.7, zorder=3)
        ax.set_xscale("log")

        if ok.sum() >= 6:
            edges = np.logspace(np.log10(dur[ok].min()), np.log10(dur[ok].max()), n_bins + 1)
            mids, meds = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = ok & (dur >= lo) & (dur <= hi)
                if m.sum() >= 3:
                    mids.append(np.sqrt(lo * hi))
                    meds.append(np.median(spd[m]))
            if len(mids) >= 2:
                ax.plot(mids, meds, color=COLORS["trend"], linewidth=1.5, marker="o",
                        markersize=4, zorder=4)

        candidates = [np.nanmin(spd[ok]), np.nanmax(spd[ok])]
        if test is not None:
            candidates += [test * (1 - self.band_pct / 100), test * (1 + self.band_pct / 100)]
        lo, hi = min(candidates), max(candidates)
        pad = 0.1 * (hi - lo if hi > lo else 1)
        ax.set_ylim(lo - pad, hi + pad)

        if test is not None:
            ax_r = ax.twinx()
            ax_r.set_ylim(np.array(ax.get_ylim()) / test * 100)
            ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(format_percent))
            ax_r.tick_params(axis="y", colors=COLORS["test"])
            ax_r.set_ylabel("% of test", color=COLORS["test"])
            for spine in ax_r.spines.values():
                spine.set_visible(False)

        ticks = [t for t in (5, 10, 15, 30, 60, 120, 300, 600, 1800)
                 if dur[ok].min() * 0.8 <= t <= dur[ok].max() * 1.2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_duration_tick(t) for t in ticks])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel("Walking-bout duration (log scale)")
        ax.set_ylabel("Gait velocity (m/s)")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

        handles, labels = self.wb_legend_handles()
        handles.append(Line2D([], [], color=COLORS["trend"], linewidth=1.5, marker="o",
                              markersize=4))
        labels.append("Median by duration")
        if test is not None:
            handles.append(Line2D([], [], color=COLORS["test"], linewidth=1.5))
            labels.append(f"Test {test:.2f} m/s")
        self.panel_header(ax, "", handles, labels,
                          ncol=len(labels))

    # -----------------------------------------------------------------
    # PAGE LAYOUT (same row system as the DMOs report)
    #   ("space", inches, [anchor]) / ("header", inches, title, subtitle)
    #   ("flex", weight, [keys])    / ("fixed", inches, [keys])
    # -----------------------------------------------------------------
    def layout_page(self, fig, rows):
        page_w, page_h = fig.get_size_inches()
        used = sum(r[1] for r in rows if r[0] != "flex")
        weights = sum(r[1] for r in rows if r[0] == "flex")
        unit = (page_h - MARGIN_TOP - MARGIN_BOTTOM - used) / weights if weights else 0
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
        fig.text(x0, y - 0.3 / page_h,
                 f"Patient {self.patient_id}  ·  {self.day:%a %d %b %Y}  ·  "
                 f"{len(self.wb)} {self.wb_text}", ha="left", va="top",
                 fontsize=9.5, color=COLORS["ink_muted"])
        fig.text(x0, y - 0.52 / page_h, self.test_text, ha="left", va="top",
                 fontsize=7.8, fontweight="bold", color=COLORS["test"])
        fig.text(x1, y, f"Page {page} of {n_pages}", ha="right", va="top",
                 fontsize=8.5, color=COLORS["ink_muted"])
        y_rule = y - 0.8 / page_h
        fig.add_artist(Line2D([x0, x1], [y_rule, y_rule], transform=fig.transFigure,
                              color=COLORS["grid"], linewidth=1))
        fig.text(x0, 0.3 / page_h, f"Generated {date.today():%d %b %Y}  ·  "
                 "DMOs pipeline (Mobilise-D)", ha="left", va="bottom",
                 fontsize=7.5, color=COLORS["ink_muted"])

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
        if not self.sections["bout_sequence"]:
            return []
        rows = [("header", 0.5, "Walking bouts across the day",
                 "Every walking bout of the day compared with the test"),
                ("space", 0.55, "bout_legend")]
        for i in range(len(PARAMS)):
            rows += [("space", 0.3), ("flex", 1.0, [f"param_{i}"])]
        rows.append(("space", 0.5))
        return rows

    def rows_page_2(self):
        rows = []
        if self.sections["test_reached"]:
            rows += [("header", 0.5, "Test reached in daily life",
                      "Share of short / medium / long walking bouts within the test band "
                      "or better (stride time: shorter)"),
                     ("space", 0.32), ("flex", 0.85, ["test_reached"]), ("space", 0.65)]
        if self.sections["speed_vs_duration"]:
            rows += [("header", 0.5, "Gait velocity vs bout duration",
                      "How long must a bout be before the patient walks as fast as in the test?"),
                     ("space", 0.32), ("flex", 1.0, ["speed"]), ("space", 0.5)]
        return rows

    # -----------------------------------------------------------------
    # PAGES
    # -----------------------------------------------------------------
    def build_page_1(self, page, n_pages):
        fig = plt.figure(figsize=PAGE_SIZE)
        self.draw_page_header(fig, page, n_pages)
        axes, anchors = self.layout_page(fig, [("space", 1.05)] + self.rows_page_1())
        for i, (col, name, unit, dec, _) in enumerate(PARAMS):
            self.plot_bout_row(axes[f"param_{i}"], col, name, unit, dec,
                               show_xlabels=(i == len(PARAMS) - 1))
        self.draw_bout_legend(fig, anchors["bout_legend"])
        return fig

    def build_page_2(self, page, n_pages):
        fig = plt.figure(figsize=PAGE_SIZE)
        self.draw_page_header(fig, page, n_pages)
        axes, _ = self.layout_page(fig, [("space", 1.05)] + self.rows_page_2())
        if "test_reached" in axes:
            self.plot_test_reached(axes["test_reached"])
        if "speed" in axes:
            self.plot_speed_vs_duration(axes["speed"])
        return fig

    def build(self):
        plt.rcParams.update(STYLE)
        page_builders = []
        if self.rows_page_1():
            page_builders.append(self.build_page_1)
        if self.rows_page_2():
            page_builders.append(self.build_page_2)
        if not page_builders:
            raise ValueError("All sections are switched off.")
        self.figures = [builder(i + 1, len(page_builders))
                        for i, builder in enumerate(page_builders)]
        return self.figures

    def save_pdf(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.figures:
            self.build()
        with PdfPages(path) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            info = pdf.infodict()
            info["Title"] = f"{self.report_title} - {self.patient_id} - {self.day:%Y-%m-%d}"
            info["Subject"] = self.test_text
        print(f"Saved: {path}")
        return path

    def save_pngs(self, folder, dpi=200):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if not self.figures:
            self.build()
        paths = []
        for i, fig in enumerate(self.figures, start=1):
            p = folder / f"{self.patient_id}_{self.day:%Y-%m-%d}_capacity_page{i}.png"
            fig.savefig(p, dpi=dpi)
            paths.append(p)
            print(f"Saved: {p}")
        return paths

    def save_reached_table(self, path):
        """Test-reached counts as CSV (one row per parameter and WB category)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.reached.to_csv(path, index=False)
        print(f"Saved: {path}")
        return path