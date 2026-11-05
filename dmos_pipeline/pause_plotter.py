# -*- coding: utf-8 -*-
"""
Walking-bout pause plotter.

Plots every walking-bout pause detected by `WBpipeline`, showing the
filtered left/right signal around the pause, signal breaks, and the
(optionally quality-checked) IC / pre_IC gait events.

Originally: old/4_WB_Pauses_Plotter.py (created on Tue May 5 2026,
@author: francesca.boschi). Moved into the package so it can be used
from the example main scripts.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


PathLike = Union[str, Path]


class PausePlotter:
    """
    Plot walking-bout pauses.

    Inputs can be given either as file paths (CSV outputs of the pipeline)
    or directly as in-memory DataFrames. For each input, the DataFrame
    takes priority over the path if both are given.

    Parameters
    ----------
    signal_path / signal_raw
        Raw IMU signal (gaitMAP_bf_all.csv). Columns as a
        (sensor, channel) MultiIndex, e.g. ("left_sensor", "gyr_ml"),
        or flat names such as "left_sensor_gyr_ml".
    pauses_path / pauses_df
        *_wb_pauses_dataframe.csv  (columns: pause_id, start, end, ...)
    events_path / events_df
        *_events.csv  (columns: foot, ic, pre_ic, quality column, ...)
    breaks_path / breaks_df
        *_signal_break_dataframe.csv  (columns: start, end, ...)
    output_folder
        Folder where PNG files are saved (created if needed).
    config
        Optional overrides of DEFAULT_CONFIG.

    Typical use
    -----------
    >>> plotter = PausePlotter(
    ...     signal_raw=signal_raw,
    ...     pauses_path=wb_saved_paths["wb_pauses_dataframe"],
    ...     events_path=saved_paths["events"],
    ...     breaks_path=wb_saved_paths["signal_break_dataframe"],
    ...     output_folder=session_folder / "pauses_plot",
    ... )
    >>> saved = plotter.run()
    """

    DEFAULT_CONFIG = {
        # ---- Signal ----
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",

        # ---- Filtering ----
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,

        # ---- Plot window ----
        "samples_before_pause": 500,
        "samples_after_pause": 500,

        # ---- Event quality filtering ----
        # Same column name used by QualityCheck / WBpipeline in the main scripts
        "use_only_quality_checked_events": True,
        "event_quality_column": "quality_check(IC<0)",

        # ---- Plot behaviour ----
        "show_plot": False,
        "save_plots": True,
        "figsize": (15, 5),
        "dpi": 150,
    }

    def __init__(
        self,
        signal_path: Optional[PathLike] = None,
        pauses_path: Optional[PathLike] = None,
        events_path: Optional[PathLike] = None,
        breaks_path: Optional[PathLike] = None,
        output_folder: Optional[PathLike] = None,
        config: Optional[Dict[str, Any]] = None,
        signal_raw: Optional[pd.DataFrame] = None,
        pauses_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        breaks_df: Optional[pd.DataFrame] = None,
    ):
        self.signal_path = Path(signal_path) if signal_path is not None else None
        self.pauses_path = Path(pauses_path) if pauses_path is not None else None
        self.events_path = Path(events_path) if events_path is not None else None
        self.breaks_path = Path(breaks_path) if breaks_path is not None else None
        self.output_folder = Path(output_folder) if output_folder is not None else None

        # Optional in-memory inputs
        self._signal_input = signal_raw
        self._pauses_input = pauses_df
        self._events_input = events_df
        self._breaks_input = breaks_df

        self.config = self._build_config(config)

        self.fs = self.config["sampling_rate_hz"]
        self.channel = self.config["channel"]
        self.cutoff_freq_gyr = self.config["cutoff_freq_gyr"]
        self.filter_order_gyr = self.config["filter_order_gyr"]
        self.use_only_quality_checked_events = self.config["use_only_quality_checked_events"]
        self.event_quality_column = self.config["event_quality_column"]

        self.samples_before_pause = self.config["samples_before_pause"]
        self.samples_after_pause = self.config["samples_after_pause"]

        self.show_plot = self.config["show_plot"]
        self.save_plots = self.config["save_plots"]
        self.figsize = self.config["figsize"]
        self.dpi = self.config["dpi"]

        if self.save_plots and self.output_folder is None:
            raise ValueError("output_folder is required when save_plots=True.")

        self.signal_raw = None
        self.pauses_df = None
        self.events_df = None
        self.breaks_df = None

        self.left_filtered = None
        self.right_filtered = None

        self.saved_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @classmethod
    def _build_config(cls, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        user_config = user_config or {}

        unknown_keys = set(user_config) - set(cls.DEFAULT_CONFIG)
        if unknown_keys:
            raise ValueError(f"Unknown configuration key(s): {sorted(unknown_keys)}")

        config = cls.DEFAULT_CONFIG.copy()
        config.update(user_config)

        cls._validate_config(config)

        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        if config["sampling_rate_hz"] <= 0:
            raise ValueError("sampling_rate_hz must be greater than 0.")

        if config["cutoff_freq_gyr"] <= 0:
            raise ValueError("cutoff_freq_gyr must be greater than 0.")

        if config["filter_order_gyr"] <= 0:
            raise ValueError("filter_order_gyr must be greater than 0.")

        if config["samples_before_pause"] < 0:
            raise ValueError("samples_before_pause must be >= 0.")

        if config["samples_after_pause"] < 0:
            raise ValueError("samples_after_pause must be >= 0.")

        if config["dpi"] <= 0:
            raise ValueError("dpi must be greater than 0.")

        for key in ("save_plots", "show_plot", "use_only_quality_checked_events"):
            if not isinstance(config[key], bool):
                raise ValueError(f"{key} must be True or False.")

        if not isinstance(config["event_quality_column"], str):
            raise ValueError("event_quality_column must be a string.")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            keep_cols = [
                col for col in df.columns
                if "unnamed" not in " ".join(str(x) for x in col).lower()
            ]
            return df.loc[:, keep_cols]

        unnamed_cols = [
            col for col in df.columns
            if str(col).lower().startswith("unnamed")
        ]

        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        return df

    def load_signal(self) -> pd.DataFrame:
        if self._signal_input is not None:
            df = self.drop_unnamed_columns(self._signal_input.copy())
            return df.reset_index(drop=True)

        if self.signal_path is None:
            raise ValueError("Provide either signal_path or signal_raw.")

        if not self.signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {self.signal_path}")

        try:
            df = pd.read_csv(self.signal_path, header=[0, 1])

            if isinstance(df.columns, pd.MultiIndex):
                cols_text = [
                    " ".join(str(x) for x in col).lower()
                    for col in df.columns
                ]

                looks_like_multiindex = (
                    any("left_sensor" in c for c in cols_text)
                    or any("right_sensor" in c for c in cols_text)
                )

                if looks_like_multiindex:
                    df = self.drop_unnamed_columns(df)
                    return df.reset_index(drop=True)

        except Exception:
            pass

        df = pd.read_csv(self.signal_path)
        df = self.drop_unnamed_columns(df)

        return df.reset_index(drop=True)

    def load_table(
        self,
        df_input: Optional[pd.DataFrame],
        path: Optional[Path],
        name: str,
    ) -> pd.DataFrame:
        if df_input is not None:
            return self.drop_unnamed_columns(df_input.copy())

        if path is None:
            raise ValueError(f"Provide either {name}_path or {name}_df.")

        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

        df = pd.read_csv(path)
        return self.drop_unnamed_columns(df)

    def apply_event_quality_filter(self):
        """
        Optionally keep only events that passed the event-level quality check.
        """
        if self.events_df is None:
            raise RuntimeError("events_df is None. Run load_inputs() first.")

        if not self.use_only_quality_checked_events:
            return self

        if self.event_quality_column not in self.events_df.columns:
            quality_like = [
                c for c in self.events_df.columns
                if str(c).lower().startswith("quality_check")
            ]
            raise ValueError(
                f"Quality column '{self.event_quality_column}' not found in events. "
                f"Quality columns available: {quality_like}. "
                "Set 'event_quality_column' accordingly, run QualityCheck first, "
                "or set use_only_quality_checked_events=False."
            )

        quality_values = self.events_df[self.event_quality_column]

        if quality_values.dtype == bool:
            keep_mask = quality_values
        else:
            keep_mask = (
                quality_values
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

        self.events_df = self.events_df[keep_mask].copy()

        return self

    def load_inputs(self):
        self.signal_raw = self.load_signal()
        self.pauses_df = self.load_table(self._pauses_input, self.pauses_path, "pauses")
        self.events_df = self.load_table(self._events_input, self.events_path, "events")
        self.breaks_df = self.load_table(self._breaks_input, self.breaks_path, "breaks")

        self.apply_event_quality_filter()

        # Prepare events once (numeric samples, lowercase foot)
        self.events_df["foot"] = self.events_df["foot"].astype(str).str.lower()
        for col in ("ic", "pre_ic"):
            if col in self.events_df.columns:
                self.events_df[col] = pd.to_numeric(self.events_df[col], errors="coerce")

        for df in (self.pauses_df, self.breaks_df):
            for col in ("start", "end"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        return self

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------
    def get_column(self, side: str, channel: str) -> pd.Series:
        if self.signal_raw is None:
            raise RuntimeError("signal_raw is None. Run load_inputs() first.")

        df = self.signal_raw

        if isinstance(df.columns, pd.MultiIndex):
            col = (side, channel)

            if col in df.columns:
                return df[col]

            raise KeyError(
                f"Column {col} not found. "
                f"Available columns example: {list(df.columns)[:10]}"
            )

        possible_names = [
            f"{side}_{channel}",
            f"{side}.{channel}",
            f"{side} {channel}",
            f"{side}/{channel}",
            f"{channel}_{side}",
            f"{channel}.{side}",
            f"{channel} {side}",
        ]

        for name in possible_names:
            if name in df.columns:
                return df[name]

        raise KeyError(
            f"Could not find column for side='{side}', channel='{channel}'. "
            f"Available columns example: {list(df.columns)[:20]}"
        )

    @staticmethod
    def butter_lowpass_filter(
        data,
        cutoff_freq: float,
        filter_order: int,
        fs: float,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=float)

        if np.all(np.isnan(data)):
            return data

        valid = np.isfinite(data)

        if valid.sum() < filter_order * 3:
            return data

        data_filled = (
            pd.Series(data)
            .interpolate(limit_direction="both")
            .to_numpy()
        )

        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist

        if normal_cutoff >= 1:
            raise ValueError(
                f"cutoff_freq={cutoff_freq} must be smaller than "
                f"Nyquist frequency={nyquist}"
            )

        b, a = butter(filter_order, normal_cutoff, btype="low", analog=False)

        filtered = filtfilt(b, a, data_filled)
        filtered[~valid] = np.nan

        return filtered

    def filter_signal(self):
        if self.signal_raw is None:
            raise RuntimeError("signal_raw is None. Run load_inputs() first.")

        for side, attr in (("left_sensor", "left_filtered"), ("right_sensor", "right_filtered")):
            setattr(
                self,
                attr,
                self.butter_lowpass_filter(
                    data=self.get_column(side, self.channel),
                    cutoff_freq=self.cutoff_freq_gyr,
                    filter_order=self.filter_order_gyr,
                    fs=self.fs,
                ),
            )

        return self

    # Backwards-compatible name used in the old script
    filter_gyr_ml = filter_signal

    def _signal_for_foot(self, foot: str) -> Optional[np.ndarray]:
        if foot == "left":
            return self.left_filtered
        if foot == "right":
            return self.right_filtered
        return None

    @staticmethod
    def get_event_color(foot: str) -> str:
        return {"left": "blue", "right": "orange"}.get(str(foot).lower(), "black")

    @staticmethod
    def _safe_text(value) -> str:
        return (
            str(value)
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _check_ready_for_plotting(self) -> None:
        if self.pauses_df is None or self.events_df is None or self.breaks_df is None:
            raise RuntimeError("Inputs missing. Run load_inputs() first.")

        if self.left_filtered is None or self.right_filtered is None:
            raise RuntimeError("Filtered signals missing. Run filter_signal() first.")

        for df, cols, name in (
            (self.pauses_df, ["pause_id", "start", "end"], "pauses_df"),
            (self.events_df, ["foot", "ic", "pre_ic"], "events_df"),
            (self.breaks_df, ["start", "end"], "breaks_df"),
        ):
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {name}: {missing}")

    @staticmethod
    def _meta(row: pd.Series, col: str, default: str) -> str:
        if col in row and pd.notna(row[col]):
            return str(row[col])
        return default

    def _scatter_events(self, ax, plot_start: int, plot_end: int) -> None:
        """Plot IC (o) and pre_IC (x) markers that fall inside the window."""
        for event_col, marker, size, zorder, label_suffix in (
            ("ic", "o", 35, 5, "IC"),
            ("pre_ic", "x", 55, 6, "pre_IC"),
        ):
            samples_all = self.events_df[event_col]
            in_window = self.events_df[
                samples_all.notna()
                & (samples_all >= plot_start)
                & (samples_all < plot_end)
            ]

            for foot in ("left", "right"):
                sig = self._signal_for_foot(foot)
                foot_events = in_window[in_window["foot"] == foot]

                if foot_events.empty or sig is None:
                    continue

                idx = foot_events[event_col].astype(int).to_numpy()
                idx = idx[(idx >= 0) & (idx < len(sig))]

                if idx.size == 0:
                    continue

                ax.scatter(
                    idx,
                    sig[idx],
                    color=self.get_event_color(foot),
                    marker=marker,
                    s=size,
                    zorder=zorder,
                    label=f"{foot} {label_suffix}",
                )

    def plot_single_pause(self, pause_row: pd.Series) -> Optional[Path]:
        pause_id = int(pause_row["pause_id"])
        pause_start = int(pause_row["start"])
        pause_end = int(pause_row["end"])

        patient_id = self._meta(pause_row, "patient_id", "unknown_patient")
        recording_date = self._meta(pause_row, "recording_date", "unknown_date")
        session_id = self._meta(pause_row, "session_id", "unknown_session")

        n_samples = len(self.left_filtered)

        plot_start = max(0, pause_start - self.samples_before_pause)
        plot_end = min(n_samples, pause_end + self.samples_after_pause)

        samples = np.arange(plot_start, plot_end)

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            samples,
            self.left_filtered[plot_start:plot_end],
            label=f"left_sensor {self.channel} filtered",
            linewidth=1,
            color="blue",
        )
        ax.plot(
            samples,
            self.right_filtered[plot_start:plot_end],
            label=f"right_sensor {self.channel} filtered",
            linewidth=1,
            color="orange",
        )

        ax.axvspan(pause_start, pause_end, color="red", alpha=0.18, label="pause")

        breaks_in_range = self.breaks_df[
            (self.breaks_df["end"] >= plot_start)
            & (self.breaks_df["start"] <= plot_end)
        ]

        first_break_label = True
        for _, break_row in breaks_in_range.iterrows():
            ax.axvspan(
                int(break_row["start"]),
                int(break_row["end"]),
                color="lightblue",
                alpha=0.35,
                label="signal break" if first_break_label else None,
            )
            first_break_label = False

        self._scatter_events(ax, plot_start, plot_end)

        ax.set_title(
            f"{patient_id} | {recording_date} | {session_id} | "
            f"pause {pause_id} | samples {plot_start}-{plot_end}"
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel(self.channel)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

        fig.tight_layout()

        save_path = None

        if self.save_plots:
            self.output_folder.mkdir(parents=True, exist_ok=True)

            save_path = (
                self.output_folder
                / f"{patient_id}_{self._safe_text(recording_date)}_{session_id}"
                  f"_pause_{pause_id:03d}.png"
            )

            fig.savefig(save_path, dpi=self.dpi)
            self.saved_paths.append(save_path)

        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)

        return save_path

    def plot_all_pauses(self) -> List[Path]:
        self._check_ready_for_plotting()

        self.saved_paths = []

        pauses = self.pauses_df.dropna(subset=["pause_id", "start", "end"])

        if pauses.empty:
            print("[PausePlotter] No pauses to plot.")
            return self.saved_paths

        for _, pause_row in pauses.iterrows():
            self.plot_single_pause(pause_row)

        return self.saved_paths

    def run(self) -> List[Path]:
        self.load_inputs()
        self.filter_signal()
        return self.plot_all_pauses()
