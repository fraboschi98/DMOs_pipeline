# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:12:36 2026

@author: francesca.boschi
"""



from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


class PausePlotter:
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
        "use_only_quality_checked_events": True,
        "event_quality_column": "quality_check(IC>0)",

        # ---- Plot behaviour ----
        "show_plot": False,
        "save_plots": True,
        "figsize": (15, 5),
        "dpi": 150,
    }

    def __init__(
        self,
        signal_path: str,
        pauses_path: str,
        events_path: str,
        breaks_path: str,
        output_folder: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.signal_path = Path(signal_path)
        self.pauses_path = Path(pauses_path)
        self.events_path = Path(events_path)
        self.breaks_path = Path(breaks_path)
        self.output_folder = Path(output_folder)

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

        self.signal_raw = None
        self.pauses_df = None
        self.events_df = None
        self.breaks_df = None

        self.left_filtered = None
        self.right_filtered = None

        self.saved_paths = []

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

        if not isinstance(config["save_plots"], bool):
            raise ValueError("save_plots must be True or False.")

        if not isinstance(config["show_plot"], bool):
            raise ValueError("show_plot must be True or False.")
        if not isinstance(config["use_only_quality_checked_events"], bool):
            raise ValueError("use_only_quality_checked_events must be True or False.")
        
        if not isinstance(config["event_quality_column"], str):
            raise ValueError("event_quality_column must be a string.")

    @staticmethod
    def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            keep_cols = []

            for col in df.columns:
                col_text = " ".join([str(x) for x in col]).lower()

                if "unnamed" not in col_text:
                    keep_cols.append(col)

            return df.loc[:, keep_cols]

        unnamed_cols = [
            col for col in df.columns
            if str(col).lower().startswith("unnamed")
        ]

        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        return df
    def apply_event_quality_filter(self):
        """
        Optionally keep only events that passed the event-level quality check.
    
        If use_only_quality_checked_events is True:
            keep only rows where event_quality_column == True
    
        If use_only_quality_checked_events is False:
            keep all events.
        """
    
        if self.events_df is None:
            raise RuntimeError("events_df is None. Run load_inputs() first.")
    
        if not self.use_only_quality_checked_events:
            return self
    
        if self.event_quality_column not in self.events_df.columns:
            raise ValueError(
                f"Quality column '{self.event_quality_column}' not found in events CSV. "
                "Run QualityCheck first or set use_only_quality_checked_events=False."
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
    def load_signal(self) -> pd.DataFrame:
        if not self.signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {self.signal_path}")

        try:
            df = pd.read_csv(self.signal_path, header=[0, 1])

            if isinstance(df.columns, pd.MultiIndex):
                cols_text = [
                    " ".join([str(x) for x in col]).lower()
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

    def load_table(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)
        df = self.drop_unnamed_columns(df)

        return df

    def load_inputs(self):
        self.signal_raw = self.load_signal()
        self.pauses_df = self.load_table(self.pauses_path)
        self.events_df = self.load_table(self.events_path)
        self.breaks_df = self.load_table(self.breaks_path)
    
        self.apply_event_quality_filter()
    
        return self
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

        b, a = butter(
            filter_order,
            normal_cutoff,
            btype="low",
            analog=False,
        )

        filtered = filtfilt(b, a, data_filled)
        filtered[~valid] = np.nan

        return filtered

    def filter_gyr_ml(self):
        if self.signal_raw is None:
            raise RuntimeError("signal_raw is None. Run load_inputs() first.")

        left_raw = self.get_column("left_sensor", self.channel)
        right_raw = self.get_column("right_sensor", self.channel)

        self.left_filtered = self.butter_lowpass_filter(
            data=left_raw,
            cutoff_freq=self.cutoff_freq_gyr,
            filter_order=self.filter_order_gyr,
            fs=self.fs,
        )

        self.right_filtered = self.butter_lowpass_filter(
            data=right_raw,
            cutoff_freq=self.cutoff_freq_gyr,
            filter_order=self.filter_order_gyr,
            fs=self.fs,
        )

        return self

    def get_signal_value(self, sample_idx: int, foot: str):
        foot = str(foot).lower()

        if self.left_filtered is None or self.right_filtered is None:
            raise RuntimeError("Filtered signals missing. Run filter_gyr_ml() first.")

        if sample_idx < 0 or sample_idx >= len(self.left_filtered):
            return None

        if foot == "left":
            return self.left_filtered[sample_idx]

        if foot == "right":
            return self.right_filtered[sample_idx]

        return None

    @staticmethod
    def get_event_color(foot: str) -> str:
        foot = str(foot).lower()

        if foot == "left":
            return "blue"

        if foot == "right":
            return "orange"

        return "black"

    @staticmethod
    def _safe_text(value) -> str:
        return (
            str(value)
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )

    def _check_ready_for_plotting(self) -> None:
        if self.pauses_df is None:
            raise RuntimeError("pauses_df is None. Run load_inputs() first.")

        if self.events_df is None:
            raise RuntimeError("events_df is None. Run load_inputs() first.")

        if self.breaks_df is None:
            raise RuntimeError("breaks_df is None. Run load_inputs() first.")

        if self.left_filtered is None or self.right_filtered is None:
            raise RuntimeError("Filtered signals missing. Run filter_gyr_ml() first.")

        required_pause_cols = ["pause_id", "start", "end"]
        missing_pause_cols = [
            col for col in required_pause_cols
            if col not in self.pauses_df.columns
        ]

        if missing_pause_cols:
            raise ValueError(f"Missing columns in pauses_df: {missing_pause_cols}")

        required_event_cols = ["foot", "ic", "pre_ic"]
        missing_event_cols = [
            col for col in required_event_cols
            if col not in self.events_df.columns
        ]

        if missing_event_cols:
            raise ValueError(f"Missing columns in events_df: {missing_event_cols}")

        required_break_cols = ["start", "end"]
        missing_break_cols = [
            col for col in required_break_cols
            if col not in self.breaks_df.columns
        ]

        if missing_break_cols:
            raise ValueError(f"Missing columns in breaks_df: {missing_break_cols}")

    def plot_single_pause(self, pause_row: pd.Series) -> Optional[Path]:
        pause_id = int(pause_row["pause_id"])
        pause_start = int(pause_row["start"])
        pause_end = int(pause_row["end"])

        patient_id = (
            str(pause_row["patient_id"])
            if "patient_id" in pause_row and pd.notna(pause_row["patient_id"])
            else "unknown_patient"
        )

        recording_date = (
            str(pause_row["recording_date"])
            if "recording_date" in pause_row and pd.notna(pause_row["recording_date"])
            else "unknown_date"
        )

        session_id = (
            str(pause_row["session_id"])
            if "session_id" in pause_row and pd.notna(pause_row["session_id"])
            else "unknown_session"
        )

        n_samples = len(self.left_filtered)

        plot_start = max(0, pause_start - self.samples_before_pause)
        plot_end = min(n_samples, pause_end + self.samples_after_pause)

        samples = np.arange(plot_start, plot_end)

        left_plot = self.left_filtered[plot_start:plot_end]
        right_plot = self.right_filtered[plot_start:plot_end]

        breaks_in_range = self.breaks_df[
            (pd.to_numeric(self.breaks_df["end"], errors="coerce") >= plot_start)
            & (pd.to_numeric(self.breaks_df["start"], errors="coerce") <= plot_end)
        ].copy()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            samples,
            left_plot,
            label=f"left_sensor {self.channel} filtered",
            linewidth=1,
            color="blue",
        )

        ax.plot(
            samples,
            right_plot,
            label=f"right_sensor {self.channel} filtered",
            linewidth=1,
            color="orange",
        )

        ax.axvspan(
            pause_start,
            pause_end,
            color="red",
            alpha=0.18,
            label="pause",
        )

        first_break_label = True

        for _, break_row in breaks_in_range.iterrows():
            break_start = int(break_row["start"])
            break_end = int(break_row["end"])

            ax.axvspan(
                break_start,
                break_end,
                color="lightblue",
                alpha=0.35,
                label="signal break" if first_break_label else None,
            )

            first_break_label = False

        first_left_ic = True
        first_right_ic = True
        first_left_pre_ic = True
        first_right_pre_ic = True

        for _, event_row in self.events_df.iterrows():
            foot = str(event_row["foot"]).lower()
            color = self.get_event_color(foot)

            if pd.notna(event_row["ic"]):
                ic_sample = int(event_row["ic"])

                if plot_start <= ic_sample < plot_end:
                    y_ic = self.get_signal_value(ic_sample, foot)

                    if y_ic is not None:
                        if foot == "left":
                            ax.scatter(
                                ic_sample,
                                y_ic,
                                color=color,
                                marker="o",
                                s=35,
                                zorder=5,
                                label="left IC" if first_left_ic else None,
                            )
                            first_left_ic = False

                        elif foot == "right":
                            ax.scatter(
                                ic_sample,
                                y_ic,
                                color=color,
                                marker="o",
                                s=35,
                                zorder=5,
                                label="right IC" if first_right_ic else None,
                            )
                            first_right_ic = False

            if pd.notna(event_row["pre_ic"]):
                pre_ic_sample = int(event_row["pre_ic"])

                if plot_start <= pre_ic_sample < plot_end:
                    y_pre_ic = self.get_signal_value(pre_ic_sample, foot)

                    if y_pre_ic is not None:
                        if foot == "left":
                            ax.scatter(
                                pre_ic_sample,
                                y_pre_ic,
                                color=color,
                                marker="x",
                                s=55,
                                zorder=6,
                                label="left pre_IC" if first_left_pre_ic else None,
                            )
                            first_left_pre_ic = False

                        elif foot == "right":
                            ax.scatter(
                                pre_ic_sample,
                                y_pre_ic,
                                color=color,
                                marker="x",
                                s=55,
                                zorder=6,
                                label="right pre_IC" if first_right_pre_ic else None,
                            )
                            first_right_pre_ic = False

        title = (
            f"{patient_id} | {recording_date} | {session_id} | "
            f"pause {pause_id} | samples {plot_start}-{plot_end}"
        )

        ax.set_title(title)
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

            safe_recording_date = self._safe_text(recording_date)

            save_path = (
                self.output_folder
                / f"{patient_id}_{safe_recording_date}_{session_id}_pause_{pause_id:03d}.png"
            )

            fig.savefig(save_path, dpi=self.dpi)
            self.saved_paths.append(save_path)

        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)

        return save_path

    def plot_all_pauses(self):
        self._check_ready_for_plotting()

        self.saved_paths = []

        for _, pause_row in self.pauses_df.iterrows():
            self.plot_single_pause(pause_row)

        return self.saved_paths

    def run(self):
        self.load_inputs()
        self.filter_gyr_ml()
        return self.plot_all_pauses()


# ============================================================
# DEBUG MAIN
# ============================================================

if __name__ == "__main__":

    signal_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_gaitMAP_bf_all.csv"
    )

    pauses_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_wb_pauses_dataframe.csv"
    )

    events_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_events.csv"
    )

    breaks_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_signal_break_dataframe.csv"
    )

    output_folder = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\pauses_plot"
    )

    user_config = {
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,
        "samples_before_pause": 500,
        "samples_after_pause": 500,
        "show_plot": False,
        "save_plots": True,
        "figsize": (15, 5),
        "dpi": 150,
        # Event quality filtering
        "use_only_quality_checked_events": True,
        "event_quality_column": "quality_check(IC>0)",
    }

    plotter = PausePlotter(
        signal_path=signal_path,
        pauses_path=pauses_path,
        events_path=events_path,
        breaks_path=breaks_path,
        output_folder=output_folder,
        config=user_config,
    )

    saved_paths = plotter.run()

    print(f"Saved {len(saved_paths)} pause plots in:")
    print(plotter.output_folder)