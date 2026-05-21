# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:55:52 2026

@author: francesca.boschi
"""


from pathlib import Path
from typing import Dict, Optional, Any
import json
from copy import deepcopy
import pandas as pd
import numpy as np


class WB_pipeline:

    DEFAULT_CONFIG = {
        # ---- Pause / break detection ----
        "wb_pause_s_threshold": 3.0,  # seconds

        # ---- Mobilise-D walking bout definition ----
        "threshold_oneSide": 3,
        "threshold_twoSides": 5,
        # ---- Event quality filtering ----
       "use_only_quality_checked_events": True,
       "event_quality_column": "quality_check(IC>0)",


        # ---- Plotting ----
        "pause_plot_threshold_min": 10.0,  # minutes
        "save_segments_images": True,
    }

    def __init__(
        self,
        session_folder: str,
        config: Optional[Dict[str, Any]] = None,
    ):

        self.session_folder = Path(session_folder)

        if not self.session_folder.exists():
            raise FileNotFoundError(f"Session folder not found: {self.session_folder}")

        # WB config
        self.config = self._build_config(config)

        self.wb_pause_s_threshold = self.config["wb_pause_s_threshold"]
        self.threshold_oneSide = self.config["threshold_oneSide"]
        self.threshold_twoSides = self.config["threshold_twoSides"]
        self.pause_plot_threshold_min = self.config["pause_plot_threshold_min"]
        self.save_segments_images = self.config["save_segments_images"]
        self.use_only_quality_checked_events = self.config["use_only_quality_checked_events"]
        self.event_quality_column = self.config["event_quality_column"]

        # Input log file from GaitMapPipeline
        self.source_log_path = None
        self.source_log = None

        # Metadata read from GaitMapPipeline log
        self.patient_id = None
        self.recording_date = None
        self.session_id = None
        self.fs = None
        
        # Input event file from GaitMapPipeline
        self.events_path = None
        self.events = None
        self.events_left_labeled = None
        self.events_right_labeled = None

        # WB class log
        self.log = {
            "config": deepcopy(self.config),
            "meta": {},
            "events": [],
        }

        self._find_source_log()
        self._load_source_log()
        self._read_metadata_from_source_log()
        self._read_metadata_from_source_log()

    @classmethod
    def _build_config(cls, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        user_config = user_config or {}

        unknown_keys = set(user_config) - set(cls.DEFAULT_CONFIG)
        if unknown_keys:
            raise ValueError(
                f"Unknown configuration key(s): {sorted(unknown_keys)}"
            )

        config = deepcopy(cls.DEFAULT_CONFIG)
        config.update(user_config)

        cls._validate_config(config)

        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:

        if config["wb_pause_s_threshold"] <= 0:
            raise ValueError("wb_pause_s_threshold must be greater than 0.")

        if config["threshold_oneSide"] <= 0:
            raise ValueError("threshold_oneSide must be greater than 0.")

        if config["threshold_twoSides"] <= 0:
            raise ValueError("threshold_twoSides must be greater than 0.")

        if config["pause_plot_threshold_min"] < 0:
            raise ValueError("pause_plot_threshold_min cannot be negative.")

        if not isinstance(config["save_segments_images"], bool):
            raise ValueError("save_segments_images must be True or False.")
        if not isinstance(config["use_only_quality_checked_events"], bool):
            raise ValueError("use_only_quality_checked_events must be True or False.")
        
        if not isinstance(config["event_quality_column"], str):
            raise ValueError("event_quality_column must be a string.")
    def apply_event_quality_filter(self):
        """
        Optionally remove events that did not pass the event-level quality check.
    
        If use_only_quality_checked_events is True:
            keep only rows where event_quality_column == True
    
        If use_only_quality_checked_events is False:
            keep all events.
    
        This method expects the events CSV to already contain the quality column
        produced by the QualityCheck class.
        """
    
        if self.events is None:
            raise RuntimeError("self.events is None. Run load_events() first.")
    
        if not self.use_only_quality_checked_events:
            self.log["events"].append(
                "Event quality filter skipped: all events are used."
            )
            return self
    
        if self.event_quality_column not in self.events.columns:
            raise ValueError(
                f"Quality column '{self.event_quality_column}' not found in events CSV. "
                "Run QualityCheck before WB_pipeline, or set "
                "use_only_quality_checked_events=False."
            )
    
        before = len(self.events)
    
        quality_values = self.events[self.event_quality_column]
    
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
    
        self.events = self.events[keep_mask].copy()
    
        after = len(self.events)
        removed = before - after
    
        self.log["events"].append(
            f"Event quality filter applied using '{self.event_quality_column}': "
            f"{after}/{before} events kept, {removed} removed."
        )
    
        return self
    def _find_source_log(self) -> None:
        """
        Find the log JSON produced by GaitMapPipeline inside the session folder.
        """

        log_candidates = list(self.session_folder.glob("*_log.json"))

        # Avoid selecting future WB logs if they exist later
        log_candidates = [
            path for path in log_candidates
            if not path.name.endswith("_wb_log.json")
        ]

        if not log_candidates:
            raise FileNotFoundError(
                f"No GaitMapPipeline log file found in: {self.session_folder}"
            )

        if len(log_candidates) > 1:
            raise ValueError(
                "More than one source log file found. "
                f"Candidates: {[p.name for p in log_candidates]}"
            )

        self.source_log_path = log_candidates[0]
        self.log["events"].append(f"Source log file found: {self.source_log_path}")

    def _load_source_log(self) -> None:
        """
        Load the GaitMapPipeline log JSON.
        """

        with open(self.source_log_path, "r", encoding="utf-8") as f:
            self.source_log = json.load(f)

        self.log["events"].append("Source log loaded.")

    def _read_metadata_from_source_log(self) -> None:
        """
        Read patient_id, recording_date, session_id, and sampling frequency
        from the GaitMapPipeline log.
        """

        if self.source_log is None:
            raise RuntimeError("source_log is None. Run _load_source_log() first.")

        if "meta" not in self.source_log:
            raise KeyError("Missing 'meta' section in source log.")

        if "config" not in self.source_log:
            raise KeyError("Missing 'config' section in source log.")

        source_meta = self.source_log["meta"]
        source_config = self.source_log["config"]

        required_meta_keys = [
            "patient_id",
            "recording_date",
            "session_id",
        ]

        for key in required_meta_keys:
            if key not in source_meta:
                raise KeyError(f"Missing '{key}' in source log meta.")

        if "sampling_rate_hz" not in source_config:
            raise KeyError("Missing 'sampling_rate_hz' in source log config.")

        self.patient_id = source_meta["patient_id"]
        self.recording_date = source_meta["recording_date"]
        self.session_id = source_meta["session_id"]
        self.fs = source_config["sampling_rate_hz"]

        self.log["meta"] = {
            "patient_id": self.patient_id,
            "recording_date": self.recording_date,
            "session_id": self.session_id,
            "sampling_rate_hz": self.fs,
            "session_folder": str(self.session_folder),
            "source_log_path": str(self.source_log_path),
        }

        self.log["events"].append("Metadata and sampling frequency read from source log.")

    def _prepare_labeled_events_from_csv(self) -> None:
        """
        Prepare left and right event tables from the saved events CSV.
    
        Methodological note
        -------------------
        The previous GaitMapPipeline saves one combined event table with a column
        named 'foot'. For walking-bout pause detection, the left and right feet must
        be treated separately because pauses are first detected independently per
        foot and then combined.
    
        Output attributes
        -----------------
        self.events_left_labeled
            Event table for the left foot only.
    
        self.events_right_labeled
            Event table for the right foot only.
        """
    
        if self.events is None:
            raise RuntimeError("self.events is None. Load the events CSV first.")
    
        if not isinstance(self.events, pd.DataFrame):
            raise TypeError("self.events must be a pandas DataFrame.")
    
        required_cols = ["s_id", "foot", "start", "end", "ic", "pre_ic"]
    
        missing_cols = [
            col for col in required_cols
            if col not in self.events.columns
        ]
    
        if missing_cols:
            raise ValueError(
                f"Missing required columns in events table: {missing_cols}"
            )
    
        events = self.events.copy()
    
        # Ensure numeric event/sample columns.
        # Invalid values are converted to NaN and removed below.
        numeric_cols = ["s_id", "start", "end", "ic", "pre_ic"]
    
        for col in numeric_cols:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    
        events = events.dropna(subset=["s_id", "foot", "start", "end", "ic", "pre_ic"])
    
        events["s_id"] = events["s_id"].astype(int)
        events["start"] = events["start"].astype(int)
        events["end"] = events["end"].astype(int)
        events["ic"] = events["ic"].astype(int)
        events["pre_ic"] = events["pre_ic"].astype(int)
    
        # Keep the original stride ID as index because later segment assignment
        # needs to store the list of stride IDs belonging to each walking bout.
        events = events.set_index("s_id", drop=False)
    
        self.events_left_labeled = (
            events[events["foot"].astype(str).str.lower() == "left"]
            .sort_values("ic")
            .copy()
        )
    
        self.events_right_labeled = (
            events[events["foot"].astype(str).str.lower() == "right"]
            .sort_values("ic")
            .copy()
        )
    
        self.log["events"].append(
            f"Events prepared from CSV: "
            f"{len(self.events_left_labeled)} left strides, "
            f"{len(self.events_right_labeled)} right strides."
        )
    
    def _find_events_file(self) -> None:
        """
        Find the events CSV produced by GaitMapPipeline inside the session folder.
        """
    
        if self.patient_id is None or self.recording_date is None or self.session_id is None:
            raise RuntimeError(
                "Metadata are missing. Read patient_id, recording_date and session_id from the source log first."
            )
    
        file_prefix = f"{self.patient_id}_{self.session_id}_{self.recording_date}"
        events_path = self.session_folder / f"{file_prefix}_events.csv"
    
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
    
        self.events_path = events_path
        self.log["events"].append(f"Events file found: {self.events_path}")  

        
    def _get_recording_length_samples(self) -> int:
        """
        Estimate the recording length in samples.
    
        Methodological note
        -------------------
        In the original class, pause detection used:
    
            len(self.signal_filtered)
    
        In this new class, the raw/filtered signal is not loaded anymore. Therefore,
        the binary pause vectors need a reconstructed length.
    
        Priority:
        1. If recording duration is available in the source log, use:
               recording_time_hours * 3600 * sampling_rate_hz
    
        2. Otherwise, fall back to the largest event/sample index found in the
           events table.
    
        The first option is preferred because it preserves the full recording
        duration, including possible quiet periods after the last detected gait event.
        """
    
        n_samples_from_log = None
    
        if isinstance(getattr(self, "source_log", None), dict):
            recording_summary = self.source_log.get("recording_summary", {})
    
            if "recording_time_hours" in recording_summary:
                recording_time_hours = float(recording_summary["recording_time_hours"])
                n_samples_from_log = int(round(recording_time_hours * 3600 * self.fs))
    
        if n_samples_from_log is not None and n_samples_from_log > 0:
            return n_samples_from_log
    
        if self.events is None or self.events.empty:
            raise RuntimeError(
                "Cannot estimate recording length: no recording_summary in log "
                "and events table is empty."
            )
    
        candidate_cols = [
            col for col in ["start", "end", "ic", "tc", "min_vel", "pre_ic"]
            if col in self.events.columns
        ]
    
        if not candidate_cols:
            raise RuntimeError(
                "Cannot estimate recording length: no sample-index columns found."
            )
    
        max_sample = (
            self.events[candidate_cols]
            .apply(pd.to_numeric, errors="coerce")
            .max()
            .max()
        )
    
        if pd.isna(max_sample):
            raise RuntimeError("Cannot estimate recording length from event table.")
    
        return int(max_sample) + 1
    
    def load_events(self):
        """
        Load the GaitMapPipeline events CSV.
    
        If configured, keep only events with quality_check == True.
        """
    
        if self.events_path is None:
            self._find_events_file()
    
        self.events = pd.read_csv(self.events_path)
    
        self.log["events"].append(
            f"Events loaded: {len(self.events)} rows."
        )
    
        self.apply_event_quality_filter()
    
        return self  
    @staticmethod
    def calculate_time_diff(events_df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """
        Calculate the time gap between consecutive strides of the same foot.
    
        Methodological note
        -------------------
        Each row represents one stride.
    
        For stride i and the following stride i+1, the method computes:
    
            gap_samples = pre_ic(i+1) - ic(i)
    
        This gap represents the time interval between the initial contact of one
        stride and the pre-initial-contact of the next stride for the same foot.
    
        A short gap is expected during continuous walking. A long gap means that
        consecutive events are not continuous in time and therefore may indicate
        a pause or an interruption in the walking bout.
    
        Parameters
        ----------
        events_df:
            Event table for one foot only.
    
        fs:
            Sampling frequency in Hz.
    
        Returns
        -------
        pd.DataFrame
            One row per pair of consecutive strides, with gap duration in samples
            and seconds.
        """
    
        if not isinstance(events_df, pd.DataFrame) or events_df.empty:
            return pd.DataFrame(
                columns=[
                    "s_id_current",
                    "s_id_next",
                    "ic_current",
                    "pre_ic_next",
                    "gap_start",
                    "gap_end",
                    "gap_samples",
                    "gap_s",
                ]
            )
    
        required_cols = ["s_id", "ic", "pre_ic"]
    
        missing_cols = [
            col for col in required_cols
            if col not in events_df.columns
        ]
    
        if missing_cols:
            raise ValueError(
                f"Missing required columns for time-diff calculation: {missing_cols}"
            )
    
        ev = events_df.copy()
        ev = ev.sort_values("ic")
    
        rows = []
    
        for i in range(len(ev) - 1):
            current_row = ev.iloc[i]
            next_row = ev.iloc[i + 1]
    
            ic_current = int(current_row["ic"])
            pre_ic_next = int(next_row["pre_ic"])
    
            gap_start = ic_current
            gap_end = pre_ic_next
    
            gap_samples = gap_end - gap_start
            gap_s = gap_samples / fs
    
            rows.append(
                {
                    "s_id_current": int(current_row["s_id"]),
                    "s_id_next": int(next_row["s_id"]),
                    "ic_current": ic_current,
                    "pre_ic_next": pre_ic_next,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "gap_samples": gap_samples,
                    "gap_s": gap_s,
                }
            )
    
        return pd.DataFrame(rows)
    
    
    @staticmethod
    def mark_pauses(
        binary_signal: np.ndarray,
        gait_events_time_diff: pd.DataFrame,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Mark candidate non-walking intervals for one foot.
    
        Methodological note
        -------------------
        The binary vector is initialized with zeros.
    
            0 = no candidate pause for that foot
            1 = candidate pause for that foot
    
        For every gap between two consecutive strides, the samples between:
    
            ic_current and pre_ic_next
    
        are marked as 1 when the gap duration is greater than the selected threshold.
    
        In WB_pauses_detection(), this threshold is intentionally set to 0 because
        we first mark all inter-stride gaps for each foot. The real walking-bout
        pause threshold is applied later only after combining both feet.
        """
    
        if gait_events_time_diff is None or gait_events_time_diff.empty:
            return binary_signal
    
        required_cols = ["gap_start", "gap_end", "gap_s"]
    
        missing_cols = [
            col for col in required_cols
            if col not in gait_events_time_diff.columns
        ]
    
        if missing_cols:
            raise ValueError(
                f"Missing required columns for pause marking: {missing_cols}"
            )
    
        out = binary_signal.copy()
    
        for _, row in gait_events_time_diff.iterrows():
            if float(row["gap_s"]) <= threshold:
                continue
    
            start = int(row["gap_start"])
            end = int(row["gap_end"])
    
            start = max(0, start)
            end = min(len(out), end)
    
            if end > start:
                out[start:end] = 1
    
        return out
    
    
    @staticmethod
    def get_pause_segments(
        pause_flag: np.ndarray,
        sampling_rate_hz: float,
        min_duration_s: float,
    ):
        """
        Split combined no-event intervals into pauses and short breaks.
    
        Methodological note
        -------------------
        After left and right binary signals are summed:
    
            0 = walking events are present for both feet
            1 = no event interval for one foot only
            2 = no event interval for both feet simultaneously
    
        Candidate walking-bout pauses are intervals where pause_flag == 2.
    
        These candidate intervals are then separated into:
    
            pauses:
                duration >= wb_pause_s_threshold
    
            breaks:
                duration < wb_pause_s_threshold
    
        In this methodology, only simultaneous absence of events from both feet is
        considered a true walking-bout pause. Shorter simultaneous gaps are retained
        as breaks because they may represent small detection gaps or brief signal
        discontinuities within the same walking bout.
    
        Output note
        -----------
        df_pause uses pause_id.
    
        df_break uses break_id.
    
        These identifiers are different from signal_segments.segment_id, which
        identifies non-pause candidate walking-bout segments.
        """
    
        pause_flag = np.asarray(pause_flag)
    
        min_samples = int(round(min_duration_s * sampling_rate_hz))
    
        segments = []
        in_segment = False
        start = None
    
        for i, value in enumerate(pause_flag):
            if value == 2 and not in_segment:
                in_segment = True
                start = i
    
            elif value != 2 and in_segment:
                end = i
                segments.append((start, end))
                in_segment = False
                start = None
    
        if in_segment:
            segments.append((start, len(pause_flag)))
    
        pause_rows = []
        break_rows = []
    
        pause_id = 0
        break_id = 0
    
        for start, end in segments:
    
            duration_samples = end - start
            duration_s = duration_samples / sampling_rate_hz
    
            if duration_samples >= min_samples:
                pause_rows.append(
                    {
                        "pause_id": pause_id,
                        "start": int(start),
                        "end": int(end),
                        "duration_samples": int(duration_samples),
                        "duration_s": float(duration_s),
                    }
                )
                pause_id += 1
    
            else:
                break_rows.append(
                    {
                        "break_id": break_id,
                        "start": int(start),
                        "end": int(end),
                        "duration_samples": int(duration_samples),
                        "duration_s": float(duration_s),
                    }
                )
                break_id += 1
    
        df_pause = pd.DataFrame(
            pause_rows,
            columns=["pause_id", "start", "end", "duration_samples", "duration_s"],
        )
    
        df_break = pd.DataFrame(
            break_rows,
            columns=["break_id", "start", "end", "duration_samples", "duration_s"],
        )
    
        return df_pause, df_break
    
    
    @staticmethod
    def get_non_pause_segments(
        n_samples: int,
        df_pause: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract walking-bout candidate segments between detected pauses.
    
        Methodological note
        -------------------
        A walking-bout candidate is defined as a portion of the recording that is
        not interrupted by a valid pause.
    
        If valid pauses are detected, they split the full recording into multiple
        non-pause segments.
    
        If no valid pause is detected, the whole recording is treated as one
        candidate segment.
    
        These segments are not yet final Mobilise-D walking bouts. They still need
        to be checked for the number of strides per side.
        """
    
        if n_samples <= 0:
            return pd.DataFrame(columns=["segment_id", "start", "end", "duration_s"])
    
        if df_pause is None or df_pause.empty:
            return pd.DataFrame(
                [
                    {
                        "segment_id": 0,
                        "start": 0,
                        "end": int(n_samples),
                        "duration_s": np.nan,
                    }
                ]
            )
    
        pauses = df_pause.sort_values("start").copy()
    
        rows = []
        current_start = 0
        segment_id = 0
    
        for _, pause in pauses.iterrows():
            pause_start = int(pause["start"])
            pause_end = int(pause["end"])
    
            if pause_start > current_start:
                rows.append(
                    {
                        "segment_id": segment_id,
                        "start": int(current_start),
                        "end": int(pause_start),
                        "duration_s": np.nan,
                    }
                )
                segment_id += 1
    
            current_start = max(current_start, pause_end)
    
        if current_start < n_samples:
            rows.append(
                {
                    "segment_id": segment_id,
                    "start": int(current_start),
                    "end": int(n_samples),
                    "duration_s": np.nan,
                }
            )
    
        return pd.DataFrame(rows)
    
    
    @staticmethod
    def add_event_ids_to_segments(
        signal_segments: pd.DataFrame,
        events_left: pd.DataFrame,
        events_right: pd.DataFrame,
        fs: float,
    ) -> pd.DataFrame:
        """
        Assign left and right stride IDs to each non-pause segment.
    
        Methodological note
        -------------------
        A stride belongs to a segment if its initial contact falls inside the
        segment boundaries:
    
            segment_start <= ic < segment_end
    
        Initial contact is used because it is the main temporal anchor of the gait
        event table.
        """
    
        if signal_segments is None or signal_segments.empty:
            return pd.DataFrame(
                columns=[
                    "segment_id",
                    "start",
                    "end",
                    "duration_s",
                    "left_s_ids",
                    "right_s_ids",
                ]
            )
    
        out = signal_segments.copy()
    
        left_ids = []
        right_ids = []
    
        for _, seg in out.iterrows():
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])
    
            if isinstance(events_left, pd.DataFrame) and not events_left.empty:
                left_in_segment = events_left[
                    (events_left["ic"] >= seg_start)
                    & (events_left["ic"] < seg_end)
                ]["s_id"].astype(int).tolist()
            else:
                left_in_segment = []
    
            if isinstance(events_right, pd.DataFrame) and not events_right.empty:
                right_in_segment = events_right[
                    (events_right["ic"] >= seg_start)
                    & (events_right["ic"] < seg_end)
                ]["s_id"].astype(int).tolist()
            else:
                right_in_segment = []
    
            left_ids.append(left_in_segment)
            right_ids.append(right_in_segment)
    
        out["left_s_ids"] = left_ids
        out["right_s_ids"] = right_ids
    
        out["total_strides_left"] = out["left_s_ids"].apply(len)
        out["total_strides_right"] = out["right_s_ids"].apply(len)
    
        out["duration_s"] = (out["end"] - out["start"]) / fs
    
        return out
    
    
    def WB_pauses_detection(self):
        """
        Detect pauses between walking-bout candidates.
    
        Methodological overview
        -----------------------
        This method detects candidate walking-bout pauses using the temporal
        structure of gait events produced by GaitMapPipeline.
    
        The procedure is:
    
        1. Split the saved event table into left and right foot event tables.
    
        2. For each foot independently, calculate the time gaps between consecutive
           strides using:
    
               gap = pre_ic_next - ic_current
    
           A gap represents a period between two consecutive strides of the same
           foot.
    
        3. Convert the left and right gap tables into binary sample-wise signals:
    
               0 = no candidate gap for that foot
               1 = candidate gap for that foot
    
        4. Sum the two binary signals:
    
               0 = walking events are present for both feet
               1 = candidate gap for one foot only
               2 = candidate gap for both feet simultaneously
    
        5. Detect intervals where the summed signal equals 2.
    
           These are periods where neither foot shows continuous gait events.
    
        6. Classify these intervals as:
    
               pause:
                   duration >= wb_pause_s_threshold
    
               break:
                   duration < wb_pause_s_threshold
    
        7. Use valid pauses to split the recording into non-pause signal segments.
    
        8. Assign left and right stride IDs to each segment.
    
        Important
        ---------
        This method does not yet apply the Mobilise-D walking-bout stride-count
        definition. It only prepares pause-based candidate segments and assigns
        stride IDs to them.
    
        Required attributes
        -------------------
        self.events
            Combined event table loaded from the saved events CSV.
    
        self.fs
            Sampling frequency read from the previous GaitMapPipeline log.
    
        self.wb_pause_s_threshold
            Minimum duration in seconds for a simultaneous no-event interval to be
            classified as a walking-bout pause.
    
        Output attributes
        -----------------
        self.gait_events_time_diff_left
        self.gait_events_time_diff_right
        self.binary_left
        self.binary_right
        self.sum_binary
        self.df_pause
        self.df_break
        self.signal_segments
        """
    
        if self.events is None:
            raise RuntimeError("self.events is None. Load the events CSV first.")
    
        if self.fs is None:
            raise RuntimeError("self.fs is None. Read sampling_rate_hz from the source log first.")
    
        # Prepare one event table per foot from the combined saved CSV.
        self._prepare_labeled_events_from_csv()
    
        ev_left = getattr(self, "events_left_labeled", None)
        ev_right = getattr(self, "events_right_labeled", None)
    
        if not isinstance(ev_left, pd.DataFrame) or ev_left.empty:
            ev_left = pd.DataFrame(columns=["s_id", "ic", "pre_ic"])
    
        if not isinstance(ev_right, pd.DataFrame) or ev_right.empty:
            ev_right = pd.DataFrame(columns=["s_id", "ic", "pre_ic"])
    
        # Determine the length of the recording in samples.
        # This replaces len(self.signal_filtered) from the previous class.
        n_samples = self._get_recording_length_samples()
        self.n_samples = n_samples
    
        # Calculate per-foot gaps between consecutive strides.
        self.gait_events_time_diff_left = self.calculate_time_diff(
            ev_left,
            self.fs,
        )
    
        self.gait_events_time_diff_right = self.calculate_time_diff(
            ev_right,
            self.fs,
        )
    
        # Mark all inter-stride gaps per foot.
        # threshold=0 means that every positive gap is marked.
        self.binary_left = self.mark_pauses(
            np.zeros(n_samples, dtype=int),
            self.gait_events_time_diff_left,
            threshold=0.0,
        )
    
        self.binary_right = self.mark_pauses(
            np.zeros(n_samples, dtype=int),
            self.gait_events_time_diff_right,
            threshold=0.0,
        )
    
        # Combined pause signal:
        # 0 = no gap
        # 1 = gap in one foot
        # 2 = simultaneous gap in both feet
        self.sum_binary = pd.DataFrame(
            self.binary_left + self.binary_right,
            columns=["pause_flag"],
        )
    
        # Long simultaneous no-event intervals are walking-bout pauses.
        # Short simultaneous no-event intervals are retained as breaks.
        self.df_pause, self.df_break = self.get_pause_segments(
            self.sum_binary["pause_flag"].values,
            sampling_rate_hz=self.fs,
            min_duration_s=self.wb_pause_s_threshold,
        )
    
        # Extract non-pause candidate segments.
        self.signal_segments = self.get_non_pause_segments(
            n_samples=n_samples,
            df_pause=self.df_pause,
        )
    
        # Assign stride IDs to each candidate segment.
        self.signal_segments = self.add_event_ids_to_segments(
            signal_segments=self.signal_segments,
            events_left=self.events_left_labeled,
            events_right=self.events_right_labeled,
            fs=self.fs,
        )
    
        self.log["events"].append(
            f"WB pauses detected: {len(self.df_pause)} pauses, "
            f"{len(self.df_break)} breaks "
            f"(threshold = {self.wb_pause_s_threshold}s)."
        )
    
        return self

    def label_wb_validity(
        self,
        segs: pd.DataFrame,
        threshold_oneSide: int,
        threshold_twoSides: int,
    ) -> pd.DataFrame:
        """
        Label each non-pause segment as valid or not valid walking bout.
    
        Methodological rule
        -------------------
        A segment is considered a valid walking bout if:
    
        1. Both feet are represented and the total number of detected events
           between left and right is at least threshold_twoSides.
    
           Example with threshold_twoSides = 5:
               left = 4, right = 1  -> validWB
               left = 3, right = 2  -> validWB
               left = 2, right = 2  -> not_validWB
    
        2. Only one foot is represented and that foot has more than
           threshold_oneSide detected events.
    
           Example with threshold_oneSide = 3:
               left = 4, right = 0  -> validWB
               left = 3, right = 0  -> not_validWB
    
        Rationale
        ---------
        If both feet are present, validity is based on the total number of
        detected events because missing events on one side may occur in free-living
        data. A minimal valid bilateral case is therefore 4 + 1 = 5.
    
        If only one foot is present, the segment is accepted only when that side
        has more than 3 events, because 3 events alone may represent only 4
        strides/steps and is not enough for the intended WB definition.
    
        Output column
        -------------
        label_WB:
            'validWB' or 'not_validWB'
        """
    
        segs = segs.copy()
    
        if "total_strides_left" not in segs.columns:
            raise ValueError("Missing column: total_strides_left")
    
        if "total_strides_right" not in segs.columns:
            raise ValueError("Missing column: total_strides_right")
    
        labels = []
    
        for _, row in segs.iterrows():
    
            n_left = int(row["total_strides_left"])
            n_right = int(row["total_strides_right"])
    
            # Case 1: both feet are represented.
            # Mobilise-D validity is approximated by the total number of detected
            # left + right events. This accepts asymmetric but plausible cases
            # such as 4 left and 1 right, because the subject cannot realistically
            # be walking by hopping on one foot in free-living gait.
            both_sides_valid = (
                n_left > 0
                and n_right > 0
                and (n_left + n_right) >= threshold_twoSides
            )
    
            # Case 2: only the left foot is represented.
            # Here we use > threshold_oneSide, not >=, because with threshold 3,
            # exactly 3 detected events are not enough for the intended WB rule.
            left_only_valid = (
                n_left > threshold_oneSide
                and n_right == 0
            )
    
            # Case 3: only the right foot is represented.
            right_only_valid = (
                n_right > threshold_oneSide
                and n_left == 0
            )
    
            if both_sides_valid or left_only_valid or right_only_valid:
                labels.append("validWB")
            else:
                labels.append("not_validWB")
    
        segs["label_WB"] = labels
    
        return segs


    def add_wb_type(self, segs: pd.DataFrame) -> pd.DataFrame:
        """
        Add a methodological WB type label.
    
        Output column
        -------------
        WB_type:
            'two_sides'
            'left_only'
            'right_only'
            'none'
        """
    
        segs = segs.copy()
    
        wb_types = []
    
        for _, row in segs.iterrows():
    
            n_left = int(row["total_strides_left"])
            n_right = int(row["total_strides_right"])
    
            if n_left > 0 and n_right > 0:
                wb_types.append("two_sides")
            elif n_left > 0 and n_right == 0:
                wb_types.append("left_only")
            elif n_right > 0 and n_left == 0:
                wb_types.append("right_only")
            else:
                wb_types.append("none")
    
        segs["WB_type"] = wb_types
    
        return segs
    def adjust_wb_boundaries(self, segs: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust candidate WB boundaries using gait events.
    
        Methodological note
        -------------------
        The pause-detection step creates segments from sample-wise pause flags.
        Those boundaries are pause-based, not gait-event-based.
    
        For the final WB candidates:
            - the start is moved to the earliest pre_ic inside the segment
            - the end is moved to the latest ic inside the segment
    
        This makes the WB boundaries closer to actual gait-event timing.
        """
    
        segs = segs.copy()
    
        if segs.empty:
            return segs
    
        if not hasattr(self, "events_left_labeled") or not hasattr(self, "events_right_labeled"):
            raise RuntimeError(
                "events_left_labeled and events_right_labeled are missing. "
                "Run WB_pauses_detection() first."
            )
    
        adjusted_starts = []
        adjusted_ends = []
    
        for _, seg in segs.iterrows():
    
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])
    
            event_starts = []
            event_ends = []
    
            if isinstance(self.events_left_labeled, pd.DataFrame) and not self.events_left_labeled.empty:
                left_events = self.events_left_labeled[
                    (self.events_left_labeled["ic"] >= seg_start)
                    & (self.events_left_labeled["ic"] < seg_end)
                ]
    
                if not left_events.empty:
                    event_starts.extend(left_events["pre_ic"].dropna().astype(int).tolist())
                    event_ends.extend(left_events["ic"].dropna().astype(int).tolist())
    
            if isinstance(self.events_right_labeled, pd.DataFrame) and not self.events_right_labeled.empty:
                right_events = self.events_right_labeled[
                    (self.events_right_labeled["ic"] >= seg_start)
                    & (self.events_right_labeled["ic"] < seg_end)
                ]
    
                if not right_events.empty:
                    event_starts.extend(right_events["pre_ic"].dropna().astype(int).tolist())
                    event_ends.extend(right_events["ic"].dropna().astype(int).tolist())
    
            if event_starts:
                adjusted_starts.append(int(min(event_starts)))
            else:
                adjusted_starts.append(seg_start)
    
            if event_ends:
                adjusted_ends.append(int(max(event_ends)))
            else:
                adjusted_ends.append(seg_end)
    
        segs["start_original"] = segs["start"]
        segs["end_original"] = segs["end"]
    
        segs["start"] = adjusted_starts
        segs["end"] = adjusted_ends
        segs["duration_s"] = (segs["end"] - segs["start"]) / self.fs
    
        return segs
    def WB_extraction(self):
        """
        Identify valid walking bouts from non-pause segments.
    
        Required previous step
        ----------------------
        WB_pauses_detection() must be run before this method.
    
        Methodological flow
        -------------------
        1. Start from non-pause segments created by WB_pauses_detection().
        2. Adjust each segment boundary using gait events:
            - start = earliest pre_ic inside the segment
            - end = latest ic inside the segment
        3. Add WB type:
            - two_sides
            - left_only
            - right_only
            - none
        4. Apply stride-count validity rules:
            - valid two-sided WB if both feet have at least threshold_twoSides strides
            - valid one-sided WB if only one foot is present and has at least
              threshold_oneSide strides
        5. Store:
            - self.signal_segments: all labeled segments
            - self.wb: only valid walking bouts
        """
    
        if not hasattr(self, "signal_segments") or self.signal_segments is None:
            raise RuntimeError("Run WB_pauses_detection() before WB_extraction().")
    
        segs = self.signal_segments.copy()
    
        if segs.empty:
            self.signal_segments = segs
            self.wb = pd.DataFrame(columns=list(segs.columns) + ["WB_id"])
    
            self.log["events"].append(
                "WB_extraction: no signal segments available."
            )
    
            return self
    
        segs = self.adjust_wb_boundaries(segs)
    
        segs = self.add_wb_type(segs)
    
        segs = self.label_wb_validity(
            segs,
            threshold_oneSide=self.threshold_oneSide,
            threshold_twoSides=self.threshold_twoSides,
        )
    
        valid = segs[segs["label_WB"] == "validWB"].copy()
    
        if not valid.empty:
            valid = valid.reset_index(drop=True)
            valid["WB_id"] = valid.index
        else:
            valid = pd.DataFrame(columns=list(segs.columns) + ["WB_id"])
    
        self.signal_segments = segs
        self.wb = valid
    
        n_valid = len(valid)
        n_not_valid = int((segs["label_WB"] == "not_validWB").sum())
    
        self.log["events"].append(
            f"WB_extraction: {n_valid} validWB, {n_not_valid} not_validWB."
        )
    
        return self
    def save_outputs(self) -> Dict[str, Path]:
        """
        Save WB pipeline outputs in the same session folder.
    
        Saved files
        -----------
        patient_id_session_id_recording_date_wb_pauses_dataframe.csv
        patient_id_session_id_recording_date_signal_break_dataframe.csv
        patient_id_session_id_recording_date_quality_segment_check_dataframe.csv
        patient_id_session_id_recording_date_wb_dataframe.csv
    
        Each CSV starts with:
            patient_id, recording_date, session_id
        """
    
        if self.patient_id is None:
            raise ValueError("patient_id is None.")
    
        if self.recording_date is None:
            raise ValueError("recording_date is None.")
    
        if self.session_id is None:
            raise ValueError("session_id is None.")
    
        file_prefix = f"{self.patient_id}_{self.session_id}_{self.recording_date}"
    
        saved_paths = {}
    
        # -------------------------------------------------
        # Helper: add metadata columns at the beginning
        # -------------------------------------------------
        def _add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
    
            # Avoid duplicated metadata columns if the method is called twice
            for col in ["patient_id", "recording_date", "session_id"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
    
            df.insert(0, "session_id", self.session_id)
            df.insert(0, "recording_date", self.recording_date)
            df.insert(0, "patient_id", self.patient_id)
    
            return df
    
        # -------------------------------------------------
        # Save WB pauses dataframe
        # -------------------------------------------------
        if hasattr(self, "df_pause") and isinstance(self.df_pause, pd.DataFrame):
            df_pause_out = _add_metadata_columns(self.df_pause)
    
            pause_path = (
                self.session_folder
                / f"{file_prefix}_wb_pauses_dataframe.csv"
            )
    
            df_pause_out.to_csv(pause_path, index=False)
            saved_paths["wb_pauses_dataframe"] = pause_path
    
        # -------------------------------------------------
        # Save signal break dataframe
        # -------------------------------------------------
        if hasattr(self, "df_break") and isinstance(self.df_break, pd.DataFrame):
            df_break_out = _add_metadata_columns(self.df_break)
    
            break_path = (
                self.session_folder
                / f"{file_prefix}_signal_break_dataframe.csv"
            )
    
            df_break_out.to_csv(break_path, index=False)
            saved_paths["signal_break_dataframe"] = break_path
    
        # -------------------------------------------------
        # Save quality segment check dataframe
        # -------------------------------------------------
        if hasattr(self, "signal_segments") and isinstance(self.signal_segments, pd.DataFrame):
            signal_segments_out = _add_metadata_columns(self.signal_segments)
    
            segments_path = (
                self.session_folder
                / f"{file_prefix}_quality_segment_check_dataframe.csv"
            )
    
            signal_segments_out.to_csv(segments_path, index=False)
            saved_paths["quality_segment_check_dataframe"] = segments_path
    
        # -------------------------------------------------
        # Save valid WB dataframe
        # -------------------------------------------------
        if hasattr(self, "wb") and isinstance(self.wb, pd.DataFrame):
            wb_out = _add_metadata_columns(self.wb)
    
            wb_path = (
                self.session_folder
                / f"{file_prefix}_wb_dataframe.csv"
            )
    
            wb_out.to_csv(wb_path, index=False)
            saved_paths["wb_dataframe"] = wb_path
    
        self.log["events"].append("WB outputs saved.")
    
        return saved_paths
#%% DEBUG MAIN


if __name__ == "__main__":

    session_folder = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
    )

    user_config = {
        "wb_pause_s_threshold": 3.0,

        # ---- Mobilise-D WB definition ----
        "threshold_oneSide": 3,
        "threshold_twoSides": 5,
        
        # ---- Event quality filtering ----
    "use_only_quality_checked_events": True,
    "event_quality_column": "quality_check(IC>0)",

        # ---- Plotting ----
        "pause_plot_threshold_min": 10.0,
        "save_segments_images": True,
    }

    pipeline = WB_pipeline(
        session_folder=session_folder,
        config=user_config,
    )

    # This assumes your class already loads events into self.events.
    # If not, run your load_events/load_inputs method before this line.
    pipeline.load_events()
    pipeline.WB_pauses_detection()
    pipeline.WB_extraction()
    saved_paths = pipeline.save_outputs()

    df_pause = pipeline.df_pause
    df_break = pipeline.df_break
    signal_segments = pipeline.signal_segments
    wb = pipeline.wb
    segments=pipeline.signal_segments
    saved_paths = pipeline.save_outputs()

    print(saved_paths)
