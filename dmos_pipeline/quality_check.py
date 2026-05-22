# -*- coding: utf-8 -*-
"""
QualityCheck.

Standalone script.

Run after 1_Gaitmap.py.

Goal
----
1. Load signal.
2. Apply low-pass filter to left/right gyr_ml.
3. Load events CSV produced by 1_Gaitmap.py.
4. Load parameters CSV produced by 1_Gaitmap.py.
5. Add quality labels to events:
       - quality_check(IC<0)
       - notes

   Events rule:
       - if foot == left, check filtered left_sensor gyr_ml at IC sample
       - if foot == right, check filtered right_sensor gyr_ml at IC sample
       - quality_check(IC<0) = False if filtered gyr_ml at IC > 0
       - quality_check(IC<0) = True otherwise

6. Add quality labels to parameters:
       - quality_check
       - notes

   Parameters rules:
       - turning angle:
             fail if 25 <= abs(turning angle [deg]) <= 90

       - parameter outlier:
             fail if at least n_parameter_violations are outside configured ranges:
                 stride time [s]:        0.2 to 3.0
                 gait velocity [m/s]:    0.2 to 2.0
                 stride length [m]:      0.10 to 1.5

7. Overwrite:
       - *_events.csv
       - *_parameters.csv

8. Update existing GaitMap log JSON with quality_check_summary.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class QualityCheck:
    
    """
    Quality-control pipeline for gait events and stride-level gait parameters.

    This class is intended to be run after the gaitmap-based processing
    pipeline. It loads the exported IMU signal, gait-event table, and
    stride-level parameter table, then adds quality-control labels without
    removing any rows.

    Two levels of quality control are applied:

    1. Event-level quality check
       Initial-contact events are checked using the filtered gyroscope signal.
       For each event, the signal value is extracted at the detected initial
       contact sample from the corresponding foot sensor:

       - left foot events are checked on `left_sensor`
       - right foot events are checked on `right_sensor`

       By default, the selected channel is `gyr_ml`. An event fails the check
       when the filtered gyroscope value at initial contact is greater than the
       configured threshold.

       The following columns are added to the events table:

       - `quality_check(IC<0)`
       - `notes`

       By default:

       - `quality_check(IC<0) = False` if filtered `gyr_ml` at IC > 0
       - `quality_check(IC<0) = True` otherwise

    2. Parameter-level quality check
       Stride-level parameters are checked using configurable plausibility
       rules. The default rules include:

       - turning angle:
           the row fails if `25 <= abs(turning angle [deg]) <= 90`

       - parameter outliers:
           the row fails if at least `n_parameter_violations` parameters fall
           outside their configured ranges.

       The default parameter ranges are:

       - `stride time [s]`: 0.2 to 3.0
       - `gait velocity [m/s]`: 0.2 to 2.0
       - `stride length [m]`: 0.10 to 1.5

       The following columns are added to the parameters table:

       - `quality_check`
       - `notes`

    The labelled events and parameters are saved back to their original CSV
    files, and the existing gaitmap processing log is updated with a
    `quality_check_summary`.

    Notes
    -----
    This class labels data quality but does not remove events or parameter
    rows. Downstream processing can decide whether to exclude failed rows.
    """
    DEFAULT_CONFIG = {
        # ---- Signal ----
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",

        # ---- Filtering ----
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,

        # ---- Event-level quality check ----
        "ic_threshold": 0.0,
        "events_quality_col": "quality_check(IC<0)",
        "notes_col": "notes",

        # ---- Parameter-level quality check ----
        "turning_angle_abs_range": (25.0, 90.0),
        "parameter_rules": {
            "stride time [s]": (0.2, 3.0),
            "gait velocity [m/s]": (0.2, 2.0),
            "stride length [m]": (0.10, 1.5),
        },

        # Fail parameter outlier check if at least this number of rules are violated.
        "n_parameter_violations": 2,

        # ---- Enable / disable checks ----
        "apply_events_ic_check": True,
        "apply_turning_angle_check": True,
        "apply_parameter_outlier_check": True,
    }

    def __init__(
        self,
        signal_path: Optional[str],
        events_path: str,
        project_folder: str,
        parameters_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        signal_filtered: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the quality-check pipeline.
    
        Parameters
        ----------
        signal_path : str, optional
            Path to the raw IMU signal file. Required only if `signal_filtered`
            is not provided.
    
        events_path : str
            Path to the gait-event CSV file produced by the gaitmap-based pipeline.
    
        project_folder : str
            Root project folder. It is used to reconstruct the expected output
            folder structure and to locate related pipeline outputs when needed.
    
        parameters_path : str, optional
            Path to the stride-level parameter CSV file. If not provided, the
            parameter file is searched automatically from `project_folder`,
            `patient_id`, `recording_date`, and `session_id` extracted from the
            events table.
    
        config : dict, optional
            User-defined configuration values used to override `DEFAULT_CONFIG`.
        """
    
        # Input paths
        self.signal_path = Path(signal_path) if signal_path is not None else None
        self.signal_filtered = signal_filtered
        self.events_path = Path(events_path)
        self.project_folder = Path(project_folder)
        self.parameters_path = Path(parameters_path) if parameters_path is not None else None
    
        # Configuration
        self.config = self._build_config(config)
    
        self.fs = self.config["sampling_rate_hz"]
        self.channel = self.config["channel"]
        self.cutoff_freq_gyr = self.config["cutoff_freq_gyr"]
        self.filter_order_gyr = self.config["filter_order_gyr"]
    
        # Event-level quality-check settings
        self.ic_threshold = self.config["ic_threshold"]
        self.events_quality_col = self.config["events_quality_col"]
        self.notes_col = self.config["notes_col"]
        self.apply_events_ic_check = self.config["apply_events_ic_check"]
    
        # Parameter-level quality-check settings
        self.turning_angle_abs_range = self.config["turning_angle_abs_range"]
        self.parameter_rules = self.config["parameter_rules"]
        self.n_parameter_violations = self.config["n_parameter_violations"]
        self.apply_turning_angle_check = self.config["apply_turning_angle_check"]
        self.apply_parameter_outlier_check = self.config["apply_parameter_outlier_check"]
    
        # Loaded data
        self.signal_raw = None
        self.events = None
        self.parameters = None
    
        # Filtered signals used for event-level quality control
        
        self.left_filtered = None
        self.right_filtered = None
    
        # Metadata extracted from the events table
        self.patient_id = None
        self.recording_date = None
        self.session_id = None
        self.output_folder = None
    
        # Output paths
        self.saved_paths = {}

   

    @classmethod
    def _build_config(cls, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
           Build the final configuration dictionary.
        
           User-defined values are used to override selected keys in `DEFAULT_CONFIG`.
           Unknown configuration keys are rejected to avoid silent mistakes.
        
           Parameters
           ----------
           user_config : dict, optional
               Dictionary containing configuration values to override.
        
           Returns
           -------
           dict
               Validated configuration dictionary.
           """
        user_config = user_config or {}

        unknown_keys = set(user_config) - set(cls.DEFAULT_CONFIG)
        if unknown_keys:
            raise ValueError(f"Unknown configuration key(s): {sorted(unknown_keys)}")

        config = deepcopy(cls.DEFAULT_CONFIG)
        config.update(user_config)

        cls._validate_config(config)

        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate quality-check configuration values.
     
        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.
     
        Raises
        ------
        ValueError
            If one or more configuration values are invalid.
        """
        if config["sampling_rate_hz"] <= 0:
            raise ValueError("sampling_rate_hz must be greater than 0.")

        if config["cutoff_freq_gyr"] <= 0:
            raise ValueError("cutoff_freq_gyr must be greater than 0.")

        if config["filter_order_gyr"] <= 0:
            raise ValueError("filter_order_gyr must be greater than 0.")

        if len(config["turning_angle_abs_range"]) != 2:
            raise ValueError("turning_angle_abs_range must be a tuple like (25.0, 90.0).")

        if not isinstance(config["parameter_rules"], dict):
            raise ValueError("parameter_rules must be a dictionary.")

        if config["n_parameter_violations"] <= 0:
            raise ValueError("n_parameter_violations must be greater than 0.")

        if not isinstance(config["apply_events_ic_check"], bool):
            raise ValueError("apply_events_ic_check must be True or False.")

        if not isinstance(config["apply_turning_angle_check"], bool):
            raise ValueError("apply_turning_angle_check must be True or False.")

        if not isinstance(config["apply_parameter_outlier_check"], bool):
            raise ValueError("apply_parameter_outlier_check must be True or False.")

    

    @staticmethod
    def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnamed columns created during CSV import.
        
        This helper supports both regular column indexes and MultiIndex columns.
        It is mainly used to remove index columns accidentally saved by pandas,
        such as `Unnamed: 0`.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.
        
        Returns
        -------
        pandas.DataFrame
            Dataframe without unnamed columns.
        """
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

    @staticmethod
    def format_recording_date(recording_date) -> str:
        """
        Convert a recording date to YYYY-MM-DD format.
    
        Parameters
        ----------
        recording_date
            Input recording date. Any value accepted by `pandas.to_datetime`
            can be used.
    
        Returns
        -------
        str
            Formatted recording date.
        """
        return pd.to_datetime(recording_date, dayfirst=True).strftime("%Y-%m-%d")

    @staticmethod
    def _count_pass_fail(df: pd.DataFrame, quality_col: str) -> Dict[str, int]:
        """
        Count passed and failed rows in a quality-check column.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing a boolean quality-check column.
    
        quality_col : str
            Name of the quality-check column.
    
        Returns
        -------
        dict
            Dictionary with total, passed, and failed row counts.
        """
        if df is None or df.empty or quality_col not in df.columns:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
            }

        passed = df[quality_col] == True
        failed = df[quality_col] == False

        return {
            "total": int(len(df)),
            "passed": int(passed.sum()),
            "failed": int(failed.sum()),
        }

    # ============================================================
    # LOAD INPUTS
    # ============================================================

    def load_signal(self) -> pd.DataFrame:
        """
        Load the IMU signal used for event-level quality control.
    
        If a filtered signal dataframe was provided at initialization, it is used
        directly. Otherwise, the signal is loaded from `signal_path`.
        """
    
        if self.signal_filtered is not None:
            return self.signal_filtered.copy()
    
        if self.signal_path is None:
            raise ValueError(
                "signal_path is None and no signal_filtered dataframe was provided."
            )
    
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

    def load_events(self) -> pd.DataFrame:
        if not self.events_path.exists():
            raise FileNotFoundError(f"Events file not found: {self.events_path}")

        events = pd.read_csv(self.events_path)
        events = self.drop_unnamed_columns(events)

        required_cols = [
            "patient_id",
            "recording_date",
            "session_id",
            "s_id",
            "foot",
            "start",
            "end",
            "ic",
            "tc",
            "min_vel",
            "pre_ic",
        ]

        missing_cols = [
            col for col in required_cols
            if col not in events.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns in events CSV: {missing_cols}")

        numeric_cols = ["s_id", "start", "end", "ic", "tc", "min_vel", "pre_ic"]

        for col in numeric_cols:
            events[col] = pd.to_numeric(events[col], errors="coerce")

        events = events.dropna(
            subset=[
                "patient_id",
                "recording_date",
                "session_id",
                "s_id",
                "foot",
                "ic",
            ]
        ).copy()

        for col in numeric_cols:
            events[col] = events[col].astype(int)

        events["foot"] = events["foot"].astype(str).str.lower()

        unique_patients = events["patient_id"].dropna().unique()
        unique_dates = events["recording_date"].dropna().unique()
        unique_sessions = events["session_id"].dropna().unique()

        if len(unique_patients) != 1:
            raise ValueError(f"Expected one patient_id, found: {unique_patients}")

        if len(unique_dates) != 1:
            raise ValueError(f"Expected one recording_date, found: {unique_dates}")

        if len(unique_sessions) != 1:
            raise ValueError(f"Expected one session_id, found: {unique_sessions}")

        self.patient_id = str(unique_patients[0])
        self.recording_date = self.format_recording_date(unique_dates[0])
        self.session_id = str(unique_sessions[0])

        self.output_folder = (
            self.project_folder
            / self.patient_id
            / self.recording_date
            / self.session_id
        )

        return events

    def _get_parameters_path(self) -> Path:
        if self.parameters_path is not None:
            if not self.parameters_path.exists():
                raise FileNotFoundError(f"Parameters file not found: {self.parameters_path}")

            return self.parameters_path

        if self.output_folder is None:
            raise RuntimeError("output_folder is None. Run load_events() first.")

        file_prefix = f"{self.patient_id}_{self.session_id}_{self.recording_date}"

        parameters_path = self.output_folder / f"{file_prefix}_parameters.csv"

        if not parameters_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {parameters_path}")

        return parameters_path

    def load_parameters(self) -> pd.DataFrame:
        parameters_path = self._get_parameters_path()

        parameters = pd.read_csv(parameters_path)
        parameters = self.drop_unnamed_columns(parameters)

        required_cols = [
            "patient_id",
            "recording_date",
            "session_id",
            "s_id",
            "foot",
        ]

        missing_cols = [
            col for col in required_cols
            if col not in parameters.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns in parameters CSV: {missing_cols}")

        parameters["s_id"] = pd.to_numeric(parameters["s_id"], errors="coerce")
        parameters = parameters.dropna(subset=["s_id", "foot"]).copy()
        parameters["s_id"] = parameters["s_id"].astype(int)
        parameters["foot"] = parameters["foot"].astype(str).str.lower()

        return parameters

    def load_inputs(self):
        """
        Load all input files required by the quality-check pipeline.
    
        This method loads:
        - the raw IMU signal file,
        - the gait-event CSV file,
        - the stride-level parameter CSV file.
    
        The loaded data are stored as class attributes:
    
        - `self.signal_raw`
        - `self.events`
        - `self.parameters`
    
        Returns
        -------
        self
            The quality-check instance, allowing method chaining.
        """
        self.signal_raw = self.load_signal()
        self.events = self.load_events()
        self.parameters = self.load_parameters()

        return self

    # ============================================================
    # FILTERING
    # ============================================================

    def get_signal_column(self, side: str, channel: str) -> pd.Series:
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

    def filter_signal(self):
        """
        Prepare left and right filtered signals for event-level quality control.
    
        If a filtered signal dataframe was provided, the selected channel is
        extracted directly. Otherwise, the raw signal is filtered first.
        """
    
        if self.signal_raw is None:
            raise RuntimeError("signal_raw is None. Run load_inputs() first.")
    
        left_signal = self.get_signal_column("left_sensor", self.channel)
        right_signal = self.get_signal_column("right_sensor", self.channel)
    
        if self.signal_filtered is not None:
            self.left_filtered = left_signal.to_numpy()
            self.right_filtered = right_signal.to_numpy()
        else:
            self.left_filtered = self.butter_lowpass_filter(
                data=left_signal,
                cutoff_freq=self.cutoff_freq_gyr,
                filter_order=self.filter_order_gyr,
                fs=self.fs,
            )
    
            self.right_filtered = self.butter_lowpass_filter(
                data=right_signal,
                cutoff_freq=self.cutoff_freq_gyr,
                filter_order=self.filter_order_gyr,
                fs=self.fs,
            )
    
        return self

    def get_ic_signal_value(self, foot: str, ic_sample: int) -> float:
        foot = str(foot).lower()

        if self.left_filtered is None or self.right_filtered is None:
            raise RuntimeError("Filtered signals are missing. Run filter_signal() first.")

        if foot == "left":
            signal = self.left_filtered

        elif foot == "right":
            signal = self.right_filtered

        else:
            return np.nan

        if ic_sample < 0 or ic_sample >= len(signal):
            return np.nan

        return float(signal[ic_sample])

    # ============================================================
    # EVENTS QUALITY CHECK
    # ============================================================

    def label_events_quality_check(self):
        """
        Add:
            quality_check(IC<0)
            notes

        True means the event passed.
        False means the event failed because filtered gyr_ml at IC > threshold.
        """

        if self.events is None:
            raise RuntimeError("events is None. Run load_inputs() first.")

        if self.left_filtered is None or self.right_filtered is None:
            raise RuntimeError("Filtered signals are missing. Run filter_signal() first.")

        events = self.events.copy()

        # Remove previous columns if rerunning.
        for col in [self.events_quality_col, self.notes_col]:
            if col in events.columns:
                events = events.drop(columns=[col])

        quality_values = []
        notes_values = []

        for _, row in events.iterrows():
            foot = row["foot"]
            ic_sample = int(row["ic"])

            if not self.apply_events_ic_check:
                quality_values.append(True)
                notes_values.append("")
                continue

            ic_value = self.get_ic_signal_value(
                foot=foot,
                ic_sample=ic_sample,
            )

            failed = ic_value > self.ic_threshold

            if failed:
                quality_values.append(False)
                notes_values.append(
                    f"filtered {self.channel} at IC sample {ic_sample} is > {self.ic_threshold}"
                )
            else:
                quality_values.append(True)
                notes_values.append("")

        events[self.events_quality_col] = quality_values
        events[self.notes_col] = notes_values

        self.events = events

        return self

    # ============================================================
    # PARAMETERS QUALITY CHECK
    # ============================================================

    def label_parameters_quality_check(self):
        """
        Add:
            quality_check
            notes

        True means the parameter row passed.
        False means the parameter row failed at least one selected rule.
        """

        if self.parameters is None:
            raise RuntimeError("parameters is None. Run load_inputs() first.")

        params = self.parameters.copy()

        # Remove previous columns if rerunning.
        for col in ["quality_check", self.notes_col]:
            if col in params.columns:
                params = params.drop(columns=[col])

        quality_values = []
        notes_values = []

        missing_rule_cols = [
            col for col in self.parameter_rules
            if col not in params.columns
        ]

        if self.apply_parameter_outlier_check and missing_rule_cols:
            raise ValueError(f"Missing parameter rule columns: {missing_rule_cols}")

        turning_col = "turning angle [deg]"

        if self.apply_turning_angle_check and turning_col not in params.columns:
            raise ValueError(f"Missing required column in parameters: {turning_col}")

        # Precompute parameter-rule violations.
        violations_df = pd.DataFrame(index=params.index)
        violation_count = pd.Series(0, index=params.index, dtype=int)

        if self.apply_parameter_outlier_check:
            for col, (low, high) in self.parameter_rules.items():
                params[col] = pd.to_numeric(params[col], errors="coerce")

                violations_df[col] = (
                    params[col] < low
                ) | (
                    params[col] > high
                )

            violation_count = violations_df.sum(axis=1)

        # Precompute turning-angle violations.
        turning_failed = pd.Series(False, index=params.index)

        if self.apply_turning_angle_check:
            low_turn, high_turn = self.turning_angle_abs_range

            params[turning_col] = pd.to_numeric(params[turning_col], errors="coerce")

            abs_turning = params[turning_col].abs()

            turning_failed = (
                abs_turning >= low_turn
            ) & (
                abs_turning <= high_turn
            )

        for idx, row in params.iterrows():
            notes = []

            if self.apply_turning_angle_check and bool(turning_failed.loc[idx]):
                notes.append(
                    f"abs(turning angle [deg]) is between {self.turning_angle_abs_range[0]} and {self.turning_angle_abs_range[1]}"
                )

            if self.apply_parameter_outlier_check:
                if int(violation_count.loc[idx]) >= self.n_parameter_violations:
                    violated_cols = [
                        col for col in self.parameter_rules
                        if bool(violations_df.loc[idx, col])
                    ]

                    rule_text = "; ".join(
                        [
                            f"{col} outside {self.parameter_rules[col]}"
                            for col in violated_cols
                        ]
                    )

                    notes.append(
                        f"{int(violation_count.loc[idx])} parameter violations: {rule_text}"
                    )

            passed = len(notes) == 0

            quality_values.append(passed)
            notes_values.append(" | ".join(notes) if notes else "")

        params["quality_check"] = quality_values
        params[self.notes_col] = notes_values

        self.parameters = params

        return self

    # ============================================================
    # LOG
    # ============================================================

    def _get_log_path(self) -> Path:
        events_name = self.events_path.name
        log_name = events_name.replace("_events.csv", "_log.json")
        log_path = self.events_path.parent / log_name
    
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
    
        return log_path

    def build_quality_check_summary(self) -> Dict[str, Any]:
        events_summary = self._count_pass_fail(
            self.events,
            self.events_quality_col,
        )

        parameters_summary = self._count_pass_fail(
            self.parameters,
            "quality_check",
        )

        return {
            "QualityCheck": {
                "description": "Events and parameters were labelled with quality_check columns. No rows were removed.",
                "events_quality_column": self.events_quality_col,
                "parameters_quality_column": "quality_check",
                "notes_column": self.notes_col,
                "events_ic_rule": f"event passes if filtered {self.channel} at IC <= {self.ic_threshold}",
                "turning_angle_rule": (
                    f"parameter row fails if abs(turning angle [deg]) is between "
                    f"{self.turning_angle_abs_range[0]} and {self.turning_angle_abs_range[1]}"
                ),
                "parameter_rules": self.parameter_rules,
                "n_parameter_violations": self.n_parameter_violations,
                "sampling_rate_hz": self.fs,
                "filter_channel": self.channel,
                "cutoff_freq_gyr": self.cutoff_freq_gyr,
                "filter_order_gyr": self.filter_order_gyr,
            },
            "events": {
                "total": events_summary["total"],
                "passed": events_summary["passed"],
                "failed": events_summary["failed"],
            },
            "parameters": {
                "total": parameters_summary["total"],
                "passed": parameters_summary["passed"],
                "failed": parameters_summary["failed"],
            },
        }

    def update_log_file(self):
        log_path = self._get_log_path()

        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)

        log["quality_check_summary"] = self.build_quality_check_summary()

        if "events" not in log:
            log["events"] = []

        log["events"].append(
            "QualityCheck: events and parameters labelled with quality_check columns. No rows removed."
        )

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, default=str)

        self.saved_paths["updated_log"] = log_path

        return log_path

    def propagate_event_quality_to_parameters(self):
        """
        Propagate failed event-level QC to stride-level parameters.
        
        If an event has quality_check(IC<0) == False, then the parameter row
        with the same s_id is also marked as quality_check == False.
        
        This does not remove rows. It only updates:
            - quality_check
            - notes
        """
        
        if self.events is None:
            raise RuntimeError("events is None. Run label_events_quality_check() first.")
        
        if self.parameters is None:
            raise RuntimeError("parameters is None. Run label_parameters_quality_check() first.")
        
        event_qc_col = self.events_quality_col
        param_qc_col = "quality_check"
        notes_col = self.notes_col
        
        if event_qc_col not in self.events.columns:
            raise ValueError(f"Missing event quality column: {event_qc_col}")
        
        if param_qc_col not in self.parameters.columns:
            raise ValueError(f"Missing parameter quality column: {param_qc_col}")
        
        if "s_id" not in self.events.columns:
            raise ValueError("Missing s_id column in events.")
        
        if "s_id" not in self.parameters.columns:
            raise ValueError("Missing s_id column in parameters.")
        
        events = self.events.copy()
        params = self.parameters.copy()
        
        events["s_id"] = pd.to_numeric(events["s_id"], errors="coerce")
        params["s_id"] = pd.to_numeric(params["s_id"], errors="coerce")
        
        failed_event_s_ids = set(
            events.loc[
                events[event_qc_col] == False,
                "s_id"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )
        
        if not failed_event_s_ids:
            self.parameters = params
            return self
        
        affected = params["s_id"].astype("Int64").isin(failed_event_s_ids)
        
        params.loc[affected, param_qc_col] = False
        
        propagation_note = f"associated event failed {event_qc_col}"
        
        def add_note(existing_note):
            if pd.isna(existing_note) or str(existing_note).strip() == "":
                return propagation_note
        
            existing_note = str(existing_note)
        
            if propagation_note in existing_note:
                return existing_note
        
            return f"{existing_note} | {propagation_note}"
        
        params.loc[affected, notes_col] = params.loc[affected, notes_col].apply(add_note)
        
        self.parameters = params
        
        return self
    # ============================================================
    # SAVE
    # ============================================================

    def save_outputs(self):
        if self.events is None:
            raise RuntimeError("events is None.")
    
        if self.parameters is None:
            raise RuntimeError("parameters is None.")
    
        events_path = self.events_path
        parameters_path = self._get_parameters_path()
    
        self.events.to_csv(events_path, index=False)
        self.parameters.to_csv(parameters_path, index=False)
    
        self.saved_paths = {
            "events_labelled": events_path,
            "parameters_labelled": parameters_path,
        }
    
        self.update_log_file()
    
        return self.saved_paths

    # ============================================================
    # RUN
    # ============================================================

    def run(self):
        self.load_inputs()
        self.filter_signal()
        self.label_events_quality_check()
        self.label_parameters_quality_check()
        self.propagate_event_quality_to_parameters()
        print("QC patient_id:", self.patient_id)
        print("QC session_id:", self.session_id)
        print("QC recording_date:", self.recording_date)
        print("QC output_folder:", self.output_folder)
        print("QC events_path input:", self.events_path)
        print("QC parameters_path input:", self.parameters_path)
        self.save_outputs()

        return self


