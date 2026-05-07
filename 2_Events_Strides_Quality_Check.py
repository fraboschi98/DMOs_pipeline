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
       - quality_check(IC>0)
       - notes

   Events rule:
       - if foot == left, check filtered left_sensor gyr_ml at IC sample
       - if foot == right, check filtered right_sensor gyr_ml at IC sample
       - quality_check(IC>0) = False if filtered gyr_ml at IC > 0
       - quality_check(IC>0) = True otherwise

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

import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class QualityCheck:
    DEFAULT_CONFIG = {
        # ---- Signal ----
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",

        # ---- Filtering ----
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,

        # ---- Events IC rule ----
        "ic_threshold": 0.0,
        "events_quality_col": "quality_check(IC>0)",
        "notes_col": "notes",

        # ---- Parameters turning angle rule ----
        "turning_angle_abs_range": (25.0, 90.0),

        # ---- Parameters outlier rules ----
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
        signal_path: str,
        events_path: str,
        project_folder: str,
        parameters_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.signal_path = Path(signal_path)
        self.events_path = Path(events_path)
        self.project_folder = Path(project_folder)
        self.parameters_path = Path(parameters_path) if parameters_path is not None else None

        self.config = self._build_config(config)

        self.fs = self.config["sampling_rate_hz"]
        self.channel = self.config["channel"]
        self.cutoff_freq_gyr = self.config["cutoff_freq_gyr"]
        self.filter_order_gyr = self.config["filter_order_gyr"]

        self.ic_threshold = self.config["ic_threshold"]

        self.events_quality_col = self.config["events_quality_col"]
        self.notes_col = self.config["notes_col"]

        self.turning_angle_abs_range = self.config["turning_angle_abs_range"]
        self.parameter_rules = self.config["parameter_rules"]
        self.n_parameter_violations = self.config["n_parameter_violations"]

        self.apply_events_ic_check = self.config["apply_events_ic_check"]
        self.apply_turning_angle_check = self.config["apply_turning_angle_check"]
        self.apply_parameter_outlier_check = self.config["apply_parameter_outlier_check"]

        self.signal_raw = None
        self.events = None
        self.parameters = None

        self.left_filtered = None
        self.right_filtered = None

        self.patient_id = None
        self.recording_date = None
        self.session_id = None
        self.output_folder = None

        self.saved_paths = {}

    # ============================================================
    # CONFIG
    # ============================================================

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

    # ============================================================
    # HELPERS
    # ============================================================

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

    @staticmethod
    def format_recording_date(recording_date) -> str:
        return pd.to_datetime(recording_date, dayfirst=True).strftime("%Y-%m-%d")

    @staticmethod
    def _count_pass_fail(df: pd.DataFrame, quality_col: str) -> Dict[str, int]:
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
        if self.signal_raw is None:
            raise RuntimeError("signal_raw is None. Run load_inputs() first.")

        left_raw = self.get_signal_column("left_sensor", self.channel)
        right_raw = self.get_signal_column("right_sensor", self.channel)

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
            quality_check(IC>0)
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
        print("QC patient_id:", self.patient_id)
        print("QC session_id:", self.session_id)
        print("QC recording_date:", self.recording_date)
        print("QC output_folder:", self.output_folder)
        print("QC events_path input:", self.events_path)
        print("QC parameters_path input:", self.parameters_path)
        self.save_outputs()

        return self


# ============================================================
# DEBUG MAIN
# ============================================================

if __name__ == "__main__":

    signal_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_gaitMAP_bf_all.csv"
    )

    events_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_events.csv"
    )

    parameters_path = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3"
        r"\PAT401_week_3_2023-07-10_parameters.csv"
    )

    project_folder = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)"
        r"\MobilityAPP_Pipeline\Prova"
    )

    user_config = {
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,

        # Events IC rule
        "ic_threshold": 0.0,
        "events_quality_col": "quality_check(IC>0)",
        "notes_col": "notes",

        # Parameters turning angle rule
        "turning_angle_abs_range": (25.0, 90.0),

        # Parameters outlier rules
        "parameter_rules": {
            "stride time [s]": (0.2, 3.0),
            "gait velocity [m/s]": (0.2, 2.0),
            "stride length [m]": (0.10, 1.5),
        },
        "n_parameter_violations": 2,

        # Switches
        "apply_events_ic_check": True,
        "apply_turning_angle_check": True,
        "apply_parameter_outlier_check": True,
    }

    quality_check = QualityCheck(
        signal_path=signal_path,
        events_path=events_path,
        parameters_path=parameters_path,
        project_folder=project_folder,
        config=user_config,
    )

    quality_check.run()

    print("Events quality check:")
    print(quality_check.events["quality_check(IC>0)"].value_counts(dropna=False))

    print("\nParameters quality check:")
    print(quality_check.parameters["quality_check"].value_counts(dropna=False))

    print("\nSaved files:")
    print(quality_check.saved_paths)