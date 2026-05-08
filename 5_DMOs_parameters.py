# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:53:55 2026

@author: francesca.boschi

DMOs class:
    1. collect wb, pauses, parameters by PATIENT and DAY
    2. label wb 
    3. Stride-Performance: map paramters to wb, remove not assigned s_id, or strides with False quality check
    4: WB-level: average strides on the wb
    5: WB-level: Cadence, PA state
    
    6: Day level: avg on short, medium, long WB of stride-performance parameters
    7: Day level: Mobility -> strides count etc
    8: Day level: Complexity
    9: KDE: single day or interval of days, -> KDEs + mode, median, mean p95 (min 10WB)
    class collector
    class wb level: labelling, mapping, cadence, pa
    class day: stride performance avg, macro dmos
    class kde
    class complexity
"""


from pathlib import Path
import pandas as pd
import json
import ast
import numpy as np
class CollectorCSV:
    def __init__(self, patient_id, date, patient_directory):
        self.patient_id = patient_id
        self.date = date
        self.patient_directory = Path(patient_directory)
        self.date_folder = self.patient_directory / self.date

        self.parameters = pd.DataFrame()
        self.wb_dataframe = pd.DataFrame()
        self.wb_pauses_dataframe = pd.DataFrame()
        self.log = []

    def collect(self):
        if not self.date_folder.exists():
            raise FileNotFoundError(f"Date folder not found: {self.date_folder}")

        parameters_list = []
        wb_list = []
        pauses_list = []
        log_list = []

        for session_folder in self.date_folder.iterdir():
            if not session_folder.is_dir():
                continue

            session_id = session_folder.name
            base_name = f"{self.patient_id}_{session_id}_{self.date}"

            parameters_file = session_folder / f"{base_name}_parameters.csv"
            log_file = session_folder / f"{base_name}_log.json"
            wb_file = session_folder / f"{base_name}_wb_dataframe.csv"
            pauses_file = session_folder / f"{base_name}_wb_pauses_dataframe.csv"

            if parameters_file.exists():
                df = pd.read_csv(parameters_file)
                df["session_id"] = session_id
                parameters_list.append(df)

            if wb_file.exists():
                df = pd.read_csv(wb_file)
                df["session_id"] = session_id
                wb_list.append(df)

            if pauses_file.exists():
                df = pd.read_csv(pauses_file)
                df["session_id"] = session_id
                pauses_list.append(df)

            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    log_data = json.load(f)

                log_list.append({
                    "patient_id": self.patient_id,
                    "date": self.date,
                    "session_id": session_id,
                    "log": log_data,
                })

        if parameters_list:
            self.parameters = pd.concat(parameters_list, ignore_index=True)

        if wb_list:
            self.wb_dataframe = pd.concat(wb_list, ignore_index=True)

        if pauses_list:
            self.wb_pauses_dataframe = pd.concat(pauses_list, ignore_index=True)

        self.log = log_list

        return self

    def summary(self):
        print(f"Patient: {self.patient_id}")
        print(f"Date: {self.date}")
        print(f"Folder: {self.date_folder}")

        print("\nLoaded data:")
        print(f"  parameters rows: {len(self.parameters)}")
        print(f"  wb_dataframe rows: {len(self.wb_dataframe)}")
        print(f"  wb_pauses_dataframe rows: {len(self.wb_pauses_dataframe)}")
        print(f"  logs loaded: {len(self.log)}")
class WalkingBouts:
    def __init__(self, parameters, wb_dataframe, wb_pauses_dataframe):
        self.parameters = parameters.copy()
        self.wb_dataframe = wb_dataframe.copy()
        self.wb_pauses_dataframe = wb_pauses_dataframe.copy()

        self.parameters_before_cleaning = pd.DataFrame()
        self.wb_parameters_average = pd.DataFrame()

    def add_wb_label(self):
        """
        Adds WB_label to wb_dataframe using duration_s.

        duration_s < 10          -> very_short
        10 <= duration_s <= 30   -> short
        30 < duration_s <= 60    -> medium
        duration_s > 60          -> long
        """

        if "duration_s" not in self.wb_dataframe.columns:
            raise KeyError("Column 'duration_s' not found in wb_dataframe")

        duration = self.wb_dataframe["duration_s"]

        self.wb_dataframe["WB_label"] = pd.NA
        self.wb_dataframe.loc[duration < 10, "WB_label"] = "very_short"
        self.wb_dataframe.loc[(duration >= 10) & (duration <= 30), "WB_label"] = "short"
        self.wb_dataframe.loc[(duration > 30) & (duration <= 60), "WB_label"] = "medium"
        self.wb_dataframe.loc[duration > 60, "WB_label"] = "long"

        return self

    def _parse_s_ids(self, value):
        """
        Converts left_s_ids / right_s_ids into a Python list.
        """

        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            value = value.strip()

            if value == "":
                return []

            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except Exception:
                return [x.strip() for x in value.split(",") if x.strip() != ""]

        return [value]

    def assign_wb_id_to_parameters(self):
        """
        Adds original session-level WB_id to parameters.
        Matching is done by:
        patient_id, recording_date, session_id, foot, s_id.
        """

        required_parameters_cols = [
            "patient_id",
            "recording_date",
            "session_id",
            "s_id",
            "foot",
        ]

        required_wb_cols = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "left_s_ids",
            "right_s_ids",
        ]

        for col in required_parameters_cols:
            if col not in self.parameters.columns:
                raise KeyError(f"Column '{col}' not found in parameters")

        for col in required_wb_cols:
            if col not in self.wb_dataframe.columns:
                raise KeyError(f"Column '{col}' not found in wb_dataframe")

        self.parameters["WB_id"] = pd.NA

        for _, wb_row in self.wb_dataframe.iterrows():
            patient_id = wb_row["patient_id"]
            recording_date = wb_row["recording_date"]
            session_id = wb_row["session_id"]
            wb_id = wb_row["WB_id"]

            left_s_ids = self._parse_s_ids(wb_row["left_s_ids"])
            right_s_ids = self._parse_s_ids(wb_row["right_s_ids"])

            left_mask = (
                (self.parameters["patient_id"] == patient_id)
                & (self.parameters["recording_date"] == recording_date)
                & (self.parameters["session_id"] == session_id)
                & (self.parameters["foot"].astype(str).str.lower() == "left")
                & (self.parameters["s_id"].isin(left_s_ids))
            )

            right_mask = (
                (self.parameters["patient_id"] == patient_id)
                & (self.parameters["recording_date"] == recording_date)
                & (self.parameters["session_id"] == session_id)
                & (self.parameters["foot"].astype(str).str.lower() == "right")
                & (self.parameters["s_id"].isin(right_s_ids))
            )

            self.parameters.loc[left_mask | right_mask, "WB_id"] = wb_id

        return self


    def clean_parameters(self, use_quality_check=True):
        """
        Cleans parameters after WB_id assignment.
    
        Saves:
        self.parameters_before_cleaning
    
        Then removes:
        1. rows not assigned to a WB_id
        2. rows with quality_check == False, if use_quality_check=True
        """
    
        if "WB_id" not in self.parameters.columns:
            raise KeyError(
                "Column 'WB_id' not found in parameters. "
                "Run assign_wb_id_to_parameters() before clean_parameters()."
            )
    
        self.parameters = self.parameters.dropna(subset=["WB_id"]).copy()
    
        self.parameters_before_cleaning = self.parameters.copy()
    
        if use_quality_check:
            if "quality_check" not in self.parameters.columns:
                raise KeyError("Column 'quality_check' not found in parameters")
    
            quality = self.parameters["quality_check"]
    
            if quality.dtype == bool:
                self.parameters = self.parameters[quality].copy()
            else:
                self.parameters = self.parameters[
                    quality.astype(str).str.lower().isin(["true", "1", "yes"])
                ].copy()
    
        self.parameters = self.parameters.reset_index(drop=True)
    
        return self
    def _compute_n_strides(self, df, output_column):
        """
        Computes max number of unique s_id between left and right foot.
        """
    
        group_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
        ]
    
        required_columns = group_columns + ["foot", "s_id"]
    
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataframe")
    
        n_strides = (
            df
            .groupby(group_columns + ["foot"])["s_id"]
            .nunique()
            .reset_index(name="n_strides_foot")
        )
    
        n_strides = (
            n_strides
            .groupby(group_columns, as_index=False)["n_strides_foot"]
            .max()
            .rename(columns={"n_strides_foot": output_column})
        )
    
        return n_strides
    def compute_wb_parameters_average(self):
        """
        Computes WB-level average gait parameters.
    
        Output:
        - patient_id
        - recording_date
        - session_id
        - WB_id
        - start
        - end
        - WB_label
        - duration_s
        - n_strides_total
        - n_strides_valid
        - cadence
        -PA type
        - average gait parameters after cleaning
        """
    
        if self.parameters_before_cleaning.empty:
            raise ValueError(
                "parameters_before_cleaning is empty. "
                "Run clean_parameters() before compute_wb_parameters_average()."
            )
    
        if "WB_label" not in self.wb_dataframe.columns:
            raise KeyError("Column 'WB_label' not found in wb_dataframe")
    
        columns_to_average = [
            "stride time [s]",
            "swing time [s]",
            "stance time [s]",
            "percent_stance_phase",
            "percent_swing_phase",
            "arc length [m]",
            "gait velocity [m/s]",
            "ic angle [deg]",
            "max. lateral excursion [m]",
            "max. orientation change [deg]",
            "max. sensor lift [m]",
            "stride length [m]",
            "tc angle [deg]",
            "turning angle [deg]",
        ]
    
        group_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
        ]
    
        required_columns = group_columns + columns_to_average + ["foot", "s_id"]
    
        for col in required_columns:
            if col not in self.parameters.columns:
                raise KeyError(f"Column '{col}' not found in parameters")
    
        wb_average = (
            self.parameters
            .groupby(group_columns, as_index=False)[columns_to_average]
            .mean()
        )
    
        n_strides_total = self._compute_n_strides(
            self.parameters_before_cleaning,
            output_column="n_strides_total",
        )
    
        n_strides_valid = self._compute_n_strides(
            self.parameters,
            output_column="n_strides_valid",
        )
    
        wb_info_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "start",
            "end",
            "duration_s",
            "WB_label",
            "cadence_[steps_per_min]",
            "PA_type"
        ]
    
        for col in wb_info_columns:
            if col not in self.wb_dataframe.columns:
                raise KeyError(f"Column '{col}' not found in wb_dataframe")
    
        wb_info = (
            self.wb_dataframe[wb_info_columns]
            .drop_duplicates(subset=group_columns)
        )
    
        self.wb_parameters_average = (
            wb_average
            .merge(n_strides_total, on=group_columns, how="left")
            .merge(n_strides_valid, on=group_columns, how="left")
            .merge(wb_info, on=group_columns, how="left")
        )
    
        ordered_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "start",
            "end",
            "duration_s",
            "WB_label",
            "n_strides_total",
            "n_strides_valid",
            "cadence_[steps_per_min]",
            "PA_type"
        ] + columns_to_average
    
        self.wb_parameters_average = self.wb_parameters_average[ordered_columns]
    
        return self

    def add_cadence_to_wb_dataframe(self, use_valid_strides=False):
        """
        Adds cadence to wb_dataframe.
    
        Cadence is defined as steps per minute.
    
        steps = n_strides * 2
        cadence = steps / duration_s * 60
    
        If use_valid_strides=False:
            cadence is computed using parameters_before_cleaning
            -> all strides assigned to the WB before quality filtering.
    
        If use_valid_strides=True:
            cadence is computed using parameters
            -> only valid strides after quality filtering.
        """
    
        required_wb_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "duration_s",
        ]
    
        for col in required_wb_columns:
            if col not in self.wb_dataframe.columns:
                raise KeyError(f"Column '{col}' not found in wb_dataframe")
    
        if use_valid_strides:
            stride_source = self.parameters
            stride_column = "n_strides_valid"
        else:
            if self.parameters_before_cleaning.empty:
                raise ValueError(
                    "parameters_before_cleaning is empty. "
                    "Run clean_parameters() before add_cadence_to_wb_dataframe()."
                )
    
            stride_source = self.parameters_before_cleaning
            stride_column = "n_strides_total"
    
        n_strides = self._compute_n_strides(
            stride_source,
            output_column=stride_column,
        )
    
        group_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
        ]
    
        self.wb_dataframe = self.wb_dataframe.merge(
            n_strides,
            on=group_columns,
            how="left",
        )
    
        self.wb_dataframe["cadence_[steps_per_min]"] = pd.NA
    
        valid_duration = self.wb_dataframe["duration_s"] > 0
    
        self.wb_dataframe.loc[valid_duration, "cadence_[steps_per_min]"] = (
            self.wb_dataframe.loc[valid_duration, stride_column]
            * 2
            / self.wb_dataframe.loc[valid_duration, "duration_s"]
            * 60
        )
    
        return self

    def add_stance_swing_percentages(
        self,
        stance_col="stance time [s]",
        swing_col="swing time [s]",
        stride_col="stride time [s]",
        stance_percent_col="percent_stance_phase",
        swing_percent_col="percent_swing_phase"
    ):
        """
        Adds stance and swing percentages to parameters.
    
        percent_stance_phase = stance time / stride time * 100
        percent_swing_phase  = swing time / stride time * 100
        """
    
        required_cols = [stance_col, swing_col, stride_col]
    
        for col in required_cols:
            if col not in self.parameters.columns:
                raise KeyError(f"Column '{col}' not found in parameters")
    
        stride = self.parameters[stride_col]
    
        self.parameters[stance_percent_col] = np.where(
            stride > 0,
            self.parameters[stance_col] / stride * 100,
            np.nan
        )
    
        self.parameters[swing_percent_col] = np.where(
            stride > 0,
            self.parameters[swing_col] / stride * 100,
            np.nan
        )
    
        return self
    def add_pa_type_to_wb_dataframe(self, cadence_col="cadence_[steps_per_min]"):
        """
        Adds PA_type to wb_dataframe based on cadence.
    
        PA_type:
        - MVPA if cadence > 90
        - LPA  if cadence <= 90
        """
    
        if cadence_col not in self.wb_dataframe.columns:
            raise KeyError(
                f"Column '{cadence_col}' not found in wb_dataframe. "
                "Run add_cadence_to_wb_dataframe() before add_pa_type_to_wb_dataframe()."
            )
    
        self.wb_dataframe["PA_type"] = pd.NA
    
        self.wb_dataframe.loc[
            self.wb_dataframe[cadence_col] > 90,
            "PA_type"
        ] = "MVPA"
    
        self.wb_dataframe.loc[
            self.wb_dataframe[cadence_col] <= 90,
            "PA_type"
        ] = "LPA"
    
        return self   
class DayLevelDMOs:
    def __init__(self, wb_parameters_average, log=None):
        self.wb_parameters_average = wb_parameters_average.copy()
        self.log = log if log is not None else []
        self.day_dmos = pd.DataFrame()

    def count_wb_by_label(self):
        """
        Computes day-level number of walking bouts by WB_label.
    
        Output columns:
        - patient_id
        - recording_date
        - n_very_short_WB
        - n_short_WB
        - n_medium_WB
        - n_long_WB
        """
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "WB_label",
        ]
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        wb_counts = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .groupby(
                [
                    "patient_id",
                    "recording_date",
                    "WB_label",
                ]
            )
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
    
        expected_labels = [
            "very_short",
            "short",
            "medium",
            "long",
        ]
    
        for label in expected_labels:
            if label not in wb_counts.columns:
                wb_counts[label] = 0
    
        wb_counts = wb_counts.rename(
            columns={
                "very_short": "n_very_short_WB",
                "short": "n_short_WB",
                "medium": "n_medium_WB",
                "long": "n_long_WB",
            }
        )
    
        ordered_columns = [
            "patient_id",
            "recording_date",
            "n_very_short_WB",
            "n_short_WB",
            "n_medium_WB",
            "n_long_WB",
        ]
    
        self.day_dmos = wb_counts[ordered_columns]
    
        return self

    def count_total_strides(self):
        """
        Computes day-level total number of strides.
    
        Starting from wb_parameters_average:
        - sums n_strides_total across all walking bouts of the same patient and day
    
        Output column:
        - n_strides_total_day
        """
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "n_strides_total",
        ]
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        strides_day = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .groupby(
                [
                    "patient_id",
                    "recording_date",
                ],
                as_index=False
            )["n_strides_total"]
            .sum()
            .rename(columns={"n_strides_total": "n_strides_total_day"})
        )
    
        if self.day_dmos.empty:
            self.day_dmos = strides_day
        else:
            self.day_dmos = self.day_dmos.merge(
                strides_day,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self

    def compute_estimated_distance_walked(self, use_valid_strides=False):
        """
        Computes day-level estimated distance walked.
    
        For each walking bout:
            distance_WB = n_strides * average_stride_length
    
        Then sums distance_WB across the day.
    
        If use_valid_strides=False:
            uses n_strides_total
    
        If use_valid_strides=True:
            uses n_strides_valid
    
        Output column:
        - estimated_distance_walked_m
        """
    
        stride_count_col = "n_strides_valid" if use_valid_strides else "n_strides_total"
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            stride_count_col,
            "stride length [m]",
        ]
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        wb_distance = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .copy()
        )
    
        wb_distance["estimated_distance_walked_m_WB"] = (
            wb_distance[stride_count_col] * wb_distance["stride length [m]"]
        )
    
        distance_day = (
            wb_distance
            .groupby(
                [
                    "patient_id",
                    "recording_date",
                ],
                as_index=False
            )["estimated_distance_walked_m_WB"]
            .sum()
            .rename(
                columns={
                    "estimated_distance_walked_m_WB": "estimated_distance_walked_m"
                }
            )
        )
    
        if self.day_dmos.empty:
            self.day_dmos = distance_day
        else:
            self.day_dmos = self.day_dmos.merge(
                distance_day,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self
    def compute_time_spent_pa(self):
        """
        Computes day-level time spent in LPA and MVPA.
    
        For each patient and recording_date:
        - time_spent_LPA_s  = sum(duration_s where PA_type == "LPA")
        - time_spent_MVPA_s = sum(duration_s where PA_type == "MVPA")
    
        Output columns:
        - time_spent_LPA_s
        - time_spent_MVPA_s
        """
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "duration_s",
            "PA_type",
        ]
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        wb_unique = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .copy()
        )
    
        pa_time = (
            wb_unique
            .groupby(
                [
                    "patient_id",
                    "recording_date",
                    "PA_type",
                ]
            )["duration_s"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
    
        expected_pa_types = ["LPA", "MVPA"]
    
        for pa_type in expected_pa_types:
            if pa_type not in pa_time.columns:
                pa_time[pa_type] = 0
    
        pa_time = pa_time.rename(
            columns={
                "LPA": "time_spent_LPA_s",
                "MVPA": "time_spent_MVPA_s",
            }
        )
    
        ordered_columns = [
            "patient_id",
            "recording_date",
            "time_spent_LPA_s",
            "time_spent_MVPA_s",
        ]
    
        pa_time = pa_time[ordered_columns]
    
        if self.day_dmos.empty:
            self.day_dmos = pa_time
        else:
            self.day_dmos = self.day_dmos.merge(
                pa_time,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self
    def compute_wearing_time_from_log(self):
        """
        Computes day-level wearing time from log.
    
        If log is missing or wearing_time_hours is not found:
        - wearing_time_hours = NaN
    
        Output column:
        - wearing_time_hours
        """
    
        required_columns = [
            "patient_id",
            "recording_date",
        ]
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        base_days = (
            self.wb_parameters_average[
                [
                    "patient_id",
                    "recording_date",
                ]
            ]
            .drop_duplicates()
            .copy()
        )
    
        if not self.log:
            base_days["wearing_time_hours"] = np.nan
            wearing_day = base_days
    
        else:
            rows = []
    
            for item in self.log:
                patient_id = item.get("patient_id")
                recording_date = item.get("date")
                session_id = item.get("session_id")
                log_data = item.get("log", {})
    
                recording_summary = log_data.get("recording_summary", {})
                wearing_time_hours = recording_summary.get("wearing_time_hours")
    
                if wearing_time_hours is None:
                    continue
    
                rows.append(
                    {
                        "patient_id": patient_id,
                        "recording_date": recording_date,
                        "session_id": session_id,
                        "wearing_time_hours": wearing_time_hours,
                    }
                )
    
            if not rows:
                base_days["wearing_time_hours"] = np.nan
                wearing_day = base_days
            else:
                wearing_session = pd.DataFrame(rows)
    
                wearing_day = (
                    wearing_session
                    .groupby(
                        [
                            "patient_id",
                            "recording_date",
                        ],
                        as_index=False,
                    )["wearing_time_hours"]
                    .sum()
                )
    
        if self.day_dmos.empty:
            self.day_dmos = wearing_day
        else:
            self.day_dmos = self.day_dmos.merge(
                wearing_day,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self
    def compute_percentage_time_spent_walking(self):
        """
        Computes percentage of wearing time spent walking.
    
        Internally:
            walking_time_s = time_spent_LPA_s + time_spent_MVPA_s
            wearing_time_s = wearing_time_hours * 3600
    
        Output column:
        - percentage_time_spent_walking
    
        If wearing_time_hours is missing:
        - percentage_time_spent_walking = NaN
        """
    
        required_columns = [
            "time_spent_LPA_s",
            "time_spent_MVPA_s",
            "wearing_time_hours",
        ]
    
        for col in required_columns:
            if col not in self.day_dmos.columns:
                raise KeyError(
                    f"Column '{col}' not found in day_dmos. "
                    "Run compute_time_spent_pa() and compute_wearing_time_from_log() first."
                )
    
        walking_time_s = (
            self.day_dmos["time_spent_LPA_s"].fillna(0)
            + self.day_dmos["time_spent_MVPA_s"].fillna(0)
        )
    
        wearing_time_s = self.day_dmos["wearing_time_hours"] * 3600
    
        self.day_dmos["percentage_time_spent_walking"] = np.where(
            wearing_time_s.notna() & (wearing_time_s > 0),
            walking_time_s / wearing_time_s * 100,
            np.nan,
        )
    
        return self
    def compute_max_wb_duration(self):
        """
        Computes day-level maximum walking bout duration.
    
        Output column:
        - max_WB_duration_s
        """
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "duration_s",
        ]
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        max_duration_day = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .groupby(
                [
                    "patient_id",
                    "recording_date",
                ],
                as_index=False,
            )["duration_s"]
            .max()
            .rename(columns={"duration_s": "max_WB_duration_s"})
        )
    
        if self.day_dmos.empty:
            self.day_dmos = max_duration_day
        else:
            self.day_dmos = self.day_dmos.merge(
                max_duration_day,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self
    def compute_wb_label_parameter_summary(
        self,
        aggregation="mean",
        include_very_short=False,
    ):
        """
        Computes day-level mean or median of WB-level parameters by WB_label.
    
        Parameters
        ----------
        aggregation : str
            Either "mean" or "median".
    
        include_very_short : bool
            If False, excludes very_short walking bouts.
            If True, includes very_short walking bouts.
    
        Output columns are named as:
            WB_label_wb_parameter_name
    
        Example:
            short_wb_cadence_[steps_per_min]
            medium_wb_stride time [s]
            long_wb_gait velocity [m/s]
        """
    
        if aggregation not in ["mean", "median"]:
            raise ValueError("aggregation must be either 'mean' or 'median'")
    
        parameter_columns = [
            "cadence_[steps_per_min]",
            "stride time [s]",
            "swing time [s]",
            "stance time [s]",
            "percent_stance_phase",
            "percent_swing_phase",
            "arc length [m]",
            "gait velocity [m/s]",
            "ic angle [deg]",
            "max. lateral excursion [m]",
            "max. orientation change [deg]",
            "max. sensor lift [m]",
            "stride length [m]",
            "tc angle [deg]",
            "turning angle [deg]",
        ]
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "WB_label",
        ] + parameter_columns
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        wb_unique = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .copy()
        )
    
        if not include_very_short:
            wb_unique = wb_unique[wb_unique["WB_label"] != "very_short"].copy()
    
        if aggregation == "mean":
            summary = (
                wb_unique
                .groupby(
                    [
                        "patient_id",
                        "recording_date",
                        "WB_label",
                    ],
                    as_index=False,
                )[parameter_columns]
                .mean()
            )
        else:
            summary = (
                wb_unique
                .groupby(
                    [
                        "patient_id",
                        "recording_date",
                        "WB_label",
                    ],
                    as_index=False,
                )[parameter_columns]
                .median()
            )
    
        summary_wide = summary.pivot(
            index=[
                "patient_id",
                "recording_date",
            ],
            columns="WB_label",
            values=parameter_columns,
        )
    
        summary_wide.columns = [
            f"{wb_label}_wb_{parameter}"
            for parameter, wb_label in summary_wide.columns
        ]
    
        summary_wide = summary_wide.reset_index()
    
        expected_labels = [
            "short",
            "medium",
            "long",
        ]
    
        if include_very_short:
            expected_labels = [
                "very_short",
                "short",
                "medium",
                "long",
            ]
    
        expected_columns = [
            f"{label}_wb_{parameter}"
            for label in expected_labels
            for parameter in parameter_columns
        ]
    
        for col in expected_columns:
            if col not in summary_wide.columns:
                summary_wide[col] = np.nan
    
        ordered_columns = [
            "patient_id",
            "recording_date",
        ] + expected_columns
    
        summary_wide = summary_wide[ordered_columns]
    
        if self.day_dmos.empty:
            self.day_dmos = summary_wide
        else:
            self.day_dmos = self.day_dmos.merge(
                summary_wide,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self
    def compute_between_wb_diversity_by_label(
        self,
        include_very_short=False,
    ):
        """
        Computes day-level between-WB diversity stratified by WB_label.
    
        For each patient, recording_date, and WB_label:
            COV = std / mean * 100
    
        Parameters are:
        - gait velocity [m/s]
        - stride length [m]
        - cadence_[steps_per_min]
        - stride time [s]
    
        Parameters
        ----------
        include_very_short : bool
            If False, excludes very_short walking bouts.
            If True, includes very_short walking bouts.
    
        Output columns:
            cov_<WB_label>_WB_<parameter>
    
        Example:
            cov_short_WB_gait velocity [m/s]
            cov_medium_WB_stride length [m]
            cov_long_WB_cadence_[steps_per_min]
        """
    
        parameter_columns = [
            "gait velocity [m/s]",
            "stride length [m]",
            "cadence_[steps_per_min]",
            "stride time [s]",
        ]
    
        required_columns = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
            "WB_label",
        ] + parameter_columns
    
        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        wb_unique = (
            self.wb_parameters_average
            .drop_duplicates(
                subset=[
                    "patient_id",
                    "recording_date",
                    "session_id",
                    "WB_id",
                ]
            )
            .copy()
        )
    
        if not include_very_short:
            wb_unique = wb_unique[wb_unique["WB_label"] != "very_short"].copy()
    
        labels = [
            "short",
            "medium",
            "long",
        ]
    
        if include_very_short:
            labels = [
                "very_short",
                "short",
                "medium",
                "long",
            ]
    
        cov_parts = []
    
        base_days = (
            self.wb_parameters_average[
                [
                    "patient_id",
                    "recording_date",
                ]
            ]
            .drop_duplicates()
            .copy()
        )
    
        for label in labels:
            label_df = wb_unique[wb_unique["WB_label"] == label].copy()
    
            if label_df.empty:
                temp = base_days.copy()
    
                for parameter in parameter_columns:
                    temp[f"cov_{label}_WB_{parameter}"] = np.nan
    
                cov_parts.append(temp)
                continue
    
            stats = (
                label_df
                .groupby(
                    [
                        "patient_id",
                        "recording_date",
                    ]
                )[parameter_columns]
                .agg(["mean", "std"])
            )
    
            stats.columns = [
                f"{parameter}_{stat}"
                for parameter, stat in stats.columns
            ]
    
            stats = stats.reset_index()
    
            for parameter in parameter_columns:
                mean_col = f"{parameter}_mean"
                std_col = f"{parameter}_std"
                output_col = f"cov_{label}_WB_{parameter}"
    
                stats[output_col] = np.where(
                    stats[mean_col].notna()
                    & (stats[mean_col] != 0),
                    stats[std_col] / stats[mean_col] * 100,
                    np.nan,
                )
    
            output_columns = [
                "patient_id",
                "recording_date",
            ] + [
                f"cov_{label}_WB_{parameter}"
                for parameter in parameter_columns
            ]
    
            cov_parts.append(stats[output_columns])
    
        cov_day = cov_parts[0]
    
        for part in cov_parts[1:]:
            cov_day = cov_day.merge(
                part,
                on=[
                    "patient_id",
                    "recording_date",
                ],
                how="outer",
            )
    
        if self.day_dmos.empty:
            self.day_dmos = cov_day
        else:
            self.day_dmos = self.day_dmos.merge(
                cov_day,
                on=["patient_id", "recording_date"],
                how="outer",
            )
    
        return self
if __name__ == "__main__":
    patient_id = "PAT401"
    date = "2023-07-10"

    patient_directory = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)"
        r"\MobilityAPP_Pipeline\Prova\PAT401"
    )

    collector = CollectorCSV(patient_id, date, patient_directory).collect()
    
    wb = WalkingBouts(
        collector.parameters,
        collector.wb_dataframe,
        collector.wb_pauses_dataframe,
    )
    
    wb.add_wb_label()
    wb.assign_wb_id_to_parameters()
    wb.clean_parameters(use_quality_check=True)
    wb.add_stance_swing_percentages()
    wb.add_cadence_to_wb_dataframe(use_valid_strides=False)
    wb.add_pa_type_to_wb_dataframe()
    wb.compute_wb_parameters_average()
    
    
    wb_parameters_average = wb.wb_parameters_average

    parameters = wb.parameters
    parameters_before_cleaning = wb.parameters_before_cleaning
    wb_dataframe = wb.wb_dataframe
    wb_pauses_dataframe = wb.wb_pauses_dataframe
    wb_parameters_average = wb.wb_parameters_average
    log = collector.log
    
    
    day = DayLevelDMOs(wb_parameters_average, log=collector.log)
    day.count_wb_by_label()
    day.count_total_strides()
    day.compute_estimated_distance_walked(use_valid_strides=False)
    day.compute_time_spent_pa()
    day.compute_wearing_time_from_log()
    day.compute_percentage_time_spent_walking()
    day.compute_max_wb_duration()
    day.compute_wb_label_parameter_summary(
    aggregation="mean", #or "median"
    include_very_short=False,
)
    day.compute_between_wb_diversity_by_label(
    include_very_short=False,
)
    
    day_dmos = day.day_dmos
    
    print("\nDay-level DMOs:")
    print(day_dmos)

 