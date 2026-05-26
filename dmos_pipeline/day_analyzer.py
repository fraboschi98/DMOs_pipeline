# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:52:01 2026

@author: francesca.boschi
"""

from pathlib import Path
import pandas as pd

import numpy as np


class DayLevel_analyzer:
    """
    Compute day-level Digital Mobility Outcomes from walking-bout-level data.

    This class is intended to be used after `WalkingBouts_analyzer`. It takes
    the walking-bout-level table `wb_parameters_average` and aggregates it into
    one row per patient and recording date.

    The class computes day-level mobility, physical-activity, and walking-bout
    distribution outcomes, including walking-bout counts by duration label,
    total stride count, estimated walking distance, time spent in LPA and MVPA,
    wearing time, percentage of wearing time spent walking, maximum walking-bout
    duration, walking-bout-label-specific parameter summaries, and between-WB
    diversity metrics.

    Parameters
    ----------
    wb_parameters_average : pandas.DataFrame
        Walking-bout-level dataframe produced by `WalkingBouts_analyzer`.
        It should contain one row per walking bout, including patient/date
        metadata, WB descriptors, stride counts, cadence, PA labels, and average
        gait parameters.

    log : list of dict, optional
        Processing logs collected by `CollectorCSV`. These logs are used to
        extract session-level wearing time and aggregate it at day level. If not
        provided, wearing-time-based outcomes are set to missing values when
        requested.

    wb_pauses_dataframe : pandas.DataFrame, optional
        Walking-bout pause dataframe collected by `CollectorCSV`. It is stored
        for traceability and future pause-based analyses, but it is not required
        by the current day-level workflow.

    Attributes
    ----------
    wb_parameters_average : pandas.DataFrame
        Copy of the input walking-bout-level dataframe.

    log : list of dict
        Processing logs used for wearing-time extraction.

    wb_pauses_dataframe : pandas.DataFrame
        Optional walking-bout pause dataframe. Currently stored for future use.

    day_dmos : pandas.DataFrame
        Final day-level DMO table. Each row represents one patient and one
        recording date.

    Notes
    -----
    This class does not compute walking bouts or WB-level averages. Those steps
    must already have been completed before initializing this class.

    The full ordered day-level workflow is implemented in `run()`.
    """

    def __init__(self, wb_parameters_average, log=None, wb_pauses_dataframe=None):
        """
        Initialize the day-level DMO analyzer.
        """
    
        self.wb_parameters_average = wb_parameters_average.copy()
        self.log = log if log is not None else []
    
        if wb_pauses_dataframe is None:
            self.wb_pauses_dataframe = pd.DataFrame()
        else:
            self.wb_pauses_dataframe = wb_pauses_dataframe.copy()
    
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
        Compute day-level wearing time from collected processing logs.
    
        The method extracts `wearing_time_hours` from each session log and sums it
        across sessions belonging to the same patient and recording date.
    
        Output column
        -------------
        wearing_time_hours
    
        Notes
        -----
        The method supports both possible collector keys:
        - "recording_date"
        - "date"
    
        If logs are missing, or if wearing_time_hours is not found, the output is
        set to NaN for the available patient-date rows.
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
    
                recording_date = (
                    item.get("recording_date")
                    if item.get("recording_date") is not None
                    else item.get("date")
                )
    
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
    
                wearing_session["wearing_time_hours"] = pd.to_numeric(
                    wearing_session["wearing_time_hours"],
                    errors="coerce",
                )
    
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
    
                wearing_day = base_days.merge(
                    wearing_day,
                    on=["patient_id", "recording_date"],
                    how="left",
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
    def run(
        self,
        use_valid_strides_for_distance=False,
        aggregation="mean",
        include_very_short=False,
    ):
        """
        Run the complete day-level DMO pipeline.
    
        This method computes the standard day-level Digital Mobility Outcomes
        from wb_parameters_average.
    
        Parameters
        ----------
        use_valid_strides_for_distance : bool, default=False
            If False, estimated distance is computed using n_strides_total.
            If True, estimated distance is computed using n_strides_valid.
    
        aggregation : {"mean", "median"}, default="mean"
            Aggregation used for WB-label parameter summaries.
    
        include_very_short : bool, default=False
            If False, excludes very_short walking bouts from WB-label parameter
            summaries and between-WB diversity.
            If True, includes very_short walking bouts.
    
        Returns
        -------
        self
            Updated DayLevelDMOs instance with day_dmos computed.
        """
    
        self.count_wb_by_label()
        self.count_total_strides()
        self.compute_estimated_distance_walked(
            use_valid_strides=use_valid_strides_for_distance
        )
        self.compute_time_spent_pa()
        self.compute_wearing_time_from_log()
        self.compute_percentage_time_spent_walking()
        self.compute_max_wb_duration()
        self.compute_wb_label_parameter_summary(
            aggregation=aggregation,
            include_very_short=include_very_short,
        )
        self.compute_between_wb_diversity_by_label(
            include_very_short=include_very_short,
        )
    
        return self   

    def save_day_dmos(self, output_folder):
        """
        Save day_dmos as a CSV file.
    
        The output filename is created using patient_id and recording_date:
    
            <patient_id>_<recording_date>_day_dmos.csv
    
        Parameters
        ----------
        output_folder : str or pathlib.Path
            Folder where the CSV file will be saved.
    
        Returns
        -------
        pathlib.Path
            Path of the saved CSV file.
    
        Raises
        ------
        ValueError
            If day_dmos is empty.
    
        KeyError
            If patient_id or recording_date is missing from day_dmos.
        """
    
        if self.day_dmos.empty:
            raise ValueError(
                "day_dmos is empty. Run run() before saving."
            )
    
        required_cols = ["patient_id", "recording_date"]
    
        for col in required_cols:
            if col not in self.day_dmos.columns:
                raise KeyError(f"Column '{col}' not found in day_dmos")
    
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
        patient_id = self.day_dmos["patient_id"].iloc[0]
        recording_date = self.day_dmos["recording_date"].iloc[0]
    
        filename = f"{patient_id}_{recording_date}_day_dmos.csv"
        output_path = output_folder / filename
    
        self.day_dmos.to_csv(output_path, index=False)
    
        return output_path    