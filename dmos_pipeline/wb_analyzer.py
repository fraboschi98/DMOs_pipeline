# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:26:46 2026

@author: francesca.boschi
"""
from pathlib import Path
import pandas as pd
import ast
import numpy as np

class WalkingBouts_analyzer:
    """
        Analyze walking-bout-level DMOs from collected pipeline outputs.
    
        This class is intended to be used after `CollectorCSV`. It takes the
        collected stride-level parameters, walking-bout dataframe, and walking-bout
        pauses dataframe, then prepares walking-bout-level outcomes for downstream
        analysis.
    
        The class links stride-level parameters to walking bouts using patient ID,
        recording date, session ID, foot, and stride ID. It can then clean the
        assigned stride-level parameters, compute additional derived variables, add
        cadence and physical-activity labels, and aggregate stride-level parameters
        into walking-bout-level averages.
    
        Parameters
        ----------
        parameters : pandas.DataFrame
            Combined stride-level parameter table collected from the processed
            session outputs.
    
        wb_dataframe : pandas.DataFrame
            Combined walking-bout table containing walking-bout timing, duration,
            stride IDs, and metadata.
    
        wb_pauses_dataframe : pandas.DataFrame
            Combined walking-bout pause table. This table is stored for traceability
            and later analysis, but is not modified by the current workflow.
    
        Attributes
        ----------
        parameters : pandas.DataFrame
            Stride-level parameters after WB assignment and optional cleaning.
    
        wb_dataframe : pandas.DataFrame
            Walking-bout dataframe enriched with WB labels, cadence, PA type, and
            PA state.
    
        wb_pauses_dataframe : pandas.DataFrame
            Walking-bout pause dataframe copied from the collector.
    
        parameters_before_cleaning : pandas.DataFrame
            Stride-level parameters assigned to walking bouts before optional
            quality-check filtering.
    
        wb_parameters_average : pandas.DataFrame
            Final walking-bout-level table containing descriptors, stride counts,
            cadence, PA labels, and averaged gait parameters.
    
        Notes
        -----
        This class does not detect walking bouts. Walking bouts must already have
        been extracted by `WBpipeline` and collected by `CollectorCSV`.
    
        The full ordered processing workflow is implemented in `run()`.
        """
    def __init__(self, parameters, wb_dataframe, wb_pauses_dataframe=None):
        """
        Initialize the walking-bout analyzer.
    
        Parameters
        ----------
        parameters : pandas.DataFrame
            Combined stride-level parameter table.
    
        wb_dataframe : pandas.DataFrame
            Combined walking-bout table.
    
        wb_pauses_dataframe : pandas.DataFrame, optional
            Combined walking-bout pause table. It is stored for future analyses but
            is not required by the current workflow.
        """
    
        self.parameters = parameters.copy()
        self.wb_dataframe = wb_dataframe.copy()
    
        if wb_pauses_dataframe is None:
            self.wb_pauses_dataframe = pd.DataFrame()
        else:
            self.wb_pauses_dataframe = wb_pauses_dataframe.copy()
    
        self.parameters_before_cleaning = pd.DataFrame()
        self.wb_parameters_average = pd.DataFrame()

    def add_wb_label(self):
        """
        Add duration-based walking-bout labels to `wb_dataframe`.
    
        The method creates a new column, `WB_label`, using the walking-bout duration
        stored in `duration_s`.
    
        Label rules
        -----------
        - `very_short`: duration_s < 10
        - `short`:      10 <= duration_s <= 30
        - `medium`:     30 < duration_s <= 60
        - `long`:       duration_s > 60
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If `duration_s` is missing from `wb_dataframe`.
        """
    
        if "duration_s" not in self.wb_dataframe.columns:
            raise KeyError("Column 'duration_s' not found in wb_dataframe")
    
        duration = pd.to_numeric(
            self.wb_dataframe["duration_s"],
            errors="coerce"
        )
    
        self.wb_dataframe["WB_label"] = pd.NA
    
        self.wb_dataframe.loc[duration < 10, "WB_label"] = "very_short"
        self.wb_dataframe.loc[
            (duration >= 10) & (duration <= 30),
            "WB_label"
        ] = "short"
        self.wb_dataframe.loc[
            (duration > 30) & (duration <= 60),
            "WB_label"
        ] = "medium"
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
        Assign walking-bout IDs to stride-level parameter rows.
    
        The method adds a `WB_id` column to `parameters`. Each parameter row is
        matched to a walking bout using:
    
        - patient_id
        - recording_date
        - session_id
        - foot
        - s_id
    
        For each walking bout, stride IDs are read from `left_s_ids` and
        `right_s_ids` in `wb_dataframe`. Left-foot parameter rows are matched only
        against `left_s_ids`, and right-foot parameter rows are matched only against
        `right_s_ids`.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If one of the required columns is missing from `parameters` or
            `wb_dataframe`.
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
    
        # Normalize matching columns to reduce CSV type mismatch problems.
        self.parameters["s_id"] = pd.to_numeric(
            self.parameters["s_id"],
            errors="coerce"
        )
    
        self.parameters["foot"] = (
            self.parameters["foot"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    
        for _, wb_row in self.wb_dataframe.iterrows():
            patient_id = wb_row["patient_id"]
            recording_date = wb_row["recording_date"]
            session_id = wb_row["session_id"]
            wb_id = wb_row["WB_id"]
    
            left_s_ids = pd.to_numeric(
                pd.Series(self._parse_s_ids(wb_row["left_s_ids"])),
                errors="coerce"
            ).dropna().tolist()
    
            right_s_ids = pd.to_numeric(
                pd.Series(self._parse_s_ids(wb_row["right_s_ids"])),
                errors="coerce"
            ).dropna().tolist()
    
            left_mask = (
                (self.parameters["patient_id"] == patient_id)
                & (self.parameters["recording_date"] == recording_date)
                & (self.parameters["session_id"] == session_id)
                & (self.parameters["foot"] == "left")
                & (self.parameters["s_id"].isin(left_s_ids))
            )
    
            right_mask = (
                (self.parameters["patient_id"] == patient_id)
                & (self.parameters["recording_date"] == recording_date)
                & (self.parameters["session_id"] == session_id)
                & (self.parameters["foot"] == "right")
                & (self.parameters["s_id"].isin(right_s_ids))
            )
    
            self.parameters.loc[left_mask | right_mask, "WB_id"] = wb_id
    
        return self


    def clean_parameters(self, use_quality_check=True):
        """
        Clean stride-level parameters after walking-bout assignment.
    
        The method keeps only parameter rows assigned to a walking bout, stores this
        pre-quality-filtered table in `parameters_before_cleaning`, and optionally
        removes rows that failed the stride-level quality check.
    
        Parameters
        ----------
        use_quality_check : bool, default=True
            If True, keep only rows where `quality_check` evaluates to True.
            Accepted true-like values are boolean True, "true", "1", and "yes".
    
            If False, keep all rows assigned to a walking bout.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If `WB_id` is missing from `parameters`.
            If `use_quality_check=True` and `quality_check` is missing.
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
                keep_mask = quality
            else:
                keep_mask = (
                    quality
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                )
    
            self.parameters = self.parameters[keep_mask].copy()
    
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



    def add_stance_swing_percentages(
        self,
        stance_col="stance time [s]",
        swing_col="swing time [s]",
        stride_col="stride time [s]",
        stance_percent_col="percent_stance_phase",
        swing_percent_col="percent_swing_phase"
    ):
        """
        Add stance and swing phase percentages to `parameters`.
    
        The method computes stance and swing duration as percentages of stride time
        and stores them in two new columns.
    
        Formulas
        --------
        percent_stance_phase = stance time / stride time * 100
        percent_swing_phase = swing time / stride time * 100
    
        Parameters
        ----------
        stance_col : str, default="stance time [s]"
            Column containing stance time in seconds.
    
        swing_col : str, default="swing time [s]"
            Column containing swing time in seconds.
    
        stride_col : str, default="stride time [s]"
            Column containing stride time in seconds.
    
        stance_percent_col : str, default="percent_stance_phase"
            Name of the output column for stance percentage.
    
        swing_percent_col : str, default="percent_swing_phase"
            Name of the output column for swing percentage.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If one of the required input columns is missing from `parameters`.
    
        Notes
        -----
        Rows with missing, non-numeric, or non-positive stride time receive NaN.
        """
    
        required_cols = [stance_col, swing_col, stride_col]
    
        for col in required_cols:
            if col not in self.parameters.columns:
                raise KeyError(f"Column '{col}' not found in parameters")
    
        stance = pd.to_numeric(self.parameters[stance_col], errors="coerce")
        swing = pd.to_numeric(self.parameters[swing_col], errors="coerce")
        stride = pd.to_numeric(self.parameters[stride_col], errors="coerce")
    
        valid_stride = stride > 0
    
        self.parameters[stance_percent_col] = np.where(
            valid_stride,
            stance / stride * 100,
            np.nan
        )
    
        self.parameters[swing_percent_col] = np.where(
            valid_stride,
            swing / stride * 100,
            np.nan
        )
    
        return self
    def add_cadence_to_wb_dataframe(self, use_valid_strides=False):
        """
        Add walking-bout cadence to `wb_dataframe`.
    
        Cadence is computed as steps per minute using the number of strides assigned
        to each walking bout and the walking-bout duration.
    
        Formula
        -------
        steps = n_strides * 2
        cadence_[steps_per_min] = steps / duration_s * 60
    
        By default, cadence is computed using all strides assigned to the walking
        bout before quality-check filtering. This reflects the original WB
        structure detected from gait events.
    
        Parameters
        ----------
        use_valid_strides : bool, default=False
            If False, compute cadence using `parameters_before_cleaning`, meaning
            all strides assigned to the walking bout before quality-check filtering.
    
            If True, compute cadence using `parameters`, meaning only valid strides
            remaining after optional quality-check filtering.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If one of the required columns is missing from `wb_dataframe`.
    
        ValueError
            If `parameters_before_cleaning` is empty when
            `use_valid_strides=False`.
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
    
        # Avoid duplicate stride-count columns if the method is called more than once.
        if stride_column in self.wb_dataframe.columns:
            self.wb_dataframe = self.wb_dataframe.drop(columns=[stride_column])
    
        self.wb_dataframe = self.wb_dataframe.merge(
            n_strides,
            on=group_columns,
            how="left",
        )
    
        duration = pd.to_numeric(
            self.wb_dataframe["duration_s"],
            errors="coerce"
        )
    
        n_stride_values = pd.to_numeric(
            self.wb_dataframe[stride_column],
            errors="coerce"
        )
    
        self.wb_dataframe["cadence_[steps_per_min]"] = pd.NA
    
        valid_duration = duration > 0
    
        self.wb_dataframe.loc[valid_duration, "cadence_[steps_per_min]"] = (
            n_stride_values.loc[valid_duration]
            * 2
            / duration.loc[valid_duration]
            * 60
        )
    
        return self
    def add_pa_type_to_wb_dataframe(self, cadence_col="cadence_[steps_per_min]"):
        """
        Add physical-activity type labels to `wb_dataframe`.
    
        The method creates a new column, `PA_type`, using walking-bout cadence.
    
        Label rules
        -----------
        - `MVPA`: cadence > 90 steps/min
        - `LPA`: cadence <= 90 steps/min
    
        Parameters
        ----------
        cadence_col : str, default="cadence_[steps_per_min]"
            Name of the cadence column used to classify physical-activity type.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If `cadence_col` is missing from `wb_dataframe`.
    
        Notes
        -----
        This method must be called after `add_cadence_to_wb_dataframe()`.
        Rows with missing or non-numeric cadence keep `PA_type` as missing.
        """
    
        if cadence_col not in self.wb_dataframe.columns:
            raise KeyError(
                f"Column '{cadence_col}' not found in wb_dataframe. "
                "Run add_cadence_to_wb_dataframe() before add_pa_type_to_wb_dataframe()."
            )
    
        cadence = pd.to_numeric(
            self.wb_dataframe[cadence_col],
            errors="coerce"
        )
    
        self.wb_dataframe["PA_type"] = pd.NA
    
        self.wb_dataframe.loc[cadence > 90, "PA_type"] = "MVPA"
        self.wb_dataframe.loc[cadence <= 90, "PA_type"] = "LPA"
    
        return self


    def add_pa_state_to_wb_dataframe(self, choice="traditional"):
        """
        Add physical-activity state labels to `wb_dataframe`.
    
        The method assigns a numeric `PA_state` to each walking bout using
        walking-bout duration and cadence.
    
        Two definitions are available:
    
        - "traditional":
          3 duration ranges and 4 cadence ranges, producing states 7 to 18.
    
        - "modified":
          4 duration ranges and 5 cadence ranges, producing states 7 to 26.
    
        Parameters
        ----------
        choice : {"traditional", "modified"}, default="traditional"
            PA-state definition to apply.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        KeyError
            If one of the required columns is missing from `wb_dataframe`.
    
        ValueError
            If `choice` is not "traditional" or "modified".
    
        Notes
        -----
        This method must be called after `add_cadence_to_wb_dataframe()`.
        Rows with missing, non-numeric, or out-of-range duration/cadence keep
        `PA_state` as NaN.
        """
    
        required_cols = [
            "patient_id",
            "recording_date",
            "WB_id",
            "start",
            "end",
            "duration_s",
            "cadence_[steps_per_min]",
        ]
    
        for col in required_cols:
            if col not in self.wb_dataframe.columns:
                raise KeyError(f"Column '{col}' not found in wb_dataframe")
    
        cadence = pd.to_numeric(
            self.wb_dataframe["cadence_[steps_per_min]"],
            errors="coerce"
        )
    
        duration = pd.to_numeric(
            self.wb_dataframe["duration_s"],
            errors="coerce"
        )
    
        if choice == "traditional":
            thc1, thc2, thc3 = 50, 80, 140
            thd1, thd2 = 30, 120
    
            conditions = [
                (cadence <= thc1) & (duration <= thd1),
                (cadence > thc1) & (cadence <= thc2) & (duration <= thd1),
                (cadence > thc2) & (cadence <= thc3) & (duration <= thd1),
                (cadence > thc3) & (duration <= thd1),
    
                (cadence <= thc1) & (duration > thd1) & (duration <= thd2),
                (cadence > thc1) & (cadence <= thc2) & (duration > thd1) & (duration <= thd2),
                (cadence > thc2) & (cadence <= thc3) & (duration > thd1) & (duration <= thd2),
                (cadence > thc3) & (duration > thd1) & (duration <= thd2),
    
                (cadence <= thc1) & (duration > thd2),
                (cadence > thc1) & (cadence <= thc2) & (duration > thd2),
                (cadence > thc2) & (cadence <= thc3) & (duration > thd2),
                (cadence > thc3) & (duration > thd2),
            ]
    
            choices = list(range(7, 19))
    
        elif choice == "modified":
            cadence_bins = [70, 90, 110, 130]
            duration_bins = [30, 120, 360]
    
            conditions = []
            choices = []
            state = 7
    
            for dur_idx in range(4):
                dur_low = 0 if dur_idx == 0 else duration_bins[dur_idx - 1]
                dur_high = duration_bins[dur_idx] if dur_idx < 3 else np.inf
    
                for cad_idx in range(5):
                    cad_low = 0 if cad_idx == 0 else cadence_bins[cad_idx - 1]
                    cad_high = cadence_bins[cad_idx] if cad_idx < 4 else np.inf
    
                    cond = (
                        (duration > dur_low)
                        & (duration <= dur_high)
                        & (cadence > cad_low)
                        & (cadence <= cad_high)
                    )
    
                    conditions.append(cond)
                    choices.append(state)
                    state += 1
    
        else:
            raise ValueError("Invalid choice. Use 'traditional' or 'modified'.")
    
        self.wb_dataframe["PA_state"] = np.select(
            conditions,
            choices,
            default=np.nan,
        )
    
        return self 

    def compute_wb_parameters_average(self):
        """
        Compute walking-bout-level average gait parameters.
    
        The method aggregates cleaned stride-level parameters into one row per
        walking bout. It also adds walking-bout descriptors and stride counts.
    
        The averages are computed using `self.parameters`, which should already be
        cleaned by `clean_parameters()`. Therefore, if `use_quality_check=True` was
        used during cleaning, failed stride-level rows are excluded from the
        averages.
    
        Output
        ------
        The resulting table is stored in `self.wb_parameters_average`.
    
        It contains:
    
        - patient_id
        - recording_date
        - session_id
        - WB_id
        - start
        - end
        - duration_s
        - WB_label
        - n_strides_total
        - n_strides_valid
        - cadence_[steps_per_min]
        - PA_type
        - PA_state
        - average stride-level gait parameters
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
    
        Raises
        ------
        ValueError
            If `parameters_before_cleaning` is empty.
    
        KeyError
            If required columns are missing from `parameters` or `wb_dataframe`.
    
        Notes
        -----
        `n_strides_total` is computed from `parameters_before_cleaning`, meaning all
        strides assigned to the walking bout before quality-check filtering.
    
        `n_strides_valid` is computed from `parameters`, meaning the stride-level
        table after optional quality-check filtering.
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
    
        parameters_for_average = self.parameters.copy()
    
        for col in columns_to_average:
            parameters_for_average[col] = pd.to_numeric(
                parameters_for_average[col],
                errors="coerce"
            )
    
        wb_average = (
            parameters_for_average
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
            "PA_type",
            "PA_state",
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
            "PA_type",
            "PA_state",
        ] + columns_to_average
    
        self.wb_parameters_average = self.wb_parameters_average[ordered_columns]
    
        return self




    def run(
        self,
        use_quality_check=True,
        use_valid_strides_for_cadence=False,
        pa_state_choice="modified",
    ):
        """
        Run the standard walking-bout analysis workflow.
    
        This method applies the in-memory processing steps needed to create the
        walking-bout-level output table `wb_parameters_average`.
    
        The method does not save files. To export the final table, call
        `save_wb_parameters_average()` after `run()`.
    
        Processing steps
        ----------------
        1. Add duration-based walking-bout labels to `wb_dataframe`.
        2. Assign `WB_id` to stride-level parameter rows.
        3. Remove parameter rows not assigned to a walking bout.
        4. Optionally remove parameter rows that failed quality control.
        5. Add stance and swing percentages to `parameters`.
        6. Compute walking-bout cadence.
        7. Add physical-activity type labels to `wb_dataframe`.
        8. Add physical-activity state labels to `wb_dataframe`.
        9. Compute walking-bout-level average gait parameters.
    
        Parameters
        ----------
        use_quality_check : bool, default=True
            If True, remove rows where `quality_check` is False after WB assignment.
            If False, keep all parameter rows assigned to a walking bout.
    
        use_valid_strides_for_cadence : bool, default=False
            If False, compute cadence using all strides assigned to the walking bout
            before quality-check filtering.
    
            If True, compute cadence using only valid strides remaining after
            quality-check filtering.
    
        pa_state_choice : {"traditional", "modified"}, default="modified"
            PA-state definition used by `add_pa_state_to_wb_dataframe()`.
    
        Returns
        -------
        self
            Updated `WalkingBouts_analyzer` instance.
        """
    
        self.add_wb_label()
        self.assign_wb_id_to_parameters()
        self.clean_parameters(use_quality_check=use_quality_check)
        self.add_stance_swing_percentages()
        self.add_cadence_to_wb_dataframe(
            use_valid_strides=use_valid_strides_for_cadence
        )
        self.add_pa_type_to_wb_dataframe()
        self.add_pa_state_to_wb_dataframe(choice=pa_state_choice)
        self.compute_wb_parameters_average()
    
        return self
    def save_wb_parameters_average(self, output_folder, filename_suffix=None):
        """
        Save `wb_parameters_average` as a CSV file.
    
        The method saves the final walking-bout-level output table created by
        `compute_wb_parameters_average()` or `run()`.
    
        The filename is built automatically from the patient ID and the date scope:
    
        - one date:
          <patient_id>_<recording_date>_wb_parameters_average.csv
    
        - multiple dates:
          <patient_id>_multiple_dates_wb_parameters_average.csv
    
        A custom suffix can also be provided through `filename_suffix`.
    
        Parameters
        ----------
        output_folder : str or pathlib.Path
            Folder where the CSV file will be saved.
    
        filename_suffix : str, optional
            Custom suffix used in the output filename. If provided, the filename is:
    
            <patient_id>_<filename_suffix>_wb_parameters_average.csv
    
            Example:
            filename_suffix="selected_dates"
    
        Returns
        -------
        pathlib.Path
            Path of the saved CSV file.
    
        Raises
        ------
        ValueError
            If `wb_parameters_average` is empty.
    
        KeyError
            If `patient_id` or `recording_date` is missing from
            `wb_parameters_average`.
        """
    
        if self.wb_parameters_average.empty:
            raise ValueError(
                "wb_parameters_average is empty. "
                "Run compute_wb_parameters_average() or run() before saving."
            )
    
        required_cols = ["patient_id", "recording_date"]
    
        for col in required_cols:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")
    
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
        patient_ids = (
            self.wb_parameters_average["patient_id"]
            .dropna()
            .astype(str)
            .unique()
        )
    
        recording_dates = (
            self.wb_parameters_average["recording_date"]
            .dropna()
            .astype(str)
            .unique()
        )
    
        if len(patient_ids) != 1:
            raise ValueError(
                f"Expected one patient_id, found: {list(patient_ids)}"
            )
    
        patient_id = patient_ids[0]
    
        if filename_suffix is not None:
            date_scope = filename_suffix
    
        elif len(recording_dates) == 1:
            date_scope = recording_dates[0]
    
        else:
            date_scope = "multiple_dates"
    
        filename = f"{patient_id}_{date_scope}_wb_parameters_average.csv"
        output_path = output_folder / filename
    
        self.wb_parameters_average.to_csv(output_path, index=False)
    
        return output_path