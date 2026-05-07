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

    def add_wb_label(self):
        """
        Adds column WB_label to wb_dataframe using duration_s.

        Rules:
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

    def get_wb_dataframe(self):
        return self.wb_dataframe
    def _parse_s_ids(self, value):
           """
           Converts left_s_ids / right_s_ids into a Python list.
    
           Works with:
           [1, 2, 3]
           "1,2,3"
           "[1, 2, 3]"
           NaN
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
        Adds WB_id to parameters.

        For each row in wb_dataframe:
        - uses patient_id
        - uses recording_date
        - uses session_id
        - uses left_s_ids for foot == left
        - uses right_s_ids for foot == right
        - assigns the corresponding WB_id to matching parameters rows
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
    def _parse_s_ids(self, value):
        """
        Converts left_s_ids / right_s_ids into a Python list.
    
        Works with:
        [1, 2, 3]
        "1,2,3"
        "[1, 2, 3]"
        NaN
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
            Adds WB_id to parameters.
    
            For each row in wb_dataframe:
            - uses patient_id
            - uses recording_date
            - uses session_id
            - uses left_s_ids for foot == left
            - uses right_s_ids for foot == right
            - assigns the corresponding WB_id to matching parameters rows
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
    
        Steps:
        1. Removes rows not assigned to a WB_id.
        2. If use_quality_check=True, removes rows where quality_check is False.
    
        Returns self.
        """
    
        if "WB_id" not in self.parameters.columns:
            raise KeyError(
                "Column 'WB_id' not found in parameters. "
                "Run assign_wb_id_to_parameters() before clean_parameters()."
            )
    
        # Remove strides not assigned to any walking bout
        self.parameters = self.parameters.dropna(subset=["WB_id"]).copy()
    
        if use_quality_check:
            if "quality_check" not in self.parameters.columns:
                raise KeyError("Column 'quality_check' not found in parameters")
    
            # Works if quality_check contains True/False booleans or strings
            quality = self.parameters["quality_check"]
    
            if quality.dtype == bool:
                self.parameters = self.parameters[quality].copy()
            else:
                self.parameters = self.parameters[
                    quality.astype(str).str.lower().isin(["true", "1", "yes"])
                ].copy()
    
        self.parameters = self.parameters.reset_index(drop=True)
    
        return self
    def create_daily_wb_id(self):
        """
        Creates a unique WB_id_day across all sessions of the same patient and date.
    
        Example:
        week_2 WB_id 0 -> WB_id_day 0
        week_2 WB_id 1 -> WB_id_day 1
        week_3 WB_id 0 -> WB_id_day 2
        week_3 WB_id 1 -> WB_id_day 3
    
        Updates both:
        - self.wb_dataframe
        - self.parameters
        """
    
        required_wb_cols = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
        ]
    
        required_parameters_cols = [
            "patient_id",
            "recording_date",
            "session_id",
            "WB_id",
        ]
    
        for col in required_wb_cols:
            if col not in self.wb_dataframe.columns:
                raise KeyError(f"Column '{col}' not found in wb_dataframe")
    
        for col in required_parameters_cols:
            if col not in self.parameters.columns:
                raise KeyError(f"Column '{col}' not found in parameters")
    
        # Make one row per original walking bout
        wb_keys = (
            self.wb_dataframe[
                ["patient_id", "recording_date", "session_id", "WB_id"]
            ]
            .drop_duplicates()
            .sort_values(["patient_id", "recording_date", "session_id", "WB_id"])
            .reset_index(drop=True)
        )
    
        # New unique walking bout id for the whole day
        wb_keys["WB_id_day"] = wb_keys.groupby(
            ["patient_id", "recording_date"]
        ).cumcount()
    
        # Add WB_id_day to wb_dataframe
        self.wb_dataframe = self.wb_dataframe.merge(
            wb_keys,
            on=["patient_id", "recording_date", "session_id", "WB_id"],
            how="left",
        )
    
        # Add WB_id_day to parameters
        self.parameters = self.parameters.merge(
            wb_keys,
            on=["patient_id", "recording_date", "session_id", "WB_id"],
            how="left",
        )
    
        return self
    def compute_wb_parameters_average(self):
        """
        Computes average gait parameters for each walking bout of the day.
    
        Output:
        - patient_id
        - recording_date
        - WB_id_day
        - WB_label
        - duration_s
        - n_strides
        - average values of selected gait parameters
        """
    
        if "WB_id_day" not in self.parameters.columns:
            raise KeyError(
                "Column 'WB_id_day' not found in parameters. "
                "Run create_daily_wb_id() before compute_wb_parameters_average()."
            )
    
        if "WB_id_day" not in self.wb_dataframe.columns:
            raise KeyError(
                "Column 'WB_id_day' not found in wb_dataframe. "
                "Run create_daily_wb_id() before compute_wb_parameters_average()."
            )
    
        if "WB_label" not in self.wb_dataframe.columns:
            raise KeyError(
                "Column 'WB_label' not found in wb_dataframe. "
                "Run add_wb_label() before compute_wb_parameters_average()."
            )
    
        columns_to_average = [
            "stride time [s]",
            "swing time [s]",
            "stance time [s]",
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
            "WB_id_day",
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
    
        n_strides = (
            self.parameters
            .groupby(group_columns + ["foot"])["s_id"]
            .nunique()
            .reset_index(name="n_strides_foot")
        )
    
        n_strides = (
            n_strides
            .groupby(group_columns, as_index=False)["n_strides_foot"]
            .max()
            .rename(columns={"n_strides_foot": "n_strides"})
        )
    
        wb_info = (
            self.wb_dataframe[
                [
                    "patient_id",
                    "recording_date",
                    "WB_id_day",
                    "WB_label",
                    "duration_s",
                ]
            ]
            .drop_duplicates()
        )
    
        self.wb_parameters_average = (
            wb_average
            .merge(n_strides, on=group_columns, how="left")
            .merge(wb_info, on=group_columns, how="left")
        )
    
        ordered_columns = [
            "patient_id",
            "recording_date",
            "WB_id_day",
            "WB_label",
            "duration_s",
            "n_strides",
        ] + columns_to_average
    
        self.wb_parameters_average = self.wb_parameters_average[ordered_columns]
    
        return self
if __name__ == "__main__":
    patient_id = "PAT401"
    date = "2023-07-10"

    patient_directory = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)"
        r"\MobilityAPP_Pipeline\Prova\PAT401"
    )

    collector = CollectorCSV(patient_id, date, patient_directory).collect()
    collector.summary()

    wb = WalkingBouts(
        parameters=collector.parameters,
        wb_dataframe=collector.wb_dataframe,
        wb_pauses_dataframe=collector.wb_pauses_dataframe,
    )

    wb.add_wb_label()
    wb.assign_wb_id_to_parameters()
    wb.clean_parameters(use_quality_check=True)
    wb.create_daily_wb_id()
    wb.compute_wb_parameters_average()

    parameters = wb.parameters
    wb_dataframe = wb.wb_dataframe
    wb_pauses_dataframe = wb.wb_pauses_dataframe
    wb_parameters_average = wb.wb_parameters_average
    log = collector.log

    print("\nParameters with original WB_id and daily WB_id:")
    print(parameters[[
        "patient_id",
        "recording_date",
        "session_id",
        "foot",
        "s_id",
        "WB_id",
        "WB_id_day",
    ]].head(20))

    print("\nWalking bouts with original WB_id and daily WB_id:")
    print(wb_dataframe[[
        "patient_id",
        "recording_date",
        "session_id",
        "WB_id",
        "WB_id_day",
        "duration_s",
        "WB_label",
    ]].head(20))

    print("\nAverage parameters per walking bout of the day:")
    print(wb_parameters_average.head())

    print("\nColumns:")
    print(wb_parameters_average.columns)