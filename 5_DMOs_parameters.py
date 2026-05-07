# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:53:55 2026

@author: francesca.boschi

DMOs class:
    1. collect wb, pauses, parameters by PATIENT and DAY
    2. label wb 
    3. Stride-Performance: map paramters to wb
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

if __name__ == "__main__":
    patient_id = "PAT401"
    date = "2023-07-10"

    patient_directory = (
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)"
        r"\MobilityAPP_Pipeline\Prova\PAT401"
    )

    # Collect previous output files
    collector = CollectorCSV(patient_id, date, patient_directory).collect()
    collector.summary()

    # Create WalkingBouts object
    wb = WalkingBouts(
        parameters=collector.parameters,
        wb_dataframe=collector.wb_dataframe,
        wb_pauses_dataframe=collector.wb_pauses_dataframe,
    )

    # Add WB_label column
    wb.add_wb_label()

    # Get updated dataframes
    parameters = wb.parameters
    wb_dataframe = wb.wb_dataframe
    wb_pauses_dataframe = wb.wb_pauses_dataframe
    log = collector.log

    # Debug prints
    print("\nWalking bouts dataframe:")
    print(wb_dataframe.head())

    print("\nWB labels count:")
    print(wb_dataframe["WB_label"].value_counts(dropna=False))

    print("\nSelected columns:")
    print(wb_dataframe[["patient_id", "recording_date", "session_id", "WB_id", "duration_s", "WB_label"]].head())