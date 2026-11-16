# -*- coding: utf-8 -*-
"""
Example script: compute PA complexity metrics from existing WB-level files.

Must be run after the WB-level analysis script has created, for each patient, a
`wb_parameters_average` CSV file that contains the `PA_state` column
(WalkingBouts_analyzer.add_pa_state_to_wb_dataframe("modified")).

The script:
1. loads the existing WB-level file of each patient and collects the logs of
   the corresponding dates with Collector;
2. puts all the patients together and uses ONE PAComplexity_analyzer for all of
   them: it builds the 1 Hz PA-state series, smooths it and computes Hn, LZC and
   PLZC for each day. One analyzer for all the patients matters because the lag
   of the PLZC is chosen once, from the mean autocorrelation of all the days
   analyzed (as in the PLZC paper, where the lag is chosen once for the
   analysis), so all the patients get the same lag and their values are
   comparable. If each patient were analyzed alone, each would get its own lag;
3. saves, in each patient's own outputs_ folder, the daily table and the
   weekly summary of that patient (mean or median of the daily values over the
   valid days, n_valid_days, and the SampEn computed on the concatenated valid
   days as in the barcode paper), plus the barcode figure (`plot_barcode`) of
   that patient's valid days. As in the other example scripts of this
   package, everything stays in the patient's own folder: there is no separate
   combined file for the cohort (the weekly tables of several patients can
   always be concatenated afterwards if one is needed).

The PLZC lag chosen for the whole analysis (`analyzer.plzc_lag_`) is only
printed, not saved as a column: re-read it from the console output when
comparing runs with a different set of patients or days.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import Collector
from dmos_pipeline import PAComplexity_analyzer


def main():
    # ------------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------------
    patient_ids = ["PAT401"]

    project_folder = Path(
        #r"C:\Users\francesca.terranova\Data\MobilityAPP_Lux"
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova"
        )

    # ------------------------------------------------------------------
    # Complexity options
    # ------------------------------------------------------------------
    window_seconds = 30        # smoothing window (barcode duration threshold)
    k_folds = 10
    pa_state_choice = "modified"    # PA-state definition used in the WB analysis: sets the number of states
    window_hours = 8           # hours of worn time per day used for the series (8 h as in the barcode paper)
    min_wearing_time_hours = 8.0    # skip days with less worn time (None = no filter)
    random_state = 0           # reproducible smoothing (one generator per patient and day)
    aggregation = "mean"       # weekly summary: "mean" or "median" of the daily values
    plzc_delay = "acf"         # PLZC lag: "acf" (chosen once from all the days analyzed) or an integer (s)

    # Optional: restrict to some recording dates (None = all dates in the WB files)
    selected_dates = None  # e.g. ["2023-07-10", "2023-07-14", "2023-07-17"]

    # ------------------------------------------------------------------
    # Load the WB-level table and the logs of every patient
    # ------------------------------------------------------------------
    wb_tables = []
    logs = []
    patient_directories = {}

    for patient_id in patient_ids:
        patient_directory = project_folder / patient_id
        patient_directories[patient_id] = patient_directory
        wb_path = (
            patient_directory / "outputs_"
            / f"{patient_id}_multiple_dates_wb_parameters_average.csv"
        )

        if not wb_path.exists():
            raise FileNotFoundError(f"WB-level input file not found: {wb_path}")

        wb_parameters_average = pd.read_csv(wb_path)

        if "PA_state" not in wb_parameters_average.columns:
            raise KeyError(
                "Column 'PA_state' not found. Run add_pa_state_to_wb_dataframe('modified') "
                "in the WB-level analysis first."
            )

        if selected_dates is not None:
            wb_parameters_average = wb_parameters_average[
                wb_parameters_average["recording_date"].astype(str).isin(selected_dates)
            ]

        available_dates = sorted(
            wb_parameters_average["recording_date"].dropna().astype(str).unique()
        )
        print(f"{patient_id}: dates {available_dates}")

        # Logs (wearing_start_sample, non-wearing windows, wearing_time_hours)
        collector = Collector(
            patient_id=patient_id,
            date=available_dates,
            patient_directory=patient_directory,
        )
        collector.collect()

        wb_tables.append(wb_parameters_average)
        logs.extend(collector.log)

    wb_all = pd.concat(wb_tables, ignore_index=True)

    # ------------------------------------------------------------------
    # Complexity: ONE analyzer for all the patients
    # ------------------------------------------------------------------
    analyzer = PAComplexity_analyzer(
        wb_parameters_average=wb_all,
        log=logs,
        window_seconds=window_seconds,
        k_folds=k_folds,
        pa_state_choice=pa_state_choice,
        window_hours=window_hours,
        plzc_delay=plzc_delay,
        min_wearing_time_hours=min_wearing_time_hours,
        random_state=random_state,
    ).run()

    if analyzer.complexity_df.empty:
        print("\nNo valid day for any patient: nothing to save.")
        return {"complexity": pd.DataFrame(), "weekly": pd.DataFrame()}

    complexity = analyzer.complexity_df
    weekly = analyzer.weekly_summary(period_col="session_id", aggregation=aggregation)
    print(f"\nPLZC lag used for all the patients: {analyzer.plzc_lag_} s")

    # ------------------------------------------------------------------
    # Save: daily table and weekly summary in each patient's outputs_ folder
    # ------------------------------------------------------------------
    for patient_id, patient_directory in patient_directories.items():
        daily_patient = complexity[complexity["patient_id"] == patient_id]
        weekly_patient = weekly[weekly["patient_id"] == patient_id]

        # A patient with no valid day (for example less worn time than the
        # threshold on every day) has nothing to save.
        if daily_patient.empty:
            print(f"  {patient_id}: no valid day, nothing saved")
            continue

        patient_output_folder = patient_directory / "outputs_"
        daily_path = patient_output_folder / f"{patient_id}_complexity_metrics.csv"
        weekly_path = patient_output_folder / f"{patient_id}_complexity_weekly.csv"
        barcode_path = patient_output_folder / f"{patient_id}_barcode.png"
        daily_patient.to_csv(daily_path, index=False)
        weekly_patient.to_csv(weekly_path, index=False)
        print(f"  saved: {daily_path}")
        print(f"  saved: {weekly_path}")

        # Barcode figure (one row per valid day, smoothed series), for a
        # visual check of the pattern behind this patient's numbers.
        analyzer.plot_barcode(
            patient_id, use_smoothed=True, output_path=barcode_path, show=False
        )
        print(f"  saved: {barcode_path}")

    return {
        "complexity": complexity,
        "weekly": weekly,
        "plzc_lag_s": analyzer.plzc_lag_,
    }


if __name__ == "__main__":
    results = main()