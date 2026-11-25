# -*- coding: utf-8 -*-
"""
Example script: compute PA complexity metrics from an existing WB-level file.

This script must be run after the WB-level analysis script has already
created a `wb_parameters_average` CSV file that contains the `PA_state`
column (WalkingBouts_analyzer.add_pa_state_to_wb_dataframe("modified")).

The user specifies the input WB-level file, for example:

    PAT401_multiple_dates_wb_parameters_average.csv

The script:
1. loads the existing WB-level file;
2. prints the recording dates available in the file;
3. uses Collector to collect the corresponding logs;
4. computes day-level complexity metrics (Hn, LZC, PLZC) and saves them;
5. aggregates those days into a week-level summary (mean/median of Hn/LZC/
   PLZC over the valid days, plus SampEn computed once on the concatenated
   valid days, as in the barcode paper) and saves it, as a separate step from
   the day-level one above;
6. saves the barcode figure of this patient's valid days.

The PLZC lag chosen for the analysis (`analyzer.plzc_lag_`) is only printed,
not saved as a column: re-read it from the console output when comparing
runs with a different set of days.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import Collector
from dmos_pipeline import PAComplexity_analyzer


def main():
    """
    Load an existing WB-level file, collect logs, and compute PA complexity
    metrics for one patient: first at the day level, then, as a separate
    step, aggregated at the week level.
    """

    # ------------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------------
    patient_id = "PAT401"

    project_folder = Path(
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova"
    )

    patient_directory = project_folder / patient_id

    input_folder = patient_directory / "outputs_"
    input_filename = f"{patient_id}_multiple_dates_wb_parameters_average.csv"

    wb_parameters_average_path = input_folder / input_filename
    output_folder = patient_directory / "outputs_"

    # ------------------------------------------------------------------
    # Complexity options
    # ------------------------------------------------------------------
    window_seconds = 30        # smoothing window (barcode duration threshold)
    k_folds = 10
    pa_state_choice = "modified"    # PA-state definition used in the WB analysis: sets the number of states
    apply_time_window = True   # True: cap each day at window_hours (as in the barcode paper); False: use all available worn time of the day instead
    window_hours = 8           # hours of worn time per day used when apply_time_window=True
    min_wearing_time_hours = 8.0    # skip days with less worn time (None = no filter)
    random_state = 0           # reproducible smoothing (one generator per day)
    aggregation = "mean"       # weekly summary: "mean" or "median" of the daily values
    plzc_delay = "acf"         # PLZC lag: "acf" (chosen once from all the days analyzed) or an integer (s)

    # Optional: restrict to some recording dates (None = all dates in the file)
    selected_dates = None  # e.g. ["2023-07-10", "2023-07-14", "2023-07-17"]

    # ------------------------------------------------------------------
    # Load WB-level input
    # ------------------------------------------------------------------
    if not wb_parameters_average_path.exists():
        raise FileNotFoundError(
            f"WB-level input file not found: {wb_parameters_average_path}"
        )

    wb_parameters_average = pd.read_csv(wb_parameters_average_path)

    print("\nLoaded WB-level input:")
    print(f"  file: {wb_parameters_average_path}")
    print(f"  rows: {len(wb_parameters_average)}")

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

    print("\nAvailable recording dates in WB-level input:")
    for date in available_dates:
        print(f"  {date}")

    # ------------------------------------------------------------------
    # Collect logs for these dates
    # ------------------------------------------------------------------
    collector = Collector(
        patient_id=patient_id,
        date=available_dates,
        patient_directory=patient_directory,
    )

    collector.collect()

    log = collector.log

    print("\nCollected logs:")
    print(f"  logs loaded: {len(log)}")

    # ------------------------------------------------------------------
    # Day-level complexity metrics (Hn, LZC, PLZC)
    # ------------------------------------------------------------------
    analyzer = PAComplexity_analyzer(
        wb_parameters_average=wb_parameters_average,
        log=log,
        window_seconds=window_seconds,
        k_folds=k_folds,
        pa_state_choice=pa_state_choice,
        apply_time_window=apply_time_window,
        window_hours=window_hours,
        plzc_delay=plzc_delay,
        min_wearing_time_hours=min_wearing_time_hours,
        random_state=random_state,
    )

    analyzer.run()

    if analyzer.complexity_df.empty:
        print("\nNo valid day for this patient: nothing to save.")
        return {"complexity": pd.DataFrame(), "weekly": pd.DataFrame()}

    complexity = analyzer.complexity_df

    print("\nDay-level complexity metrics:")
    print(f"  valid days: {len(complexity)}")
    print(f"  PLZC lag used: {analyzer.plzc_lag_} s")

    # ------------------------------------------------------------------
    # Save day-level output
    # ------------------------------------------------------------------
    saved_daily_path = analyzer.save_complexity(
        output_folder=output_folder,
        filename=f"{patient_id}_complexity_metrics.csv",
    )

    print("\nSaved day-level output:")
    print(f"  complexity_metrics: {saved_daily_path}")

    # ------------------------------------------------------------------
    # Week-level aggregation -- a separate step from the day-level one above:
    # mean/median of Hn/LZC/PLZC over the valid days, plus SampEn computed
    # once on the concatenated valid days (as in the barcode paper).
    # ------------------------------------------------------------------
    weekly = analyzer.weekly_summary(
        period_col="session_id",
        aggregation=aggregation,
    )

    print("\nWeek-level aggregation:")
    print(f"  periods: {len(weekly)}")

    # ------------------------------------------------------------------
    # Save week-level output
    # ------------------------------------------------------------------
    saved_weekly_path = output_folder / f"{patient_id}_complexity_weekly.csv"
    weekly.to_csv(saved_weekly_path, index=False)

    print("\nSaved week-level output:")
    print(f"  complexity_weekly: {saved_weekly_path}")

    # ------------------------------------------------------------------
    # Barcode figure (one row per valid day, smoothed series), for a visual
    # check of the pattern behind this patient's numbers.
    # ------------------------------------------------------------------
    saved_barcode_path = output_folder / f"{patient_id}_barcode.png"
    analyzer.plot_barcode(
        patient_id, use_smoothed=True, output_path=saved_barcode_path, show=False
    )

    print("\nSaved barcode figure:")
    print(f"  barcode: {saved_barcode_path}")

    # ------------------------------------------------------------------
    # Outputs available in Spyder
    # ------------------------------------------------------------------
    return {
        "wb_parameters_average": wb_parameters_average,
        "log": log,
        "complexity": complexity,
        "weekly": weekly,
        "plzc_lag_s": analyzer.plzc_lag_,
        "saved_daily_path": saved_daily_path,
        "saved_weekly_path": saved_weekly_path,
        "saved_barcode_path": saved_barcode_path,
    }


if __name__ == "__main__":
    results = main()
