

# -*- coding: utf-8 -*-
"""
Example script: generate KDE plots from WB-level gait parameters.

This script starts from one patient's existing `wb_parameters_average` CSV file.

Optionally, it can also load the patient's `day_dmos` CSV file to exclude
recording dates with wearing time below a selected threshold.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import KDEPlotter


def main():
    """
    Load one patient's WB-level file and generate KDE plots.
    """

    # ------------------------------------------------------------------
    # Patient and project folder
    # ------------------------------------------------------------------
    patient_id = "PAT401"

    project_folder = Path(
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova"
    )

    patient_directory = project_folder / patient_id

    # ------------------------------------------------------------------
    # Input files
    # ------------------------------------------------------------------
    wb_parameters_average_path = (
        patient_directory
        / "outputs_"
        / "PAT401_multiple_dates_wb_parameters_average.csv"
    )

    day_dmos_path = (
        patient_directory
        / "outputs_"
        / "PAT401_day_dmos.csv"
    )

    # ------------------------------------------------------------------
    # Date selection
    # ------------------------------------------------------------------
    # Choose ONE option only.

    # Option 1: one specific date
    # date = "2023-07-10"

    # Option 2: selected list of dates
    # date = ["2023-07-10", "2023-07-14", "2023-07-17"]

    # Option 3: all available dates
    date = "all"

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    # Choose ONE option only.

    # Option 1: one specific feature
    features = "gait velocity [m/s]"

    # Option 2: selected list of features
    # features = [
    #     "gait velocity [m/s]",
    #     "stride length [m]",
    #     "cadence_[steps_per_min]",
    # ]

    # Option 3: all default features
    # features = "all"

    # ------------------------------------------------------------------
    # Wearing-time control
    # ------------------------------------------------------------------
    check_wearing_time = True
    min_wearing_time_hours = 8.0

    # ------------------------------------------------------------------
    # Plot options
    # ------------------------------------------------------------------
    include_very_short = False
    min_wb = 1 #Minum number of WB to be plotted
    bins = 30
    show = False
    dpi = 150

    output_folder = (
        patient_directory
        / "outputs_"
        / "KDE_plots"
    )

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

    if "recording_date" not in wb_parameters_average.columns:
        raise KeyError("Column 'recording_date' not found in wb_parameters_average")

    available_dates = sorted(
        wb_parameters_average["recording_date"]
        .dropna()
        .astype(str)
        .unique()
    )

    print("\nAvailable dates in WB-level input:")
    for available_date in available_dates:
        print(f"  {available_date}")

    # ------------------------------------------------------------------
    # Load optional day-level input
    # ------------------------------------------------------------------
    day_dmos = None

    if check_wearing_time:
        if not day_dmos_path.exists():
            raise FileNotFoundError(
                f"Day-level DMO file not found: {day_dmos_path}"
            )

        day_dmos = pd.read_csv(day_dmos_path)

        print("\nLoaded day-level input:")
        print(f"  file: {day_dmos_path}")
        print(f"  rows: {len(day_dmos)}")

        if "wearing_time_hours" not in day_dmos.columns:
            raise KeyError("Column 'wearing_time_hours' not found in day_dmos")

    # ------------------------------------------------------------------
    # Run KDE plotter
    # ------------------------------------------------------------------
    kde = KDEPlotter(
        wb_parameters_average=wb_parameters_average,
        patient_id=patient_id,
        day_dmos=day_dmos,
        date=date,
        features=features,
        check_wearing_time=check_wearing_time,
        min_wearing_time_hours=min_wearing_time_hours,
        include_very_short=include_very_short,
    )

    kde.run(
        output_folder=output_folder,
        min_wb=min_wb,
        bins=bins,
        show=show,
        dpi=dpi,
        plot_histogram=False,
        save_json=True,
    )

    print("\nKDE plotting completed.")
    print(f"  output folder: {output_folder}")

    return {
        "kde": kde,
        "wb_parameters_average": wb_parameters_average,
        "day_dmos": day_dmos,
    }


if __name__ == "__main__":
    results = main()