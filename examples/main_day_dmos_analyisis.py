# -*- coding: utf-8 -*-
"""
Example script: compute day-level DMOs from an existing WB-level file.

This script must be run after the WB-level analysis script has already created
a `wb_parameters_average` CSV file.

The user specifies the input WB-level file, for example:

    PAT401_multiple_dates_wb_parameters_average.csv

The script:
1. loads the existing WB-level file;
2. prints the recording dates available in the file;
3. uses Collector to collect the corresponding logs;
4. computes day-level DMOs;
5. saves the final day_dmos table.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import Collector
from dmos_pipeline import DayLevel_analyzer


def main():
    """
    Load an existing WB-level file, collect logs, and compute day-level DMOs.
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
    input_filename = "PAT401_multiple_dates_wb_parameters_average.csv"

    wb_parameters_average_path = input_folder / input_filename
    output_folder = patient_directory / "outputs_"

    # ------------------------------------------------------------------
    # Day-level analysis options
    # ------------------------------------------------------------------
    use_valid_strides_for_distance = False
    aggregation = "mean"
    include_very_short = False

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
    wb_pauses_dataframe = collector.wb_pauses_dataframe

    print("\nCollected logs:")
    print(f"  logs loaded: {len(log)}")
    print(f"  wb_pauses_dataframe rows: {len(wb_pauses_dataframe)}")

    # ------------------------------------------------------------------
    # Compute day-level DMOs
    # ------------------------------------------------------------------
    day = DayLevel_analyzer(
        wb_parameters_average=wb_parameters_average,
        log=log,
        wb_pauses_dataframe=wb_pauses_dataframe,
    )

    day.run(
        use_valid_strides_for_distance=use_valid_strides_for_distance,
        aggregation=aggregation,
        include_very_short=include_very_short,
    )

    # ------------------------------------------------------------------
    # Save day-level output
    # ------------------------------------------------------------------
    saved_day_dmos_path = day.save_day_dmos(
        output_folder=output_folder
    )

    print("\nSaved day-level output:")
    print(f"  day_dmos: {saved_day_dmos_path}")

    # ------------------------------------------------------------------
    # Outputs available in Spyder
    # ------------------------------------------------------------------
    day_dmos = day.day_dmos

    print("\nDay-level analysis:")
    print(f"  day_dmos rows: {len(day_dmos)}")

    return {
        "wb_parameters_average": wb_parameters_average,
        "day_dmos": day_dmos,
        "log": log,
        "wb_pauses_dataframe": wb_pauses_dataframe,
        "saved_day_dmos_path": saved_day_dmos_path,
    }


if __name__ == "__main__":
    results = main()