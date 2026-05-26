# -*- coding: utf-8 -*-
"""
Example script: collect processed DMOs pipeline outputs and compute
walking-bout-level analysis tables.

The script has two stages:

1. Collect already processed outputs for one patient.
2. Analyze the collected walking bouts and compute WB-level average parameters.

The user can choose one of three date-selection modes:

1. One specific recording date.
2. A selected list of recording dates.
3. All available recording dates.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import Collector
from dmos_pipeline import WalkingBouts_analyzer

def main():
    """
    Collect patient/date data and compute walking-bout-level outcomes.
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
    # Date selection
    # ------------------------------------------------------------------
    # Choose ONE option only.

    # Option 1: one specific date
    # date = "2023-07-14"

    # Option 2: selected list of dates
    date = ["2023-07-10", "2023-07-14", "2023-07-17"]

    # Option 3: all available date folders
    # date = "all"

    # ------------------------------------------------------------------
    # Step 1: collect processed outputs
    # ------------------------------------------------------------------
    collector = Collector(
        patient_id=patient_id,
        date=date,
        patient_directory=patient_directory,
    )

    collector.collect()
    collector.summary()

    # ------------------------------------------------------------------
    # Step 2: analyze walking bouts
    # ------------------------------------------------------------------
    wb = WalkingBouts_analyzer(
        parameters=collector.parameters,
        wb_dataframe=collector.wb_dataframe,
        #wb_pauses_dataframe=collector.wb_pauses_dataframe,
    )

    wb.run(
        use_quality_check=True,
        use_valid_strides_for_cadence=False,
        pa_state_choice="modified",
    )
    
    # ------------------------------------------------------------------
    # Step 3: save walking-bout analysis output
    # ------------------------------------------------------------------
    analysis_output_folder = patient_directory / "outputs_"
    
    saved_wb_average_path = wb.save_wb_parameters_average(
        output_folder=analysis_output_folder
    )
    
    print("\nSaved walking-bout analysis output:")
    print(f"  wb_parameters_average: {saved_wb_average_path}")

    # ------------------------------------------------------------------
    # Outputs for downstream analysis
    # ------------------------------------------------------------------
    parameters = wb.parameters
    parameters_before_cleaning = wb.parameters_before_cleaning
    wb_dataframe = wb.wb_dataframe
    wb_pauses_dataframe = wb.wb_pauses_dataframe
    wb_parameters_average = wb.wb_parameters_average
    log = collector.log

    print("\nWalking-bout analysis:")
    print(f"  parameters rows after cleaning: {len(parameters)}")
    print(f"  parameters rows before cleaning: {len(parameters_before_cleaning)}")
    print(f"  wb_dataframe rows: {len(wb_dataframe)}")
    print(f"  wb_pauses_dataframe rows: {len(wb_pauses_dataframe)}")
    print(f"  wb_parameters_average rows: {len(wb_parameters_average)}")

    return {
        "parameters": parameters,
        "parameters_before_cleaning": parameters_before_cleaning,
        "wb_dataframe": wb_dataframe,
        "wb_pauses_dataframe": wb_pauses_dataframe,
        "wb_parameters_average": wb_parameters_average,
        "log": log,
    }


if __name__ == "__main__":
    results = main()