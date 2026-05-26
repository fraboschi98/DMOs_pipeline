# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Example script: collect DMOs pipeline outputs before downstream analysis.

The script collects the already processed outputs for one patient. The user can
choose one of three date-selection modes:

1. One specific recording date.
2. A selected list of recording dates.
3. All available recording dates.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import CollectorCSV


def main():
    """
    Select patient/date data and collect the available DMOs pipeline outputs.
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
    # Collect data
    # ------------------------------------------------------------------
    collector = CollectorCSV(
        patient_id=patient_id,
        date=date,
        patient_directory=patient_directory,
    )

    collector.collect()
    collector.summary()

    # ------------------------------------------------------------------
    # Collected outputs for downstream analysis
    # ------------------------------------------------------------------
    parameters = collector.parameters
    wb_dataframe = collector.wb_dataframe
    wb_pauses_dataframe = collector.wb_pauses_dataframe
    logs = collector.log

    return parameters, wb_dataframe, wb_pauses_dataframe, logs


if __name__ == "__main__":
    parameters, wb_dataframe, wb_pauses_dataframe, logs = main()