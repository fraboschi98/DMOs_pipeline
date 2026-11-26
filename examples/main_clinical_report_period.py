
# -*- coding: utf-8 -*-
"""
Example script: build the DMOs clinical report (PDF) for one patient.

This script must be run after:
1. the WB-level analysis (main_dmos_extraction.py), which creates
       <patient_id>_multiple_dates_wb_parameters_average.csv
2. the day-level analysis (main_day_dmos_analyisis.py), which creates
       <patient_id>_day_dmos.csv

The script:
1. loads both files for one patient and one period (max 7 days);
2. builds the report pages with DMOsReport;
3. saves one PDF (and optionally one PNG per page).

Settings that must match the pipeline:
- VARIABILITY_WB below must match `asym_label` used in DayLevel_analyzer.run()
- ASYM_MIN_STRIDES in dmos_pipeline/clinical_report.py must match
  `min_strides_per_foot_asym` used in WalkingBouts_analyzer.run()
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dmos_pipeline import DMOsReport


def main():
    """
    Build and save the clinical report for one patient and period.
    """

    # ------------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------------
    patient_id = "PAT401"

    project_folder = Path(
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit"
    )

    patient_directory = project_folder / patient_id
    input_folder = patient_directory / "outputs_"

    day_dmos_path = input_folder / f"{patient_id}_day_dmos.csv"
    wb_level_path = input_folder / f"{patient_id}_multiple_dates_wb_parameters_average.csv"

    output_folder = project_folder / "prova_clinical_report"
    

    # Period (max 7 days) - choose ONE option
    # dates = "all"                                  # all available days
    # dates = ["2023-07-10", "2023-07-14"]           # selected days
    dates = ("2023-07-10", "2023-07-16")             # from day -> to day (inclusive)

    # ------------------------------------------------------------------
    # Report options
    # ------------------------------------------------------------------
    sections = {"amount_pattern": True, "variability": True, "kde": True, "pauses": True}
    amount_panels = {"wear_strip": True, "time": True, "wb_counts": True,
                     "strides_distance": True}
    pause_panels = {"counts": True, "kde": True}

    show_values = True              # value labels on bars / dots / lines
    wear_valid_hours = None         # e.g. 12 -> tick on the wear strip
    variability_wb = "long"         # "short", "medium" or "long"
    kde_min_wb = 5                  # min WBs of a category to draw its curve
    pause_kde_by_day = False        # True = one pause curve per day
    pause_reference_lines = False   # True = 1 / 5 / 10 min lines on pause KDE

    save_png = False                # also save one PNG per page
    show_plots = True               # open the pages in a window

    # ------------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------------
    for path in (day_dmos_path, wb_level_path):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    print("\nInputs:")
    print(f"  day-level DMOs: {day_dmos_path}")
    print(f"  WB-level:       {wb_level_path}")

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    plt.close("all")

    report = DMOsReport(
        patient_id=patient_id,
        day_dmos_path=day_dmos_path,
        wb_level_path=wb_level_path,
        dates=dates,
        sections=sections,
        amount_panels=amount_panels,
        pause_panels=pause_panels,
        show_values=show_values,
        wear_valid_hours=wear_valid_hours,
        variability_wb=variability_wb,
        kde_min_wb=kde_min_wb,
        pause_kde_by_day=pause_kde_by_day,
        pause_reference_lines=pause_reference_lines,
        
    )

    report.build()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    pdf_path = report.save_pdf(output_folder / f"{patient_id}_DMOs_report.pdf")

    png_paths = []
    if save_png:
        png_paths = report.save_pngs(output_folder)

    if show_plots:
        plt.show()

    # ------------------------------------------------------------------
    # Outputs available in Spyder
    # ------------------------------------------------------------------
    return {
        "report": report,
        "day_df": report.day_df,
        "wb_df": report.wb_df,
        "pauses_df": report.pauses_df,
        "stats": report.stats,
        "pdf_path": pdf_path,
        "png_paths": png_paths,
    }


if __name__ == "__main__":
    results = main()