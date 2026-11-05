

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:31:18 2026

@author: francesca.boschi
"""

# -*- coding: utf-8 -*-
"""
Example script: run gaitmap processing followed by quality control.

This script:
1. Loads one gaitmap-compatible IMU signal file.
2. Runs the gaitmap-based processing pipeline.
3. Saves gait events, stride-level parameters, and the processing log.
4. Runs quality control on the exported events and parameters.
5. Updates the exported CSV files and the processing log.
6. Extracts walking bouts and pauses.
7. Optionally plots every walking-bout pause.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import re
import pandas as pd

from dmos_pipeline import GaitMapPipeline, QualityCheck, WBpipeline, PausePlotter


def parse_filename_metadata(signal_path: str):
    """
    Extract patient ID, session ID, and recording date from the signal filename.

    Expected filename format
    ------------------------
    PAT401_week_3_2023-07-10_gaitMAP_bf_all.csv
    """

    filename = Path(signal_path).name

    pattern = r"^(PAT\d+)_([A-Za-z0-9_]+)_(\d{4}-\d{2}-\d{2})_gaitMAP"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(
            "Filename does not match the expected pattern: "
            f"{filename}"
        )

    patient_id = match.group(1)
    session_id = match.group(2)
    recording_date = match.group(3)

    return patient_id, session_id, recording_date


def main():
    # ---------------------------------------------------------------------
    # User inputs
    # ---------------------------------------------------------------------
    signal_path = Path(
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\Desktop\Luxembourg_Analysis\HOME_MONITORING\PAT401\week_1\2023-07-14\PAT401_week_1_2023-07-14_gaitMAP_bf_all.csv"
    )

    project_folder = Path(
        r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova_dmos_kit"
    )

    if not signal_path.exists():
        raise FileNotFoundError(
            "Please update `signal_path` with the path to your gaitmap-compatible "
            "IMU signal file."
        )

    project_folder.mkdir(parents=True, exist_ok=True)

    gaitmap_config = {
        "sampling_rate_hz": 102.4,
    }

    quality_check_config = {
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,
        "ic_threshold": 0.0,
        "events_quality_col": "quality_check(IC<0)",
        "notes_col": "notes",
        "turning_angle_abs_range": (25.0, 90.0),
        "parameter_rules": {
            "stride time [s]": (0.2, 3.0),
            "gait velocity [m/s]": (0.2, 2.0),
            "stride length [m]": (0.10, 1.5),
        },
        "n_parameter_violations": 2,
        "apply_events_ic_check": True,
        "apply_turning_angle_check": True,
        "apply_parameter_outlier_check": True,
    }

    wb_config = {
        "wb_pause_s_threshold": 3.0,
        "threshold_one_side": 3,
        "threshold_two_sides": 5,
        "use_only_quality_checked_events": True,
        "event_quality_column": "quality_check(IC<0)",
    }

    plot_pauses = True

    pause_plot_config = {
        "sampling_rate_hz": 102.4,
        "channel": "gyr_ml",
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,
        "samples_before_pause": 500,
        "samples_after_pause": 500,
        "show_plot": False,
        "save_plots": True,
        "figsize": (15, 5),
        "dpi": 150,
        "use_only_quality_checked_events": True,
        "event_quality_column": "quality_check(IC<0)",
    }

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------
    patient_id, session_id, recording_date = parse_filename_metadata(signal_path)

    # ---------------------------------------------------------------------
    # Load signal
    # ---------------------------------------------------------------------
    signal_raw = pd.read_csv(signal_path, header=[0, 1], index_col=0)
    signal_raw = signal_raw.reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Step 1: gaitmap-based processing
    # ---------------------------------------------------------------------
    pipeline = GaitMapPipeline(
        signal_raw=signal_raw,
        config=gaitmap_config,
        patient_id=patient_id,
        session_id=session_id,
        recording_date=recording_date,
        output_root=project_folder,
    )

    pipeline.filter_signal()
    pipeline.run_gaitmap_pipeline()
    pipeline.compute_recording_and_wearing_time()

    saved_paths = pipeline.save_outputs(project_folder=project_folder)

    # ---------------------------------------------------------------------
    # Step 2: quality check
    # ---------------------------------------------------------------------
    quality_check = QualityCheck(
        signal_path=None,
        events_path=saved_paths["events"],
        parameters_path=saved_paths["parameters"],
        project_folder=project_folder,
        signal_filtered=pipeline.signal_filtered,
        config=quality_check_config,
    )
    quality_check.run()

    # ---------------------------------------------------------------------
    # Step 3: walking-bout extraction
    # ---------------------------------------------------------------------
    session_folder = saved_paths["events"].parent

    wb_pipeline = WBpipeline(
        session_folder=session_folder,
        config=wb_config,
    )

    wb_pipeline.load_events()
    wb_pipeline.detect_wb_pauses()
    wb_pipeline.extract_walking_bouts()

    wb_saved_paths = wb_pipeline.save_outputs()

    # ---------------------------------------------------------------------
    # Step 4: plot walking-bout pauses
    # ---------------------------------------------------------------------
    pauses_path = wb_saved_paths.get("wb_pauses_dataframe")
    breaks_path = wb_saved_paths.get("signal_break_dataframe")

    if plot_pauses and pauses_path is not None and breaks_path is not None:
        pause_plotter = PausePlotter(
            signal_raw=signal_raw,
            pauses_path=pauses_path,
            events_path=saved_paths["events"],
            breaks_path=breaks_path,
            output_folder=session_folder / "pauses_plot",
            config=pause_plot_config,
        )
        pause_plot_paths = pause_plotter.run()
        print(
            f"Saved {len(pause_plot_paths)} pause plots in: "
            f"{session_folder / 'pauses_plot'}"
        )
    elif plot_pauses:
        print("No pauses/breaks file saved for this recording - skipping pause plots.")


if __name__ == "__main__":
    main()