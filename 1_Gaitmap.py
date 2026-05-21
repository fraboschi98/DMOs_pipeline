# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:31:16 2026

@author: francesca.boschi

"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Callable
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import json, datetime
import traceback as tb
from gaitmap.gait_detection import UllrichGaitSequenceDetection
from gaitmap.stride_segmentation import BarthDtw, RoiStrideSegmentation
from gaitmap.event_detection import RamppEventDetection
from gaitmap.trajectory_reconstruction import RegionLevelTrajectory, RtsKalman
from gaitmap.utils.datatype_helper import get_multi_sensor_names
from gaitmap.parameters import TemporalParameterCalculation
from gaitmap.parameters import SpatialParameterCalculation 
import traceback as tb
import re
from scipy.ndimage import uniform_filter1d
import os
from copy import deepcopy


class GaitMapPipeline:
    """
    End-to-end pipeline for IMU-based gait analysis using gaitmap framework by Küderle et al. (2024).
    
    This class processes bilateral foot-mounted IMU signals to detect gait
    sequences, segment strides, identify gait events, reconstruct trajectories,
    and compute stride-level temporal and spatial parameters.
    
    It was developed as a preprocessing framework for extracting Digital Mobility
    Outcomes (DMOs) from real-world recordings using gaitmap algorithms. Since gaitmap was primarily
    designed for structured laboratory data, this pipeline adds additional
    processing steps for free-living recordings, including signal filtering,
    wearing-time estimation, diagnostic visualization, processing logs, and export
    of intermediate and final results.
    """
    # Default processing parameters. Values can be selectively overridden by user_config.
    DEFAULT_CONFIG = {
        
        # ---- General ----
        "sampling_rate_hz": 102.4,

        # ---- Signal filtering ----
        "cutoff_freq_gyr": 5.0,
        "filter_order_gyr": 4,
        "cutoff_freq_acc": 10.0,
        "filter_order_acc": 4,

        # ---- Gait sequence detection ----
        "sensor_channel_config": "gyr_ml",
        "peak_prominence": 10,
        "window_size_s": 10.0,
        "active_signal_threshold": 10,
        "additional_margin_s": None,
        "locomotion_band": (0.4, 3.0),
        "harmonic_tolerance_hz": 0.8,

        # ---- Stride segmentation ----
        "dtw_find_matches_method": "find_peaks",
        "dtw_max_cost": 4.0,
        "dtw_min_match_length_s": 0.6,
        "dtw_max_match_length_s": 3.0,
        "dtw_max_template_stretch_ms": None,
        "dtw_max_signal_stretch_ms": None,
        "dtw_snap_to_min_win_ms": 300,
        "dtw_snap_to_min_axis": "gyr_ml",
        "dtw_conflict_resolution": True,

        # ---- Rampp event detection ----
        "rampp_ic_search_region_ms": (80, 50),
        "rampp_min_vel_search_win_size_ms": 100,
        "rampp_enforce_consistency": True,
        "rampp_detect_only": None,

        # ---- Trajectory reconstruction ----
        "ori_method": None,
        "pos_method": None,
        "steady_duration_s": 0.0,
        "trim_ratio": 0.0,
        
        # ---- Recording / wearing time ----
        "acc_wearing_threshold": 0.05 * 9.81,
        "gyr_wearing_threshold": 2.0,
        "wearing_initial_window_min": 0,
        "wearing_window_size_min": 30,
        "wearing_overlap": 0.5,
        "wearing_highpass_cutoff_hz": 0.25,
        "wearing_highpass_order": 4,
        "wearing_smoothing_window_s": 3,

    }

    def __init__(
        self,
        signal_raw: Any,
        config: Optional[Dict[str, Any]] = None,
        path_config: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        recording_date: Optional[str] = None,
        output_root: Optional[str] = None,
    ):
        """
        Initialize the gaitmap-based processing pipeline.
    
        Parameters
        ----------
        signal_raw : pandas.DataFrame
            Raw bilateral IMU signal in gaitmap-compatible format.
    
        config : dict, optional
            User-defined configuration values used to override DEFAULT_CONFIG.
    
        patient_id : str, optional
            Patient or participant identifier.
    
        session_id : str, optional
            Recording session identifier.
    
        recording_date : str, optional
            Recording date.
    
        output_root : str, optional
            Root directory where pipeline outputs can be saved.
        """

        # Input data
        self.signal_raw = signal_raw

        # Configuration
        self.config = self._build_config(config)
        self.fs = self.config["sampling_rate_hz"]
        
        # Metadata
        self.patient_id = patient_id
        self.session_id = session_id
        self.recording_date = recording_date
        self.output_root = Path(output_root) if output_root is not None else None
        
        #Signal attributes
        self.signal_filtered = None
        self.timestamps_unix = None
        self.acc_norm = None
        self.gyr_norm = None
        
        # Gaitmap outputs
        self.gs = None
        self.stride_list = None
        self.events = None
        self.events_clean = None
        self.temporal_left = None
        self.temporal_right = None
        self.spatial_left = None
        self.spatial_right = None
        self.orientations = None
        self.positions = None
        
        # Quality-control outputs
        self.removed_events = {
            "left_sensor": pd.DataFrame(),
            "right_sensor": pd.DataFrame(),
        }
        
        # Recording and wearing-time outputs
        self.recording_time_hours = None
        self.wearing_time_hours = None
        self.recording_time_str = None
        self.wearing_time_str = None
        self.merged_windows = []
                        
        # Log
        self.log = {
            "config": deepcopy(self.config),
            "meta": {
                "patient_id": self.patient_id,
                "session_id": self.session_id,
                "recording_date": self.recording_date,
            },
            "events": [],
        }

    @classmethod
    def _build_config(cls, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge default configuration with user-defined overrides.

        The user may override any key in DEFAULT_CONFIG.
        Unknown keys are rejected to avoid silent mistakes.
        """

        user_config = user_config or {}

        unknown_keys = set(user_config) - set(cls.DEFAULT_CONFIG)
        if unknown_keys:
            raise ValueError(
                "Unknown configuration key(s): "
                f"{sorted(unknown_keys)}. "
                "Please use only keys declared in DEFAULT_CONFIG."
            )

        config = deepcopy(cls.DEFAULT_CONFIG)
        config.update(user_config)

        cls._validate_config(config)

        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the final configuration after merging defaults and user overrides.
        """

        if config["sampling_rate_hz"] <= 0:
            raise ValueError("sampling_rate_hz must be greater than 0.")

        if config["cutoff_freq_gyr"] <= 0:
            raise ValueError("cutoff_freq_gyr must be greater than 0.")

        if config["cutoff_freq_acc"] <= 0:
            raise ValueError("cutoff_freq_acc must be greater than 0.")

        if config["filter_order_gyr"] <= 0:
            raise ValueError("filter_order_gyr must be greater than 0.")

        if config["filter_order_acc"] <= 0:
            raise ValueError("filter_order_acc must be greater than 0.")


        if config["dtw_min_match_length_s"] <= 0:
            raise ValueError("dtw_min_match_length_s must be greater than 0.")

        if config["dtw_max_match_length_s"] <= 0:
            raise ValueError("dtw_max_match_length_s must be greater than 0.")

        if config["dtw_min_match_length_s"] >= config["dtw_max_match_length_s"]:
            raise ValueError(
                "dtw_min_match_length_s must be smaller than dtw_max_match_length_s."
            )

        locomotion_band = config["locomotion_band"]
        if (
            not isinstance(locomotion_band, tuple)
            or len(locomotion_band) != 2
            or locomotion_band[0] <= 0
            or locomotion_band[1] <= locomotion_band[0]
        ):
            raise ValueError(
                "locomotion_band must be a tuple like (0.4, 3.0), "
                "with lower frequency smaller than upper frequency."
            )

    @staticmethod
    def _butter_lowpass_filter(data, cutoff, order, fs):
        """Apply a low-pass Butterworth filter to a one-dimensional signal."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data, method="gust")    


    def filter_signal(self):
        """
        Apply low-pass Butterworth filtering to the available IMU channels.
        
        Gyroscope and accelerometer channels are filtered separately using the
        cutoff frequencies and filter orders defined in the pipeline configuration.
        
        The filtered signal is stored in `self.signal_filtered`.
        
        Returns
        -------
        self
            The pipeline instance, allowing method chaining.
        """
        try:
            df = self.signal_raw.copy()
    
            fs = self.fs
            cutoff_gyr = self.config["cutoff_freq_gyr"]
            order_gyr = self.config["filter_order_gyr"]
            cutoff_acc = self.config["cutoff_freq_acc"]
            order_acc = self.config["filter_order_acc"]
    
            
    
            columns = [
                ("left_sensor", "gyr_pa"), ("left_sensor", "gyr_ml"), ("left_sensor", "gyr_si"),
                ("left_sensor", "acc_pa"), ("left_sensor", "acc_ml"), ("left_sensor", "acc_si"),
                ("right_sensor", "gyr_pa"), ("right_sensor", "gyr_ml"), ("right_sensor", "gyr_si"),
                ("right_sensor", "acc_pa"), ("right_sensor", "acc_ml"), ("right_sensor", "acc_si"),
            ]
    
            df_filtered = df.copy()
    
            for col in columns:
                if col not in df.columns:
                    
                    continue
    
                data = df[col].values
                cutoff, order = (
                    (cutoff_gyr, order_gyr) if "gyr" in col[1] else (cutoff_acc, order_acc)
                )
    
                df_filtered[col] = self._butter_lowpass_filter(data, cutoff, order, fs)
    
            self.signal_filtered = df_filtered
    
            # Updating Log file
            self.log["events"].append("filter_signal: completed")
    
    
            return self
    
        except Exception as e:
    
            
            self.log["events"].append(f"filter_signal: ERROR {str(e)}")
    
            
            raise

    def plot_raw_filtered_channel(
        self,
        channel: str = "gyr_ml",
        start: Optional[int] = None,
        end: Optional[int] = None,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """
        Plot raw and filtered signals for one IMU channel on both sensors.
    
        The method creates two subplots:
        one for the left sensor and one for the right sensor.
    
        Parameters
        ----------
        channel : str, default="gyr_ml"
            IMU channel to plot.
    
        start : int, optional
            First sample index to include.
    
        end : int, optional
            Last sample index to include.
    
        figsize : tuple, default=(12, 6)
            Figure size.
    
        save_path : str, optional
            Path where the figure is saved.
    
        show : bool, default=True
            If True, display the figure. If False, close it after saving.
        """

    
        if self.signal_raw is None:
            raise ValueError("self.signal_raw is None.")
    
        if self.signal_filtered is None:
            raise ValueError("self.signal_filtered is None. Run filtering first.")
    
        left_col = ("left_sensor", channel)
        right_col = ("right_sensor", channel)
    
        for col in [left_col, right_col]:
            if col not in self.signal_raw.columns:
                raise KeyError(f"Column {col} not found in self.signal_raw.")
            if col not in self.signal_filtered.columns:
                raise KeyError(f"Column {col} not found in self.signal_filtered.")
    
        start = 0 if start is None else start
        end = len(self.signal_raw) if end is None else end
    
        raw_left = self.signal_raw.loc[:, left_col].iloc[start:end]
        filt_left = self.signal_filtered.loc[:, left_col].iloc[start:end]
    
        raw_right = self.signal_raw.loc[:, right_col].iloc[start:end]
        filt_right = self.signal_filtered.loc[:, right_col].iloc[start:end]
    
        
        x = np.arange(start, end)
        x_label = "Samples"
    
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
        # Left sensor
        axes[0].plot(x, raw_left.values, label="Raw")
        axes[0].plot(x, filt_left.values, label="Filtered")
        axes[0].set_title(f"Left sensor - {channel}")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        axes[0].legend()
    
        # Right sensor
        axes[1].plot(x, raw_right.values, label="Raw")
        axes[1].plot(x, filt_right.values, label="Filtered")
        axes[1].set_title(f"Right sensor - {channel}")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel(x_label)
        axes[1].grid(True)
        axes[1].legend()
    
        fig.suptitle(f"Raw vs Filtered - Channel: {channel}")
        plt.tight_layout()
    
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
        plt.show()
    def _compute_trajectories(self, df, fs, gs_list, stride_list):
        """
        Reconstruct stride-level orientations and positions.
    
        This helper prepares the filtered IMU signal for gaitmap trajectory
        reconstruction and applies `RegionLevelTrajectory` with `RtsKalman`.
    
        Optional preprocessing steps can be applied before reconstruction:
    
        1. Static padding
           A short artificial steady-state segment can be added before the signal.
           This is controlled by `steady_duration_s`. When padding is applied, gait
           sequence and stride indices are shifted by the same number of samples.
    
        2. Gait-sequence trimming
           The start and end of each gait sequence can be trimmed before trajectory
           reconstruction. This is controlled by `trim_ratio`.
    
        3. Column renaming
           IMU channel names are converted from the body-frame convention used in
           this pipeline to the axis names expected by gaitmap:
           `acc_pa`, `acc_ml`, `acc_si` become `acc_x`, `acc_y`, `acc_z`;
           `gyr_pa`, `gyr_ml`, `gyr_si` become `gyr_x`, `gyr_y`, `gyr_z`.
    
        
    
        Parameters
        ----------
        df : pandas.DataFrame
            Filtered IMU signal with bilateral sensor columns.
    
        fs : float
            Sampling frequency in Hz.
    
        gs_list : dict
            Gait sequence tables for each sensor side.
    
        stride_list : dict
            Stride tables for each sensor side.
    
        Returns
        -------
        orientations : dict
            Reconstructed stride-level orientations for each sensor side.
    
        positions : dict
            Reconstructed stride-level positions for each sensor side.
    
        Notes
        -----
        If trajectory reconstruction fails, the error is logged and two empty
        dictionaries are returned.
        """

    
        # Parameters from config (trajectories)
        steady_duration = self.config["steady_duration_s"]
        trim_ratio = self.config["trim_ratio"]
        ori_method = self.config["ori_method"]
        pos_method = self.config["pos_method"]
    
        n_steady = int(steady_duration * fs)
    
        # ---------------------------------------------------------
        # 1) STATIC PADDING (if steady duration > 0)
        # ---------------------------------------------------------
        df_extended_list = []
        if steady_duration > 0:
            self.log["events"].append(f"Traj: static padding applied ({steady_duration}s → {n_steady} samples).")
        else:
            self.log["events"].append("Traj: no static padding applied.")
    
        for side in ["left_sensor", "right_sensor"]:
            if side in df.columns.get_level_values(0):
                df_side = df[side]
                if steady_duration > 0:
                    mean_vals = df_side.iloc[:100].mean().to_frame().T
                    steady_block = pd.concat([mean_vals] * n_steady, ignore_index=True)
                    steady_block.columns = df_side.columns
                    df_side_extended = pd.concat([steady_block, df_side], ignore_index=True)
                else:
                    df_side_extended = df_side.copy()
    
                df_side_extended.columns = pd.MultiIndex.from_product([[side], df_side.columns])
                df_extended_list.append(df_side_extended)
    
        if not df_extended_list:
            self.log["events"].append("Traj: no IMU sides available for reconstruction.")
            return {}, {}
    
        df_extended = pd.concat(df_extended_list, axis=1)
    
        
        # Shift GS + Strides if padding was applied
       
        if steady_duration > 0:
            gs_extended = {k: v.assign(start=v["start"] + n_steady, end=v["end"] + n_steady)
                           for k, v in gs_list.items()}
            stride_extended = {k: v.assign(start=v["start"] + n_steady, end=v["end"] + n_steady)
                               for k, v in stride_list.items()}
            self.log["events"].append("Traj: GS and stride indices shifted due to padding.")
        else:
            gs_extended = {k: v.copy() for k, v in gs_list.items()}
            stride_extended = {k: v.copy() for k, v in stride_list.items()}
    
        # ---------------------------------------------------------
        # 2) TRIMMING
        # ---------------------------------------------------------
        trim = int(trim_ratio * fs)
        if trim_ratio > 0:
            self.log["events"].append(f"Traj: trimming applied ({trim_ratio*100:.1f}% → {trim} samples).")
        else:
            self.log["events"].append("Traj: no trimming applied.")
    
        gs_trimmed = {
            side: roi.assign(
                start=(roi["start"] + trim).clip(0),
                end=(roi["end"] - trim).clip(lower=roi["start"] + 5),
            )
            for side, roi in gs_extended.items()
        }
    
        # ---------------------------------------------------------
        # 3) Columns rename
        # ---------------------------------------------------------
        col_map = {
            "acc_pa": "acc_x", "acc_ml": "acc_y", "acc_si": "acc_z",
            "gyr_pa": "gyr_x", "gyr_ml": "gyr_y", "gyr_si": "gyr_z",
        }
        df_extended = df_extended.rename(columns=col_map, level=1)
        self.log["events"].append("Traj: column rename applied for gaitmap compatibility.")
    
        
        try:
            traj = RegionLevelTrajectory(
                trajectory_method=RtsKalman(),
                ori_method=ori_method,
                pos_method=pos_method,
            )
    
            traj.estimate(df_extended, regions_of_interest=gs_trimmed, sampling_rate_hz=fs)
            traj_stride = traj.clone()
            traj_stride.estimate_intersect(
                data=df_extended,
                regions_of_interest=gs_trimmed,
                stride_event_list=stride_extended,
                sampling_rate_hz=fs,
            )
    
            self.log["events"].append("Traj: reconstruction completed successfully.")
            return traj_stride.orientation_, traj_stride.position_
    
        except Exception as e:
            self.log["events"].append(f"Traj ERROR: {str(e)}")
            self.log["events"].append(tb.format_exc())
            return {}, {}
    def _apply_removed_sids(self):
        """
        Remove all removed s_id from temporal and event tables.
    
        Uses:
            self.removed_events["left_sensor"].index
            self.removed_events["right_sensor"].index
        """
    
        if not hasattr(self, "removed_events"):
            return
    
        for side in ["left_sensor", "right_sensor"]:
    
            if side not in self.removed_events:
                continue
    
            removed_df = self.removed_events[side]
    
            if not isinstance(removed_df, pd.DataFrame) or removed_df.empty:
                continue
    
            removed_sids = removed_df.index
    
            if side == "left_sensor":
                if isinstance(self.temporal_left, pd.DataFrame):
                    self.temporal_left = self.temporal_left[
                        ~self.temporal_left.index.isin(removed_sids)
                    ]
    
            elif side == "right_sensor":
                if isinstance(self.temporal_right, pd.DataFrame):
                    self.temporal_right = self.temporal_right[
                        ~self.temporal_right.index.isin(removed_sids)
                    ]
    
            if (
                hasattr(self, "events")
                and side in self.events
                and isinstance(self.events[side], pd.DataFrame)
            ):
                self.events[side] = self.events[side][
                    ~self.events[side].index.isin(removed_sids)
                ]
    def _extract_bad_tuples_from_keyerror(self, msg):
        # regex for tuples like (291, 100)
        pattern = r"\((\d+)\s*,\s*(\d+)\)"
        matches = re.findall(pattern, msg)
    
        if not matches:
            return None  
    
        # convert to list of tuples of ints
        return [(int(a), int(b)) for a, b in matches]    
    
    
    def run_gaitmap_pipeline(self):
        """
           Run the core gaitmap-based gait analysis pipeline.
           
            This method applies the main processing steps required to extract
            stride-level gait information from filtered IMU signals. It uses gaitmap
            algorithms for gait sequence detection, stride segmentation, gait event
            detection, trajectory reconstruction, and temporal/spatial parameter
            calculation.
            
            For trajectory reconstruction, this pipeline includes optional preprocessing
            adjustments designed for real-world recordings: a configurable static padding
            period can be added before the signal, gait sequence and stride indices are
            shifted accordingly, and gait sequence boundaries can be trimmed before
            trajectory estimation. These steps are intended to improve robustness when
            working with noisy free-living data.
           
           The method expects `self.signal_filtered` to be available. Therefore,
           `filter_signal()` must be called before this method.
           
           Processing steps
           ----------------
           1. Detect gait sequences independently for the left and right sensors.
           2. Segment strides within the detected gait sequences using dynamic time
              warping.
           3. Detect gait events, including initial contact, terminal contact,
              minimum velocity, and previous initial contact.
           4. Reconstruct stride-level orientations and positions.
           5. Compute temporal gait parameters.
           6. Compute spatial gait parameters.
           7. If spatial parameter calculation fails because some stride IDs are
              missing from the reconstructed trajectories, iteratively remove only
              those problematic stride IDs from the spatial-parameter inputs.
           
           Notes
           -----
           The original gait event tables stored in `self.events` are preserved.
           Cleanup for spatial-parameter calculation is applied only to
           `self.events_clean`, `self.positions`, and `self.orientations`.
           
           Results are stored as attributes of the class.
           
           Main outputs
           ------------
           self.gs : dict
               Detected gait sequences for left and right sensors.
           
           self.stride_list : dict
               Segmented strides for left and right sensors.
           
           self.events : dict
               Detected gait events for left and right sensors.
           
           self.events_clean : dict
               Copy of the gait events after removing only the stride IDs that are
               incompatible with spatial-parameter calculation.
           
           self.orientations : dict
               Reconstructed stride-level orientations.
           
           self.positions : dict
               Reconstructed stride-level positions.
           
           self.temporal_left, self.temporal_right : pandas.DataFrame or None
               Temporal gait parameters for the left and right sensors.
           
           self.spatial_left, self.spatial_right : pandas.DataFrame or None
               Spatial gait parameters for the left and right sensors.
           
           self.removed_events : dict
               Events removed only from the spatial-parameter calculation workflow.
           
           Raises
           ------
           RuntimeError
               If `self.signal_filtered` is missing.
           
           Exception
               Re-raises unexpected errors from the global pipeline execution after
               logging them.
           """
    
        self.log["events"].append("run_gaitmap_pipeline: started")
        empty_gs = pd.DataFrame(columns=["gs_id", "start", "end"])

    
        try:
            if self.signal_filtered is None:
                raise RuntimeError("Filtered signal missing. Run filter_signal() first.")
    
            df_fil = self.signal_filtered.copy()
            fs = self.fs   
            
    
           
                
            # ----------------------------------------
            # Gait sequences (GS) detection
            # ----------------------------------------
            
            # Parameters from config (GS detection)
            sensor_channel = self.config["sensor_channel_config"]
            peak_prominence = self.config["peak_prominence"]
            window_size_s = self.config["window_size_s"]
            active_signal_threshold = self.config["active_signal_threshold"]
            additional_margin_s = self.config["additional_margin_s"]
            locomotion_band = self.config["locomotion_band"]
            harmonic_tolerance_hz = self.config["harmonic_tolerance_hz"]            

            #Empty GS dictionnary
            self.gs = {
                "left_sensor": empty_gs.copy(),
                "right_sensor": empty_gs.copy(),
            }
            
            for side in ["left_sensor", "right_sensor"]:
                try:
                    # Check if left and right sensor side exist
                    if side not in df_fil.columns.get_level_values(0):
                        self.log["events"].append(f"GS detection skipped: {side} missing.")
                        continue
            
                    # Check if  sensor channel exist
                    if sensor_channel not in df_fil[side].columns:
                        self.log["events"].append(f"GS detection skipped: channel '{sensor_channel}' missing for {side}.")
                        continue
            
                    #  Ullrich GS detection algorithm 
                    det = UllrichGaitSequenceDetection(
                        sensor_channel_config=sensor_channel,
                        peak_prominence=peak_prominence,
                        window_size_s=window_size_s,
                        active_signal_threshold=active_signal_threshold,
                        additional_margin_s=additional_margin_s,
                        locomotion_band=locomotion_band,
                        harmonic_tolerance_hz=harmonic_tolerance_hz,
                    ).detect(
                        data=df_fil[side],
                        sampling_rate_hz=fs
                    )
            
                    gsd = det.gait_sequences_.copy()
            
                    
                    if not isinstance(gsd, pd.DataFrame) or gsd.empty:
                        self.log["events"].append(f"GS detection: no GS found for {side}.")
                        self.gs[side] = empty_gs.copy()
                    else:
                       
                        if "gs_id" not in gsd.columns:
                            gsd = gsd.reset_index(drop=True)
                            gsd["gs_id"] = gsd.index
            
                        for col in ["start", "end"]:
                            if col not in gsd.columns:
                                gsd[col] = np.nan
            
                        gsd = gsd[["gs_id", "start", "end"]]
            
                        
                        for c in ["gs_id", "start", "end"]:
                            gsd[c] = pd.to_numeric(gsd[c], errors="coerce").astype("Int64")
            
                        gsd = gsd.dropna(subset=["start", "end"], how="any")
                        gsd = gsd.astype({"gs_id": "int", "start": "int", "end": "int"})
            
                        self.gs[side] = gsd
            
                except Exception as e:
                    msg = f"GS detection ERROR at {side}: {str(e)}"
                    self.log["events"].append(msg)
                    self.log["events"].append(tb.format_exc())
                    self.gs[side] = empty_gs.copy()
                                    
            # ----------------------------------------
            # Stride segmentation
            # ----------------------------------------
            
            # Parameters from config (DTW)
            dtw_find_matches_method = self.config["dtw_find_matches_method"]
            dtw_max_cost = self.config["dtw_max_cost"]
            dtw_min_match_length_s = self.config["dtw_min_match_length_s"]
            dtw_max_match_length_s = self.config["dtw_max_match_length_s"]
            dtw_max_template_stretch_ms = self.config["dtw_max_template_stretch_ms"]
            dtw_max_signal_stretch_ms = self.config["dtw_max_signal_stretch_ms"]
            dtw_snap_to_min_win_ms = self.config["dtw_snap_to_min_win_ms"]
            dtw_snap_to_min_axis = self.config["dtw_snap_to_min_axis"]
            dtw_conflict_resolution = self.config["dtw_conflict_resolution"]
                        
                    
            empty_stride = pd.DataFrame(columns=["start", "end"])
            
            # Empty stride list dictionnary                
            self.stride_list = {
                "left_sensor": empty_stride.copy(),
                "right_sensor": empty_stride.copy(),
            }
            #Dynamic Time Warping
            dtw = BarthDtw(
                find_matches_method=dtw_find_matches_method,
                max_cost=dtw_max_cost,
                min_match_length_s=dtw_min_match_length_s,
                max_match_length_s=dtw_max_match_length_s,
                max_template_stretch_ms=dtw_max_template_stretch_ms,
                max_signal_stretch_ms=dtw_max_signal_stretch_ms,
                snap_to_min_win_ms=dtw_snap_to_min_win_ms,
                snap_to_min_axis=dtw_snap_to_min_axis,
                conflict_resolution=dtw_conflict_resolution,
            )
    
            roi_seg = RoiStrideSegmentation(segmentation_algorithm=dtw)
            
            for side in ["left_sensor", "right_sensor"]:
                try:
                    
                    if self.gs[side].empty:
                        self.log["events"].append(f"Stride segmentation skipped: no GS for {side}.")
                        self.stride_list[side] = empty_stride.copy()
                        continue
            
                    
                    if side not in df_fil.columns.get_level_values(0):
                        self.log["events"].append(f"Stride segmentation skipped: {side} missing.")
                        continue
            
                    #  Segmentation
                    res = roi_seg.segment(
                        data=df_fil[side],
                        sampling_rate_hz=fs,
                        regions_of_interest=self.gs[side]
                    )
            
                    stride_df = res.stride_list_.copy()
            
                    
                    if not isinstance(stride_df, pd.DataFrame) or stride_df.empty:
                        self.log["events"].append(f"Stride segmentation: no strides found for {side}.")
                        self.stride_list[side] = empty_stride.copy()
                    else:
                        
                        for col in ["start", "end"]:
                            if col not in stride_df.columns:
                                stride_df[col] = np.nan
            
                        stride_df = stride_df[["start", "end"]].copy()
                        stride_df["start"] = pd.to_numeric(stride_df["start"], errors="coerce")
                        stride_df["end"] = pd.to_numeric(stride_df["end"], errors="coerce")
            
                        stride_df = stride_df.dropna(subset=["start", "end"], how="any")
                        stride_df["start"] = stride_df["start"].astype(int)
                        stride_df["end"] = stride_df["end"].astype(int)
            
                        self.stride_list[side] = stride_df
            
                except Exception as e:
                    msg = f"Stride segmentation ERROR at {side}: {str(e)}"
                    self.log["events"].append(msg)
                    self.log["events"].append(tb.format_exc())
                    self.stride_list[side] = empty_stride.copy()                    

            # ----------------------------------------
            # Gait Events detection
            # ----------------------------------------
            
            # Parameters from config (Rampp event detection)    
            rampp_ic_search_region_ms = self.config["rampp_ic_search_region_ms"]
            rampp_min_vel_search_win_size_ms = self.config["rampp_min_vel_search_win_size_ms"]
            rampp_enforce_consistency = self.config["rampp_enforce_consistency"]
            rampp_detect_only = self.config["rampp_detect_only"]
            
            
            required_cols = ["start", "end", "ic", "tc", "min_vel", "pre_ic"]
            
            empty_events = pd.DataFrame(columns=["s_id"] + required_cols)
            
            self.events = {
                "left_sensor": empty_events.copy(),
                "right_sensor": empty_events.copy(),
            }
            
            try:
                if self.stride_list["left_sensor"].empty and self.stride_list["right_sensor"].empty:
                    self.log["events"].append("Event detection skipped: no strides on either side.")
                    return
            
                
                ed = RamppEventDetection(
                    ic_search_region_ms=rampp_ic_search_region_ms,
                    min_vel_search_win_size_ms=rampp_min_vel_search_win_size_ms,
                    enforce_consistency=rampp_enforce_consistency,
                    detect_only=rampp_detect_only,
                    
                ).detect(
                    data=df_fil,
                    stride_list=self.stride_list,
                    sampling_rate_hz=self.fs,
                )
            
                detected = ed.min_vel_event_list_
            
                for side in ["left_sensor", "right_sensor"]:
                    ev = detected.get(side, empty_events).copy()
                
                    if not isinstance(ev, pd.DataFrame) or ev.empty:
                        self.events[side] = empty_events.copy()
                        self.log["events"].append(f"Event detection: no events found for {side}.")
                        continue
                
                    
                    for col in required_cols:
                        if col not in ev.columns:
                            ev[col] = np.nan
                
                    ev = ev[required_cols]
                
                    for col in required_cols:
                        ev[col] = pd.to_numeric(ev[col], errors="coerce")
                
                    ev = ev.dropna(subset=["start", "end"], how="any")
                
                    ev.index.name = "s_id"   # index = stride id
                    self.events[side] = ev
                    self.events_clean = {
                    "left_sensor": self.events["left_sensor"].copy(),
                    "right_sensor": self.events["right_sensor"].copy(),
                }
                    
            
            except Exception as e:
                self.log["events"].append(f"Event detection ERROR: {str(e)}")
                self.log["events"].append(tb.format_exc())
                self.events = {
                    "left_sensor": empty_events.copy(),
                    "right_sensor": empty_events.copy(),
                }
        
        
            # ----------------------------------------
            # Trajectory reconstruction
            # ----------------------------------------
            try:
                self.orientations, self.positions = self._compute_trajectories(
                    df=df_fil,
                    fs=fs,
                    gs_list=self.gs,
                    stride_list=self.stride_list,
                )
            
                
            
            except Exception as e:
                self.log["events"].append(f"Trajectory reconstruction ERROR: {str(e)}")
                self.log["events"].append(tb.format_exc())
            
            # ----------------------------------------
            # Temporal parameters
            # ----------------------------------------
            try:
                tp = TemporalParameterCalculation().calculate(
                    stride_event_list=self.events,
                    sampling_rate_hz=fs
                )
            
                
                
            
                if hasattr(tp, "parameters_pretty_") and isinstance(tp.parameters_pretty_, dict):
            
                    if "left_sensor" in tp.parameters_pretty_:
                        self.temporal_left = tp.parameters_pretty_["left_sensor"].copy()
                        self.temporal_left.index.name = "s_id"
            
                    if "right_sensor" in tp.parameters_pretty_:
                        self.temporal_right = tp.parameters_pretty_["right_sensor"].copy()
                        self.temporal_right.index.name = "s_id"
            
                self.log["events"].append("Temporal parameters computed.")
            
            except Exception as e:
                self.log["events"].append(f"Temporal parameters ERROR: {str(e)}")
                self.log["events"].append(tb.format_exc())
                self.temporal_left = None
                self.temporal_right = None
            
            # ----------------------------------------
            # Spatial parameters (with cleanup)
            # self.events is NOT modified.
            # Cleanup is applied only to self.events_clean, positions, and orientations.
            # ----------------------------------------
            try:
                max_iter = 20
                iter_count = 0
            
                self.spatial_left = None
                self.spatial_right = None
            
                self.events_clean = {
                    "left_sensor": self.events["left_sensor"].copy(),
                    "right_sensor": self.events["right_sensor"].copy(),
                }
            
                while iter_count < max_iter:
                    try:
                        sp = SpatialParameterCalculation().calculate(
                            stride_event_list=self.events_clean,
                            positions=self.positions,
                            orientations=self.orientations,
                            sampling_rate_hz=fs,
                        )
            
                        self.spatial_left = sp.parameters_pretty_.get("left_sensor", None)
                        self.spatial_right = sp.parameters_pretty_.get("right_sensor", None)
            
                        if isinstance(self.spatial_left, pd.DataFrame):
                            self.spatial_left = self.spatial_left.copy()
                            self.spatial_left.index.name = "s_id"
            
                        if isinstance(self.spatial_right, pd.DataFrame):
                            self.spatial_right = self.spatial_right.copy()
                            self.spatial_right.index.name = "s_id"
            
                        self.log["events"].append(
                            f"Spatial parameters computed successfully after {iter_count} cleanup cycles."
                        )
            
                        break
            
                    except KeyError as e:
                        msg = str(e)
                        self.log["events"].append(f"Spatial crash detected: {msg}")
            
                        bad_tuples = self._extract_bad_tuples_from_keyerror(msg)
            
                        if not bad_tuples:
                            self.log["events"].append(
                                "Unable to extract bad s_id tuples from KeyError. Aborting spatial."
                            )
                            raise
            
                        bad_sids_by_side = {
                            "left_sensor": [],
                            "right_sensor": [],
                        }
            
                        for side in ["left_sensor", "right_sensor"]:
            
                            if side not in self.events_clean or not isinstance(self.events_clean[side], pd.DataFrame):
                                continue
            
                            if side not in self.positions or not isinstance(self.positions[side], pd.DataFrame):
                                continue
            
                            if side not in self.orientations or not isinstance(self.orientations[side], pd.DataFrame):
                                continue
            
                            pos_index = self.positions[side].index
                            ori_index = self.orientations[side].index
            
                            for bad_tuple in bad_tuples:
                                sid = bad_tuple[0]
            
                                if sid not in self.events_clean[side].index:
                                    continue
            
                                missing_in_pos = bad_tuple not in pos_index
                                missing_in_ori = bad_tuple not in ori_index
            
                                if missing_in_pos or missing_in_ori:
                                    bad_sids_by_side[side].append(sid)
            
                        for side in ["left_sensor", "right_sensor"]:
                            bad_sids_by_side[side] = sorted(set(bad_sids_by_side[side]))
            
                        self.log["events"].append(
                            f"Removing bad s_id by side for spatial only: {bad_sids_by_side}"
                        )
            
                        if (
                            len(bad_sids_by_side["left_sensor"]) == 0
                            and len(bad_sids_by_side["right_sensor"]) == 0
                        ):
                            self.log["events"].append(
                                "Bad tuples were found, but no side-specific bad s_id could be assigned."
                            )
                            raise
            
                        # Store removed event rows, but do NOT remove them from self.events
                        for side in ["left_sensor", "right_sensor"]:
                            bad_in_side = bad_sids_by_side[side]
            
                            if not bad_in_side:
                                continue
            
                            removed_rows = self.events_clean[side].loc[
                                self.events_clean[side].index.intersection(bad_in_side)
                            ].copy()
            
                            if removed_rows.empty:
                                continue
            
                            if self.removed_events[side].empty:
                                self.removed_events[side] = removed_rows
                            else:
                                self.removed_events[side] = pd.concat(
                                    [self.removed_events[side], removed_rows],
                                    axis=0,
                                )
            
                            self.removed_events[side] = (
                                self.removed_events[side]
                                .sort_index()
                                .loc[~self.removed_events[side].index.duplicated(keep="first")]
                            )
            
                        # Remove bad s_id only from spatial inputs
                        for side in ["left_sensor", "right_sensor"]:
                            bad_in_side = bad_sids_by_side[side]
            
                            if not bad_in_side:
                                continue
            
                            if side in self.orientations and isinstance(self.orientations[side], pd.DataFrame):
                                df = self.orientations[side]
            
                                if "s_id" in df.index.names:
                                    self.orientations[side] = df[
                                        ~df.index.get_level_values("s_id").isin(bad_in_side)
                                    ]
            
                            if side in self.positions and isinstance(self.positions[side], pd.DataFrame):
                                df = self.positions[side]
            
                                if "s_id" in df.index.names:
                                    self.positions[side] = df[
                                        ~df.index.get_level_values("s_id").isin(bad_in_side)
                                    ]
            
                            if side in self.events_clean and isinstance(self.events_clean[side], pd.DataFrame):
                                df = self.events_clean[side]
                                self.events_clean[side] = df[
                                    ~df.index.isin(bad_in_side)
                                ]
            
                        self.log["events"].append(
                            "Bad s_id removed from spatial inputs only. Original self.events was preserved."
                        )
            
                        iter_count += 1
                        continue
            
                else:
                    self.log["events"].append(
                        f"Spatial parameters FAILED after {max_iter} cleanup cycles."
                    )
                    self.spatial_left = None
                    self.spatial_right = None
            
            except Exception as e:
                self.log["events"].append(f"Spatial parameters ERROR: {str(e)}")
                self.log["events"].append(tb.format_exc())
                self.spatial_left = None
                self.spatial_right = None
        
        
        
        
        except Exception as e:
                msg = f"run_gaitmap_pipeline ERROR: {str(e)}"
                self.log["events"].append(msg)
                self.log["events"].append(tb.format_exc())
        
                self.gs = {
                    "left_sensor": empty_gs.copy(),
                    "right_sensor": empty_gs.copy(),
                }
        
                raise
                            

    def plot_gs_events(
        self,
        channel: str = "gyr_ml",
        start: Optional[int] = None,
        end: Optional[int] = None,
        signal_type: str = "filtered",
        show_events: bool = True,
        show_removed_events: bool = False,
        show_strides: bool = False,
        events_attr: str = "events",
        removed_events_attr: str = "removed_events",
        strides_attr: str = "stride_list",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """
        Plot signal, gait sequences/strides, valid gait events, and optionally removed gait events.
    
        Layout
        ------
        - top subplot: left_sensor
        - bottom subplot: right_sensor
    
        On each subplot:
        - signal is plotted
        - if show_strides=False:
            gait sequence areas are shaded in light blue
        - if show_strides=True:
            stride areas are shaded in light green and stride start/end are shown as thin vertical lines
        - valid pre_ic events are shown as green dots
        - valid ic events are shown as red dots
        - removed pre_ic events are shown as orange x markers
        - removed ic events are shown as purple x markers
        """
    
        # -------------------------
        # Check gait sequences
        # -------------------------
        if not hasattr(self, "gs"):
            raise ValueError("self.gs does not exist. Run the gait pipeline first.")
    
        # -------------------------
        # Select signal
        # -------------------------
        if signal_type == "filtered":
            if self.signal_filtered is None:
                raise ValueError("self.signal_filtered is None. Run filter_signal() first.")
            signal_df = self.signal_filtered
    
        elif signal_type == "raw":
            if self.signal_raw is None:
                raise ValueError("self.signal_raw is None.")
            signal_df = self.signal_raw
    
        else:
            raise ValueError("signal_type must be either 'filtered' or 'raw'.")
    
        # -------------------------
        # Select event / stride containers
        # -------------------------
        events = None
        removed_events = None
        strides = None
    
        if show_events:
            if not hasattr(self, events_attr):
                raise ValueError(f"self.{events_attr} does not exist.")
            events = getattr(self, events_attr)
    
        if show_removed_events:
            if not hasattr(self, removed_events_attr):
                raise ValueError(f"self.{removed_events_attr} does not exist.")
            removed_events = getattr(self, removed_events_attr)
    
        if show_strides:
            if not hasattr(self, strides_attr):
                raise ValueError(f"self.{strides_attr} does not exist.")
            strides = getattr(self, strides_attr)
    
        sensors = ["left_sensor", "right_sensor"]
    
        start = 0 if start is None else start
        end = len(signal_df) if end is None else end
    
        if start < 0 or end > len(signal_df) or start >= end:
            raise ValueError("Invalid start/end interval.")
    
        x = np.arange(start, end)
    
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
        # ------------------------------------------------------------------
        # Helper function to plot events
        # ------------------------------------------------------------------
        def _plot_event_column(
            ax,
            ev_df,
            ev_col: str,
            col: tuple,
            marker: str,
            color: str,
            label: str,
            size: int,
            zorder: int,
        ):
            if ev_df is None:
                return
    
            if not isinstance(ev_df, pd.DataFrame):
                return
    
            if ev_df.empty:
                return
    
            if ev_col not in ev_df.columns:
                return
    
            idx = pd.to_numeric(ev_df[ev_col], errors="coerce")
            idx = idx.dropna().astype(int)
    
            idx = idx[(idx >= start) & (idx < end)]
            idx = idx[idx < len(signal_df)]
    
            if len(idx) == 0:
                return
    
            y = signal_df.loc[idx, col].values
    
            ax.scatter(
                idx,
                y,
                label=label,
                color=color,
                marker=marker,
                s=size,
                zorder=zorder,
            )
    
        # ------------------------------------------------------------------
        # Plot sensor by sensor
        # ------------------------------------------------------------------
        for ax, sensor in zip(axes, sensors):
            col = (sensor, channel)
    
            if col not in signal_df.columns:
                raise KeyError(f"Column {col} not found in signal dataframe.")
    
            if sensor not in self.gs:
                raise KeyError(f"{sensor} not found in self.gs.")
    
            # -------------------------
            # Signal
            # -------------------------
            signal = signal_df.loc[:, col].iloc[start:end]
    
            ax.plot(
                x,
                signal.values,
                label=f"{signal_type.capitalize()} signal",
                linewidth=1.0,
            )
    
            # -------------------------
            # GS areas OR stride areas
            # -------------------------
            if show_strides:
                if sensor not in strides:
                    raise KeyError(f"{sensor} not found in self.{strides_attr}.")
    
                stride_df = strides[sensor]
    
                if isinstance(stride_df, pd.DataFrame) and not stride_df.empty:
                    for _, row in stride_df.iterrows():
                        stride_start = int(row["start"])
                        stride_end = int(row["end"])
    
                        if stride_end < start or stride_start > end:
                            continue
    
                        shade_start = max(stride_start, start)
                        shade_end = min(stride_end, end)
    
                        ax.axvspan(
                            shade_start,
                            shade_end,
                            color="lightgreen",
                            alpha=0.18,
                            label="Stride",
                        )
    
                        ax.axvline(
                            shade_start,
                            color="green",
                            linewidth=0.5,
                            alpha=0.45,
                        )
    
                        ax.axvline(
                            shade_end,
                            color="green",
                            linewidth=0.5,
                            alpha=0.45,
                        )
    
            else:
                gs_df = self.gs[sensor]
    
                if gs_df is not None and not gs_df.empty:
                    for _, row in gs_df.iterrows():
                        gs_start = int(row["start"])
                        gs_end = int(row["end"])
    
                        if gs_end < start or gs_start > end:
                            continue
    
                        shade_start = max(gs_start, start)
                        shade_end = min(gs_end, end)
    
                        ax.axvspan(
                            shade_start,
                            shade_end,
                            color="lightblue",
                            alpha=0.35,
                            label="GS",
                        )
    
            # -------------------------
            # Valid events
            # -------------------------
            if show_events:
                if sensor not in events:
                    raise KeyError(f"{sensor} not found in self.{events_attr}.")
    
                ev_df = events[sensor]
    
                _plot_event_column(
                    ax=ax,
                    ev_df=ev_df,
                    ev_col="pre_ic",
                    col=col,
                    marker="o",
                    color="green",
                    label="pre_ic",
                    size=18,
                    zorder=5,
                )
    
                _plot_event_column(
                    ax=ax,
                    ev_df=ev_df,
                    ev_col="ic",
                    col=col,
                    marker="o",
                    color="red",
                    label="ic",
                    size=18,
                    zorder=6,
                )
    
            # -------------------------
            # Removed events
            # -------------------------
            if show_removed_events:
                if sensor not in removed_events:
                    raise KeyError(f"{sensor} not found in self.{removed_events_attr}.")
    
                removed_ev_df = removed_events[sensor]
    
                _plot_event_column(
                    ax=ax,
                    ev_df=removed_ev_df,
                    ev_col="pre_ic",
                    col=col,
                    marker="x",
                    color="orange",
                    label="removed pre_ic",
                    size=40,
                    zorder=7,
                )
    
                _plot_event_column(
                    ax=ax,
                    ev_df=removed_ev_df,
                    ev_col="ic",
                    col=col,
                    marker="x",
                    color="purple",
                    label="removed ic",
                    size=40,
                    zorder=8,
                )
    
            ax.set_title(f"{sensor} - {channel}")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
    
            # Remove duplicate legend items
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())
    
        axes[1].set_xlabel("Samples")
    
        title_parts = ["Signal"]
    
        if show_strides:
            title_parts.append("Strides")
        else:
            title_parts.append("GS")
    
        if show_events:
            title_parts.append("Events")
    
        if show_removed_events:
            title_parts.append("Removed events")
    
        fig.suptitle(f"{' + '.join(title_parts)} - Channel: {channel}")
    
        plt.tight_layout()
    
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
        plt.show()
    def compute_recording_and_wearing_time(self):

        """
        Estimate total recording duration and sensor wearing time.
    
        This method identifies potential non-wearing periods from low-activity
        windows computed on accelerometer and gyroscope signals. Accelerometer
        channels are first high-pass filtered to remove the gravity component,
        while gyroscope channels are analyzed directly.
    
        For each sliding window, the vector norm of the available accelerometer
        and gyroscope channels is computed and smoothed. A window is classified as
        non-wearing when both signal norms remain below their respective activity
        thresholds.
    
        The method supports:
        - configurable sliding-window duration and overlap,
        - optional forced exclusion of an initial recording segment,
        - merging of overlapping or contiguous non-wearing windows,
        - estimation of the first valid wearing sample.
    
        Results are stored both as class attributes and in the processing log.
    
        Processing steps
        ----------------
        1. Compute total recording duration from signal length and sampling rate.
        2. High-pass filter accelerometer channels.
        3. Compute accelerometer and gyroscope vector norms.
        4. Smooth signal norms using a moving-average filter.
        5. Detect candidate non-wearing windows using sliding-window thresholding.
        6. Merge overlapping non-wearing windows.
        7. Estimate total wearing time and first valid wearing sample.
    
        Notes
        -----
        Non-wearing detection is based on low signal activity and is intended for
        long real-world recordings. Thresholds and window parameters may require
        adjustment depending on sensor placement, acquisition system, and study
        population.
    
        Returns
        -------
        dict
            Dictionary containing recording duration, wearing duration,
            non-wearing windows, and related summary metrics.
    
        Raises
        ------
        RuntimeError
            If `self.signal_filtered` is not available.
    
        ValueError
            If the configured overlap is outside the valid range.
        """
        if self.signal_filtered is None:
            raise RuntimeError(
                "Filtered signal missing. Run filter_signal() before compute_recording_and_wearing_time()."
            )
    
        df = self.signal_filtered.copy()
        fs = self.fs
    
        # -------------------------
        # Parameters from config
        # -------------------------
        acc_th = self.config["acc_wearing_threshold"]
        gyr_th = self.config["gyr_wearing_threshold"]
        initial_window = self.config["wearing_initial_window_min"]
        window_size = self.config["wearing_window_size_min"]
        overlap = float(self.config["wearing_overlap"])
        highpass_cutoff = self.config["wearing_highpass_cutoff_hz"]
        highpass_order = self.config["wearing_highpass_order"]
        smoothing_window_s = self.config["wearing_smoothing_window_s"]
    
        if not 0 <= overlap < 1:
            raise ValueError("wearing_overlap must be >= 0 and < 1.")
    
        # -------------------------
        # Timestamps
        # -------------------------
        timestamps_unix = self.timestamps_unix
    
        if timestamps_unix is None:
            timestamps_unix = []
        else:
            timestamps_unix = np.asarray(timestamps_unix)
    
        # -------------------------
        # Basic recording duration
        # -------------------------
        df = df.mask(df.abs() < 1e-6, 0)
    
        n_samples = len(df)
        total_seconds = n_samples / fs
        rec_time_hours = total_seconds / 3600.0
        rec_time_str = f"{int(rec_time_hours)}h {int((rec_time_hours % 1) * 60)}m"
    
        # -------------------------
        # High-pass filter
        # -------------------------
        def highpass_filter(signal):
            nyq = 0.5 * fs
            normal_cutoff = highpass_cutoff / nyq
    
            if normal_cutoff <= 0 or normal_cutoff >= 1:
                raise ValueError(
                    "Invalid wearing_highpass_cutoff_hz. "
                    "It must be greater than 0 and lower than Nyquist frequency."
                )
    
            b, a = butter(
                highpass_order,
                normal_cutoff,
                btype="high",
                analog=False,
            )
    
            return filtfilt(b, a, signal)
    
        # -------------------------
        # Available columns
        # -------------------------
        acc_columns = [
            ("left_sensor", "acc_pa"),
            ("left_sensor", "acc_ml"),
            ("left_sensor", "acc_si"),
            ("right_sensor", "acc_pa"),
            ("right_sensor", "acc_ml"),
            ("right_sensor", "acc_si"),
        ]
    
        gyr_columns = [
            ("left_sensor", "gyr_pa"),
            ("left_sensor", "gyr_ml"),
            ("left_sensor", "gyr_si"),
            ("right_sensor", "gyr_pa"),
            ("right_sensor", "gyr_ml"),
            ("right_sensor", "gyr_si"),
        ]
    
        acc_cols_present = [col for col in acc_columns if col in df.columns]
        gyr_cols_present = [col for col in gyr_columns if col in df.columns]
    
        if not acc_cols_present:
            self.log["events"].append(
                "compute_recording_and_wearing_time: no accelerometer columns found."
            )
    
        if not gyr_cols_present:
            self.log["events"].append(
                "compute_recording_and_wearing_time: no gyroscope columns found."
            )
    
        # -------------------------
        # Accelerometer norm
        # -------------------------
        if acc_cols_present:
            acc_filtered = np.zeros((n_samples, len(acc_cols_present)))
    
            for i, col in enumerate(acc_cols_present):
                acc_filtered[:, i] = highpass_filter(df[col].to_numpy())
    
            self.acc_norm = np.linalg.norm(acc_filtered, axis=1)
    
        else:
            self.acc_norm = np.zeros(n_samples)
    
        # -------------------------
        # Gyroscope norm
        # -------------------------
        if gyr_cols_present:
            gyr_array = np.zeros((n_samples, len(gyr_cols_present)))
    
            for i, col in enumerate(gyr_cols_present):
                gyr_array[:, i] = df[col].to_numpy()
    
            self.gyr_norm = np.linalg.norm(gyr_array, axis=1)
    
        else:
            self.gyr_norm = np.zeros(n_samples)
    
        # -------------------------
        # Smooth norms
        # -------------------------
        smoothing_samples = max(1, int(fs * smoothing_window_s))
    
        self.acc_norm = uniform_filter1d(
            self.acc_norm,
            size=smoothing_samples,
        )
    
        self.gyr_norm = uniform_filter1d(
            self.gyr_norm,
            size=smoothing_samples,
        )
    
        # -------------------------
        # Sliding window non-wearing detection
        # -------------------------
        win_samples = max(1, int(window_size * 60 * fs))
        step = max(1, int(win_samples * (1.0 - overlap)))
    
        non_wearing_windows = []
    
        initial_window_samples = int(initial_window * 60 * fs)
    
        if initial_window > 0 and n_samples >= initial_window_samples:
            non_wearing_windows.append((0, initial_window_samples))
    
        if n_samples < win_samples:
            if (
                np.nanmax(self.acc_norm) < acc_th
                and np.nanmax(self.gyr_norm) < gyr_th
            ):
                non_wearing_windows.append((0, n_samples))
    
        else:
            for start_idx in range(0, n_samples - win_samples + 1, step):
                end_idx = start_idx + win_samples
    
                acc_segment = self.acc_norm[start_idx:end_idx]
                gyr_segment = self.gyr_norm[start_idx:end_idx]
    
                if (
                    np.nanmax(acc_segment) < acc_th
                    and np.nanmax(gyr_segment) < gyr_th
                ):
                    non_wearing_windows.append((start_idx, end_idx))
    
        # -------------------------
        # Adjust initial forced non-wearing window
        # -------------------------
        if len(non_wearing_windows) > 1:
            first_window = non_wearing_windows[0]
            second_window = non_wearing_windows[1]
    
            if (
                first_window[1] == initial_window_samples
                and second_window[0] > initial_window_samples
            ):
                non_wearing_windows.pop(0)
    
        # -------------------------
        # Merge overlapping / contiguous windows
        # -------------------------
        merged_windows = []
    
        for s, e in sorted(non_wearing_windows):
            if not merged_windows:
                merged_windows.append([s, e])
            else:
                last_s, last_e = merged_windows[-1]
    
                if s <= last_e:
                    merged_windows[-1][1] = max(last_e, e)
                else:
                    merged_windows.append([s, e])
    
        # -------------------------
        # Compute wearing time
        # -------------------------
        non_wearing_sec = (
            sum((e - s) / fs for s, e in merged_windows)
            if merged_windows
            else 0.0
        )
    
        wearing_sec = max(0.0, total_seconds - non_wearing_sec)
    
        wearing_time_hours = wearing_sec / 3600.0
        wearing_time_str = f"{int(wearing_time_hours)}h {int((wearing_time_hours % 1) * 60)}m"
    
        # -------------------------
        # First wearing sample
        # -------------------------
        if merged_windows and merged_windows[0][0] == 0:
            wearing_start_sample = merged_windows[0][1]
        else:
            wearing_start_sample = 0
    
        if len(timestamps_unix) > 0 and 0 <= wearing_start_sample < len(timestamps_unix):
            wearing_start_unix_time = int(timestamps_unix[wearing_start_sample])
        elif len(timestamps_unix) > 0:
            wearing_start_unix_time = int(timestamps_unix[-1])
        else:
            wearing_start_unix_time = 0
    
        # -------------------------
        # Save attributes
        # -------------------------
        self.recording_time_hours = round(rec_time_hours, 2)
        self.wearing_time_hours = round(wearing_time_hours, 2)
        self.recording_time_str = rec_time_str
        self.wearing_time_str = wearing_time_str
        self.merged_windows = merged_windows
    
        # -------------------------
        # Save in log
        # -------------------------
        if "recording_summary" not in self.log:
            self.log["recording_summary"] = {}
    
        self.log["recording_summary"].update(
            {
                "recording_time_hours": self.recording_time_hours,
                "wearing_time_hours": self.wearing_time_hours,
                "non_wearing_time_hours": round(non_wearing_sec / 3600.0, 2),
                "non_wearing_windows_samples": self.merged_windows,
                "overlap_used": overlap,
                "window_size_min": window_size,
                "initial_window_min": initial_window,
                "wearing_start_sample": int(wearing_start_sample),
                "wearing_start_unix_time": wearing_start_unix_time,
            }
        )
    
        self.log["events"].append(
            "compute_recording_and_wearing_time: completed"
        )
    
        return self.log["recording_summary"]
    def plot_signal_with_non_wearing(
        self,
        sensor: str = "left_sensor",
        channel: str = "gyr_ml",
        signal_type: str = "filtered",
        figsize: tuple = (16, 5),
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot one selected signal channel for one selected sensor, highlighting
        non-wearing windows.
    
        Parameters
        ----------
        sensor : str, default="left_sensor"
            Sensor side to plot. Must be "left_sensor" or "right_sensor".
    
        channel : str, default="gyr_ml"
            Signal channel to plot, e.g. "gyr_ml", "acc_ml", "gyr_pa", "acc_si".
    
        signal_type : str, default="filtered"
            Which signal dataframe to use: "filtered" or "raw".
    
        figsize : tuple, default=(16, 5)
            Figure size.
    
        save_path : str, optional
            If provided, save the figure to this path.
    
        show : bool, default=True
            If True, display the figure. If False, close it after creation.
        """
    
        # -------------------------
        # Select dataframe
        # -------------------------
        if signal_type == "filtered":
            if self.signal_filtered is None:
                raise ValueError("self.signal_filtered is None. Run filter_signal() first.")
            df = self.signal_filtered.copy()
    
        elif signal_type == "raw":
            if self.signal_raw is None:
                raise ValueError("self.signal_raw is None.")
            df = self.signal_raw.copy()
    
        else:
            raise ValueError("signal_type must be either 'filtered' or 'raw'.")
    
        if df is None or df.empty:
            raise ValueError("Selected signal dataframe is empty.")
    
        # -------------------------
        # Check sensor / channel
        # -------------------------
        if sensor not in ["left_sensor", "right_sensor"]:
            raise ValueError("sensor must be 'left_sensor' or 'right_sensor'.")
    
        col = (sensor, channel)
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in selected signal dataframe.")
    
        # -------------------------
        # Check non-wearing windows
        # -------------------------
        if not hasattr(self, "merged_windows") or self.merged_windows is None:
            raise ValueError(
                "self.merged_windows not found. Run compute_recording_and_wearing_time() first."
            )
    
        merged_windows = self.merged_windows
    
        # -------------------------
        # Signal values
        # -------------------------
        y = df.loc[:, col].to_numpy()
    
        # -------------------------
        # X axis: timestamps if available, otherwise samples
        # -------------------------
        use_timestamps = False
        if hasattr(self, "timestamps_unix") and self.timestamps_unix is not None:
            timestamps_unix = np.asarray(self.timestamps_unix)
            if len(timestamps_unix) == len(df):
                x = pd.to_datetime(timestamps_unix, unit="s", utc=True)
                use_timestamps = True
            else:
                x = np.arange(len(df))
        else:
            x = np.arange(len(df))
    
        # -------------------------
        # Create plot
        # -------------------------
        fig, ax = plt.subplots(figsize=figsize)
    
        ax.plot(x, y, label=f"{sensor} - {channel}", linewidth=1.0)
    
        # -------------------------
        # Shade non-wearing windows
        # -------------------------
        first_patch = True
        for start_idx, end_idx in merged_windows:
            if start_idx >= len(df):
                continue
    
            end_idx = min(end_idx, len(df))
    
            if end_idx <= start_idx:
                continue
    
            if use_timestamps:
                x_start = x[start_idx]
                x_end = x[end_idx - 1]
            else:
                x_start = start_idx
                x_end = end_idx - 1
    
            ax.axvspan(
                x_start,
                x_end,
                color="red",
                alpha=0.25,
                label="Non-wearing" if first_patch else None,
            )
            first_patch = False
    
        # -------------------------
        # Labels and title
        # -------------------------
        ax.set_title(f"Non wearing time windows - {sensor} - {channel}")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
    
        if use_timestamps:
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("Samples")
    
        ax.legend()
    
        plt.tight_layout()
    
        # -------------------------
        # Save figure
        # -------------------------
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
        # -------------------------
        # Show or close
        # -------------------------
        if show:
            plt.show()
        else:
            plt.close()
    def _combine_left_right_tables(
        self,
        left_df: Optional[pd.DataFrame],
        right_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Combine left and right dataframes into one dataframe.
    
        Adds:
            foot = "left" or "right"
    
        Keeps:
            s_id as index
        """
    
        tables = []
    
        if isinstance(left_df, pd.DataFrame) and not left_df.empty:
            left = left_df.copy()
            left.index.name = "s_id"
            left.insert(0, "foot", "left")
            tables.append(left)
    
        if isinstance(right_df, pd.DataFrame) and not right_df.empty:
            right = right_df.copy()
            right.index.name = "s_id"
            right.insert(0, "foot", "right")
            tables.append(right)
    
        if not tables:
            return pd.DataFrame()
    
        out = pd.concat(tables, axis=0)
        out.index.name = "s_id"
    
        return out
    def save_outputs(self, project_folder: str) -> Dict[str, Path]:
        """
            Save pipeline outputs to disk.
        
            Outputs are organized using the following folder structure:
        
                project_folder /
                    patient_id /
                        recording_date /
                            session_id /
        
            The method exports:
            - processing log (`.json`),
            - gait events (`.csv`),
            - combined temporal and spatial gait parameters (`.csv`).
        
            Metadata columns (`patient_id`, `recording_date`, `session_id`) are
            automatically added to all exported tables.
        
            Temporal and spatial parameter tables from the left and right sensors are
            merged into a single stride-level parameter table before export.
        
            Processing steps
            ----------------
            1. Validate recording metadata.
            2. Create the output directory structure.
            3. Save the processing log as JSON.
            4. Combine and export gait-event tables.
            5. Combine temporal and spatial parameter tables.
            6. Flatten potential MultiIndex column names.
            7. Export the final stride-level parameter table.
        
            Parameters
            ----------
            project_folder : str
                Root directory where outputs will be saved.
        
            Returns
            -------
            dict
                Dictionary containing the paths of the saved files.
        
            Raises
            ------
            ValueError
                If `patient_id`, `session_id`, or `recording_date` are missing.
            """
        if self.patient_id is None:
            raise ValueError("patient_id is None.")
    
        if self.session_id is None:
            raise ValueError("session_id is None.")
    
        if self.recording_date is None:
            raise ValueError("recording_date is None.")
    
        # Ensure date format: YYYY-MM-DD
        recording_date_fmt = pd.to_datetime(self.recording_date).strftime("%Y-%m-%d")
    
        # New folder structure:
        # project_folder / patient_id / recording_date / session_id /
        output_dir = (
            Path(project_folder)
            / str(self.patient_id)
            / recording_date_fmt
            / str(self.session_id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
    
        # Filename prefix
        file_prefix = f"{self.patient_id}_{self.session_id}_{recording_date_fmt}"
    
        saved_paths = {}
    
        # -------------------------
        # Helper: add metadata columns
        # -------------------------
        def _add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
            """
            Add patient_id, recording_date, session_id as the first columns.
            """
    
            df = df.copy()
    
            # Avoid duplicate metadata columns if method is called twice
            for col in ["patient_id", "recording_date", "session_id"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
    
            df.insert(0, "session_id", self.session_id)
            df.insert(0, "recording_date", recording_date_fmt)
            df.insert(0, "patient_id", self.patient_id)
    
            return df
    
        # -------------------------
        # Save log JSON
        # -------------------------
        log_path = output_dir / f"{file_prefix}_log.json"
    
        # Update metadata in log with formatted date
        self.log["meta"].update(
            {
                "patient_id": self.patient_id,
                "recording_date": recording_date_fmt,
                "session_id": self.session_id,
            }
        )
    
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=4, default=str)
    
        saved_paths["log"] = log_path
    
        # -------------------------
        # Save events
        # -------------------------
        if self.events is not None:
            events_df = self._combine_left_right_tables(
                left_df=self.events.get("left_sensor"),
                right_df=self.events.get("right_sensor"),
            )
    
            # Keep s_id as a normal column, not as index
            if not events_df.empty:
                events_df = events_df.reset_index()
    
            events_df = _add_metadata_columns(events_df)
    
            events_path = output_dir / f"{file_prefix}_events.csv"
            events_df.to_csv(events_path, index=False)
    
            saved_paths["events"] = events_path
    
        # -------------------------
        # Save combined parameters
        # -------------------------
        temporal_df = self._combine_left_right_tables(
            left_df=self.temporal_left,
            right_df=self.temporal_right,
        )
    
        spatial_df = self._combine_left_right_tables(
            left_df=self.spatial_left,
            right_df=self.spatial_right,
        )
    
        # Make sure columns are flat strings, not MultiIndex
        if not temporal_df.empty:
            temporal_df = temporal_df.copy()
            temporal_df.columns = [
                "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col)
                for col in temporal_df.columns
            ]
            temporal_df = temporal_df.reset_index()
    
        if not spatial_df.empty:
            spatial_df = spatial_df.copy()
            spatial_df.columns = [
                "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col)
                for col in spatial_df.columns
            ]
            spatial_df = spatial_df.reset_index()
    
        if not temporal_df.empty and not spatial_df.empty:
            parameters_df = pd.merge(
                temporal_df,
                spatial_df,
                on=["s_id", "foot"],
                how="outer",
                suffixes=("_temporal", "_spatial"),
            )
    
        elif not temporal_df.empty:
            parameters_df = temporal_df.copy()
    
        elif not spatial_df.empty:
            parameters_df = spatial_df.copy()
    
        else:
            parameters_df = pd.DataFrame()
    
        parameters_df = _add_metadata_columns(parameters_df)
    
        parameters_path = output_dir / f"{file_prefix}_parameters.csv"
        parameters_df.to_csv(parameters_path, index=False)
    
        saved_paths["parameters"] = parameters_path
    
        self.log["events"].append("save_outputs: completed")
    
        return saved_paths
#%% DEBUG MAIN


#Parse function to extract filename pattern, to be adpated
def parse_filename_metadata(path):
    fn = Path(path).name
    
    # pattern: PAT404_xxxx_2023-07-11_gaitMAP_bf_all.csv
    pattern = r"^(PAT\d+)_([A-Za-z0-9_]+)_(\d{4}-\d{2}-\d{2})_gaitMAP"
    match = re.search(pattern, fn)
    
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {fn}")
    
    patient_id = match.group(1)
    session_id = match.group(2)
    date_str = match.group(3)
    
    return patient_id, session_id, date_str

# Paths
path = r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova\PAT401\2023-07-10\week_3\PAT401_week_3_2023-07-10_gaitMAP_bf_all.csv"
patient_id, session_id, recording_date = parse_filename_metadata(path)
output_root = r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova"

# Loadings
df_raw = pd.read_csv(path, header=[0, 1], index_col=0)
df_raw = df_raw.reset_index(drop=True)


# Optional user config
user_config = {
    "sampling_rate_hz": 102.4,
    
}


# Pipeline
#Step 1 Initialisation
pipeline = GaitMapPipeline(
    signal_raw=df_raw,
    config=user_config,
    patient_id=patient_id,
    session_id=session_id,
    recording_date=recording_date,
    output_root=output_root
)    



# Step 2 Filtering
pipeline.filter_signal()

#Step 2.1 Plot Filtering
pipeline.plot_raw_filtered_channel(channel="gyr_ml")

#Step 3 Gaitmap
pipeline.run_gaitmap_pipeline()

#Step 3.1 gait Sequences
gs=pipeline.gs
#pipeline.plot_gs(channel="gyr_ml")

#Step 3.2 Strides Segmentation
stride_list=pipeline.stride_list

#Step 3.3 Events detection
events_list=pipeline.events

#Step 3.4 Trajectories
log=pipeline.log

#Step 3.5 Temporal Parameters
temporal_l=pipeline.temporal_left
temporal_r=pipeline.temporal_right
#Step 3.6 Spatial Parameters
spatial_l=pipeline.spatial_left
spatial_r=pipeline.spatial_right
removed_events = pipeline.removed_events

events_cleaned=pipeline.events_clean
# pipeline.plot_gs_events(
#     channel="gyr_ml",
#     show_events=True,
#     show_removed_events=False,
#     show_strides=True,
# )
# pipeline.plot_gs_events(
#     channel="gyr_ml",
#     show_events=True,
#     show_removed_events=True,
#     show_strides=True,
# )

pipeline.compute_recording_and_wearing_time()


# pipeline.plot_signal_with_non_wearing(
#     sensor="left_sensor",
#     channel="gyr_ml",
#     signal_type="filtered",
# )

saved_paths = pipeline.save_outputs(
    project_folder=r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\MobilityAPP_Pipeline\Prova"
)

print(saved_paths)