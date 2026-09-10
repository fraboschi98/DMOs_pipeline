# -*- coding: utf-8 -*-
"""
KDE plotting utilities for WB-level gait-parameter distributions.

This module contains the KDEPlotter class, which generates KDE distribution
plots from walking-bout-level gait parameters.
"""

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class KDEPlotter:
    """
    Generate KDE distribution plots for one patient's WB-level gait parameters.

    This class takes the walking-bout-level dataframe of one patient and creates
    one distribution plot per selected gait feature. Walking bouts are stratified
    by `WB_label`.

    The input `wb_parameters_average` should already be produced by the
    walking-bout analyzer and should contain one row per walking bout.

    The optional `day_dmos` dataframe can be provided to apply a wearing-time
    control. When `check_wearing_time=True`, only recording dates with wearing
    time greater than or equal to `min_wearing_time_hours` are kept.

    Date selection
    --------------
    The `date` argument controls which recording dates are included:

    - one date string, for example `"2023-07-14"`;
    - a list of date strings, for example `["2023-07-10", "2023-07-14"]`;
    - `"all"` to use all available dates.

    Feature selection
    -----------------
    The `features` argument controls which gait features are plotted:

    - one feature string, for example `"gait velocity [m/s]"`;
    - a list of feature strings;
    - `"all"` to plot all default gait features.

    Parameters
    ----------
    wb_parameters_average : pandas.DataFrame
        WB-level dataframe for one patient. It should contain patient/date
        metadata, `WB_label`, and the gait features to plot.

    patient_id : str, optional
        Patient identifier. If provided, the dataframe is filtered to this
        patient. If None, the patient is inferred from the dataframe.

    day_dmos : pandas.DataFrame, optional
        Day-level DMO dataframe for the same patient. Required only when
        `check_wearing_time=True`. It must contain `recording_date` and
        `wearing_time_hours`.

    date : str or list of str, default="all"
        Recording dates to include.

    features : str, list of str, or "all", default="all"
        Gait features to plot.

    check_wearing_time : bool, default=False
        If True, remove dates with wearing time lower than
        `min_wearing_time_hours`.

    min_wearing_time_hours : float, default=8.0
        Minimum wearing time required to keep a recording date.

    include_very_short : bool, default=False
        If False, very short walking bouts are excluded.

    Attributes
    ----------
    df : pandas.DataFrame
        Filtered dataframe used for plotting.

    selected_dates : list of str
        Recording dates kept after date selection and optional wearing-time
        filtering.

    selected_features : list of str
        Gait features selected for plotting.

    Notes
    -----
    This class only creates KDE plots. It does not compute WB-level or day-level
    DMOs.
    """

    DEFAULT_FEATURES = [
        "cadence_[steps_per_min]",
        "stride time [s]",
        "swing time [s]",
        "stance time [s]",
        "percent_stance_phase",
        "percent_swing_phase",
        "arc length [m]",
        "gait velocity [m/s]",
        "ic angle [deg]",
        "max. lateral excursion [m]",
        "max. orientation change [deg]",
        "max. sensor lift [m]",
        "stride length [m]",
        "tc angle [deg]",
        "turning angle [deg]",
    ]

    DEFAULT_LABEL_ORDER = [
        "short",
        "medium",
        "long",
    ]
    WB_COLORS = {
        "short": "#7AAACE",   
        "medium": "#BF1700",  
        "long": "#093C5D",    
        "very_short": "#B0B0B0",
    }

    def __init__(
        self,
        wb_parameters_average,
        patient_id=None,
        day_dmos=None,
        date="all",
        features="all",
        check_wearing_time=False,
        min_wearing_time_hours=8.0,
        include_very_short=False,
    ):
        self.wb_parameters_average = wb_parameters_average.copy()
        self.day_dmos = day_dmos.copy() if day_dmos is not None else None

        self.patient_id = patient_id
        self.date = date
        self.features = features
        self.check_wearing_time = check_wearing_time
        self.min_wearing_time_hours = min_wearing_time_hours
        self.include_very_short = include_very_short

        self.label_order = self.DEFAULT_LABEL_ORDER.copy()

        if self.include_very_short:
            self.label_order = ["very_short"] + self.label_order

        self.selected_features = self._select_features()
        self.selected_dates = self._select_dates()
        self.df = self._filter_data()
        self.kde_summary = {
    "config": {},
    "results": [],
}

    def _get_dates_from_df(self, df):
        """
        Return sorted recording dates from a dataframe.
        """
    
        if "recording_date" not in df.columns:
            return []
    
        return sorted(
            df["recording_date"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    
    def _select_features(self):
        """
        Resolve and validate selected gait features.
        """

        if self.features == "all":
            selected_features = self.DEFAULT_FEATURES.copy()

        elif isinstance(self.features, str):
            selected_features = [self.features]

        elif isinstance(self.features, list):
            selected_features = self.features

        else:
            raise ValueError(
                "features must be 'all', a string, or a list of strings."
            )

        missing_features = [
            feature
            for feature in selected_features
            if feature not in self.wb_parameters_average.columns
        ]

        if missing_features:
            raise KeyError(
                "Selected feature(s) not found in wb_parameters_average: "
                f"{missing_features}"
            )

        return selected_features
    def _build_kde_config(
        self,
        output_folder,
        min_wb,
        bins,
        show,
        dpi,
        plot_histogram,
    ):
        """
        Build the configuration dictionary saved in the KDE JSON summary.
        """
    
        return {
            "patient_id": self.patient_id,
            "selected_dates": self.selected_dates,
            "selected_features": self.selected_features,
            "check_wearing_time": self.check_wearing_time,
            "min_wearing_time_hours": self.min_wearing_time_hours,
            "include_very_short": self.include_very_short,
            "label_order": self.label_order,
            "min_wb": int(min_wb),
            "bins": int(bins),
            "show": bool(show),
            "dpi": int(dpi),
            "plot_histogram": bool(plot_histogram),
            "output_folder": str(output_folder),
            "n_rows_after_filtering": int(len(self.df)),
        }
    def save_kde_summary_json(self, output_folder, filename=None):
        """
        Save KDE configuration and computed distribution summaries as JSON.
        """
    
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
        if filename is None:
            patient_id = self.patient_id
    
            if patient_id is None:
                patient_ids = sorted(
                    self.df["patient_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                patient_id = "multiple_patients" if len(patient_ids) != 1 else patient_ids[0]
    
            date_part = self._clean_filename("_".join(self.selected_dates))
            filename = f"{patient_id}_{date_part}_kde_summary.json"
    
        output_path = output_folder / filename
    
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(self.kde_summary, file, indent=4)
    
        print(f"Saved KDE summary JSON: {output_path}")
    
        return output_path
    def _select_dates(self):
        """
        Resolve and validate selected recording dates.
        """

        if "recording_date" not in self.wb_parameters_average.columns:
            raise KeyError(
                "Column 'recording_date' not found in wb_parameters_average"
            )

        available_dates = sorted(
            self.wb_parameters_average["recording_date"]
            .dropna()
            .astype(str)
            .unique()
        )

        if self.date == "all":
            selected_dates = available_dates

        elif isinstance(self.date, str):
            selected_dates = [self.date]

        elif isinstance(self.date, list):
            selected_dates = [str(date) for date in self.date]

        else:
            raise ValueError(
                "date must be 'all', a string date, or a list of date strings."
            )

        missing_dates = [
            date for date in selected_dates
            if date not in available_dates
        ]

        if missing_dates:
            raise ValueError(
                "Selected date(s) not found in wb_parameters_average: "
                f"{missing_dates}"
            )

        return selected_dates

    def _filter_data(self):
        """
        Filter input dataframe by patient, date, WB label, and wearing time.
        """

        required_columns = [
            "patient_id",
            "recording_date",
            "WB_id",
            "WB_label",
        ]

        for col in required_columns:
            if col not in self.wb_parameters_average.columns:
                raise KeyError(
                    f"Column '{col}' not found in wb_parameters_average"
                )

        df = self.wb_parameters_average.copy()

        df["patient_id"] = df["patient_id"].astype(str)
        df["recording_date"] = df["recording_date"].astype(str)
        df["WB_label"] = df["WB_label"].astype(str)

        if self.patient_id is not None:
            df = df[df["patient_id"] == str(self.patient_id)].copy()

        df = df[df["recording_date"].isin(self.selected_dates)].copy()
        df = df[df["WB_label"].isin(self.label_order)].copy()

        if self.check_wearing_time:
            df = self._apply_wearing_time_filter(df)

        return df.reset_index(drop=True)

    def _apply_wearing_time_filter(self, df):
        """
        Remove recording dates with wearing time below the selected threshold.
        """

        if self.day_dmos is None:
            raise ValueError(
                "day_dmos must be provided when check_wearing_time=True."
            )

        required_columns = [
            "patient_id",
            "recording_date",
            "wearing_time_hours",
        ]

        for col in required_columns:
            if col not in self.day_dmos.columns:
                raise KeyError(f"Column '{col}' not found in day_dmos")

        day_dmos = self.day_dmos.copy()

        day_dmos["patient_id"] = day_dmos["patient_id"].astype(str)
        day_dmos["recording_date"] = day_dmos["recording_date"].astype(str)
        day_dmos["wearing_time_hours"] = pd.to_numeric(
            day_dmos["wearing_time_hours"],
            errors="coerce",
        )

        if self.patient_id is not None:
            day_dmos = day_dmos[
                day_dmos["patient_id"] == str(self.patient_id)
            ].copy()

        valid_days = day_dmos[
            day_dmos["wearing_time_hours"] >= self.min_wearing_time_hours
        ][["patient_id", "recording_date"]].drop_duplicates()

        df = df.merge(
            valid_days,
            on=["patient_id", "recording_date"],
            how="inner",
        )

        return df

    def summary(self):
        """
        Print a compact summary of the data available for plotting.
        """

        print("\nKDEPlotter summary")
        print("------------------")

        if self.patient_id is None:
            patients = sorted(
                self.df["patient_id"].dropna().astype(str).unique()
            )
            print(f"Patient: inferred from dataframe: {patients}")
        else:
            print(f"Patient: {self.patient_id}")

        print(f"Selected dates: {self.selected_dates}")
        print(f"Selected features: {self.selected_features}")
        print(f"Check wearing time: {self.check_wearing_time}")

        if self.check_wearing_time:
            print(f"Minimum wearing time: {self.min_wearing_time_hours} h")

        print(f"Rows available for plotting: {len(self.df)}")

        if not self.df.empty:
            counts = (
                self.df
                .groupby(["recording_date", "WB_label"])
                .size()
                .reset_index(name="n_WB")
            )

            print("\nWB counts by date and label:")
            print(counts.to_string(index=False))

        return self

    def generate_plots(
        self,
        output_folder,
        min_wb=10,
        bins=30,
        show=False,
        dpi=150,
        plot_histogram=False,
    ):
        """
        Generate KDE plots for selected features.

        One plot is created for each selected feature. Within each plot,
        walking bouts are stratified by `WB_label`.

        Parameters
        ----------
        output_folder : str or pathlib.Path
            Folder where plots will be saved.

        min_wb : int, default=10
            Minimum number of walking bouts required for one WB label to be
            plotted.

        bins : int, default=30
            Number of histogram bins.

        show : bool, default=False
            If True, display plots interactively.

        dpi : int, default=150
            Resolution of saved PNG files.

        Returns
        -------
        self
            Updated KDEPlotter instance.
        """

        if self.df.empty:
            print("No data available after filtering. No plots generated.")
            return self

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        patient_ids = sorted(
            self.df["patient_id"].dropna().astype(str).unique()
        )

        for patient_id in patient_ids:
            patient_df = self.df[
                self.df["patient_id"].astype(str) == patient_id
            ].copy()

            patient_folder = output_folder / str(patient_id)
            patient_folder.mkdir(parents=True, exist_ok=True)

            for feature in self.selected_features:
                self._plot_one_feature(
                    patient_df=patient_df,
                    patient_id=patient_id,
                    feature=feature,
                    output_folder=patient_folder,
                    min_wb=min_wb,
                    bins=bins,
                    show=show,
                    dpi=dpi,
                    plot_histogram=plot_histogram,
                )

        return self


    def _plot_one_feature(
        self,
        patient_df,
        patient_id,
        feature,
        output_folder,
        min_wb,
        bins,
        show,
        dpi,
        plot_histogram=True,
    ):
        """
        Create and save one KDE plot for one gait feature.
        """
    
        fig, ax_count = plt.subplots(figsize=(12, 6))
        ax_density = ax_count.twinx()
    
        plotted_anything = False
    
        plot_result = {
            "patient_id": str(patient_id),
            "feature": feature,
            "recording_dates": self._get_dates_from_df(patient_df),
            "statistics_by_wb_label": {},
        }
    
        for label in self.label_order:
            values = pd.to_numeric(
                patient_df.loc[
                    patient_df["WB_label"] == label,
                    feature,
                ],
                errors="coerce",
            ).dropna()
    
            n_wb = int(len(values))
    
            label_result = {
                "n_WB": n_wb,
                "plotted": False,
                "skip_reason": None,
                "mode": None,
                "p95": None,
                "mean": None,
                "median": None,
                "std": None,
            }
    
            if n_wb < min_wb:
                label_result["skip_reason"] = "n_WB_below_min_wb"
                plot_result["statistics_by_wb_label"][label] = label_result
                continue
    
            if values.nunique() <= 1:
                label_result["skip_reason"] = "no_variability"
                plot_result["statistics_by_wb_label"][label] = label_result
                continue
    
            plotted_anything = True
    
            color = self.WB_COLORS.get(label, "#333333")
    
            if plot_histogram:
                ax_count.hist(
                    values,
                    bins=bins,
                    alpha=0.3,
                    color=color,
                    label="_nolegend_",
                )
    
            kde = gaussian_kde(values)
    
            x_min = values.min()
            x_max = values.max()
    
            x_padding = (x_max - x_min) * 0.10
            x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 300)
            y_kde = kde(x_grid)
    
            ax_density.plot(
                x_grid,
                y_kde,
                linewidth=2,
                color=color,
                label="_nolegend_",
            )
    
            ax_density.fill_between(
                x_grid,
                0,
                y_kde,
                color=color,
                alpha=0.08,
            )
    
            mode_value = float(x_grid[np.argmax(y_kde)])
            p95_value = float(values.quantile(0.95))
            mean_value = float(values.mean())
            median_value = float(values.median())
            std_value = float(values.std())
    
            label_result.update(
                {
                    "plotted": True,
                    "skip_reason": None,
                    "mode": mode_value,
                    "p95": p95_value,
                    "mean": mean_value,
                    "median": median_value,
                    "std": std_value,
                }
            )
    
            plot_result["statistics_by_wb_label"][label] = label_result
    
            ax_count.axvline(
                mode_value,
                color=color,
                linestyle="-",
                linewidth=1.2,
            )
    
            ax_count.axvline(
                p95_value,
                color=color,
                linestyle="--",
                linewidth=1.2,
            )
    
            ax_count.plot(
                [],
                [],
                color=color,
                linestyle="-",
                linewidth=1.2,
                label=f"{label} mode = {mode_value:.2f}",
            )
    
            ax_count.plot(
                [],
                [],
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"{label} p95 = {p95_value:.2f}",
            )
    
        if not plotted_anything:
            plt.close(fig)
            self.kde_summary["results"].append(plot_result)
    
            print(
                f"Skipped {feature} for patient {patient_id}: "
                f"not enough WBs per label or no variability."
            )
            return
    
        title_suffix = self._make_title_suffix(patient_df)
    
        ax_count.set_xlabel(feature)
        ax_count.set_ylabel("Number of WBs")
        ax_density.set_ylabel("Density")
    
        ax_count.set_ylim(bottom=0)
        ax_density.set_ylim(bottom=0)
    
        ax_count.set_title(
            f"{feature} distribution — Patient {patient_id}{title_suffix}"
        )
    
        ax_count.legend(
            loc="upper right",
            frameon=True,
            fontsize=9,
        )
    
        plt.tight_layout()
    
        feature_clean = self._clean_filename(feature)
        date_part = self._make_filename_date_part(patient_df)
    
        filename = f"{patient_id}_{date_part}_KDE_{feature_clean}.png"
        output_path = output_folder / filename
    
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    
        if show:
            plt.show()
    
        plt.close(fig)
    
        self.kde_summary["results"].append(plot_result)
    
        print(f"Saved KDE plot: {output_path}")
    def run(
        self,
        output_folder,
        min_wb=10,
        bins=30,
        show=False,
        dpi=150,
        plot_histogram=True,
        save_json=True,
    ):
        """
        Run KDE plot generation and optionally save a JSON summary.
        """
    
        output_folder = Path(output_folder)
    
        self.kde_summary = {
            "config": self._build_kde_config(
                output_folder=output_folder,
                min_wb=min_wb,
                bins=bins,
                show=show,
                dpi=dpi,
                plot_histogram=plot_histogram,
            ),
            "results": [],
        }
    
        self.summary()
    
        self.generate_plots(
            output_folder=output_folder,
            min_wb=min_wb,
            bins=bins,
            show=show,
            dpi=dpi,
            plot_histogram=plot_histogram,
        )
    
        if save_json:
            self.save_kde_summary_json(output_folder=output_folder)
    
        return self
    def _make_title_suffix(self, patient_df):
        """
        Create plot title suffix showing the actual recording dates.
        """
    
        dates = sorted(
            patient_df["recording_date"]
            .dropna()
            .astype(str)
            .unique()
        )
    
        if not dates:
            return ""
    
        dates_text = ", ".join(dates)
    
        return f" — Dates: {dates_text}"

    def _make_filename_date_part(self, patient_df):
        """
        Create filename-safe date information using the actual recording dates.
        """
    
        dates = sorted(
            patient_df["recording_date"]
            .dropna()
            .astype(str)
            .unique()
        )
    
        if not dates:
            return "no_date"
    
        dates_text = "_".join(dates)
    
        return self._clean_filename(dates_text)

    @staticmethod
    def _clean_filename(text):
        """
        Create a safe filename component.
        """

        text = str(text)
        text = text.replace(" ", "_")
        text = re.sub(r"[^\w\d_-]", "", text)

        return text