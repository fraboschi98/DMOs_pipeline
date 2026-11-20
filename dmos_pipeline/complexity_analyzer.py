# -*- coding: utf-8 -*-
"""
INSERT description, aligned with all the other classes
"""

import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import antropy as ant

# Number of possible PA states and range of the walking states, for each
# definition of the barcode available in `WalkingBouts_analyzer`.
#   "modified":    state 1 (sedentary) + 4 duration ranges x 5 cadence ranges
#                  (states 7-26)                                   -> 21 states
#   "traditional": state 1 (sedentary) + 3 duration ranges x 4 cadence ranges
#                  (states 7-18)                                   -> 13 states
_PA_SCHEMES = {
    "modified": {"n_states": 21, "activity_max": 26},
    "traditional": {"n_states": 13, "activity_max": 18},
}

# Duration and cadence bin edges used by `WalkingBouts_analyzer`'s
# `add_pa_state_to_wb_dataframe` to turn (duration, cadence) into a walking
# PA state (states 7+). Kept here, duplicated, only to turn a state number
# back into a human-readable duration/cadence label for `plot_barcode`'s
# legend; if those bins ever change there, update them here too.
_DURATION_BINS_S = {"modified": [30, 120, 360], "traditional": [30, 120]}
_CADENCE_BINS_STEPS_PER_MIN = {
    "modified": [70, 90, 110, 130],
    "traditional": [50, 80, 140],
}


class PAComplexity_analyzer:
    """
    Compute complexity metrics of the daily physical-activity (PA) state
    sequence, from walking-bout-level data to day-level Hn, LZC and PLZC, plus
    a week-level SampEn (`weekly_summary`).

    Workflow (see `run()`): build the 1 Hz PA-state series per day
    (`build_pa_time_series`), smooth it (`smooth`), then compute the daily
    metrics (`compute_metrics`). Call `weekly_summary()` afterwards for the
    weekly aggregation and SampEn.

    Parameters
    ----------
    wb_parameters_average : pandas.DataFrame
        Walking-bout-level table from `WalkingBouts_analyzer`, with at least
        patient_id, recording_date, session_id, WB_id, start, end, PA_state.
    log : list of dict
        Logs from `Collector`, used for wearing_start_sample,
        wearing_start_unix_time, non_wearing_windows_samples and
        wearing_time_hours.
    fs : float, default=102.4
        Sampling frequency (Hz) of `start`/`end` in `wb_parameters_average`.
    apply_time_window : bool, default=True
        If True, each day's series is cut to the first `window_hours` of worn
        time. If False, the full worn time of the day is kept.
    window_hours : float, default=8
        Length (hours) of the daily series when `apply_time_window=True`.
    window_seconds : int, default=30
        Smoothing window length (seconds).
    k_folds : int, default=10
        Number of random-shift folds in the smoothing.
    pa_state_choice : {"modified", "traditional"}, default="modified"
        PA-state scheme used upstream in `add_pa_state_to_wb_dataframe`. Sets
        the defaults of `n_states` and `activity_max`.
    activity_min : int, default=7
        Lowest PA state counted as "active" (used only if "%Activity" is
        requested in `compute_metrics`/`weekly_summary`).
    activity_max : int or None, default=None
        Highest PA state counted as "active". None uses the value implied by
        `pa_state_choice` (26 for "modified", 18 for "traditional").
    n_states : int or None, default=None
        Alphabet size (number of possible PA states) used to normalize Hn and
        LZC. None uses the value implied by `pa_state_choice` (21 for
        "modified", 13 for "traditional").
    lzc_base : {"n_states", "distinct"} or int, default="n_states"
        Alphabet size used in the LZC normalization: the scheme's `n_states`
        (fixed), the number of distinct states actually observed in the
        series ("distinct"), or an explicit integer.
    plzc_order : int, default=5
        Embedding dimension m of the PLZC's ordinal motifs (3-7).
    plzc_delay : "acf" or int, default="acf"
        Lag (seconds) of the PLZC's motifs. "acf" selects it once for the
        whole run, from the mean autocorrelation of all valid days analyzed
        together; an integer fixes it explicitly.
    sampen_order : int, default=3
        Embedding dimension m of the sample entropy.
    sampen_tolerance : float, default=1.0
        Tolerance r of the sample entropy.
    random_state : int or None, default=0
        Seed for the smoothing's random shifts (one reproducible generator
        per patient and date). None makes the shifts non-reproducible.
    min_wearing_time_hours : float, default=8.0
        Minimum worn time (hours) for a day to be analyzed; shorter days are
        skipped.

    Attributes
    ----------
    complexity_df : pandas.DataFrame
        Daily metrics, filled by `compute_metrics` (called by `run()`).
    time_series, smoothed_series : pandas.DataFrame
        The 1 Hz series before and after smoothing (filled by
        `build_pa_time_series` and `smooth`).
    plzc_lag_ : int or None
        The PLZC lag actually used in the last `compute_metrics` call.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from dmos_pipeline import PAComplexity_analyzer
    >>> # minimal toy input: one 90 s walking bout every 15 min over 8 h of wear
    >>> fs = 102.4
    >>> starts = [int(m * 15 * 60 * fs) for m in range(32)]
    >>> wb = pd.DataFrame({
    ...     "patient_id": "P1", "recording_date": "2023-01-01", "session_id": "s1",
    ...     "WB_id": range(len(starts)),
    ...     "start": starts,
    ...     "end": [s + int(90 * fs) for s in starts],
    ...     "PA_state": np.random.default_rng(0).integers(7, 27, len(starts)),
    ... })
    >>> log = [{"patient_id": "P1", "recording_date": "2023-01-01",
    ...         "session_id": "s1", "log": {"recording_summary": {
    ...             "recording_time_hours": 8.0, "wearing_time_hours": 8.0,
    ...             "wearing_start_sample": 0, "wearing_start_unix_time": 0,
    ...             "non_wearing_windows_samples": []}}}]
    >>> analyzer = PAComplexity_analyzer(wb, log, min_wearing_time_hours=0).run()
    >>> analyzer.complexity_df[["recording_date", "Hn", "LZC", "PLZC"]]
      recording_date        Hn       LZC      PLZC
    0     2023-01-01  0.192322  0.004801  0.000459
    """

    # State used for every second not covered by a walking bout. In this
    # pipeline lying/sitting/standing are not distinguished, so all non-walking
    # time is one "sedentary" state (state 1 of the barcode).
    SEDENTARY_STATE = 1

    # Constructor: stores every option below (with its default) and checks
    # the few that only accept a fixed set of values.
    def __init__(
        self,
        wb_parameters_average,
        log,
        fs=102.4,
        apply_time_window=True,
        window_hours=8,
        window_seconds=30,
        k_folds=10,
        pa_state_choice="modified",
        activity_min=7,
        activity_max=None,
        n_states=None,
        lzc_base="n_states",
        plzc_order=5,
        plzc_delay="acf",
        sampen_order=3,
        sampen_tolerance=1.0,
        random_state=0,
        min_wearing_time_hours=8.0,
    ):
        # Check these four options now, in the constructor, instead of
        # discovering a wrong one only deep inside a metric computation,
        # where it would raise no error and just turn into NaN.

        # pa_state_choice selects which barcode scheme to use. Only the two
        # schemes defined in _PA_SCHEMES are known ("modified" or
        # "traditional"); anything else has no entry there.
        if pa_state_choice not in _PA_SCHEMES:
            raise ValueError("pa_state_choice must be 'modified' or 'traditional'")

        # lzc_base is the alphabet size (number of possible states) used to
        # normalize the LZC. It is either the scheme's fixed n_states, the
        # string "distinct" (count only the states actually observed that
        # day), or an integer the caller fixes explicitly.
        if lzc_base not in ("n_states", "distinct") and not isinstance(lzc_base, int):
            raise ValueError("lzc_base must be 'n_states', 'distinct' or an integer")

        # plzc_order (m) is the embedding dimension of the PLZC's motifs.
        # Bai et al. (2015) [50]: m < 3 gives too few possible motifs to be
        # meaningful; m > 7 is too costly to compute, since the number of
        # motifs grows as m! (e.g. 5040 for m=7).
        if not isinstance(plzc_order, int) or not 3 <= plzc_order <= 7:
            raise ValueError("plzc_order must be an integer between 3 and 7")

        # plzc_delay (tau) is the lag, in seconds, between the values that
        # make up a PLZC motif. "acf" picks it automatically (see
        # _select_plzc_lag); otherwise it must be a fixed positive integer.
        if plzc_delay != "acf" and not (isinstance(plzc_delay, int) and plzc_delay >= 1):
            raise ValueError("plzc_delay must be 'acf' or a positive integer")

        # From here on, every parameter is only STORED on `self` (no
        # computation yet), so the methods below can read it later.

        # n_states and activity_max have a "None" default (see __init__'s
        # signature): when left as None, they are filled in from the chosen
        # PA-state scheme (_PA_SCHEMES, defined near the top of this file),
        # e.g. "modified" -> n_states=21, activity_max=26.
        scheme = _PA_SCHEMES[pa_state_choice]
        self.pa_state_choice = pa_state_choice
        self.n_states = scheme["n_states"] if n_states is None else n_states
        self.activity_min = activity_min
        self.activity_max = scheme["activity_max"] if activity_max is None else activity_max

        # --- parameters of the 4 complexity metrics (see class docstring
        # for what each one means) ---
        self.sampen_order = sampen_order
        self.sampen_tolerance = sampen_tolerance
        self.lzc_base = lzc_base
        self.plzc_order = plzc_order
        self.plzc_delay = plzc_delay

        # --- parameters of the daily series, the smoothing, and the day
        # validity filter ---
        self.min_wearing_time_hours = min_wearing_time_hours
        self.fs = fs
        self.apply_time_window = apply_time_window
        self.window_hours = window_hours
        self.window_seconds = window_seconds
        self.k_folds = k_folds

        # The actual data. `.copy()` on the WB table: methods below (e.g.
        # build_pa_time_series) edit this dataframe in place; without the
        # copy, those edits would also change the dataframe the caller
        # passed in and still holds a reference to, outside this class.
        # `log if log is not None else []`: so a missing log does not crash
        # later, when the code iterates over self.log, but simply behaves
        # as if there were no sessions at all.
        self.wb = wb_parameters_average.copy()
        self.log = log if log is not None else []

        # Seed of the random smoothing shifts. Each day gets its own generator,
        # derived from this seed, the patient and the date (see `_day_rng`), so
        # the smoothing of a day does not depend on the other days analyzed with
        # it. With None the shifts are different at each run.
        self.random_state = random_state

        # Outputs filled by the steps below.
        self.plzc_lag_ = None  # lag (s) used for the PLZC, set in compute_metrics
        self.sessions_info = pd.DataFrame()
        self.time_series = pd.DataFrame()
        self.smoothed_series = pd.DataFrame()
        self.complexity_df = pd.DataFrame()

    # ==================================================================
    # STEP 1-2. Day validity and daily PA-state series
    # ==================================================================
    def get_sessions_info(self):
        """
        Read from the logs the information of each session (recording).

        Reads directly from `Collector.log` (`recording_summary`).

        Output columns: patient_id, recording_date, session_id,
        recording_time_hours, wearing_time_hours, wearing_start_sample,
        wearing_start_unix_time, non_wearing_windows (list of [start, end]
        samples).
        """
        rows = []
        for item in self.log:
            summary = item.get("log", {}).get("recording_summary", {})
            # The collector can store the date under either key.
            date = item.get("recording_date", item.get("date"))
            rows.append(
                {
                    "patient_id": item.get("patient_id"),
                    "recording_date": date,
                    "session_id": str(item.get("session_id")),
                    "recording_time_hours": summary.get("recording_time_hours"),
                    "wearing_time_hours": summary.get("wearing_time_hours"),
                    "wearing_start_sample": summary.get("wearing_start_sample"),
                    "wearing_start_unix_time": summary.get("wearing_start_unix_time"),
                    "non_wearing_windows": summary.get("non_wearing_windows_samples") or [],
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "patient_id",
                "recording_date",
                "session_id",
                "recording_time_hours",
                "wearing_time_hours",
                "wearing_start_sample",
                "wearing_start_unix_time",
                "non_wearing_windows",
            ],
        )
        # Missing log files (or logs without a recording summary) give no
        # usable session information.
        df = df.dropna(subset=["recording_time_hours", "wearing_start_sample"])
        self.sessions_info = df.reset_index(drop=True)
        return self.sessions_info

    @staticmethod
    def _order_sessions(info):
        """
        Chronological order of the sessions of one day.

        By wearing start unix time when every session has one (> 0), otherwise
        by session_id (a guess: a message is printed in that case).
        """
        unix = pd.to_numeric(info["wearing_start_unix_time"], errors="coerce").fillna(0)
        if len(info) > 1 and (unix > 0).all():
            return info.assign(_t=unix).sort_values("_t").drop(columns="_t")
        return info.sort_values("session_id")

    def _session_series(self, wbs, rec_hours, start_sample, windows):
        """
        PA-state series (1 Hz) of the WORN time of ONE session (one
        recording).

        Parameters
        ----------
        wbs : pandas.DataFrame
            The walking bouts of this session only (with `start`, `end` in
            raw samples, and `PA_state`).
        rec_hours : float
            Total duration of the recording (hours), from the log.
        start_sample : float
            The first WORN sample of this session (raw samples): everything
            before it (e.g. the sensor lying on a table before being put on)
            is not worn.
        windows : list of [start, end]
            Non-wearing windows inside the recording (raw samples), e.g. the
            sensor taken off and put back on.

        Returns
        -------
        numpy.ndarray
            One state per WORN second only: the unworn seconds are removed,
            not kept as gaps. So the length of the result is the worn time of
            this session in seconds, not the recording's length, and index 0
            of the result is the first worn second, not the first second of
            the recording.
        """
        fs = self.fs

        # Step 1: lay out ONE state per second for the WHOLE recording
        # (worn or not), as if every second were sedentary, then overwrite
        # with the real PA_state wherever a walking bout occurred.
        # `n_sec` must be large enough to hold every walking bout even if a
        # bout's end falls slightly after the log's own recording duration
        # (rounding): hence the max() with the last walking bout's end.
        last_wb_sec = int(np.floor(wbs["end"].max() / fs)) + 1 if len(wbs) else 0
        n_rec = int(round(float(rec_hours) * 3600)) if pd.notna(rec_hours) else 0
        n_sec = max(n_rec, last_wb_sec)

        states = np.full(n_sec, self.SEDENTARY_STATE, dtype=int)
        for _, row in wbs.iterrows():
            # Convert raw samples (sampling frequency) to seconds. Using
            # seconds (1 Hz) is what makes the 30 s smoothing window and the
            # entropy parameters mean what the papers describe.
            s = int(np.floor(row["start"] / fs))
            e = int(np.floor(row["end"] / fs))
            s, e = max(s, 0), min(e, n_sec - 1)
            if s <= e:
                states[s : e + 1] = int(row["PA_state"])

        # Step 2: mark which of those seconds were actually WORN: False
        # before the wearing start, and False inside every non-wearing
        # window (the sensor taken off and put back on during the session).
        worn = np.ones(n_sec, dtype=bool)
        worn[: int(np.floor(start_sample / fs))] = False
        for s, e in windows:
            worn[int(np.floor(s / fs)) : int(np.ceil(e / fs))] = False

        # Step 3: keep only the worn seconds. This is a selection, not a
        # mask over the same positions: the array returned is shorter than
        # `states`, and its seconds are no longer at their original clock
        # position, only in the same relative order.
        return states[worn]

    def build_pa_time_series(self):
        """
        Build the 1 Hz PA-state series per patient and recording date.

        Output (`self.time_series`): patient_id, recording_date, session_id
        (the sessions used, joined by "+"), second, PA_state, where `second`
        is the time in worn seconds from the start of the series.
        """
        # The WB table must carry the PA state computed by the WB analyzer.
        for col in ["patient_id", "recording_date", "session_id", "start", "end", "PA_state"]:
            if col not in self.wb.columns:
                raise KeyError(f"Column '{col}' not found in wb_parameters_average")

        sessions = self.get_sessions_info()

        # One row per walking bout (the WB table can repeat a WB), and only WBs
        # that received a PA state (WBs with missing cadence/duration have none).
        wb = (
            self.wb.drop_duplicates(["patient_id", "recording_date", "session_id", "WB_id"])
            .dropna(subset=["PA_state"])
            .copy()
        )
        wb["session_id"] = wb["session_id"].astype(str)

        parts = []
        for (pid, date), day_wb in wb.groupby(["patient_id", "recording_date"]):
            info = sessions[
                (sessions["patient_id"] == pid) & (sessions["recording_date"] == date)
            ]

            if info.empty:
                # Fallback when this day has no usable log at all (no match
                # in `sessions`): build a fake `info` table instead of
                # skipping the day. One row per session that has walking
                # bouts; the wearing start is guessed as that session's
                # first walking bout, and there are no known non-wearing
                # windows. `has_log = False` below then skips the wearing-
                # time filter for this day, since there is no wearing_time_hours
                # to check it against.
                info = pd.DataFrame(
                    [
                        {
                            "session_id": sid,
                            "recording_time_hours": np.nan,
                            "wearing_time_hours": np.nan,
                            "wearing_start_sample": float(g["start"].min()),
                            "wearing_start_unix_time": 0,
                            "non_wearing_windows": [],
                        }
                        for sid, g in day_wb.groupby("session_id")
                    ]
                )
                has_log = False
            else:
                has_log = True

            # --- STEP 1: day validity (wearing-time filter) ---
            # Worn time of the whole day = sum over its sessions (same
            # definition as DayLevel_analyzer / KDEPlotter). Days with less
            # worn time than the threshold are skipped.
            if self.min_wearing_time_hours is not None and has_log:
                wear_h = pd.to_numeric(info["wearing_time_hours"], errors="coerce").sum(
                    min_count=1
                )
                if pd.isna(wear_h) or wear_h < self.min_wearing_time_hours:
                    print(
                        f"[!] {pid} {date}: wearing time {wear_h} h < "
                        f"{self.min_wearing_time_hours} h, day skipped"
                    )
                    continue

            # --- order of the sessions of the day ---
            # If a day has more than one recording (e.g. the sensor was
            # taken off and put on again later), all of them contribute:
            # `ordered` lists them in the order their worn series will be
            # concatenated in (see STEP 2 below). This order only matters
            # when there is more than one session; with one, there is
            # nothing to order.
            ordered = self._order_sessions(info)
            if len(ordered) > 1:
                unix = pd.to_numeric(ordered["wearing_start_unix_time"], errors="coerce")
                how = "unix start time" if (unix.fillna(0) > 0).all() else "session_id (guess)"
                print(
                    f"[i] {pid} {date}: {len(ordered)} sessions joined in order of {how}: "
                    f"{', '.join(ordered['session_id'])}"
                )

            # A walking bout whose session_id has no matching row in `info`
            # (no log for that session) cannot be placed in time: there is
            # no wearing_start_sample to convert its samples into seconds
            # from. Such bouts are silently excluded from `ordered`'s loop
            # below; this only reports how many, so they are not lost
            # without a trace.
            known = set(ordered["session_id"])
            n_lost = int((~day_wb["session_id"].isin(known)).sum())
            if n_lost:
                print(f"[!] {pid} {date}: {n_lost} WB of sessions without log ignored")

            # --- STEP 2: worn series of each session, joined in order ---
            # For each session of the day (in the order decided above), get
            # only its own walking bouts and build its worn-time series
            # (`_session_series`); then lay all those series end to end.
            # The seconds between two sessions (e.g. the time the sensor was
            # off and the day's recording was paused) are NOT inserted: the
            # last worn second of one session is immediately followed by the
            # first worn second of the next.
            pieces = []
            for _, s in ordered.iterrows():
                wbs = day_wb[day_wb["session_id"] == s["session_id"]]
                pieces.append(
                    self._session_series(
                        wbs,
                        s["recording_time_hours"],
                        float(s["wearing_start_sample"]),
                        s["non_wearing_windows"],
                    )
                )
            states = np.concatenate(pieces)

            # Length: keep only the first `window_hours` hours of worn time
            # of the day (cutting a longer day short). If the day has less
            # worn time than that, it is simply shorter; the day-validity
            # filter above already ensures it has at least
            # `min_wearing_time_hours`.
            if self.apply_time_window:
                states = states[: int(self.window_hours * 3600)]

            parts.append(
                pd.DataFrame(
                    {
                        "patient_id": pid,
                        "recording_date": date,
                        "session_id": "+".join(ordered["session_id"]),
                        "second": np.arange(len(states)),
                        "PA_state": states,
                    }
                )
            )

        self.time_series = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return self

    # ==================================================================
    # STEP 3. Smoothing ([44], Methods section 2.2)
    # ==================================================================
    @staticmethod
    def _window_mode(windows):
        """
        Most frequent value of each row (NaN ignored; ties -> lowest value).

        Vectorized with a count matrix (rows x states) instead of looping or
        using scipy.stats.mode, because it runs on ~1000 windows x K folds per
        day.
        """
        n_values = int(np.nanmax(windows)) + 1
        counts = np.zeros((windows.shape[0], n_values), dtype=int)
        rows, cols = np.nonzero(~np.isnan(windows))
        np.add.at(counts, (rows, windows[rows, cols].astype(int)), 1)
        return counts.argmax(axis=1)

    def _single_fold(self, pa, shift):
        """
        One smoothing fold: non-overlapping windows of `window_seconds`, each
        replaced by its most frequent state, with the window grid shifted by
        `shift` seconds.
        """
        n = len(pa)
        # Shift the grid: pad NaN at the start (NaN are ignored by the mode).
        pad = shift % self.window_seconds
        padded = np.pad(pa.astype(float), (pad, 0), constant_values=np.nan)
        # Pad NaN at the end so the length is a multiple of the window.
        remainder = len(padded) % self.window_seconds
        if remainder:
            padded = np.pad(padded, (0, self.window_seconds - remainder), constant_values=np.nan)

        # One row per window -> its mode -> repeated back to 1 Hz -> remove the
        # padding to get the original length again.
        modes = self._window_mode(padded.reshape(-1, self.window_seconds))
        return np.repeat(modes, self.window_seconds)[pad : pad + n]

    def _day_rng(self, patient_id, recording_date):
        """
        Random generator of one day, derived from `random_state`, the patient
        and the date.

        [44] does not say whether the random shifts are the same for all the
        days, nor mention a seed. Here each day has its own reproducible
        generator, so the smoothing of a day is the same whether it is analyzed
        alone or together with other days or patients (a single generator used
        in sequence would give different shifts depending on the days that come
        before). The patient and date are turned into a number with a CRC
        (stable between runs, unlike Python's `hash`). With
        `random_state=None` the generator is not reproducible.
        """
        if self.random_state is None:
            return np.random.default_rng()
        key = zlib.crc32(f"{patient_id}|{recording_date}".encode("utf-8"))
        return np.random.default_rng(np.random.SeedSequence([int(self.random_state), key]))

    def smooth_series(self, pa, rng):
        """
        K-fold smoothing of one PA-state array.

        K folds, each with a random shift of 1..`window_seconds` seconds drawn
        from `rng`; the final state of each second is the majority vote across
        the folds ([44] Methods section 2.2). Random shifts avoid that the
        result depends on where the first window happens to start.
        """
        shifts = rng.integers(1, self.window_seconds + 1, size=self.k_folds)
        folds = np.column_stack([self._single_fold(pa, s) for s in shifts])
        return self._window_mode(folds.astype(float))

    def smooth(self):
        """Smooth each patient/day series. Adds `PA_state_smoothed`."""
        if self.time_series.empty:
            raise ValueError("time_series is empty. Run build_pa_time_series() first.")

        parts = []
        for (pid, date), day in self.time_series.groupby(["patient_id", "recording_date"]):
            day = day.sort_values("second").copy()
            day["PA_state_smoothed"] = self.smooth_series(
                day["PA_state"].values, self._day_rng(pid, date)
            )
            parts.append(day)

        self.smoothed_series = pd.concat(parts, ignore_index=True)
        return self

    # ==================================================================
    # STEP 4. Metrics
    # ==================================================================
    @staticmethod
    def _to_letters(values):
        """
        Encode integer states as letters (state s -> chr(65 + s)).

        Needed for the Lempel-Ziv algorithm, which works on symbol strings.
        The mapping is one-to-one, so the complexity does not depend on it.
        """
        return "".join(chr(65 + int(v)) for v in values)

    def _shannon_normalized(self, series):
        """
        Normalized Shannon entropy Hn of the state distribution.

        Measures the VARIETY of states used during the day (insensitive to
        their temporal order). Normalized by the maximal entropy log(alpha),
        alpha = number of POSSIBLE states (`n_states`), so Hn is in [0, 1]
        ([26]). The ratio does not depend on the log base. The denominator is
        FIXED, so days and patients are comparable (a per-day denominator would
        make a day with 2 states look maximally even).
        """
        _, counts = np.unique(series, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum() / np.log2(self.n_states))

    def _lzc(self, series):
        """
        Normalized Lempel-Ziv complexity (number of new sub-patterns along the
        sequence): measures how hard the sequence is to compress, i.e. how
        irregular its temporal structure is. Computed on the whole daily
        series, as in [26].

        As defined in [26]:

            LZC = c_alpha(S(N, alpha)) / (N / log_alpha N)

        with N the length of the sequence, alpha the size of its alphabet (the
        number of possible states) and c the number of sub-patterns, counted
        with the Kaspar-Schuster algorithm (the last phrase is counted). 
        [27] cites [26] for the LZC; [3] and [27] give no formula.

        Base alpha: `lzc_base="n_states"` is the number of possible states of
        the PA-state scheme in use (fixed for all days); "distinct" is the
        number of different states present in the sequence (antropy's own
        choice); an integer fixes it explicitly.
        """
        if len(series) < 2:
            return np.nan
        letters = self._to_letters(series)
        if self.lzc_base == "distinct":
            base = max(len(np.unique(series)), 2)
        elif self.lzc_base == "n_states":
            base = self.n_states
        else:
            base = self.lzc_base
        c = ant.lziv_complexity(letters, normalize=False)
        n = len(series)
        return float(c / (n / np.log(n) * np.log(base)))

    @staticmethod
    def _acf(series):
        """
        Autocorrelation function of the series, normalized so that its peak
        (lag 0) is 1. Computed through the FFT (zero-padded to avoid
        wrap-around). Returns None for a constant or too short series.
        """
        x = np.asarray(series, dtype=float)
        x = x - x.mean()
        n = len(x)
        if n < 2 or not np.any(x):
            return None
        f = np.fft.rfft(x, 2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n]
        return acf / acf[0]

    @staticmethod
    def _lag_from_acf(acf):
        """
        Lag (in samples) at which the autocorrelation function has decayed to
        1/e of its peak value: the rule used by Bai et al. (2015) [50] to
        choose the lag of the PLZC. If it never decays that far, 1 is returned.
        """
        below = np.nonzero(np.asarray(acf) <= np.exp(-1))[0]
        return int(below[0]) if len(below) else 1

    def _select_plzc_lag(self, series_list):
        """
        One lag for all the days analyzed.

        Bai et al. (2015) [50] choose the lag once, from the autocorrelation
        function of a signal (the lag where it decays to 1/e of its peak), and
        use it for the PLZC of the whole analysis; they do not describe
        recomputing it for each window. Here the autocorrelation function of
        each valid day is computed, the functions are averaged (up to the
        length of the shortest day) and the 1/e rule is applied to the mean
        function. If `plzc_delay` is an integer it is used as it is.

        Note: with "acf" the lag depends on the days that are analyzed together;
        for comparisons between runs, set `plzc_delay` to the value printed.
        """
        if self.plzc_delay != "acf":
            return int(self.plzc_delay)
        acfs = [a for a in (self._acf(s) for s in series_list) if a is not None]
        if not acfs:
            return 1
        n = min(len(a) for a in acfs)
        mean_acf = np.mean([a[:n] for a in acfs], axis=0)
        return max(self._lag_from_acf(mean_acf), 1)

    @staticmethod
    def _bai_lz_count(s):
        """
        Lempel-Ziv complexity c(n) of a symbol string, following Blocks 1-8 of
        Bai et al. (2015) [50] (the "primary" LZC):

        - S is the first symbol, Q the second, c = 1;
        - SQ = S + Q, and SQv is SQ without its last character;
        - if Q has reached the last symbol of the sequence, stop;
        - if Q is a substring of SQv, Q is extended by the next symbol;
        - otherwise Q is a new pattern: S becomes SQ, Q is the next symbol and
          c is increased by one.

        The last, unfinished Q is not counted (the paper does not add it), so
        the value can be one lower than the phrase count of antropy.
        """
        n = len(s)
        if n < 2:
            return n
        c = 1
        s_len = 1
        q_len = 1
        while s_len + q_len < n:
            end = s_len + q_len
            if s[s_len:end] in s[: end - 1]:
                q_len += 1
            else:
                c += 1
                s_len = end
                q_len = 1
        return c

    def _ordinal_symbols(self, series, tau):
        """
        Bandt-Pompe ordinal patterns of `plzc_order` (m) values spaced by
        `tau` seconds, one integer symbol per position.

        The embedding vector is [x(t-(m-1)tau), ..., x(t-tau), x(t)] (oldest
        value first) and its pattern is the permutation that sorts it in
        ascending order. Exact ties are broken by the index: the older value
        counts as the smaller one (a stable sort), as in Bai et al. (2015)
        [50]. There are m! possible patterns. The number of patterns is
        T - (m-1)*tau.
        """
        m = self.plzc_order
        n = len(series) - (m - 1) * tau
        if n < 2:
            return None
        x = np.column_stack([series[i * tau : i * tau + n] for i in range(m)])
        # Stable argsort: among equal values the lower column (older) comes first.
        perm = np.argsort(x, axis=1, kind="stable")
        # The permutation is a one-to-one code of the pattern (base-m number).
        return (perm * (m ** np.arange(m))).sum(axis=1)

    def _plzc(self, series, tau):
        """
        Permutation Lempel-Ziv complexity as defined by Bai et al. (2015) [50]:
        Lempel-Ziv complexity c(n) of the sequence of n ordinal patterns
        (motifs), normalized with their Eq. (6):

            PLZC = c(n) * (log_{m!}(c(n)) + 1) / n

        with m! the number of possible motifs (the alphabet) and n the length
        of the motif sequence, n = T - (m-1)*tau. (Their Eq. (7),
        c(n) * log_{m!}(n) / n, is the simplification for very large n.)

        Steps as in the paper: motifs of m values spaced by the lag tau, ties
        broken by the index (the older value is the smaller one), c(n) counted
        with their Blocks 1-8 (`_bai_lz_count`). The lag `tau` is selected once
        for all the days (`_select_plzc_lag`).

        Why: [45] describes it as more robust to signal artifacts than the
        plain Lempel-Ziv complexity, because it only uses the order relations
        between values and not the values themselves. Caution: PA state labels
        are categories whose numeric order is not a physical magnitude, so the
        order relations have a limited meaning here.
        """
        m = self.plzc_order
        codes = self._ordinal_symbols(series, tau)
        if codes is None:
            return np.nan
        n = len(codes)
        # Bai et al. (2015): every motif can occur only if m! <= n, and n should
        # be much larger than m! to avoid under-sampling.
        if n < 10 * math.factorial(m):
            print(
                f"[!] PLZC: only {n} patterns for m={m} (m! = {math.factorial(m)}): "
                "the series is short for this order"
            )
        # One symbol (character) per distinct pattern; code points start at
        # 0x100 so they never collide with control characters.
        _, inverse = np.unique(codes, return_inverse=True)
        letters = "".join(chr(0x100 + int(k)) for k in inverse)
        c = self._bai_lz_count(letters)
        alpha = math.factorial(m)
        return float(c * (np.log(c) / np.log(alpha) + 1) / n)

    def _sampen(self, series):
        """
        Sample entropy of a series: -ln of the conditional probability that two
        patterns similar for m points stay similar at the next point (m+1)
        ([26], Eq. 1), with m = `sampen_order` and r = `sampen_tolerance`
        ([26]: m = 3, r = 1).

        [26] computes it on the barcode obtained by concatenating the data of
        the 5 recording days, not on single days; here it is
        computed in `weekly_summary` on the concatenation of the valid days of
        the period, and not as a daily metric.

        Caution [45]: SampEn is "very susceptible to resting periods", giving
        lower values when the signal contains extended resting periods (here
        the sedentary state), so low values are expected.
        """
        tol = float(np.nextafter(self.sampen_tolerance, 0))
        return float(ant.sample_entropy(series, order=self.sampen_order, tolerance=tol))

    def compute_metrics(self, metrics=("Hn", "LZC", "PLZC")):
        """
        Compute the daily complexity metrics per patient and date, on the
        smoothed series. The sample entropy is not a daily metric: it is
        computed in `weekly_summary` on the concatenated days, as in [26].

        Days with fewer than 10 s or a single PA state get NaN (and
        %Activity = 0 if it is requested): the entropies are undefined on a
        constant series. `%Activity` is not computed by default: pass
        `metrics=("%Activity", ...)` to get it.
        """
        if self.smoothed_series.empty:
            raise ValueError("smoothed_series is empty. Run smooth() first.")

        days = {
            key: day.sort_values("second")
            for key, day in self.smoothed_series.groupby(["patient_id", "recording_date"])
        }

        # PLZC lag: selected once for all the valid days analyzed (see
        # `_select_plzc_lag`, Bai et al. 2015 [50]).
        self.plzc_lag_ = None
        if "PLZC" in metrics:
            valid = [
                d["PA_state_smoothed"].astype(int).values
                for d in days.values()
                if len(d) >= 10 and d["PA_state_smoothed"].nunique() > 1
            ]
            self.plzc_lag_ = self._select_plzc_lag(valid)
            print(
                f"[i] PLZC: lag = {self.plzc_lag_} s "
                f"({'from the mean autocorrelation of ' + str(len(valid)) + ' day(s)' if self.plzc_delay == 'acf' else 'given'})"
            )

        rows = []
        for (pid, date), day in days.items():
            series = day["PA_state_smoothed"].astype(int).values
            row = {
                "patient_id": pid,
                "recording_date": date,
                "session_id": day["session_id"].iloc[0],
                "n_seconds": len(series),
            }

            # Edge case: nothing to measure.
            if len(series) < 10 or len(np.unique(series)) <= 1:
                print(f"[!] {pid} {date}: insufficient variation, metrics skipped")
                for m in metrics:
                    row[m] = 0 if m == "%Activity" else np.nan
                rows.append(row)
                continue

            # %Activity: share of the day spent in "active" (walking) states.
            # A QUANTITY measure, to be read together with the complexity
            # (QUALITY) measures ([44]).
            if "%Activity" in metrics:
                row["%Activity"] = (
                    np.sum((series >= self.activity_min) & (series <= self.activity_max))
                    / len(series)
                    * 100
                )

            # Hn: variety of states (see _shannon_normalized).
            if "Hn" in metrics:
                row["Hn"] = self._shannon_normalized(series)

            # LZC: compressibility / irregularity (see _lzc).
            if "LZC" in metrics:
                try:
                    row["LZC"] = self._lzc(series)
                except Exception as e:
                    print(f"[!] {pid} {date}: LZC failed: {e}")
                    row["LZC"] = np.nan

            # PLZC: LZC on ordinal patterns, more robust to artifacts [45].
            # The lag used to build the motifs (`self.plzc_lag_`, chosen once
            # for every day and patient analyzed together, see above) is NOT a
            # complexity metric and is not saved as a column: it is only
            # printed once above. It changes if the days analyzed together
            # change, so re-read the printed line when comparing runs.
            if "PLZC" in metrics:
                try:
                    row["PLZC"] = self._plzc(series, self.plzc_lag_)
                except Exception as e:
                    print(f"[!] {pid} {date}: PLZC failed: {e}")
                    row["PLZC"] = np.nan
            rows.append(row)

        self.complexity_df = pd.DataFrame(rows)
        return self

    # ==================================================================
    # STEP 5. Weekly summary (separate step, called by the user)
    # ==================================================================
    def weekly_summary(
        self,
        period_col="session_id",
        metrics=("Hn", "LZC", "PLZC"),
        aggregation="mean",
    ):
        """
        Summary of the valid days of each measurement period.

        A method of `PAComplexity_analyzer` (not another script): called by
        the user after `run()`. It has two parts:

        - the daily metrics (`metrics`) aggregated over the days of the period
          with the mean (or median), as in [44] (metrics computed for each day
          and averaged over the one-week measurement);
        - the sample entropy (SampEn), computed on the concatenation of the
          smoothed series of the valid days of the period, in chronological
          order, as in [26]: the barcode analyzed there was obtained by
          concatenating the data of the 5 recording days (N = 5 x 8 h x 3600 s
          = 144,000), with one value per subject. No paper of the reference
          list computes SampEn per day on PA-state barcodes, so it is not a
          daily metric here.

        Other papers: [3] has one value per valid day,
        3-7 per period, and [27] averages the 2 days of its design.

        Parameters
        ----------
        period_col : str, default="session_id"
            Column of `complexity_df` identifying the measurement period (e.g.
            "week_1"). The log does not contain the week, so with the current
            data it is the `session_id`. When a day joins several sessions,
            `session_id` holds them joined by "+".
        metrics : tuple of str
            Daily columns to aggregate. Days with NaN in a metric are ignored
            for that metric (the aggregation uses the available days).
        aggregation : {"mean", "median"}, default="mean"
            Statistic used to aggregate the daily values of each period. The
            mean follows [44]; the median is an alternative added here (same
            parameter name as `DayLevel_analyzer`).

        Returns
        -------
        pandas.DataFrame
            One row per patient and period, with:
            - `n_valid_days`: how many days the row is based on;
            - `<metric>_mean` or `<metric>_median` (matching `aggregation`) for
              each of `metrics`, e.g. `Hn_mean`, `LZC_mean`, `PLZC_mean`: the
              MEAN (or MEDIAN) of that metric's DAILY values, not a value
              recomputed on the days put together. The lag used for the PLZC is
              not a column: it is only printed once in `compute_metrics`;
            - `SampEn`: computed ONCE on the concatenation of the valid days
              (not a mean of daily values, see above) and `SampEn_n_seconds`
              (length in seconds of that concatenated series).
        """
        if aggregation not in ["mean", "median"]:
            raise ValueError("aggregation must be either 'mean' or 'median'")
        if self.complexity_df.empty:
            raise ValueError("complexity_df is empty. Run run() first.")
        df = self.complexity_df
        if period_col not in df.columns:
            raise KeyError(f"Column '{period_col}' not found in complexity_df")

        # The aggregated columns are named "<metric>_mean" or "<metric>_median"
        # (whichever `aggregation` is): these are the MEAN (or MEDIAN) of the
        # daily values computed in `compute_metrics`, one per valid day, NOT a
        # value recomputed on the days put together. This differs from SampEn
        # below, which is computed once on the concatenated days.
        cols = [m for m in metrics if m in df.columns]
        agg_names = {m: f"{m}_{aggregation}" for m in cols}
        weekly = (
            df.groupby(["patient_id", period_col])
            .agg(
                n_valid_days=("recording_date", "nunique"),
                **{agg_names[m]: (m, aggregation) for m in cols},
            )
            .reset_index()
        )

        # SampEn on the concatenated valid days of each patient and period.
        sampen, lengths = [], []
        for _, row in weekly.iterrows():
            days = df[
                (df["patient_id"] == row["patient_id"]) & (df[period_col] == row[period_col])
            ]["recording_date"]
            series = (
                self.smoothed_series[
                    (self.smoothed_series["patient_id"] == row["patient_id"])
                    & (self.smoothed_series["recording_date"].isin(days))
                ]
                .sort_values(["recording_date", "second"])["PA_state_smoothed"]
                .astype(int)
                .values
            )
            lengths.append(len(series))
            if len(series) < 10 or len(np.unique(series)) <= 1:
                sampen.append(np.nan)
                continue
            try:
                sampen.append(self._sampen(series))
            except Exception as e:
                print(f"[!] {row['patient_id']} {row[period_col]}: SampEn failed: {e}")
                sampen.append(np.nan)
        weekly["SampEn"] = sampen
        weekly["SampEn_n_seconds"] = lengths
        return weekly

    # ------------------------------------------------------------------
    def run(self):
        """Run steps 1-4: daily series, smoothing and metrics. Returns self."""
        self.build_pa_time_series()
        if self.time_series.empty:
            # Every day was skipped (for example by the wearing-time filter):
            # say so and leave `complexity_df` empty instead of failing in the
            # smoothing step with a misleading error.
            print("[!] No valid day to analyze: complexity_df is empty.")
            return self
        self.smooth()
        self.compute_metrics()
        return self

    def save_complexity(self, output_folder, filename="complexity_metrics.csv"):
        """Save `complexity_df` as CSV and return its path."""
        if self.complexity_df.empty:
            raise ValueError("complexity_df is empty. Run run() before saving.")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        path = output_folder / filename
        self.complexity_df.to_csv(path, index=False)
        return path

    # ==================================================================
    # Barcode plot (not a metric: a visual check / figure)
    # ==================================================================
    def _state_duration_cadence_idx(self, state):
        """
        Decode a walking state (7+) into its (duration bin, cadence bin)
        indices, using the same `divmod` layout
        `WalkingBouts_analyzer.add_pa_state_to_wb_dataframe` uses to build
        the state number in the first place: states are numbered in blocks
        of `n_cad` (one block per duration bin), consecutively by cadence
        bin within each block.

        Returns
        -------
        dur_idx, cad_idx, n_dur, n_cad : int
            Zero-based duration/cadence bin indices of `state`, and the
            total number of duration/cadence bins of the scheme in use.
        """
        dur_bins = _DURATION_BINS_S[self.pa_state_choice]
        cad_bins = _CADENCE_BINS_STEPS_PER_MIN[self.pa_state_choice]
        n_dur, n_cad = len(dur_bins) + 1, len(cad_bins) + 1
        dur_idx, cad_idx = divmod(int(state) - 7, n_cad)
        return dur_idx, cad_idx, n_dur, n_cad

    def _state_label(self, state):
        """
        Human-readable label of a PA state, for the barcode's legend.

        Decodes the (duration, cadence) bin a walking state (7+) was built
        from in `WalkingBouts_analyzer.add_pa_state_to_wb_dataframe`, using
        `_DURATION_BINS_S` / `_CADENCE_BINS_STEPS_PER_MIN` above. State 1
        (`SEDENTARY_STATE`) is always "Sedentary".
        """
        if state == self.SEDENTARY_STATE:
            return "Sedentary"
        dur_bins = _DURATION_BINS_S[self.pa_state_choice]
        cad_bins = _CADENCE_BINS_STEPS_PER_MIN[self.pa_state_choice]
        dur_idx, cad_idx, _, _ = self._state_duration_cadence_idx(state)
        dur_low = 0 if dur_idx == 0 else dur_bins[dur_idx - 1]
        dur_high = dur_bins[dur_idx] if dur_idx < len(dur_bins) else None
        cad_low = 0 if cad_idx == 0 else cad_bins[cad_idx - 1]
        cad_high = cad_bins[cad_idx] if cad_idx < len(cad_bins) else None
        dur_label = f"{dur_low}-{dur_high}s" if dur_high is not None else f">{dur_low}s"
        cad_label = f"{cad_low}-{cad_high}" if cad_high is not None else f">{cad_low}"
        return f"Walking {dur_label}, {cad_label} steps/min"

    def plot_barcode(
        self,
        patient_id,
        use_smoothed=True,
        dates=None,
        figsize=None,
        output_path=None,
        dpi=200,
        show=True,
    ):
        """
        Draw the PA-state barcode of one patient's valid days, one row per
        day, in chronological order, in a SINGLE figure (not a separate
        image per day).

        Not a complexity metric: a visual check, to see the pattern behind
        the numbers. 
        The barcode itself is defined in [26] as the 1 Hz PA-state sequence; by default
        this plots the version actually used for the complexity metrics,
        i.e. AFTER the 30 s majority-vote smoothing of [44]
        (`use_smoothed=True`). Pass `use_smoothed=False` to plot the
        ORIGINAL sequence instead, i.e. the one `build_pa_time_series`
        built, before that smoothing step (this is still the sequence
        already classified into PA states by `WalkingBouts_analyzer`, not
        the raw sensor signal).

        Each day is its own row, only for the days that passed the
        wearing-time filter (the days present in `complexity_df`). Rows are
        NOT concatenated into one continuous strip (unlike the SampEn
        computation in `weekly_summary`): consecutive rows can be far apart
        in the calendar, and the gap (in days) between two consecutive valid
        days is printed above the second one, so this is never mistaken for
        a continuous recording.

        Parameters
        ----------
        patient_id : str
            Patient to plot. Must have at least one valid day in
            `complexity_df` (run `run()` first).
        use_smoothed : bool, default=True
            True: the smoothed barcode (what the metrics are computed on).
            False: the original barcode, before smoothing.
        dates : list of str, optional
            Restrict to these recording dates (must already be valid days).
            None plots every valid day of this patient.
        figsize : (float, float), optional
            Matplotlib figure size. None scales it with the number of days.
        output_path : str or pathlib.Path, optional
            If given, the figure is saved there (passed to `savefig`).
        dpi : int, default=150
            Resolution used when saving.
        show : bool, default=True
            If True, display the figure (`plt.show()`); if False, close it
            after saving (use this in non-interactive scripts).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        if self.complexity_df.empty:
            raise ValueError("complexity_df is empty. Run run() first.")

        days_df = self.complexity_df[self.complexity_df["patient_id"] == patient_id]
        if dates is not None:
            days_df = days_df[days_df["recording_date"].isin(dates)]
        if days_df.empty:
            raise ValueError(f"No valid day found for patient {patient_id!r}.")
        # Chronological order; `valid_dates` keeps the values exactly as they
        # appear in complexity_df (same dtype as in time_series/smoothed_series,
        # so they can be compared directly below, with no type conversion).
        valid_dates = sorted(
            days_df["recording_date"].unique(), key=lambda d: pd.to_datetime(d)
        )

        series_source = self.smoothed_series if use_smoothed else self.time_series
        state_col = "PA_state_smoothed" if use_smoothed else "PA_state"
        series_source = series_source[series_source["patient_id"] == patient_id]

        day_arrays = {}
        for d in valid_dates:
            day_rows = series_source[series_source["recording_date"] == d].sort_values("second")
            day_arrays[d] = day_rows[state_col].astype(int).values

        # Colors: sedentary is always the same fixed light gray. Every
        # walking state gets a color from its (duration bin, cadence bin):
        # one distinct HUE per duration bin (so states of a different
        # duration are never confusable, unlike a single smooth colormap
        # where neighboring states can look almost identical), and, within
        # a duration bin, a lighter-to-darker SHADE of that hue as cadence
        # increases. This is more colors to tell apart than a sequential
        # colormap can give while staying perceptually ordered, at the cost
        # of the previous single warm/cool gradient across all states.
        import colorsys

        all_states = sorted(set(np.concatenate(list(day_arrays.values())).tolist()))
        walking_states = [s for s in all_states if s != self.SEDENTARY_STATE]
        # Hues spaced around the wheel, chosen to stay distinguishable from
        # each other and from the gray background: blue, green, red-orange,
        # purple, teal, magenta (enough for any scheme in `_PA_SCHEMES`).
        hues = [0.60, 0.33, 0.04, 0.78, 0.47, 0.92]

        # Lightness AND saturation both vary with cadence (not lightness
        # alone): a pure lightness ramp leaves mid-tones of some hues (green
        # especially, where the eye is least sensitive to this kind of
        # change) looking nearly the same. Pairing a pale/low-saturation
        # slow-cadence end with a vivid/dark-saturation fast-cadence end
        # spreads every hue family further apart.
        colors = {self.SEDENTARY_STATE: "#F0F0F0"}
        for s in walking_states:
            dur_idx, cad_idx, _, n_cad = self._state_duration_cadence_idx(s)
            hue = hues[dur_idx % len(hues)]
            frac = cad_idx / (n_cad - 1) if n_cad > 1 else 0.5
            lightness = 0.80 - 0.42 * frac  # lighter (slow) -> darker (fast)
            saturation = 0.40 + 0.50 * frac  # muted (slow) -> vivid (fast)
            colors[s] = colorsys.hls_to_rgb(hue, lightness, saturation)
        listed_cmap = ListedColormap([colors[s] for s in all_states])
        norm = BoundaryNorm(np.arange(len(all_states) + 1) - 0.5, listed_cmap.N)
        state_to_index = {s: i for i, s in enumerate(all_states)}

        # Every size below is fixed in INCHES, not in figure-fraction, and
        # the fractions matplotlib actually needs (`top`/`bottom`/`hspace`)
        # are derived from them as inches / total_figure_height. This is
        # what keeps a day's row the same physical thickness regardless of
        # how many days are plotted: a figure-fraction constant (the
        # previous approach) instead gives each row a SHRINKING share of
        # the figure as more days are added, because the title/legend/
        # x-label need the same absolute space whether there are 1 or 10
        # rows, which a fixed fraction does not guarantee.
        row_h_in = 0.9  # physical height of one day's bar (the "amplitude" of the barcode)
        row_gap_in = 0.4  # gap between two rows (room for the day label and the gap annotation)
        top_in = 0.55  # suptitle
        xtick_label_in = 0.25  # the last row's own x-axis tick numbers, drawn below its axes
        xlabel_in = 0.3  # one line of x-axis label text
        pad_above_xlabel_in = 0.1
        pad_above_legend_in = 0.05
        pad_below_legend_in = 0.08
        legend_row_in = 0.2  # one row of legend entries

        n_legend_rows = -(-len(all_states) // 4)  # ceil(len / 4)
        bottom_in = (
            xtick_label_in
            + pad_above_xlabel_in
            + xlabel_in
            + pad_above_legend_in
            + n_legend_rows * legend_row_in
            + pad_below_legend_in
        )

        n_days = len(valid_dates)
        total_height_in = top_in + n_days * row_h_in + (n_days - 1) * row_gap_in + bottom_in
        if figsize is None:
            figsize = (12, total_height_in)

        bottom_margin = bottom_in / total_height_in
        top_margin = 1 - top_in / total_height_in
        hspace = row_gap_in / row_h_in
        xlabel_y = bottom_margin - (xtick_label_in + pad_above_xlabel_in) / total_height_in
        legend_y = pad_below_legend_in / total_height_in

        fig, axes = plt.subplots(
            n_days, 1, figsize=figsize, squeeze=False, gridspec_kw={"hspace": hspace}
        )
        axes = axes[:, 0]
        fig.subplots_adjust(bottom=bottom_margin, top=top_margin)

        for i, (d, ax) in enumerate(zip(valid_dates, axes)):
            arr = day_arrays[d]
            img = np.array([state_to_index[s] for s in arr])[None, :]
            ax.imshow(img, aspect="auto", cmap=listed_cmap, norm=norm, extent=[0, len(arr), 0, 1])
            ax.set_yticks([])
            ax.set_xlim(0, len(arr))

            # This day's Hn/LZC/PLZC (if present), next to its row.
            row = days_df[days_df["recording_date"] == d].iloc[0]
            metrics = ", ".join(
                f"{m}={row[m]:.4f}"
                for m in ("Hn", "LZC", "PLZC")
                if m in row.index and pd.notna(row[m])
            )
            ax.set_ylabel(
                f"{d}\n{metrics}", rotation=0, ha="right", va="center", fontsize=8
            )

            # Gap (in calendar days) since the PREVIOUS valid day plotted,
            # so two rows next to each other are never mistaken for a
            # continuous recording when they are actually far apart.
            if i > 0:
                gap_days = (pd.to_datetime(d) - pd.to_datetime(valid_dates[i - 1])).days
                if gap_days > 1:
                    ax.set_title(
                        f"— {gap_days} days with no valid data in between —",
                        fontsize=8,
                        color="gray",
                        loc="left",
                        pad=4,
                    )

        fig.text(
            0.5,
            xlabel_y,
            "Seconds of worn time",
            ha="center",
            va="top",
            fontsize=10,
        )

        # Placed explicitly below the x-axis label text above (both
        # positioned in figure fraction coordinates, independent of the
        # subplots' own layout), so the two never overlap regardless of how
        # many legend rows there are.
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in all_states]
        labels = [self._state_label(s) for s in all_states]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(all_states), 4),
            bbox_to_anchor=(0.5, legend_y),
            fontsize=7,
            frameon=False,
        )

        fig.suptitle(
            f"PA-state barcode — patient {patient_id} "
            f"({'smoothed' if use_smoothed else 'original, before smoothing'})"
        )

        if output_path is not None:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig