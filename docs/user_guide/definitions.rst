:html_theme.sidebar_secondary.remove: true

Definitions and thresholds
==========================

dmokit represents walking as a sequence of **walking bouts (WBs)** — periods
containing at least two consecutive strides of both feet, delimited by
non-walking periods, following the Mobilise-D terminology.

.. important::

   The thresholds below are the **default** configuration values; each is set
   through the ``config`` of the corresponding class and can be overridden per
   run.

Core terms
----------

Stride
   One gait cycle of one foot, anchored at its initial contact (IC).
Pause
   An interval in which no gait events are detected on either foot for at least
   the pause threshold. Pauses delimit segments.
Break
   A shorter event-free interval, below the pause threshold, kept inside a
   segment so that brief detection dropouts do not split a bout.
Segment
   The portion of recording between two pauses.
Walking bout (WB)
   A segment that meets the validity rule.

Walking-bout thresholds
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Rule
     - Default
     - Meaning
   * - Pause threshold
     - ``3.0`` s
     - A pause of at least 3.0 s ends a bout (``wb_pause_s_threshold``); shorter
       event-free intervals are kept as breaks.
   * - Validity (two feet)
     - ``>= 5``
     - A segment is a WB if both feet contribute and the combined number of
       detected events is at least 5 (``threshold_two_sides``).
   * - Validity (one foot)
     - ``> 3``
     - ...or if only one foot contributes and its count is above 3
       (``threshold_one_side``).

Duration labels (``add_wb_label``): **very_short** (< 10 s), **short**
(10–30 s), **medium** (30–60 s), **long** (> 60 s).

Cadence and physical activity
-----------------------------

Cadence
   Taken from the foot contributing more strides: ``strides × 2 ÷ duration × 60``
   = steps/min (``add_cadence_to_wb_dataframe``).
Physical-activity type
   **MVPA** if cadence > 90 steps/min, **LPA** otherwise
   (``add_pa_type_to_wb_dataframe``).
Physical-activity state
   Each WB is assigned a state from a cadence × duration matrix
   (``add_pa_state_to_wb_dataframe``): ``traditional`` (12 walking states) or
   ``modified`` (20 locomotion states). dmokit outputs the per-WB state
   sequence; complexity metrics are computed downstream.

Quality control
---------------

Quality control adds labels **without removing rows**.

Parameter level — a stride fails when at least ``n_parameter_violations = 2`` of
the plausibility rules are violated:

.. list-table::
   :header-rows: 1
   :widths: 40 30

   * - Parameter
     - Plausible range
   * - stride time
     - 0.2 – 3.0 s
   * - gait velocity
     - 0.2 – 2.0 m/s
   * - stride length
     - 0.10 – 1.5 m

Strides with an absolute turning angle between 25° and 90° are additionally
flagged.

Event level — the low-pass-filtered medio-lateral angular velocity at each
initial contact is inspected on the corresponding foot (Butterworth, 5 Hz, 4th
order; fails if > 0).

By default, event-level QC runs **before** WB detection
(``use_only_quality_checked_events = True``), while parameter-level QC is applied
at **aggregation** (``use_quality_check = True``); both are configurable.
