:html_theme.sidebar_secondary.remove: true

User guide
==========

This guide explains what **dmokit** does and the concepts behind it: how to
install it, what Digital Mobility Outcomes are, the definitions and thresholds
used to build walking bouts, and the processing workflow.

.. note::

   **TODO:** integrate the *report* module and the *complexity metrics*
   (per-WB PA-state sequence → Hn / LZC / PLZC) into this guide once finalized.

Installation
------------

dmokit is currently distributed as a **repository** (not yet a published
package). Clone it and install its dependencies into a dedicated environment.

.. code-block:: bash

   git clone https://github.com/fraboschi98/DMOs_pipeline.git
   cd DMOs_pipeline

   conda create -n dmos python=3.11
   conda activate dmos
   pip install -r requirements.txt

dmokit builds on the **gaitmap** algorithms for the stride-level processing
stage; see ``requirements.txt`` for the full dependency list. An end-to-end
example script is provided in ``examples/main_dmos_extraction.py``.

.. note::

   **TODO:** state the required Python and gaitmap versions, the licence, and a
   citable versioned release (DOI via Zenodo). See :doc:`../citing`.

What are DMOs?
--------------

Digital Mobility Outcomes (DMOs) are quantitative measures of real-world walking
derived from wearable IMU signals — walking speed, cadence, stride length, daily
walking volume, and so on — obtained objectively and over long periods of
everyday life.

Outcomes are grouped into **micro** characteristics — spatiotemporal measures
computed within a walking bout, available both in the clinic and at home — and
**macro** characteristics, which describe the amount and pattern of walking and
require prolonged free-living recording.

dmokit organises its outputs by the **Mobilise-D DMO domains**: amount, pattern,
pace, rhythm and bout-to-bout variability, extended to day- and week-level
summaries.

.. note::

   **TODO:** add the DMO **sunburst chart** (micro vs macro parameters). Place it
   in ``docs/_static/`` and reference it here, e.g.::

      .. image:: ../_static/dmos_sunburst.png
         :alt: DMO parameter families
         :align: center

See `Definitions and thresholds`_ below for the operational definitions, and the
:doc:`../api` for the classes that compute each outcome.

Definitions and thresholds
--------------------------

dmokit represents walking as a sequence of **walking bouts (WBs)** — periods
containing at least two consecutive strides of both feet, delimited by
non-walking periods, following the Mobilise-D terminology.

.. important::

   The thresholds below are the **default** configuration values; each is set
   through the ``config`` of the corresponding class and can be overridden per
   run.

Core terms
~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

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

Workflow
--------

dmokit is organised as a sequence of processing blocks, each implemented as a
Python class with an explicit configuration dictionary. Blocks can be run on
their own or chained into an end-to-end pipeline, and any block can be replaced
by a custom implementation that respects the same input/output contract.

High-level flow
~~~~~~~~~~~~~~~~

.. code-block:: text

   bilateral foot-worn IMU
        -> GaitMapPipeline    (stride-level processing, wraps gaitmap)
        -> QualityCheck       (event- & parameter-level labels)
        -> WBpipeline         (walking bouts, pauses, breaks)
        -> Collector          (collect per-session outputs)
        -> WalkingBouts_analyzer / DayLevel_analyzer   (WB-, day-, week-level DMOs)
        -> KDEPlotter         (distribution-based visualisation)

The five stages:

#. **Stride-level processing** (:class:`~dmos_pipeline.gaitmap_pipeline.GaitMapPipeline`,
   wrapping gaitmap) — filter signals, detect gait sequences, segment strides,
   reconstruct the foot trajectory, compute stride parameters, estimate wearing
   time.
#. **Quality control** (:class:`~dmos_pipeline.quality_check.QualityCheck`) —
   event- and parameter-level labels, without removing rows.
#. **Walking-bout extraction** (:class:`~dmos_pipeline.wb_pipeline.WBpipeline`) —
   detect pauses and breaks, split into segments, apply the validity rule.
#. **Collection** (:class:`~dmos_pipeline.csv_collector.Collector`) — traverse a
   participant directory and concatenate per-session outputs.
#. **Aggregation** (:class:`~dmos_pipeline.wb_analyzer.WalkingBouts_analyzer`,
   :class:`~dmos_pipeline.day_analyzer.DayLevel_analyzer`) into WB-, day- and
   week-level DMOs, with distribution plots
   (:class:`~dmos_pipeline.kde_plotter.KDEPlotter`).

.. note::

   **TODO:** replace the text flow above with the pipeline **flowchart** (class
   blocks, from the slides / paper Figure 1). Add it to ``docs/_static/`` and
   reference it, e.g.::

      .. image:: ../_static/pipeline_flowchart.png
         :alt: dmokit workflow
         :align: center
