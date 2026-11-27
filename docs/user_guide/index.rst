:html_theme.sidebar_secondary.remove: true

User guide
==========

This guide introduces **dmokit** and the ideas behind it. Following the same
logic as the paper, it builds up from a standardized gait data structure to a set
of interchangeable processing blocks: how to install the framework, what Digital
Mobility Outcomes are and why they matter, the definitions and thresholds used to
turn gait data into walking bouts, and the overall workflow that chains the
individual classes.

.. note::

   **TODO:** integrate the *report* module into this guide once finalized.

Installation
------------

dmokit is open source and available on `GitHub
<https://github.com/fraboschi98/DMOs_pipeline>`_. The recommended way to use it
is to clone the repository and install its dependencies into a dedicated
environment:

.. code-block:: bash

   git clone https://github.com/fraboschi98/DMOs_pipeline.git
   cd dmo-kit

   conda create -n dmos python=3.11
   conda activate dmos
   pip install -r requirements.txt

A short note on where dmokit starts from. As described on the home page, dmokit
does **not** process the raw IMU signal from scratch: it starts from the strides
and the stride-level gait parameters already estimated from the recordings. That
stride-level step is carried out by `gaitmap
<https://github.com/mad-lab-fau/gaitmap>`_, a validated gait-analysis library,
which dmokit's first class (:class:`~dmos_pipeline.gaitmap_pipeline.GaitMapPipeline`)
wraps and integrates. This is why gaitmap is part of the dependencies listed in
``requirements.txt``.

.. note::

   **TODO:** state the required Python and gaitmap versions, the licence, and a
   citable versioned release (DOI via Zenodo; see :doc:`../citing`). The
   repository may also be renamed to **dmo-kit** — update the clone URL and the
   ``cd`` accordingly once that is done.

What are DMOs?
--------------

**Digital Mobility Outcomes (DMOs)** are measures of a person's mobility in daily
life, computed from data recorded by digital health technologies, most commonly
body-worn inertial sensors. Because they are collected continuously and
unobtrusively during normal activity, they capture real-world mobility rather
than a single supervised test in the clinic. In dmokit, DMOs are derived from
bilateral foot-worn IMUs and focus on walking: walking speed, cadence, stride
length, daily walking volume, and so on.

DMOs span different levels of observation. The spatiotemporal **micro** measures
— stride length, gait velocity, cadence and the like — characterise a single
walking bout and can be obtained both from supervised clinical assessments and
from free-living home monitoring, which makes them a natural bridge between the
two settings. The **macro** measures instead describe the *amount* and *pattern*
of walking across the day and only emerge from prolonged, unsupervised recording
at home, complementing what the clinic can capture.

dmokit organises these outputs along the **Mobilise-D DMO domains** (amount,
pattern, pace, rhythm and bout-to-bout variability) and summarises them at the
**day** and **week** level. These summaries in turn feed a clinical **report**
that gives clinicians an easy overview of a patient's real-world mobility.

.. note::

   **TODO:** add the DMO **sunburst chart** that Fra B will upload (micro vs macro
   parameters); place it in ``docs/_static/`` and adjust the text above to match
   the families it shows. Reference it here, e.g.::

      .. image:: ../_static/dmos_sunburst.png
         :alt: DMO parameter families
         :align: center

Standardization and Mobilise-D
------------------------------

Real-world DMOs are only comparable across studies, devices and cohorts if they
are computed in the same way — from how a walking bout is defined to which days
count as valid. Without shared definitions, two pipelines can report different
values of *walking speed* or *daily walking volume* from the very same recording,
which undermines their use as clinical endpoints (Micó-Amigo et al. 2023; Kluge
et al. 2021).

The **Mobilise-D** consortium addressed this by establishing consensus
definitions and a validated framework for extracting real-world DMOs, together
with its open reference implementation, **mobgap**, built around a single
lower-back inertial sensor. These definitions — the walking-bout criteria, the
DMO domains and the valid-day and wear-time rules — are what make results
portable between cohorts and comparable with the wider literature on free-living
mobility in Parkinson's disease (Del Din et al. 2016).

**dmokit** adopts the same definitions and reporting conventions, but for a
**bilateral foot-worn IMU** setup, acting as a foot-mounted companion to the
lower-back mobgap reference. Its downstream analysis layer is deliberately
independent of how strides are obtained: the reference upstream implementation
relies on gaitmap (Küderle et al. 2024), but any stride source that respects the
same data contract can be used in its place.

The sections below give the operational definitions behind these outcomes, and
the :doc:`../api` documents the classes that compute each of them.

Definitions and thresholds
--------------------------

At the core of dmokit is the notion of a **walking bout (WB)**: a period
containing at least two consecutive strides of both feet, delimited by
non-walking periods, following the Mobilise-D terminology. The rest of this
section defines the terms and the default thresholds used to build and
characterise walking bouts.

.. important::

   The thresholds below are the **default** configuration values. Each one is set
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

Walking bouts are also labelled by duration (``add_wb_label``): **very_short**
(< 10 s), **short** (10–30 s), **medium** (30–60 s) and **long** (> 60 s).

Cadence and physical activity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From each walking bout dmokit derives a few activity descriptors:

Cadence
   Taken from the foot contributing more strides: ``strides × 2 ÷ duration × 60``
   = steps/min (``add_cadence_to_wb_dataframe``).
Physical-activity type
   **MVPA** if cadence > 90 steps/min, **LPA** otherwise
   (``add_pa_type_to_wb_dataframe``).
Physical-activity state
   Each WB is assigned a state from a cadence × duration matrix
   (``add_pa_state_to_wb_dataframe``): ``traditional`` (12 walking states) or
   ``modified`` (20 locomotion states). This per-WB state sequence is what the
   complexity metrics are computed on.

Quality control
~~~~~~~~~~~~~~~~

Quality control in dmokit **labels** data rather than removing it, so that
nothing is silently discarded. A stride fails the parameter-level check when at
least ``n_parameter_violations = 2`` of the plausibility rules below are
violated:

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
flagged. At the event level, the low-pass-filtered medio-lateral angular velocity
at each initial contact is inspected on the corresponding foot (Butterworth,
5 Hz, 4th order; the event fails if it is > 0).

By default, event-level QC runs **before** WB detection
(``use_only_quality_checked_events = True``), while parameter-level QC is applied
at **aggregation** (``use_quality_check = True``); both are configurable.

Workflow
--------

dmokit is organised as a sequence of processing blocks, each implemented as a
Python class with an explicit configuration dictionary. The blocks can be run on
their own or chained into an end-to-end pipeline, and any block can be replaced
by a custom implementation that respects the same input/output contract.

At a glance, the data flows through five stages:

.. code-block:: text

   bilateral foot-worn IMU
        -> GaitMapPipeline    (stride-level processing, wraps gaitmap)
        -> QualityCheck       (event- & parameter-level labels)
        -> WBpipeline         (walking bouts, pauses, breaks)
        -> Collector          (collect per-session outputs)
        -> WalkingBouts_analyzer / DayLevel_analyzer   (WB-, day-, week-level DMOs)
        -> PAComplexity_analyzer  (complexity of the daily PA pattern)
        -> KDEPlotter / PausePlotter   (distribution & pause visualisation)

Each block is documented below, with a link to its full API reference and to a
worked example.

:class:`~dmos_pipeline.gaitmap_pipeline.GaitMapPipeline` — stride-level processing
   Filters the signals, detects gait sequences, segments strides, reconstructs
   the foot trajectory, computes stride parameters and estimates wearing time.
   It wraps and integrates gaitmap. See :doc:`../examples/gaitmap_quality_check`.

:class:`~dmos_pipeline.quality_check.QualityCheck` — quality control
   Adds event- and parameter-level labels without removing rows, so nothing is
   silently discarded. See :doc:`../examples/gaitmap_quality_check`.

:class:`~dmos_pipeline.wb_pipeline.WBpipeline` — walking-bout extraction
   Detects pauses and breaks, splits the recording into segments and applies the
   validity rule to identify walking bouts. See :doc:`../examples/dmos_extraction`.

:class:`~dmos_pipeline.csv_collector.Collector` — collection
   Traverses a participant directory and concatenates the per-session outputs
   into a single table. See :doc:`../examples/dmos_extraction`.

:class:`~dmos_pipeline.wb_analyzer.WalkingBouts_analyzer` — walking-bout-level DMOs
   Computes the micro and macro Digital Mobility Outcomes at the walking-bout
   level. See :doc:`../examples/dmos_extraction`.

:class:`~dmos_pipeline.day_analyzer.DayLevel_analyzer` — day- and week-level DMOs
   Aggregates walking-bout outputs to day- and week-level summaries, applying the
   valid-day and wear-time criteria. See :doc:`../examples/day_dmos`.

:class:`~dmos_pipeline.complexity_analyzer.PAComplexity_analyzer` — complexity metrics
   Encodes the day as a sequence of physical-activity states and measures the
   temporal complexity of the daily walking pattern. See
   :doc:`../examples/complexity`.

:class:`~dmos_pipeline.kde_plotter.KDEPlotter` — distribution plots
   Visualises the distribution of gait parameters across walking bouts with
   kernel-density-estimate plots. See :doc:`../examples/kde`.

:class:`~dmos_pipeline.pause_plotter.PausePlotter` — pause visualisation
   Plots pauses and breaks within a recording to inspect how segments are
   delimited.

.. note::

   **TODO:** replace the text flow above with the pipeline **flowchart** (class
   blocks, from the slides / paper Figure 1). Add it to ``docs/_static/`` and
   reference it, e.g.::

      .. image:: ../_static/pipeline_flowchart.png
         :alt: dmokit workflow
         :align: center

References
----------

- Küderle et al. (2024) — *gaitmap*, Gait and Movement Analysis Package, MaD Lab, FAU Erlangen-Nürnberg. https://github.com/mad-lab-fau/gaitmap
- Micó-Amigo et al. (2023) — technical validation of real-world gait within the Mobilise-D framework.
- Kluge et al. (2021) — Mobilise-D consensus and validation of real-world DMOs.
- Del Din et al. (2016) — free-living gait monitoring in Parkinson's disease.
- Paraschiv-Ionescu et al. (2012) — locomotion-complexity (barcode) characterisation of daily activity.
- *mobgap* — Mobilise-D reference pipeline for a single lower-back inertial sensor.

.. note::

   **TODO:** complete the full bibliographic details (authors, title, venue, year,
   DOI) for each reference above from the paper's reference list, and add any
   further Mobilise-D sources behind the walking-bout definition and the DMO
   domains.
