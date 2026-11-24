Workflow
========

dmokit is organised as a sequence of processing blocks, each implemented as a
Python class with an explicit configuration dictionary. Blocks can be run on
their own or chained into an end-to-end pipeline, and any block can be replaced
by a custom implementation that respects the same input/output contract.

High-level flow
---------------

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
