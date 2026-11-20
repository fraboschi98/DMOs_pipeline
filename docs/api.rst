API reference
=============

The pipeline is organized as object-oriented classes, one per processing stage.

Gait processing
---------------

.. autoclass:: dmos_pipeline.gaitmap_pipeline.GaitMapPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Quality control
---------------

.. autoclass:: dmos_pipeline.quality_check.QualityCheck
   :members:
   :undoc-members:
   :show-inheritance:

Walking-bout extraction
-----------------------

.. autoclass:: dmos_pipeline.wb_pipeline.WBpipeline
   :members:
   :undoc-members:
   :show-inheritance:

Data collection
---------------

.. autoclass:: dmos_pipeline.csv_collector.Collector
   :members:
   :undoc-members:
   :show-inheritance:

Walking-bout-level analysis
---------------------------

.. autoclass:: dmos_pipeline.wb_analyzer.WalkingBouts_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Day-level analysis
------------------

.. autoclass:: dmos_pipeline.day_analyzer.DayLevel_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Complexity analysis
-------------------

.. autoclass:: dmos_pipeline.complexity_analyzer.PAComplexity_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. autoclass:: dmos_pipeline.kde_plotter.KDEPlotter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmos_pipeline.pause_plotter.PausePlotter
   :members:
   :undoc-members:
   :show-inheritance:
