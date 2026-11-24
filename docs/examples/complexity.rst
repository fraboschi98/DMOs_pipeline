:html_theme.sidebar_secondary.remove: true

Complexity metrics
==================

Compute physical-activity (PA) complexity metrics (``Hn``, ``LZC``, ``PLZC``)
and the weekly ``SampEn`` from walking-bout-level data.

Inputs
------

- a ``wb_parameters_average`` table (one row per walking bout) that already
  contains a ``PA_state`` column;
- the per-session logs collected by
  :class:`~dmos_pipeline.csv_collector.Collector`.

Usage
-----

.. code-block:: python

   import pandas as pd
   from dmos_pipeline import Collector, PAComplexity_analyzer

   patient_directory = "data/PAT01"
   wb = pd.read_csv(
       f"{patient_directory}/outputs_/PAT01_multiple_dates_wb_parameters_average.csv"
   )
   dates = sorted(wb["recording_date"].astype(str).unique())

   collector = Collector("PAT01", dates, patient_directory)
   collector.collect()

   analyzer = PAComplexity_analyzer(
       wb_parameters_average=wb,
       log=collector.log,
       pa_state_choice="modified",
       window_hours=8,
       min_wearing_time_hours=8.0,
   ).run()

Daily metrics are in ``analyzer.complexity_df`` (one row per patient and day,
with ``Hn``, ``LZC`` and ``PLZC``). The weekly summary and the barcode figure:

.. code-block:: python

   weekly = analyzer.weekly_summary(period_col="session_id", aggregation="mean")
   analyzer.plot_barcode("PAT01", use_smoothed=True, output_path="barcode.png")

.. note::

   The PLZC lag is chosen once for the whole analysis — read it from
   ``analyzer.plzc_lag_``.

See also the
API reference for :class:`~dmos_pipeline.complexity_analyzer.PAComplexity_analyzer`
(which includes a minimal, self-contained example).

Output
------

.. note::

   **TODO:** add the output figure(s) after running locally, e.g.::

      .. figure:: ../_static/examples/complexity.png
         :alt: Complexity metrics output
         :align: center
