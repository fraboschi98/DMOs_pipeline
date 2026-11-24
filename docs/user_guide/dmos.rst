:html_theme.sidebar_secondary.remove: true

What are DMOs?
==============

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

See :doc:`definitions` for the operational definitions and thresholds, and the
:doc:`../api` for the classes that compute each outcome.
