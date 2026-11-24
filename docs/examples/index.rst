:html_theme.sidebar_secondary.remove: true

Examples
========

Worked examples for the main stages of the pipeline. Each card opens a page with
the full script you can run and the resulting figures. These pages do not run
during the build and need no dataset.

.. note::

   **TODO:** a **report** example is still missing — add it once the report
   module is finalized.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Gait processing & quality control
      :img-top: /_static/examples/gaitmap_quality_check.png
      :link: gaitmap_quality_check
      :link-type: doc
      :class-card: ex-card

      Stride detection with gaitmap and event/parameter quality control.

   .. grid-item-card:: Walking-bout & DMO extraction
      :img-top: /_static/examples/dmos_extraction.png
      :link: dmos_extraction
      :link-type: doc
      :class-card: ex-card

      Detect walking bouts and extract micro/macro Digital Mobility Outcomes.

   .. grid-item-card:: Day-level DMO analysis
      :img-top: /_static/examples/day_dmos.png
      :link: day_dmos
      :link-type: doc
      :class-card: ex-card

      Aggregate to day- and week-level DMOs with the reporting guidance.

   .. grid-item-card:: Distribution plots (KDE)
      :img-top: /_static/examples/kde.png
      :link: kde
      :link-type: doc
      :class-card: ex-card

      Kernel-density plots of gait parameters across walking bouts.

   .. grid-item-card:: Complexity metrics
      :img-top: /_static/examples/complexity.png
      :link: complexity
      :link-type: doc
      :class-card: ex-card

      PA-state complexity (Hn, LZC, PLZC) and weekly SampEn.

.. toctree::
   :hidden:

   gaitmap_quality_check
   dmos_extraction
   day_dmos
   kde
   complexity
