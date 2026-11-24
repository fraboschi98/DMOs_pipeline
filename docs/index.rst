:sd_hide_title:
:html_theme.sidebar_secondary.remove: true

DMOs Pipeline
=============

.. raw:: html

   <div class="dmo-hero">
     <div class="hero-inner">
       <div class="hero-text">
         <h1>DMOkit Pipeline</h1>
         <p class="tagline">Standardized Digital Mobility Outcomes from bilateral
         foot-worn IMU recordings in free-living conditions.</p>
         <div class="hero-buttons">
           <a class="hero-btn primary" href="user_guide/index.html">User guide</a>
           <a class="hero-btn secondary" href="https://github.com/fraboschi98/DMOs_pipeline">GitHub</a>
         </div>
       </div>
       <div class="hero-image">
         <img src="_static/icon_white.png" alt="DMOkit">
       </div>
     </div>
   </div>

.. raw:: html

   <div class="dmo-landing">

What is DMOkit pipeline?
------------------------

DMOkit is an open Python framework that turns raw bilateral foot-worn IMU
recordings into standardized Digital Mobility Outcomes (DMOs) for real-world
physical-activity monitoring. It builds on already-validated gait algorithms and
adds standardized walking-bout identification, quality control and aggregation,
following the Mobilise-D framework. It is built around object-oriented classes
and a non-relational CSV/JSON data organization, so each step can run on its own
or as an end-to-end pipeline, and is designed for reproducible use in clinical
studies.

Which are the functionalities?
------------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`location` Walking-bout identification

      Identify walking bouts from the output of an already-validated gait
      algorithm (e.g. gaitmap), using the Mobilise-D walking-bout definition.

   .. grid-item-card:: :octicon:`checklist` Quality control & aggregation

      Event- and stride-level quality checks, and aggregation of gait
      parameters following available literature.

   .. grid-item-card:: :octicon:`graph` Multi-level DMOs

      Spatiotemporal micro-parameters, macro-parameters and complexity metrics,
      and day- and week-level summaries for physical-activity monitoring.

   .. grid-item-card:: :octicon:`database` Reproducible data structure

      Object-oriented classes and a standardized CSV/JSON organization for
      real-world clinical studies.

How it works
------------

.. raw:: html

   <div class="schema-placeholder">
     Pipeline schema — add the flowchart image at
     <code>docs/_static/pipeline_schema.png</code> and replace this placeholder.
   </div>

Documentation
-------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide/index
      :link-type: doc

      Concepts, installation, definitions and the processing workflow.

   .. grid-item-card:: :octicon:`beaker` Examples
      :link: examples/index
      :link-type: doc

      Worked examples for each stage of the pipeline, shown as code.

   .. grid-item-card:: :octicon:`code-square` API reference
      :link: api
      :link-type: doc

      All classes, grouped by processing stage.

   .. grid-item-card:: :octicon:`mark-github` Source code
      :link: https://github.com/fraboschi98/DMOs_pipeline
      :link-type: url

      Browse the repository, open issues, and contribute.

.. raw:: html

   </div>

.. toctree::
   :hidden:
   :caption: Contents

   user_guide/index
   examples/index
   api
   citing
   contact
