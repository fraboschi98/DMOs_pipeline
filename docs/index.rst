:sd_hide_title:
:html_theme.sidebar_secondary.remove: true

Home
====

.. raw:: html

   <div class="dmo-hero">
     <div class="hero-inner">
       <div class="hero-text">
         <h1>DMOkit</h1>
         <p class="tagline">Standardized Digital Mobility Outcomes from bilateral
         foot-worn IMU recordings in free-living conditions.</p>
         <div class="hero-buttons">
           <a class="hero-btn primary" href="user_guide/index.html">User guide</a>
           <a class="hero-btn secondary" href="https://github.com/fraboschi98/DMOs_pipeline">Get the code</a>
         </div>
       </div>
       <div class="hero-image">
         <img src="_static/hero_logo.png" alt="DMOkit">
       </div>
     </div>
   </div>

.. raw:: html

   <div class="dmo-landing">

What is DMOkit?
---------------

DMOkit is an open Python framework for standardized real-world Digital Mobility
Outcomes (DMOs) from bilateral foot-worn IMUs. Starting from the strides and the
gait parameters estimated on them, it adds walking bout identification, quality
control and DMO aggregation, following the literature. It also produces a
clinical report that gives clinicians an easy overview of the patient's
condition, with Parkinson's disease as the primary use case.

Each processing step is a separate, self-contained class, so you can run one
stage on its own or chain them into a full pipeline. Inputs and outputs are plain
CSV/JSON files, supporting reproducibility and standardization in research.

Key features
------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`location` Walking bout identification

      Identify walking bouts from the output of a validated gait algorithm
      (here we employ gaitmap), using the Mobilise-D walking bout definition.

   .. grid-item-card:: :octicon:`checklist` Quality control

      Event- and stride-level quality checks that flag implausible strides and
      events with labels.

   .. grid-item-card:: :octicon:`graph` DMO aggregation

      Aggregate DMOs at the walking bout, day and week level following the
      literature, including complexity metrics.

   .. grid-item-card:: :octicon:`pulse` Clinical report

      Produce a clinical report that visualizes the patient's mobility, giving
      clinicians an easy overview.

   .. grid-item-card:: :octicon:`mortar-board` Literature-based

      Definitions, thresholds and methods based on reviewing the literature,
      working toward standardization, with Parkinson's disease as the primary
      use case.

   .. grid-item-card:: :octicon:`database` Reproducible data structure

      Modular classes with plain CSV/JSON inputs and outputs, for reproducible
      use in research.

How it works
------------

.. figure:: _static/dmos_ecosystem.png
   :alt: DMOkit ecosystem
   :align: center
   :class: ecosystem-figure

   **DMOkit ecosystem** — From foot-worn IMU recordings to stride, walking-bout,
   daily and complexity DMOs and the clinical report.

Get started
-----------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide/index
      :link-type: doc

      Concepts, installation, definitions and the processing workflow.

   .. grid-item-card:: :octicon:`code-square` API reference
      :link: api
      :link-type: doc

      All classes, grouped by processing stage.

   .. grid-item-card:: :octicon:`beaker` Examples
      :link: examples/index
      :link-type: doc

      Worked examples for each stage of the pipeline, shown as code.

   .. grid-item-card:: :octicon:`mark-github` Source code
      :link: https://github.com/fraboschi98/DMOs_pipeline
      :link-type: url

      Browse the repository, open issues, and contribute.

.. raw:: html

   </div>

.. toctree::
   :hidden:
   :caption: Contents

   Home <self>
   user_guide/index
   api
   examples/index
   citing
   contact
