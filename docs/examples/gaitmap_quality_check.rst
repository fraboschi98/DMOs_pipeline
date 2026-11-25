:html_theme.sidebar_secondary.remove: true

Gait processing & quality control
=================================

Run stride-level processing with gaitmap and apply event- and parameter-level
quality control, producing the quality-checked gait events used downstream.

Script
------

.. literalinclude:: ../../examples/main_gaitmap_quality_check.py
   :language: python
   :linenos:

Output
------

.. note::

   **TODO:** add the output figure(s) produced by this script after running it
   locally. Put the image in ``docs/_static/examples/`` and reference it, e.g.::

      .. figure:: ../_static/examples/gaitmap_quality_check.png
         :alt: Gait processing & quality control output
         :align: center
