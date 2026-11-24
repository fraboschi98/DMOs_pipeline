Installation
============

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
