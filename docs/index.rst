PyTerrier Documentation
=====================================

PyTerrier is a Python framework for Information Retrieval (IR) research and experimentation.

.. grid:: 2

   .. grid-item-card:: 🚀 State-of-the-Art IR

      Including TODO

   .. grid-item-card:: 🧩 Extendable and Composable

      A common :doc:`data model <datamodel>` lets you easily :doc:`construct pipelines <operators>` that combine various methods.

   .. grid-item-card:: 🧪 Experimentation

      Tools for :doc:`conducting IR experiments <experiments>`, with built-in support for :doc:`hundreds of benchmarks <datasets>`, and :doc:`dozens of evaluation measures <experiments>`.

   .. grid-item-card:: ⚙️ Retrieval Engines

      It's not just :doc:`Terrier <terrier/index>` --- PyTerrier supports :doc:`PISA <ext/pyterrier_pisa/index>`, :doc:`Anserini <ext/pyterrier-anserini/index>`, :doc:`FAISS <ext/pyterrier-dr/index>`, and others.


.. toctree::
   :maxdepth: 1
   :caption: Guides

   installation
   datasets
   Terrier <terrier/index>
   experiments
   ltr
   Artifacts <artifacts/index>

.. toctree::
   :maxdepth: 1
   :caption: Explanations & Illustrations

   datamodel.md
   transformer
   operators
   pipeline_examples.md
   text
   neural
   tuning
   experiments/Robust04
   extending/index
   troubleshooting/index

.. toctree::
   :maxdepth: 1
   :caption: Other Modules

   io
   apply
   new
   debug
   inspect
   schematic

.. include:: ./_includes/ext_toc.rst

.. toctree::
   :maxdepth: 1
   :caption: Indices and tables

   genindex
   bibliography
