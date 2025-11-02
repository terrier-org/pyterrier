Extending PyTerrier
===============================================

.. note::

    This guide is for adding new datasets to PyTerrier, allowing them to be easily used by others.
    If you simply want to run PyTerrier with your own data, you can build Pandas DataFrames compatible
    with the :doc:`PyTerrier Data Model <../datamodel>` - for example, using ``pt.io.read_topics()`` 
    to read from a file.

    If you want to use existing built-in datasets, you can find them on :doc:`this page <../datasets>`.


PyTerrier is designed to be extensible. This section describes how to extend PyTerrier with new features, including:

.. toctree::
   :maxdepth: 1

   packages
   transformers
   artifact_types
   datasets
   indexer_retrieval
   validate
