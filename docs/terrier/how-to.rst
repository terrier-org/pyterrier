Terrier How-To Guides
============================================================

How do I index a standard corpus with Terrier?
------------------------------------------------------------

.. code-block:: python
    :caption: Indexing a standard corpus with Terrier

    import pyterrier as pt
    dataset = pt.datasets.get_dataset("irds:msmarco-passage") # [1]
    my_index = pt.terrier.TerrierIndex('/path/to/index/location.terrier') # [2]
    my_index.index(dataset.get_corpus_iter()) # [3]

.. [1] Select your dataset here. If the corpus is not available in PyTerrier datasets, you can also provide your own iterator over documents.
.. [2] Specify the location where you want to store the Terrier index. The location must not yet exist. We recommend using the ``.terrier`` extension, though this is not required.
.. [3] This performs indexing with default settings. If you need more control over the indexing settings, see :meth:`~pyterrier.terrier.TerrierIndex.indexer` and :class:`~pyterrier.terrier.Indexer` for advanced options.



How do I index a custom collection with Terrier?
------------------------------------------------------------
