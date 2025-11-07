Terrier API Reference
======================================

This page provides API documentation for the Terrier integration in PyTerrier.

High-Level API
--------------------------------------

:class:`~pyterrier.terrier.TerrierIndex` provides a high-level API. We recommended it for
most use cases.

.. autoclass:: pyterrier.terrier.TerrierIndex

   Retrieval
   ------------------------------------------

   .. automethod:: retriever
   .. automethod:: bm25
   .. automethod:: dph
   .. automethod:: pl2
   .. automethod:: dirichlet_lm
   .. automethod:: hiemstra_lm
   .. automethod:: tf
   .. automethod:: tf_idf

   Query Expansion
   ------------------------------------------

   .. automethod:: rm3
   .. automethod:: bo1
   .. automethod:: kl

   Loading
   ------------------------------------------
   .. automethod:: text_loader

   Indexing
   ------------------------------------------
   
   .. automethod:: indexer
   .. automethod:: index

   Index Data
   ------------------------------------------

   .. automethod:: collection_statistics
   .. automethod:: lexicon
   .. automethod:: inverted_index
   .. automethod:: document_index
   .. automethod:: meta_index
   .. automethod:: direct_index
   .. automethod:: index_ref
   .. automethod:: index_obj

   Miscellaneous
   ------------------------------------------
   
   .. automethod:: built
   .. automethod:: get_corpus_iter
   .. automethod:: coerce

   Sharing
   ------------------------------------------

   .. seealso::
       You can share Terrier indices using the Artifacts API:

       - HuggingFace: :meth:`~pyterrier.Artifact.from_hf` and :meth:`~pyterrier.Artifact.to_hf`
       - Zenodo: :meth:`~pyterrier.Artifact.from_zenodo` and :meth:`~pyterrier.Artifact.to_zenodo`
       - Peer-to-peer: :meth:`~pyterrier.Artifact.from_p2p` and :meth:`~pyterrier.Artifact.to_p2p`
       - URLs: :meth:`~pyterrier.Artifact.from_url`

.. autoenum:: pyterrier.terrier.TerrierModel
.. autoenum:: pyterrier.terrier.TerrierTokeniser
.. autoenum:: pyterrier.terrier.TerrierStemmer
   :members:
.. autoenum:: pyterrier.terrier.TerrierStopwords
   :members:


Mid-Level API
----------------------------------------

The Mid-Level API provides more control over Terrier functionality.

Indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyterrier.terrier.IterDictIndexer
   :members: index

.. autoclass:: pyterrier.terrier.TRECCollectionIndexer
   :members: index

.. autoclass:: pyterrier.terrier.FilesIndexer
   :members: index

.. autoclass:: pyterrier.terrier.IndexingType

.. autofunction:: pyterrier.terrier.treccollection2textgen

Retrieval & Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyterrier.terrier.Retriever
    :members: transform

.. autoclass:: pyterrier.terrier.FeaturesRetriever
    :members: transform

.. autoclass:: pyterrier.terrier.TextScorer
    :members: transform

Query Expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyterrier.terrier.rewrite.SDM
   :members: transform

.. autoclass:: pyterrier.terrier.rewrite.RM3
   :members: transform

.. autoclass:: pyterrier.terrier.rewrite.Bo1QueryExpansion
   :members: transform

.. autoclass:: pyterrier.terrier.rewrite.KLQueryExpansion
   :members: transform

.. autofunction:: pyterrier.terrier.rewrite.reset
.. autofunction:: pyterrier.terrier.rewrite.tokenise
.. autofunction:: pyterrier.terrier.rewrite.stash_results
.. autofunction:: pyterrier.terrier.rewrite.reset_results
.. autofunction:: pyterrier.terrier.rewrite.linear

Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyterrier.terrier.TerrierTextLoader
    :members: transform

Low-Level (Java) API
-----------------------------------------

Some functions return Java object wrappers (e.g., :meth:`TerrierIndex.index_obj() <pyterrier.terrier.TerrierIndex.index_obj>`)
that provide direct low-level API access to Terrier classes. You can find documentation for it in the
`Terrier Documentation <http://terrier.org/docs/current/javadoc/>`__.

.. autoclass:: pyterrier.terrier.IndexFactory
    :members: of

.. tip::

   Pyjnius Java object wrappers show which class they wrap in their string representation. For instance,
   ``str(index.index_obj()) = "<org.terrier.structures.Index at 0x10cd8ba60 ...>"``, showing that it
   wraps an instance of ``org.terrier.structures.Index``.
