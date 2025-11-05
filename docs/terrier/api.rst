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
   
   .. automethod:: index
   .. automethod:: indexer

   Miscellaneous
   ------------------------------------------
   
   .. automethod:: built
   .. automethod:: index_ref
   .. automethod:: index_obj
   .. automethod:: coerce

   Sharing
   ------------------------------------------

   .. seealso::
       You can share Terrier indices using the Artifacts API:
         - HuggingFace: :meth:`~pyterrier.Artifact.from_hf` and :meth:`~pyterrier.Artifact.to_hf`
         - Zenodo: :meth:`~pyterrier.Artifact.from_zenodo` and :meth:`~pyterrier.Artifact.to_zenodo`
         - Peer-to-peer: :meth:`~pyterrier.Artifact.from_p2p` and :meth:`~pyterrier.Artifact.to_p2p`
         - URLs: :meth:`~pyterrier.Artifact.from_url`


Mid-Level API
----------------------------------------

The Mid-Level API provides more control over Terrier functionality.

Low-Level (Java) API
-----------------------------------------

Some functions return Java object wrappers (e.g., :meth:`TerrierIndex.index_obj() <pyterrier.terrier.TerrierIndex.index_obj>`)
that provide direct low-level API access to Terrier classes. You can find documentation for it in the
`Terrier Documentation <http://terrier.org/docs/current/javadoc/>`__.

.. tip::

   Pyjnius Java object wrappers show which class they wrap in their string representation. For instance,
   ``str(index.index_obj()) = "<org.terrier.structures.Index at 0x10cd8ba60 ...>"``, showing that it
   wraps an instance of ``org.terrier.structures.Index``.
