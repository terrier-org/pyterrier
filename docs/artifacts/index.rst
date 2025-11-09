PyTerrier Artifacts
-----------------------------------------------

PyTerrier Artifacts provide a powerful way to :ref:`share <artifacts:how-to:services>`
resources, such as indexes, cached results, and more. Re-using one another's artifacts
is a great way to help achieve green (i.e., sustainable) research.

Shared artifacts are also ready-to-use in your experiments through objects that expose various
retrieval functionality through :doc:`Transformers <../transformer>`. For instance, once you load a
:class:`~pyterrier.terrier.TerrierIndex` artifact, you can use its :meth:`~pyterrier.terrier.TerrierIndex.bm25`
method to build a transformer that retrieves from the index using BM25.

.. toctree::
   :maxdepth: 1

   How-To Guides <how-to>
   Listing <listing>
   API Reference <api>

**Acknowledgements:** The design of the Artifact API was described in the following paper:

.. cite.dblp:: conf/sigir/MacAvaney25

