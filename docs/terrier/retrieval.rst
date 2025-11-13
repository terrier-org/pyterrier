Terrier Retrieval and Re-Ranking
--------------------------------

This section describes how to perform retrieval using Terrier.

Retrieval Basics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.Retriever
.. related:: pyterrier.terrier.TerrierIndex.bm25
.. related:: pyterrier.terrier.TerrierIndex.pl2

:class:`pt.terrier.Retriever <pyterrier.terrier.Retriever>` is one of the most commonly used PyTerrier transformers.
It represents a retrieval transformation, in which queries are executed over a Terrier index, returning their
retrieved documents. Retriever uses a pre-existing Terrier index data structure, typically saved on disk.

You can construct a :class:`~pyterrier.terrier.Retriever` directly. However, :class:`~pyterrier.terrier.TerrierIndex` 
provides convenience methods to create ``Retriever`` instnances, such as :meth:`~pyterrier.terrier.TerrierIndex.bm25`,
:meth:`~pyterrier.terrier.TerrierIndex.pl2`, and :meth:`~pyterrier.terrier.TerrierIndex.tf_idf`.

.. tabs::
    .. tab:: Using ``TerrierIndex``
        .. code-block:: python

            index = pt.terrier.TerrierIndex("/path/to/index")
            retriever = index.bm25()
    .. tab:: Constructing Directly
        .. code-block:: python

            index = pt.IndexFactory.of("/path/to/index")
            retriever = pt.terrier.Retriever(index, wmodel="BM25")

Retriever is a retrieval transformation, meaning that it takes as input dataframes with columns ``["qid", "query"]``,
and returns dataframes with columns ``["qid", "query", "docno", "score", "rank"]``:

.. schematic::
    :input_columns: qid,query

    pt.terrier.TerrierIndex.example().bm25()

Retriever can also act as a re-ranker. In this scenario, it takes as input dataframes with columns ``["qid", "query", "docno"]``,
and returns dataframes with columns ``["qid", "query", "docno", "score", "rank"]``:

.. schematic::
    :input_columns: qid,query,docno

    pt.terrier.TerrierIndex.example().bm25()

For instance, if you first want to retrieve the top 100 results with BM25, then re-rank those results using PL2, you can 
construct the following pipeline:

.. schematic::
    :input_columns: qid,query
    :show_code:

    index = pt.terrier.TerrierIndex.example()
    # FOLD
    index.bm25() % 100 >> index.pl2()


Query Formats for Terrier retrievers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default Terrier assumes that queries can be parsed by its `standard query parser <https://github.com/terrier-org/terrier-core/blob/5.x/doc/querylanguage.md#user-query-language>`_,
which is standard search-engine like query language. Queries provided by Dataset objects are assumed to be in this format, using the 
standard ``["qid", "query"]`` dataframe columns. 

Two alternative query formats are also supported:

- MatchOp - this is a `lower-level query language <https://github.com/terrier-org/terrier-core/blob/5.x/doc/querylanguage.md#matching-op-query-language>`_ supported by Terrier, which is Indri-like in nature, and supports operators like ``#1()``. (exact phrase and ``#combine()`` (weighting). MatchOp queries stored in the `"query"` column. 

- pre-tokenised queries - in this format, query terms are provided, with weights, in a dictionary. Query terms are assumed to be already stemmed. This
  format is useful for techniques that weight query terms, such as for Learned Sparse Retrieval (e.g. see `pyterrier_splade <https://github.com/cmacdonald/pyt_splade>`_).

The following query dataframes are therefore equivalent:

- Raw query:

=====  =============================
qid    query         
=====  =============================
    1  chemical chemical reactions
=====  =============================

- Using Terrier's QL to express weights on query terms:

=====  =============================
qid    query         
=====  =============================
    1  chemical^2 reactions
=====  =============================

- Using Terrier's MatchOpQL to express weights on stemmed and tokenised query terms:

=====  ======================================
qid    query         
=====  ======================================
    1  #combine:0=2:1=1(chemic reaction)
=====  ======================================

- Use the query_toks column (the query column is ignored):

=====  ====================================== =============================
qid    query_toks                             query         
=====  ====================================== =============================
    1  {'chemic' : 2.0, 'reaction' : 1}       chemical chemical reactions
=====  ====================================== =============================


Scoring documents without an index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.TextScorer

Sometimes we want to apply Terrier to compute the score of document for a given query when we do not yet 
have the documents indexed. :class:`~pyterrier.terrier.TextScorer` allows you do do just this. It creates a temporary
index on-the-fly  for text of the documents, and scores the provided documents.

Optionally, an index-like object can be specified as the ``background_index`` argument, which will be used for
the collection statistics (e.g. term frequencies, document lengths etc.)

Index-Like Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with Terrier indices, Retriever allows can make use of:

- a string representing an index, such as "/path/to/data.properties"
- a Terrier `IndexRef <http://terrier.org/docs/current/javadoc/org/terrier/querying/IndexRef.html>`_ object, constructed from a string, but which may also hold a reference to the existing index.
- a Terrier `Index <http://terrier.org/docs/current/javadoc/org/terrier/structures/Index.html>`_ object - the actual loaded index.

In general, there is a significant cost to loading an Index, as data structures may have to be loaded from disk.
Where possible, for faster reuse, load the actual Index.

Bad Practice::

    bm25 = pt.terrier.Retriever("/path/to/data.properties", wmodel="BM25")
    pl2 = pt.terrier.Retriever("/path/to/data.properties", wmodel="PL2")
    # here, the same index must be loaded twice

Good Practice::

    index = pt.IndexFactory.of("/path/to/data.properties")
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    pl2 = pt.terrier.Retriever(index, wmodel="PL2")
    # here, we share the index between two instances of Retriever

You can use the IndexFactory to specify that the index data structures to be loaded into memory, which can benefit efficiency::

    # load all structures into memory
    inmemindex = pt.IndexFactory.of("/path/to/data.properties", memory=True)
    bm25_fast = pt.terrier.Retriever(inmemindex, wmodel="BM25")

    # load just inverted and lexicon into memory
    inmem_inverted_index = pt.IndexFactory.of("/path/to/data.properties", memory=['inverted', 'lexicon'])
    bm25_fast = pt.terrier.Retriever(inmem_inverted_index, wmodel="BM25")


Advanced: Custom Weighting Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normally, weighting models are specified as a string class names. Terrier then loads the Java class of that name (it will search
the `org.terrier.matching.models package <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html>`_ 
unless the class name is fully qualified (e.g. `"com.example.MyTF"`).

The available models can be found in the Terrier `weighting models javadoc <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html>`_.
Some interesting models include:

- BM25 - the classic Okapi BM25 model
- PL2 - a Divergence from Randomness model
- TF_IDF - the classic vector space model
- DLH13 - a DFR model that is similar to BM25, but with fewer parameters
- DPH - a DFR model that does not require any tuning
- Hiemstra_LM, Dirichlet_LM - language models with different smoothing methods
- DFRWeightingModel - a meta-model allowing to generate arbitrary DFR weighting models, e.g. `"DFRWeightingModel(PL2, L, 2)"`.

For using on indices with multiple fields, Terrier provides some advanced field-based models as well as meta-models that can be used to wrap other weighting models:

- PL2F - a field-based variant of PL2
- BM25F - a field-based variant of BM25
- PerFieldNormWeightingModel - a meta-model that allows you to specify construct an arbitrary field-based model, e.g. `"PerFieldNormWeightingModel(BM, Normalisation2)"`. 

If you have your own Java weighting model instance (which extends the 
`WeightingModel abstract class <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/WeightingModel.html>`_, 
you can load it and pass it directly to Retriever::

    mymodel = pt.autoclass("com.example.MyTF")()
    retr = pt.terrier.Retriever(indexref, wmodel=mymodel)

More usefully, it is possible to express a weighting model entirely in Python, as a function or a lambda expression, that can be
used by Terrier for scoring. In this example, we create a Terrier Retriever instance that scores based solely on term frequency::

    Tf = lambda keyFreq, posting, entryStats, collStats: posting.getFrequency()
    retr = pt.terrier.Retriever(indexref, wmodel=Tf)

All functions passed must accept 4 arguments, as follows:

- keyFrequency(float): the weight of the term in the query, usually 1 except during PRF.
- posting(`Posting <http://terrier.org/docs/current/javadoc/org/terrier/structures/postings/Posting.html>`_): access to the information about the occurrence of the term in the current document (frequency, document length etc).
- entryStats(`EntryStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/EntryStatistics.html>`_): access to the information about the occurrence of the term in the whole index (document frequency, etc.).
- collStats(`CollectionStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/CollectionStatistics.html>`_): access to the information about the index as a whole (number of documents, etc).

Note that due to the overheads of continually traversing the JNI boundary, using a Python function for scoring has a marked efficiency overhead. This is probably too slow for retrieval using most indices of any significant size,
but allows simple explanation of weighting models and exploratory weighting model development.


Advanced: Fine-Grained Terrier Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internally, Terrier manages query execution through a set of configuration options, known as *properties* and *controls*.
Most options are made available through the Python API, but for some advanced use cases it is necessary to modify these
values directly. You can apply both controls and properties for a Retriever by passing dictionaries as the ``controls``
and ``properties`` keyword arguments.

.. note::
    **"Controls" vs "Properties"?**

    A control is a per-query configuration option, whereas a property is a global configuration option.

Common controls:

- `"wmodel"` - the name of the weighting model. (This can also be specified using the wmodel kwarg).
  Valid values are the Java class name of any Terrier weighting model. Terrier provides many,
  such as `"BM25"`, `"PL2"`. A list can be found in the Terrier `weighting models javadoc <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html>`_.

- `"qe"` - whether to run the Divergence from Randomness query expansion.

- `"qemodel"` - which Divergence from Randomness query expansion model. Default is `"Bo1"`.
  A list can be found the Terrier `query expansion models javadoc <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/queryexpansion/package-summary.html>`_.
 
Common properties:

- `"termpipelines"` - the default Terrier term pipeline configuration is `"Stopwords,PorterStemmer"`.
  If you have created an index with a different configuration, you will need to set the  `"termpipelines"`
  property for *each* Retriever constructed. NB: These are now configurable using ``stemming=`` and
  ``stopwords=`` kwargs.

**Examples**::

    # these two Retriever instances are identical, using the same weighting model
    bm25a = pt.terrier.Retriever(index, wmodel="BM25")
    bm25b = pt.terrier.Retriever(index, controls={"wmodel":"BM25"})

    # this one also applies query expansion inside Terrier
    bm25_qe = pt.terrier.Retriever(index, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"})

    # when we introduce an unstemmed Retriever, we ensure to explicitly set the termpipelines
    # for the other Retriever as well
    bm25s_unstemmed = pt.terrier.Retriever(indexUS, wmodel="BM25", properties={"termpipelines" : ""})
    bm25s_stemmed = pt.terrier.Retriever(indexSS, wmodel="BM25", properties={"termpipelines" : "Stopwords,PorterStemmer"})
