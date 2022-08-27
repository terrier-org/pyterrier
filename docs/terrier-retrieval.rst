Terrier Retrieval
-----------------

BatchRetrieve is one of the most commonly used PyTerrier objects. It represents a retrieval transformation, 
in which queries are mapped to retrieved documents. BatchRetrieve uses a pre-existing Terrier index data
structure, typically saved on disk.


Typical usage::

    index = pt.IndexFactory.of("/path/to/data.properties")
    tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    pl2 = pt.BatchRetrieve(index, wmodel="PL2")

    pt.Experiment([tf_idf, bm25, pl2], topic, qrels, eval_metrics=["map"])

As BatchRetrieve is a retrieval transformation, it takes as input dataframes with columns `["qid", "query"]`,
and returns dataframes with columns `["qid", "query", "docno", "score", "rank"]`.

However, BatchRetrieve can also act as a re-ranker. In this scenario, it takes as input dataframes with 
columns `["qid", "query", "docno"]`, and returns dataframes with columns `["qid", "query", "docno", "score", "rank"]`.

For instance, to create a re-ranking pipeline that re-scores the top 100 BM25 documents using PL2::

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    pl2 = pt.BatchRetrieve(index, wmodel="PL2")
    pipeline = (bm25 % 100) >> pl2

BatchRetrieve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.BatchRetrieve
    :members: transform, from_dataset



Terrier Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using PyTerrier, we have to be aware of the underlying Terrier configuration, 
namely *properties* and *controls*. Properties are global configuration and were 
traditionally configured by editing a `terrier.properties` file; In contrast, 
controls are per-query configuration. In PyTerrier, we specify both when we construct
the BatchRetrieve object:

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
   property for *each* BatchRetrieve constructed.

**Examples**::

    # these two BatchRetrieve instances are identical, using the same weighting model
    bm25a = pt.BatchRetrieve(index, wmodel="BM25")
    bm25b = pt.BatchRetrieve(index, controls={"wmodel":"BM25"})

    # this one also applies query expansion inside Terrier
    bm25_qe = pt.BatchRetrieve(index, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"})

    # when we introduce an unstemmed BatchRetrieve, we ensure to explicitly set the termpipelines
    # for the other BatchRetrieve as well
    bm25s_unstemmed = pt.BatchRetrieve(indexUS, wmodel="BM25", properties={"termpipelines" : ""})
    bm25s_stemmed = pt.BatchRetrieve(indexSS, wmodel="BM25", properties={"termpipelines" : "Stopwords,PorterStemmer"})
    


Index-Like Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with Terrier indices, BatchRetrieve allows can make use of:

 - a string representing an index, such as "/path/to/data.properties"
 - a Terrier `IndexRef <http://terrier.org/docs/current/javadoc/org/terrier/querying/IndexRef.html>`_ object, constructed from a string, but which may also hold a reference to the existing index.
 - a Terrier `Index <http://terrier.org/docs/current/javadoc/org/terrier/structures/Index.html>`_ object - the actual loaded index.

In general, there is a significant cost to loading an Index, as data structures may have to be loaded from disk.
Where possible, for faster reuse, load the actual Index.

Bad Practice::

    bm25 = pt.BatchRetrieve("/path/to/data.properties", wmodel="BM25")
    pl2 = pt.BatchRetrieve("/path/to/data.properties", wmodel="PL2")
    # here, the same index must be loaded twice

Good Practice::

    index = pt.IndexFactory.of("/path/to/data.properties")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    pl2 = pt.BatchRetrieve(index, wmodel="PL2")
    # here, we share the index between two instances of BatchRetrieve

TextScorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes we want to apply Terrier to compute the score of document for a given query when we do not yet 
have the documents indexed. TextScorer allows a neat-workaround, in that an index is created on-the-fly 
for the documents, and these are then scored.

Optionally, an index-like object can be specified as the `background_index` kwarg, which will be used for
the collection statistics (e.g. term frequencies, document lengths etc. 

.. autoclass:: pyterrier.batchretrieve.TextScorer

Non-English Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, PyTerrier is configured for indexing and retrieval in English. See
`our notebook <https://github.com/terrier-org/pyterrier/blob/master/examples/notebooks/non_en_retrieval.ipynb>`_
(`colab <https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/non_en_retrieval.ipynb>`_)
for details on how to configure PyTerrier in other languages.

Custom Weighting Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normally, weighting models are specified as a string class names. Terrier then loads the Java class of that name (it will search
the `org.terrier.matching.models package <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html>`_ 
unless the class name is fully qualified (e.g. `"com.example.MyTF"`).

If you have your own Java weighting model instance (which extends the 
`WeightingModel abstract class <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/WeightingModel.html>`_, 
you can load it and pass it directly to BatchRetrieve::

    mymodel = pt.autoclass("com.example.MyTF")()
    retr = pt.BatchRetrieve(indexref, wmodel=mymodel)

More usefully, it is possible to express a weighting model entirely in Python, as a function or a lambda expression, that can be
used by Terrier for scoring. In this example, we create a Terrier BatchRetrieve instance that scores based solely on term frequency::

    Tf = lambda keyFreq, posting, entryStats, collStats: posting.getFrequency()
    retr = pt.BatchRetrieve(indexref, wmodel=Tf)

All functions passed must accept 4 arguments, as follows:

 - keyFrequency(float): the weight of the term in the query, usually 1 except during PRF.
 - posting(`Posting <http://terrier.org/docs/current/javadoc/org/terrier/structures/postings/Posting.html>`_): access to the information about the occurrence of the term in the current document (frequency, document length etc).
 - entryStats(`EntryStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/EntryStatistics.html>`_): access to the information about the occurrence of the term in the whole index (document frequency, etc.).
 - collStats(`CollectionStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/CollectionStatistics.html>`_): access to the information about the index as a whole (number of documents, etc).

Note that due to the overheads of continually traversing the JNI boundary, using a Python function for scoring has a marked efficiency overhead. This is probably too slow for retrieval using most indices of any significant size,
but allows simple explanation of weighting models and exploratory weighting model development.
