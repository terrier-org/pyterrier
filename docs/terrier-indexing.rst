Terrier Indexing
----------------

PyTerrier has a number of useful classes for creating Terrier indices, which can be used for retrieval, query expansion, etc.
There are four indexer classes:

 - You can create an index from TREC-formatted files, from a TREC test collection, using TRECCollectionIndexer.
 - For indexing TXT, PDF, Microsoft Word files, etc files you can use FilesIndexer.
 - For indexing Pandas Dataframe you can use DFIndexer.
 - For any abitrary iterable dictionaries, you can use IterDictIndexer.

There are also different types of indexing supported in Terrier that are exposed in PyTerrier.

We explain both the indexing types and the indexer classes below, with examples. Further worked examples of indexing are provided in the `example indexing notebook <https://github.com/terrier-org/pyterrier/blob/master/examples/notebooks/indexing.ipynb>`_.


Index Types
===========

.. autoclass:: pyterrier.index.IndexingType
   :inherited-members: CLASSIC, SINGLEPASS, MEMORY

Indexer Classes
===============


TRECCollectionIndexer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.TRECCollectionIndexer
   :members: index

Example indexing the TREC WT2G corpus::

    import pyterrier as pt
    pt.init()
    # list of filenames to index
    files = pt.io.find_files("/path/to/WT2G/wt2g-corpus/")

    # build the index
    indexer = pt.TRECCollectionIndexer("./wt2g_index", verbose=True, blocks=False)
    indexref = indexer.index(files)
    
    # load the index, print the statistics
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())


FilesIndexer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.FilesIndexer
   :members: index

DFIndexer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.DFIndexer
   :members: index

Example indexing a dataframe::

    # define an example dataframe of documents
    import pandas as pd
    df = pd.DataFrame({ 
        'docno':
        ['1', '2', '3'],
        'url': 
        ['url1', 'url2', 'url3'],
        'text': 
        ['He ran out of money, so he had to stop playing',
        'The waves were crashing on the shore; it was a',
        'The body may perhaps compensates for the loss']
    })

    # index the text, record the docnos as metadata
    pd_indexer = pt.DFIndexer("./pd_index")
    indexref = pd_indexer.index(df["text"], df["docno"])


IterDictIndexer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.IterDictIndexer
   :members: index

**Examples using IterDictIndexer**

An iterdict can just be a list of dictionaries::

    docs = [ { 'docno' : 'doc1', 'text' : 'a b c' }  ]
    iter_indexer = pt.IterDictIndexer("./index")
    indexref1 = iter_indexer.index(docs, meta=['docno', 'text'], meta_lengths=[20, 4096])

A dataframe can also be used, virtue of its ``.to_dict()`` method::

    df = pd.DataFrame([['doc1', 'a b c']], columns=['docno', 'text'])
    iter_indexer = pt.IterDictIndexer("./index")
    indexref2 = indexer.index(df.to_dict(orient="records"))

However, the main power of using IterDictIndexer is for processing indefinite iterables, such as those returned by generator functions.
For example, the tsv file of the MSMARCO Passage Ranking corpus can be indexed as follows::

    dataset = pt.get_dataset("trec-deep-learning-passages")
    def msmarco_generate():
        with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
            for l in corpusfile:
                docno, passage = l.split("\t")
                yield {'docno' : docno, 'text' : passage}
    
    iter_indexer = pt.IterDictIndexer("./passage_index")
    indexref3 = iter_indexer.index(msmarco_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096])

IterDictIndexer can be used in connection with :ref:`indexing_pipelines`.

Similarly, indexing of JSONL files is similarly a few lines of Python::

    def iter_file(filename):
      import json
      with open(filename, 'rt') as file:
        for l in file:
          # assumes that each line contains 'docno', 'text' attributes
          # yields a dictionary for each json line 
          yield json.loads(l)

    indexref4 = pt.IterDictIndexer("./index").index(iter_file("/path/to/file.jsonl"), meta=['docno', 'text'], meta_lengths=[20, 4096])
  
NB: Use ``pt.io.autoopen()`` as a drop-in replacement for ``open()`` that supports files compressed by gzip etc.

**Indexing TREC-formatted files using IterDictIndexer**

If you have TREC-formatted files that you wish to use with an IterDictIndexer-like indexer, ``pt.index.treccollection2textgen()`` can be used
as a helper function to aid in parsing such files.

.. autofunction:: pyterrier.index.treccollection2textgen

Example using Indexing Pipelines::

    files = pt.io.find_files("/path/to/Disk45")
    gen = pt.index.treccollection2textgen(files)
    indexer = pt.text.sliding() >> pt.IterDictIndexer("./index45")
    index = indexer.index(gen)

**Threading**

On UNIX-based systems, IterDictIndexer can also perform multi-threaded indexing::

    iter_indexer = pt.IterDictIndexer("./passage_index_8", threads=8)
    indexref6 = iter_indexer.index(msmarco_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096])

Note that the resulting index ordering with multiple threads is non-deterministic; if you need 
deterministic behavior you must index in single-threaded mode. Furthermore, indexing can only go
as quickly as the document iterator, so to take full advantage of multi-threaded indexing, you 
will need to keep the iterator function light-weight. Many datasets provide a fast corpus iteration 
function (``get_corpus_iter()``), see more information in the :ref:`datasets`.

Indexing Configuration
======================

Our aim is to expose all conventional Terrier indexing configuration through PyTerrier, for instance as constructor arguments
to the Indexer classes. However, as Terrier is a legacy platform, some changes will take time to integrate into Terrier. 
Moreover, the manner of the configuration needed varies a little between the Indexer classes. In the following, we list common
indexing configurations, and how to apply them when indexing using PyTerrier, noting any differences betweeen the Indexer classes.

- *stemming configuation or stopwords*: the default Terrier indexing configuration is to use a term pipeline of `Stopwords,PorterStemmer`. You can change the term pipeline configuration using the `"termpipeline"` property::

    indexer.setProperty("termpipelines", "")

 Note that any subsequent indexing or retrieval operation would also require the `"termpipeline"` property to be suitably updated.
 See the `org.terrier.terms <http://terrier.org/docs/current/javadoc/org/terrier/terms/package-summary.html>`_ package for a list of 
 the available term pipeline objects provided by Terrier.

- *languages and tokenisation*: Similarly, the choice of tokeniser is controlled by the `"tokeniser"` property. `EnglishTokeniser <http://terrier.org/docs/current/javadoc/org/terrier/indexing/tokenisation/EnglishTokeniser.html>`_ is the  default tokeniser. Other tokenisers are listed in  `org.terrier.indexing.tokenisation <http://terrier.org/docs/current/javadoc/org/terrier/indexing/tokenisation/package-summary.html>`_ package. For instance, use `indexer.setProperty("tokeniser", "UTFTokeniser")` when indexing non-English text.

- *positions (aka blocks)* - all indexers expose a `blocks` boolean constructor argument to allow position information to be recoreded in the index. Defaults to False, i.e. positions are not recorded.

- *fields* - fields refers to storing the frequency of a terms occurrence in different parts of a document, e.g. title vs body vs anchor text. In the IterDictIndexer, fields are set in the `index()` method; otherwise the `"FieldTags.process"` property must be set. See the Terrier `indexing documentation on fields <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_indexing.md#fields>`_ for more information. 

- *changing the tags parsed by TREC Collection* - use the relevant properties listed in the Terrier `indexing documentation <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_indexing.md#basic-indexing-setup>`_.

- *metaindex configuration*: metadata refers to the arbitrary strings associated to each document recorded in a Terrier index. These can range from the `"docno"` attribute of each document, as used to support experimentation, to other attributes such as the URL of the documents, or even the raw text of the document. Indeed, storing the raw text of each document is a trick often used when applying additional re-rankers such as BERT (see `pyterrier_bert <https://github.com/cmacdonald/pyterrier_bert>`_ for more information on integrating PyTerrier with BERT-based re-rankers). Indexers now expose `meta` and `meta_tags` constructor kwarg to make this easier.

- *reverse metaindex configuration*: on occasion, there is a need to lookup up documents in a Terrier index based on their metadata, e.g. "docno". The `meta_reverse` constructor kwarg allows meta keys that support reverse lookup to be specified.
