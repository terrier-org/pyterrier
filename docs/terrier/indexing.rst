Terrier Indexing
----------------


Indexing Basics
==================================

.. related:: pyterrier.terrier.TerrierIndex.indexer
.. related:: pyterrier_doc2query.Doc2Query

Common indexing options in Terrier are exposed through :meth:`TerrierIndex.indexer <pyterrier.terrier.TerrierIndex.indexer()>`.
Its parameters let you adjust various aspects of the indexing process, such as tokenisation, stemming, position indexing, and more:

.. code-block:: python
    :caption: Creating an indexer from a TerrierIndex

    import pyterrier as pt
    my_index = pt.terrier.TerrierIndex("my_index.terrier")
    indexer = index.indexer(tokeniser='twitter', store_positions=True) # :footnote: See :meth:`TerrierIndex.indexer <pyterrier.terrier.TerrierIndex.indexer>` for full list of parameters.
    indexer.index([
        {"docno" : "tweet1", "text" : "This is a tweet! #pyterrier"},
        {"docno" : "tweet2", "text" : "Another tweet, with a link: https://example.com"}
    ])

Note that Terrier indexes do not support adding additional documents after the initial indexing process.

You do not need to load all documents into memory at once when indexing. Indexers support any "iterable", including
generators that yield one document at a time. Here is an example of indexing documents from a generator function:

.. code-block:: python
    :caption: Indexing from a generator function

    def generate_docs():
        for line in open('my_humongous_collection.txt'):
            yield {"docno" : str(i), "text" : line.strip()}

    indexer.index(generate_docs())

The indexer can function as the final stage of an :ref:`indexing pipeline <indexing_pipelines>`. For example, here is an example that expands documents with
:class:`~pyterrier_doc2query.Doc2Query` before indexing them:

.. schematic::
    :show_code:

    import pyterrier as pt
    from pyterrier_doc2query import Doc2Query
    index = pt.terrier.TerrierIndex("my_index.terrier")
    pipeline = Doc2Query(append=True) >> index.indexer()


Pretokenisation
======================================

.. related:: pyterrier.terrier.TerrierIndex.toks_indexer

Sometimes you want more fine-grained control over the tokenisation directly within PyTerrier.
In this case, each document to be indexed can contain a dictionary of pre-tokenised text and their counts in the ``toks`` column:

.. code-block:: python
    :caption: Indexing pre-tokenised text

    my_index = pt.terrier.TerrierIndex('my_index.terrier')
    my_index.index([
        {'docno' : 'd1', 'toks' : {'a' : 1, '##2' : 2}},
        {'docno' : 'd2', 'toks' : {'a' : 2, '##2' : 1}}
    ])

    # or

    indexer = my_index.toks_indexer() # :footnote: Using :meth:`~pyterrier.terrier.TerrierIndex.toks_indexer` lets you configure settings about the indexing process.
    indexer.index([
        {'docno' : 'd1', 'toks' : {'a' : 1, '##2' : 2}},
        {'docno' : 'd2', 'toks' : {'a' : 2, '##2' : 1}}
    ])

.. note::
    When supplying pre-tokenized text during indexing, Terrier bypasses its tokeniser, stemmer, and stopword removal. You will need
    to apply the same tokenization rules at retrieval time, for instance using :meth:`pt.rewrite.tokenise <pyterrier.rewrite.tokenise>`.

This allows tokenisation using, for instance, the `HuggingFace tokenizers <https://huggingface.co/docs/transformers/fast_tokenizers>`_::

    from transformers import AutoTokenizer
    from collections import Counter

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    # This creates a new column called 'toks', where each row contains 
    # a dictionary of the BERT WordPiece tokens of the 'text' column.
    # This simple example tokenises one row at a time, this could be  
    # made more efficient to utilise batching support in the tokeniser. 
    token_row_apply = pt.apply.toks(lambda row: Counter(tok.tokenize(row['text'])))

    my_index = pt.terrier.TerrierIndex('my_index.terrier')
    index_pipe = token_row_apply >> my_index
    index_pipe.index([
        {'docno' : 'd1', 'text' : 'do goldfish grow?'},
        {'docno' : 'd2', 'text' : ''}
    ])

At retrieval time, WordPieces that contain special characters (e.g. `'##w'` `'[SEP]'`) need to be encoded so as to avoid Terrier's tokeniser. 
We use ``pt.rewrite.tokenise()`` to apply a tokeniser to the query, setting ``matchop`` to True, such that ``pt.terrier.Retriever.matchop()`` 
is called to ensure that rewritten query terms are properly encoded::

    br = pt.terrier.Retriever(indexref)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    query_toks = pt.rewrite.tokenise(tok.tokenize, matchop=True)
    retr_pipe = query_toks >> br


What's in a Terrier index?
===================================

.. related:: pyterrier.terrier.TerrierIndex.collection_statistics
.. related:: pyterrier.terrier.TerrierIndex.lexicon
.. related:: pyterrier.terrier.TerrierIndex.inverted_index
.. related:: pyterrier.terrier.TerrierIndex.document_index
.. related:: pyterrier.terrier.TerrierIndex.meta_index
.. related:: pyterrier.terrier.TerrierIndex.direct_index

A Terrier index contains several data structures. These structures provide low-level API access to the indexed data.
The data structures that can be present in a Terrier index are:

**Collection Statistics**
    :meth:`TerrierIndex.collection_statistics() <pyterrier.terrier.TerrierIndex.collection_statistics>` provides
    global statistics of the index, such as the number of documents, number of terms, etc.

**Lexicon**
    :meth:`TerrierIndex.lexicon() <pyterrier.terrier.TerrierIndex.lexicon>` provides an entry
    for each unique term in the index, which contains the corresponding statistics of each term (frequency etc), and a
    pointer to the inverted index posting list for that term.

**Inverted Index**
    :meth:`TerrierIndex.inverted_index() <pyterrier.terrier.TerrierIndex.inverted_index>` provides access to
    the posting list for each term, which records the documents that a given term appears in, and with what frequency
    for each document.

**Document Index**
    :meth:`TerrierIndex.document_index() <pyterrier.terrier.TerrierIndex.document_index>` provides access to
    the length of the document (and other field lengths).

**Meta Index**
    :meth:`TerrierIndex.meta_index() <pyterrier.terrier.TerrierIndex.meta_index>` provides access to
    document metadata, such as the ``docno``, and optionally the raw text and the URL of each document.

**Direct Index** (*Forward Index*)
    :meth:`TerrierIndex.direct_index() <pyterrier.terrier.TerrierIndex.direct_index>` provides a posting list for
    each document, detailing which terms occur in that document and with which frequency. The presence of the
    direct index depends on the IndexingType that has been applied - single-pass and some memory indices do not
    provide a direct index.


Advanced: IterDictIndexer
========================================

.. related:: pyterrier.terrier.IterDictIndexer

:class:`~pyterrier.terrier.IterDictIndexer` is a flexible Terrier indexer implemementation. It is returned from
:meth:`TerrierIndex.indexer() <pyterrier.terrier.TerrierIndex.indexer()>`, but can be constructed manually if additional functionality
is required.

**Examples using IterDictIndexer**

An iterdict can just be a list of dictionaries::

    docs = [ { 'docno' : 'doc1', 'text' : 'a b c' }  ]
    iter_indexer = pt.IterDictIndexer("./index", meta={'docno': 20, 'text': 4096})
    indexref1 = iter_indexer.index(docs)

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
    
    iter_indexer = pt.IterDictIndexer("./passage_index", meta={'docno': 20, 'text': 4096})
    indexref3 = iter_indexer.index(msmarco_generate())

IterDictIndexer can be used in connection with :ref:`indexing_pipelines`.

Similarly, indexing of JSONL files is similarly a few lines of Python::

    def iter_file(filename):
      import json
      with open(filename, 'rt') as file:
        for l in file:
          # assumes that each line contains 'docno', 'text' attributes
          # yields a dictionary for each json line 
          yield json.loads(l)

    indexref4 = pt.IterDictIndexer("./index", meta={'docno': 20, 'text': 4096}).index(iter_file("/path/to/file.jsonl"))
  
NB: Use ``pt.io.autoopen()`` as a drop-in replacement for ``open()`` that supports files compressed by gzip etc.

**Indexing TREC-formatted files using IterDictIndexer**

If you have TREC-formatted files that you wish to use with an IterDictIndexer-like indexer, :func:`~pyterrier.index.treccollection2textgen` can be used
as a helper function to aid in parsing such files.

Example using Indexing Pipelines::

    files = pt.io.find_files("/path/to/Disk45")
    gen = pt.index.treccollection2textgen(files)
    indexer = pt.text.sliding() >> pt.IterDictIndexer("./index45")
    index = indexer.index(gen)

**Threading**

On UNIX-based systems, IterDictIndexer can also perform multi-threaded indexing::

    iter_indexer = pt.IterDictIndexer("./passage_index_8", meta={'docno': 20, 'text': 4096}, threads=8)
    indexref6 = iter_indexer.index(msmarco_generate())

Note that the resulting index ordering with multiple threads is non-deterministic; if you need 
deterministic behavior you must index in single-threaded mode. Furthermore, indexing can only go
as quickly as the document iterator, so to take full advantage of multi-threaded indexing, you 
will need to keep the iterator function light-weight. Many datasets provide a fast corpus iteration 
function (``get_corpus_iter()``), see more information in the :ref:`datasets`.


Advanced: Specialized Indexers
============================================================

.. related:: pyterrier.terrier.TRECCollectionIndexer
.. related:: pyterrier.terrier.FilesIndexer

In most cases, you will want to use :class:`~pyterrier.terrier.IterDictIndexer` (e.g., using
:meth:`TerrierIndex.indexer() <pyterrier.terrier.TerrierIndex.indexer>`). However, several specialized indexers are available
for specific use-cases.

- :class:`~pyterrier.terrier.TRECCollectionIndexer` lets you index TREC-formaated collections by passing
  in a list of file paths. For example:

  .. code-block:: python
      :caption: Indexing a TREC collection using TRECCollectionIndexer
  
      import pyterrier as pt
      # list of filenames to index
      files = pt.io.find_files("/path/to/WT2G/wt2g-corpus/")
  
      # build the index
      indexer = pt.TRECCollectionIndexer("./wt2g_index", verbose=True, blocks=False)
      indexref = indexer.index(files)
      
      # load the index, print the statistics
      index = pt.IndexFactory.of(indexref)
      print(index.getCollectionStatistics().toString())

- :class:`~pyterrier.terrier.FilesIndexer` lets you index a list of files in various formats such as pdf, docx, and txt.

.. warning::
    The specialized indexers in this section are not compatible with indexing pipelines.


Advanced: Indexing Configuration
===========================================

Our aim is to expose all conventional Terrier indexing configuration through PyTerrier, for instance as constructor arguments
to the Indexer classes. However, as Terrier is a legacy platform, some changes will take time to integrate into Terrier. 
Moreover, the manner of the configuration needed varies a little between the Indexer classes. In the following, we list common
indexing configurations, and how to apply them when indexing using PyTerrier, noting any differences betweeen the Indexer classes.

**Choice of Indexer**

Terrier has three different types of indexer. The choice of indexer is exposed using the ``type`` kwarg 
to the indexer class. The indexer type can be set using the ``IndexingType`` enum.

**Stemming configuation or stopwords**

The default Terrier indexing configuration is to apply an English stopword list, and Porter's stemmer. You can configure this using the ``stemmer`` and ``stopwords`` kwargs for the various indexers::

    indexer = pt.IterDictIndexer(stemmer='SpanishSnowballStemmer', stopwords=None)

See also the `org.terrier.terms <http://terrier.org/docs/current/javadoc/org/terrier/terms/package-summary.html>`_ package for a list of 
the available term pipeline objects provided by Terrier.

Similarly the use of Terrier's English stopword list can be disabled using the ``stopwords`` kwarg.

A custom stopword list can be set by setting the ``stopwords`` kwarg to a list of words::

    indexer = pt.IterDictIndexer("./index", stopwords=['a', 'an', 'the'])

**Languages and Tokenisation**

Similarly, the choice of tokeniser can be controlled in the indexer constructor using the ``tokeniser`` kwarg. 
`EnglishTokeniser <http://terrier.org/docs/current/javadoc/org/terrier/indexing/tokenisation/EnglishTokeniser.html>`_ is the 
default tokeniser. Other tokenisers are listed in  `org.terrier.indexing.tokenisation <http://terrier.org/docs/current/javadoc/org/terrier/indexing/tokenisation/package-summary.html>`_ 
package. For instance, its common to use `UTFTokeniser` when indexing non-English text::

    indexer = pt.IterDictIndexer(stemmer=None, stopwords=None, tokeniser="UTFTokeniser")


**Positions (aka blocks)** 

All indexer classes expose a `blocks` boolean constructor argument to allow position information to be recoreded in the index. Defaults to False, i.e. positions are not recorded.


**Fields**

Fields refers to storing the frequency of a terms occurrence in different parts of a document, e.g. title vs. body vs. anchor text.

IterDictIndexer can be configured to record fields by setting the ``fields=True`` kwarg to the constructor. For instance, if we have two different fields to a document::

    docs = [ {'docno' : 'd1', 'title': 'This is the title', 'text' : 'This is the main document']
    indexref = pt.IterDictIndexer("./index_fields", text_attrs=['text', 'title'], fields=True).index(docs)
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().getNumberOfFields()) # will print 2
    # make a BM25F retriever, places twice as much weight on the title as the main body
    bm25 = pt.terrier.Retriever(index, wmodel='BM25F', controls={'w.0' = 1, 'w.1' = 2, 'c.0' = 0.75, 'c.1' = 0.5})

See the Terrier `indexing documentation on fields <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_indexing.md#fields>`_ for more information. 

NB: Since PyTerrier 0.13, IterDictIndexer no longer records fields by default. This speeds up indexing and retrieval when field-based models such as BM25F are not required.

**Changing the tags parsed by TREC Collection** 

Use the relevant properties listed in the Terrier `indexing documentation <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_indexing.md#basic-indexing-setup>`_.

**MetaIndex configuration** 

Metadata refers to the arbitrary strings associated to each document recorded in a Terrier index. These can range from the `"docno"` attribute of each document, as used to support experimentation, to other attributes such as the URL of the documents, or even the raw text of the document. Indeed, storing the raw text of each document is a trick often used when applying additional re-rankers such as BERT (see `pyterrier_bert <https://github.com/cmacdonald/pyterrier_bert>`_ for more information on integrating PyTerrier with BERT-based re-rankers). Indexers now expose `meta` and `meta_tags` constructor kwarg to make this easier.

**Reverse MetaIndex configuration** 

On occasion, there is a need to lookup up documents in a Terrier index based on their metadata, e.g. "docno". The `meta_reverse` constructor kwarg allows meta keys that support reverse lookup to be specified.
