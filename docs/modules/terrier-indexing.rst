Terrier Indexing
----------------

PyTerrier has a number of useful classes for creating Terrier indices, which can be used for retrieval, query expansion, etc.
There are four indexer classes:

 - You can create an index from TREC-formatted files, from a TREC test collectoin, using TRECCollectionIndexer.
 - For indexing TXT, PDF, Microsoft Word files, etc files you can use FilesIndexer.
 - For indexing Pandas Dataframe you can use DFIndexer.
 - For any abitrary iterable dictionaries, you can use IterDictIndexer.

There are also different types of indexing supported in Terrier that are exposed in PyTerrier.

We explain both the indexing types and the indexer classes below, with examples. Further worked examples of indexing are provided in the `example indexing notebook <https://github.com/terrier-org/pyterrier/blob/master/examples/notebooks/indexing.ipynb>`_.


Index Types
===========

.. autoclass:: pyterrier.index.IndexingType
   :inherited-members: CLASSIC SINGLEPASS MEMORY

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

Example indexing MSMARCO Passage Ranking dataset::

    dataset = pt.get_dataset("trec-deep-learning-passages")
    def msmarco_generate():
        with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
            for l in corpusfile:
                docno, passage = l.split("\t")
                yield {'docno' : docno, 'text' : passage}
    
    iter_indexer = pt.IterDictIndexer("./passage_index")
    indexref3 = iter_indexer.index(msmarco_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096])