Importing Datasets
-----------------------

The datasets module allows easy access to existing standard test collections, particulary those from `TREC <https://trec.nist.gov/>`_. In particular, 
each defined dataset can download and provide easy access to:

 - files containing the documents of the corpus
 - topics (queries), as a dataframe, ready for retrieval
 - relevance assessments (aka, labels or qrels), as a dataframe, ready for evaluation
 - ready-made Terrier indices, where appropriate

.. autofunction:: pyterrier.datasets.list_datasets()

.. autofunction:: pyterrier.datasets.get_dataset()

.. autoclass:: pyterrier.datasets.Dataset
    :members:

Available Datasets
==================

The table below lists the provided datasets, detailing the attributes available for each dataset.
In each column, True designates the presence of a single artefact of that type, while a list denotes the available variants.

.. include:: ./_includes/datasets-list-inc.rst


Examples
========

Many of the PyTerrier unit tests are based on the `Vaswani NPL test collection <http://ir.dcs.gla.ac.uk/resources/test_collections/npl/>`_, a corpus of scientific abstract from ~11,000 documents.
PyTerrier provides a ready-made index. This allows experiments to be easily conducted::

    dataset = pt.get_dataset("vaswani")
    bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")
    dph = pt.BatchRetrieve(dataset.get_index(), wmodel="DPH")
    pt.Experiment(
        [bm25, dph],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map"]
    )

Indexing and then retrieval of documents from the `MSMARCO document corpus <https://microsoft.github.io/msmarco/>`_ can be achieved as follows::

    dataset = pt.get_dataset("trec-deep-learning-docs")
    indexer = pt.TRECCollectionIndexer("./index")
    # this downloads the file msmarco-docs.trec.gz 
    indexref = indexer.index(dataset.get_corpus())

    DPH_br = pt.BatchRetrieve(index, wmodel="DPH") % 100
    BM25_br = pt.BatchRetrieve(index, wmodel="DPH") % 100
    # this runs an experiment to obtain results on the TREC 2019 Deep Learning track queries and qrels
    pt.Experiment(
        [DPH_br, BM25_br], 
        dataset.get_topics("test"), 
        dataset.get_qrels("test"), 
        eval_metrics=["recip_rank", "ndcg_cut_10", "map"])

For more details on use of MSMARCO, see `our MSMARCO leaderboard submission notebooks <https://github.com/cmacdonald/pyterrier-msmarco-document-leaderboard-runs>`_.