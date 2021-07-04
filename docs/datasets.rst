.. _datasets:

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

Examples
========

Many of the PyTerrier unit tests are based on the `Vaswani NPL test collection <http://ir.dcs.gla.ac.uk/resources/test_collections/npl/>`_, a corpus of scientific abstract from ~11,000 documents.
PyTerrier provides a ready-made index on the `Terrier Data Repository <http://data.terrier.org/>`_. This allows experiments to be easily conducted::

    dataset = pt.get_dataset("vaswani")
    bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
    dph = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="DPH")
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
    index = pt.IndexFactory.of(indexref)

    DPH_br = pt.BatchRetrieve(index, wmodel="DPH") % 100
    BM25_br = pt.BatchRetrieve(index, wmodel="BM25") % 100
    # this runs an experiment to obtain results on the TREC 2019 Deep Learning track queries and qrels
    pt.Experiment(
        [DPH_br, BM25_br], 
        dataset.get_topics("test"), 
        dataset.get_qrels("test"), 
        eval_metrics=["recip_rank", "ndcg_cut_10", "map"])

For more details on use of MSMARCO, see `our MSMARCO leaderboard submission notebooks <https://github.com/cmacdonald/pyterrier-msmarco-document-leaderboard-runs>`_.

You can also index datasets that include a corpus using IterDictIndexer and get_corpus_iter::

    dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
    indexer = pt.index.IterDictIndexer('./cord19-index')
    indexref = indexer.index(dataset.get_corpus_iter(), fields=('title', 'abstract'))
    index = pt.IndexFactory.of(indexref)

    DPH_br = pt.BatchRetrieve(index, wmodel="DPH") % 100
    BM25_br = pt.BatchRetrieve(index, wmodel="BM25") % 100
    # this runs an experiment to obtain results on the TREC COVID queries and qrels
    pt.Experiment(
        [DPH_br, BM25_br], 
        dataset.get_topics('title'), 
        dataset.get_qrels(), 
        eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"])

Available Datasets
==================

The table below lists the provided datasets, detailing the attributes available for each dataset.
In each column, True designates the presence of a single artefact of that type, while a list denotes the available variants.
Datasets with the ``irds:`` prefix are from the `ir_datasets package <https://github.com/allenai/ir_datasets>`_; further
documentation on these datasets can be found `here <https://ir-datasets.com/all.html>`_.

.. include:: ./_includes/datasets-list-inc.rst