pyterrier.anserini - Anserini/Lucene Support
--------------------------------------------


Through an integration of pyserini, PyTerrier can integrate results from the Lucene-based Anserini platform into retrieval pipelines.

.. automodule:: pyterrier.anserini
    :members:
        

Examples
========

Comparative retrieval from Anserini and Terrier::

    trIndex = "/path/to/data.properties"
    luceneIndex "/path/to/lucene-index-dir"

    BM25_tr = pt.BatchRetrieve(trIndex, wmodel="BM25")
    BM25_ai = pt.anserini.AnseriniBatchRetrieve(luceneIndex, wmodel="BM25")

    pt.Experiment([BM25_tr, BM25_ai], topics, qrels, eval_metrics=["map"])


AnseriniBatchRetrieve can also be used as a re-ranker::

    BM25_tr = pt.BatchRetrieve(trIndex, wmodel="BM25")
    QLD_ai = pt.anserini.AnseriniBatchRetrieve(luceneIndex, wmodel="QLD")

    pipe = BM25_tr >> QLD_ai

