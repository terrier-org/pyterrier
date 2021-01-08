Runnning Experiments
--------------------

PyTerrier aims to make it easy to conduct an information retrieval experiment, namely in running a transformer 
pipeline over a set of queries, and evaluating the outcome using standard information retrieval metrics (as calculated
by the `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ tool.

The main way to achieve this is using `pt.Experiment()`.

API
========

.. autofunction:: pyterrier.Experiment()


Examples
========

Average Effectiveness
~~~~~~~~~~~~~~~~~~~~~


Getting average effectiveness over a set of topics::

    dataset = pt.get_dataset("vaswani")
    # vaswani dataset provides an index, topics and qrels

    # lets generate two BRs to compare
    tfidf = pt.BatchRetrieve(dataset.get_index(), wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"]
    )

The returned dataframe is as follows:

.. include:: ../_includes/experiment-basic.rst

Each row represents one system. We can manually set the names of the systems, using the `names=` kwarg, as follows::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"]
    )

This produces dataframes that are more easily interpretable.

.. include:: ../_includes/experiment-names.rst

Significance Testing
~~~~~~~~~~~~~~~~~~~~

We can perform significance testing by declaring the index of our baseline using `baseline=0`::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"],
        baseline=0
    )

In this case, additional columns are returned for each measure, indicating 
the number of queries improved compared to the baseline, the number of queries
degraded, as well as the paired t-test p-value in the difference between each
row and the baseline row. NB: For the baseline, these values are NaN.

.. include:: ../_includes/experiment-sig.rst

For this test collection, between the TF_IDF and BM25 weighting models, there is no 
significant difference observed for in MAP, but there is for mean reciprocal rank (*p<0.05*). Indeed,
while BM25 improves average precision for 46 queries over TF_IDF, it degrades it for 45; on the 
other hand, the rank of the first relevant document is improved for 16 queries by BM25 over
TD_IDF.


Per-query Effectiveness
~~~~~~~~~~~~~~~~~~~~~~~

Finally, if necessary, we can request per-query performances using the `perquery=True` kwarg::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"],
        perquery=True
    )

This provides a dataframe where each row is the performance of a given system for a give query on a particular evaluation measure.

.. include:: ../_includes/experiment-perq.rst

NB: For brevity, we only show the top 5 rows of the returned table.

Available Evaluation Measures
=============================

All `trec_eval <https://github.com/usnistgov/trec_eval>`_ evaluation measure are available. 
Often used measures, including the name that must be used, are:

 - Mean Average Precision (`map`)
 - Mean Reciprocal Rank (`recip_rank`)
 - Normalized Discounted Cumulative Gain (`ndcg`), or calculated at a given rank cutoff (e.g. `ndcg_cut_5`).
 - Number of queries (`num_q`) - not averaged
 - Number of retrieved documents (`num_ret`) - not averaged
 - Number of relevant documents (`num_rel`) - not averaged
 - Number of relevant documents retrieved (`num_rel_ret`) - not averaged
 - Interpolated recall precision curves (`iprec_at_recall`). This is family of measures, so requesting this will produce output measurements for `iprec_at_recall_0.00`, `iprec_at_recall_0.10`, etc.
 - Precision at rank cutoff (e.g. `P_5`)
 - Mean response time (`mrt`) will report the average number of milliseconds to conduct a query (this is calculated by pt.Experiment() directly, not pytrec_eval).

See also a `list of common TREC eval measures <http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system>`_.