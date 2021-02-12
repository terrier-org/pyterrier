Running Experiments
-------------------

PyTerrier aims to make it easy to conduct an information retrieval experiment, namely, to run a transformer 
pipeline over a set of queries, and evaluating the outcome using standard information retrieval evaluation 
metrics based on known relevant documents (obtained from a set relevance assessments, also known as *qrels*).
The evaluation metrics are calculated by the `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ library,
a Python wrapper around the widely-used `trec_eval evaluation tool <https://github.com/usnistgov/trec_eval>`_.

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

.. include:: ./_includes/experiment-basic.rst

Each row represents one system. We can manually set the names of the systems, using the `names=` kwarg, as follows::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"]
    )

This produces dataframes that are more easily interpretable.

.. include:: ./_includes/experiment-names.rst

We can also reduce the number of decimal places reported using the `round=` kwarg, as follows::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        round={"map" : 4, "recip_rank" : 3},
        names=["TF_IDF", "BM25"]
    )

The result is as follows:

.. include:: ./_includes/experiment-round.rst

Passing an integer value to `round=` (e.g. `round=3`) applies rounding to all evaluation measures.


Significance Testing
~~~~~~~~~~~~~~~~~~~~

We can perform significance testing by specifying the index of which transformer we consider to be our baseline,
e.g. `baseline=0`::

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
row and the baseline row. NB: For the baseline, these values are NaN (not applicable).

.. include:: ./_includes/experiment-sig.rst

For this test collection, between the TF_IDF and BM25 weighting models, there is no 
significant difference observed in terms of MAP, but there is a significant different in terms of 
mean reciprocal rank (*p<0.05*). Indeed, while BM25 improves average precision for 46 queries 
over TF_IDF, it degrades it for 45; on the other hand, the rank of the first relevant document 
is improved for 16 queries by BM25 over TD_IDF.


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

.. include:: ./_includes/experiment-perq.rst

NB: For brevity, we only show the top 5 rows of the returned table.

Available Evaluation Measures
=============================

All `trec_eval <https://github.com/usnistgov/trec_eval>`_ evaluation measure are available. 
Often used measures, including the name that must be used, are:

 - Mean Average Precision (`map`).
 - Mean Reciprocal Rank (`recip_rank`).
 - Normalized Discounted Cumulative Gain (`ndcg`), or calculated at a given rank cutoff (e.g. `ndcg_cut_5`).
 - Number of queries (`num_q`) - not averaged.
 - Number of retrieved documents (`num_ret`) - not averaged.
 - Number of relevant documents (`num_rel`) - not averaged.
 - Number of relevant documents retrieved (`num_rel_ret`) - not averaged.
 - Interpolated recall precision curves (`iprec_at_recall`). This is family of measures, so requesting `iprec_at_recall` will output measurements for `iprec_at_recall_0.00`, `iprec_at_recall_0.10`, etc.
 - Precision at rank cutoff (e.g. `P_5`).
 - Recall (`recall`) will generate recall at different cutoffs, such as `recall_5`, etc.).
 - Mean response time (`mrt`) will report the average number of milliseconds to conduct a query (this is calculated by `pt.Experiment()` directly, not pytrec_eval).
 - trec_eval measure *families* such as `official`, `set` and `all_trec` will be expanded. These result in many measures being returned. For instance, asking for `official` results in the following (very wide) output reporting the usual default metrics of trec_eval:

.. include:: ./_includes/experiment-official.rst


See also a `list of common TREC eval measures <http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system>`_.