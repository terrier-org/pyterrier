Running Experiments
-------------------

PyTerrier aims to make it easy to conduct an information retrieval experiment, namely, to run a transformer 
pipeline over a set of queries, and evaluating the outcome using standard information retrieval evaluation 
metrics based on known relevant documents (obtained from a set relevance assessments, also known as *qrels*).


NB: For calculating evaluation metrics, we use `ir_measures <https://github.com/terrierteam/ir_measures>`_ library,
which includes implementations of many standard metrics. By default, to calculate more measures, ir_measures uses our fork of 
the `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ library, which itself is a Python wrapper around 
the widely-used `trec_eval evaluation tool <https://github.com/usnistgov/trec_eval>`_.

The main way to achieve this is using ``pt.Experiment()``. If you have an existing results dataframe, you can use
``pt.Evaluate()``.

API
========

.. autofunction:: pyterrier.Experiment()

.. autofunction:: pyterrier.Evaluate()

Examples
========

Average Effectiveness
~~~~~~~~~~~~~~~~~~~~~

Getting average effectiveness over a set of topics::

    dataset = pt.get_dataset("vaswani")
    # vaswani dataset provides an index, topics and qrels

    # lets generate two BRs to compare
    tfidf = pt.terrier.Retriever(dataset.get_index(), wmodel="TF_IDF")
    bm25 = pt.terrier.Retriever(dataset.get_index(), wmodel="BM25")

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

Further more, modern experimental convention suggests that it is important to correct for multiple
testing in the comparative evaluation of many IR systems. Experiments provides supported for the
multiple testing correction methods supported by the `statsmodels package <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html#statsmodels.stats.multitest.multipletests>`_,
such as `Bonferroni`::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map"],
        names=["TF_IDF", "BM25"],
        baseline=0,
        correction='bonferroni'
    )

This adds two further columns for each measure, denoting if the null hypothesis can be rejected (e.g. `"map reject"`),
as well as the corrected p value (`"map p-value corrected"`), as shown below:

.. include:: ./_includes/experiment-sig-corr.rst

The table below summarises the multiple testing correction methods supported:

.. include:: ./_includes/experiment-corr-methods.rst

Any value in the Aliases column can be passed to Experiment's `correction=` kwarg.

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

Saving and Reusing Results 
~~~~~~~~~~~~~~~~~~~~~~~~~~

For some research tasks, it is considered good practice to save your results files when conducting experiments. This allows
several advantages:

 - It permits additional evaluation (e.g. more measures, more signifiance tests) without re-applying potentially slow transformer pipelines.
 - It allows transformer results to be made available for other experiments, perhaps as a virtual data appendix in a paper.

Saving can be enabled by adding the ``save_dir`` as a kwarg to pt.Experiment::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"],
        save_dir="./",
    )

This will save two files, namely, TF_IDF.res.gz and BM25.res.gz to the current directory. If these files already exist,
they will be "reused", i.e. loaded and evaluated in preference to application of the tfidf and/or bm25 transformers. 
If experiments are being conducted on multiple different topic sets, care should be taken to ensure that previous 
results for a different topic set are not reused for evaluation.

If a transformer has been updated, outdated results files can be mistakenly used. To prevent this, set the ``save_mode`` 
kwarg to ``"overwrite"``::

    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"],
        save_dir="./",
        save_mode="overwrite"
    )

Missing Topics and/or Qrels
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is not always a one-to-one correspondance between the topic/query IDs (qids) that appear in
the provided ``topics`` and ``qrels``. Qids that appear in topics but not qrels can be due to incomplete judgments,
such as in sparsely labeled datasets or shared tasks that choose to omit some topics (e.g., due to cost).
Qids that appear in qrels but no in topics can happen when running a subset of topics for testing purposes
(e.g., ``topics.head(5)``).

The ``filter_by_qrels`` and ``filter_by_topics`` parameters control the behaviour of an experiment when topics and qrels
do not perfectly overlap. When ``filter_by_qrels=True``, topics are filtered down to only the ones that have qids in the
qrels. Similarly, when ``filter_by_topics=True``, qrels are filtered down to only the ones that have qids in the topics.

For example, consier topics that include qids ``A`` and ``B`` and qrels that include ``B`` and ``C``. The results with
each combination of settings are:

+----------------------+----------------------+------------------+--------------------------------------------------------------------+
| ``filter_by_topics`` | ``filter_by_qrels``  | Results consider | Notes                                                              |
+======================+======================+==================+====================================================================+
| ``True`` (default)   | ``False`` (default)  | ``A,B``          | ``C`` is removed because it does not appear in the topics.         |
+----------------------+----------------------+------------------+--------------------------------------------------------------------+
| ``True`` (default)   | ``True``             | ``B``            | Acts as an intersection of the qids found in the qrels and topics. |
+----------------------+----------------------+------------------+--------------------------------------------------------------------+
| ``False``            | ``False`` (default)  | ``A,B,C``        | Acts as a union of the qids found in qrels and topics.             |
+----------------------+----------------------+------------------+--------------------------------------------------------------------+
| ``False``            | ``True``             | ``B,C``          | ``A`` is removed because it does not appear in the qrels.          |
+----------------------+----------------------+------------------+--------------------------------------------------------------------+

Note that, following IR evaluation conventions, topics that have no relevance judgments (``A`` in the above example)
do not contribute to relevance-based measures (e.g., ``map``), but still contribute to efficiency measures (e.g., ``mrt``).
As such, aggregate relevance-based measures will not change based on the value of ``filter_by_qrels``. When ``perquery=True``,
topics that have no relevance judgments (``A``) will give a value of ``NaN``, indicating that they are not defined
and should not contribute to the average.

The defaults (``filter_by_topics=True`` and ``filter_by_qrels=False``) were chosen because they likely reflect the intent
of the user in most cases. In particular, it runs all topics requested and evaluates on only those topics. However, you
may want to change these settings in some circumstnaces. E.g.:

 - If you want to save time and avoid running topics that will not be evaluated, set ``filter_by_qrels=True``.
   This can be particularly helpful for large collections with many missing judgments, such as MS MARCO.
 - If you want to evaluate across all topics from the qrels set ``filter_by_topics=False``.

Note that in all cases, if a requested topic that appears in the qrels returns no results, it will properly contribute
a score of 0 for evaluation.



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
- Interpolated recall precision curves (`iprec_at_recall`). This is family of measures, so requesting `iprec_at_recall` will output measurements for `IPrec@0.00`, `IPrec@0.10`, etc.
- Precision at rank cutoff (e.g. `P_5`).
- Recall (`recall`) will generate recall at different cutoffs, such as `recall_5`, etc.).
- Mean response time (`mrt`) will report the average number of milliseconds to conduct a query (this is calculated by ``pt.Experiment()`` directly, not pytrec_eval).
- trec_eval measure *families* such as `official`, `set` and `all_trec` will be expanded. These result in many measures being returned. For instance, asking for `official` results in the following (very wide) output reporting the usual default metrics of trec_eval:

.. include:: ./_includes/experiment-official.rst


See also a `list of common TREC eval measures <http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system>`_.

Evaluation Measures Objects
===========================

Using the `ir_measures <https://github.com/terrierteam/ir_measures>`_ Python package, PyTerrier supports evaluation measure objects. These make it easier to 
express measure configurations such as rank cutoffs::

    from pyterrier.measures import *
    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[AP, RR, nDCG@5],
    )

NB: We have to use `from pyterrier.measures import *`, as `from pt.measures import *` wont work.

More specifically, lets consider the TREC Deep Learning track passage ranking task, which requires NDCG\@10, NDCG\@100 (using graded labels), as well as MRR\@10 and MAP using binary labels 
(where relevant is grade 2 and above). The necessary incantation of `pt.Experiment()` looks like::

    from pyterrier.measures import *
    dataset = pt.get_dataset("trec-deep-learning-passages")
    pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics("test-2019"),
        dataset.get_qrels("test-2019"),
        eval_metrics=[RR(rel=2), nDCG@10, nDCG@100, AP(rel=2)],
    )

The available evaluation measure objects are listed below.

.. autofunction:: pyterrier.measures.P

.. autofunction:: pyterrier.measures.R

.. autofunction:: pyterrier.measures.AP

.. autofunction:: pyterrier.measures.RR

.. autofunction:: pyterrier.measures.nDCG

.. autofunction:: pyterrier.measures.ERR

.. autofunction:: pyterrier.measures.Success

.. autofunction:: pyterrier.measures.Judged

.. autofunction:: pyterrier.measures.NumQ

.. autofunction:: pyterrier.measures.NumRet

.. autofunction:: pyterrier.measures.NumRelRet

.. autofunction:: pyterrier.measures.NumRel

.. autofunction:: pyterrier.measures.Rprec

.. autofunction:: pyterrier.measures.Bpref

.. autofunction:: pyterrier.measures.infAP

Validation of Transformers
==========================

When formulating pipelines for a ``pt.Experiment()``, its possible to formulate invalid pipelines, e.g. a transformer that does not produce the expected columns, or a transformer that does not accept the input columns of the previous transformer. 
To mitigate this, ``pt.Experiment()`` will validate the transformers in the pipeline, and raise an error if the pipeline is invalid.

This validation is controlled by the `validate=` kwarg, which can take the following values:
- ``"warn"`` (default): If the pipeline is invalid, a warning is issued, but the experiment proceeds. Pipelines that do not validate will still run, but may produce unexpected results.
- ``"error"``: If the pipeline is invalid, an error is raised, and the experiment does not proceed. If a pipeline is not validated, the user is informed such that the experiment fails-fast.
- ``"ignore"``: No validation is performed, and the experiment proceeds. This is useful for pipelines that are known to be valid, but cannot be validated due to transformer objects that cannot be inspected to determing their input and output columns.

Validation uses ``pt.inspect.transformer_outputs()`` to determine the output columns of each transformer in the pipeline, and whether they match the expected input columns of the next transformer, and that the overall result of the pipeline has the expected columns for the evaluation measures requested. 
Most transformers can be validated automatically, particularly if they respond correctly to an empty DataFrame input. Other transformers may require a `transform_output` method to be implemented, which returns the expected output columns of the transformer.

If a pipeline fails validation, the user is informed of the problem, and, if `validate="error"` is set, the experiment does not proceed.
On the other hand, if a pipeline cannot be validated (because a transformer cannot be inspected), a warning is issued, and the experiment proceeds.

Precomputation of Common Pipeline Prefixes
==========================================

Often we wish to evaluate multiple pipelines that have exactly the same initial stages. ``pt.Experiment`` exposes a `precompute_prefix` kwarg, will precompute the results of the common initial stages, and then use these results to call the subsequent remainder of each pipelines.

Consider the following example::

    from pyterrier_t5 import MonoT5ReRanker
    bm25 = pt.terrier.Retriever.from_dataset('vaswani', 'terrier_stemmed_text', wmodel='BM25', num_results=100)
    monoT5 = MonoT5ReRanker()

    monoT5 = bm25 >> monoT5
    pt.Experiment(
        [bm25, monoT5], 
        pt.get_dataset('vaswani').get_topics(), 
        pt.get_dataset('vaswani').get_qrels(), 
        eval_metrics=['map'], 
        precompute_prefix=True
    )

Normally, BM25 retriever would be invoked twice during this experiment - once for each pipeline, resulting in a slower executation time compared to an imperative workflow (get BM25 results, evaluate, apply monoT5, evaluate). By setting `precompute_prefix=True`, ``pt.Experiment`` will execute the `bm25` transformer only once on the input topics, and then reuse those results as input to monoT5.

NB: This is experimental functionality, but should initial usage be successful, it may be turned on by default in future versions of PyTerrier.