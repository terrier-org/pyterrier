Learning to Rank with PyTerrier
-------------------------------

Introduction
============

PyTerrier makes it easy to formulate learning to rank pipelines. Conceptually, learning to rank consist of three phases:

 1. identifying a candidate set of documents for each query
 2. computing extra features on these documents
 3. using a learned model to re-rank the candidate documents to obtain a more effective ranking

PyTerrier allows each of these phases to be expressed as transformers, and for them to be composed into a full pipeline.  

In particular, conventional retrieval transformers (such as `pt.BatchRetrieve`) can be used for the first phase.
To permit the second phase, PyTerrier data model allows for a `"features"` column to be associated to each retrieved document. 
Such features can be generated using specialised transformers, or by combining other re-ranking transformers using the  `**`
feature-union operator; Lastly, to facilitate the final phase, we provide easy ways to integrate with standard learning libraries
such as `sklearn <https://scikit-learn.org/>`_, `xgBoost <https://xgboost.readthedocs.io/en/latest/>`_  and 
`LightGBM <http://lightgbm.readthedocs.io/>`_.

In the following, we focus on the second and third phases, as well as describing ways to assist in conducting learning to rank
experiments. 

Calculating Features
====================

Feature Union (`**`)
~~~~~~~~~~~~~~~~~~~~

PyTerrier's main way to faciliate calculating extra features is through the `**` operator. Consider an example where
the candidate set should be identified using the BM25 weighting model, and then additional features computed using the
PL2F and BM25F models::

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    bm25f = pt.BatchRetrieve(index, wmodel="BM25F")
    pl2f = pt.BatchRetrieve(index, wmodel="PL2F")
    pipeline = bm25 >> (bm25f ** pl2f)

The output of the bm25 ranker would look like:

====  ==========  ========  ============
  ..  qid          docno        score  
====  ==========  ========  ============
   1  q1             d5     (bm25 score)
====  ==========  ========  ============

Application of the feature-union operator (`**`) ensures that bm25f and pl2f 
operate as *re-rankers*, i.e. they re-rank the documents identified by bm25.
For each document, the score they calculate combined into the `"features"` column, as follows:

====  ==========  ========  ============  =========================
  ..  qid          docno        score              features
====  ==========  ========  ============  =========================
   1  q1             d5     (bm25 score)  [bm25f score, pl2f score]
====  ==========  ========  ============  =========================




FeaturesBatchRetrieve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When doing executing the pipeline above, then re-ranking of the documents again can be slow, as each separate BatchRetrieve
object has to re-access the inverted index. For this reason, PyTerrier provides a class called FeaturesBatchRetrieve,
which allows multiple query dependent features to be calculated at once.


.. autoclass:: pyterrier.FeaturesBatchRetrieve
    :members: transform 

An equivalent pipeline to the example above would be::

    #pipeline = bm25 >> (bm25f ** pl2f)
    pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:BM25F", "WMODEL:PL2F"]


Apply Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a way to calculate all one or mulitple ranking features, you can use pt.apply functions to create
your feature sets.  See the :ref:`pyterrier.apply` for examples.

Learning
========

.. autofunction:: pyterrier.ltr.apply_learned_model()

SKLearn
~~~~~~~

A sklearn regressor can be passed directly to `pt.ltr.apply_learned_model()`::

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=400)
    rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)
    rf_pipe.fit(train_topics, qrels)
    pt.Experiment([bm25, rf_pipe], test_topics, qrels, ["map"], names=["BM25 Baseline", "LTR"])

Note that for analysis, the features importances identified by RandomForestRegressor can be accessed
through `rf.features_importances_`.

Gradient Boosted Trees & LambdaMART
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both `xgBoost <https://xgboost.readthedocs.io/en/latest/>`_  and `LightGBM <http://lightgbm.readthedocs.io/>`_
provide gradient boosted regression tree and LambdaMART implementations. These support a sklearn-like
interface that is supported by PyTerrier by supplying `form="ltr"` kwarg to `pt.ltr.apply_learned_model()`::

    import xgboost as xgb
    lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg', 
          learning_rate= 0.1, 
          gamma= 1.0 min_child_weight=0.1,
          max_depth=6,
          verbose=2,
          random_state=42)
    
    lmart_x_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
    lmart_x_pipe.fit(train_topics, qrels)

    import lightgbm as lgb
    lmart_l = lgb.LGBMRanker(task="train",
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=100,
        max_bin=255,
        num_leaves=7,
        objective="lambdarank",
        "metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate= .1,
        importance_type="gain",
        num_iterations=10)
    lmart_l_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
    lmart_l_pipe.fit(train_topics, qrels)

    pt.Experiment([bm25, lmart_x_pipe, lmart_l_pipe], test_topics, qrels, ["map"], names=["BM25 Baseline", "LambdaMART (xgBoost)", "LambdaMART (LightGBM)" ])

In our experience, LightGBM *tends* to be more effective than xgBoost.


Working with Features
=====================

We provide additional transformations fucntions to aid the analysis of learned model, for instance, removing (ablating) features from a 
complex ranking pipeline.

.. autofunction:: pyterrier.ltr.ablate_features()

Example::

    # assume pipeline is a retrieval pipeline that produces four ranking features
    numf=4
    rankers = []
    names = []
    # learn a model for all four features
    full = pipeline >> pt.ltr.apply_learned_model(RandomForestRegressor(n_estimators=400))
    full.fit(trainTopics, trainQrels)
    ranker.append(full)
    
    # learn a model for 3 features, removing one each time
    for fid in range(numf):
        ablated = pipeline >> pt.ltr.ablate_features(fid) >> pt.ltr.apply_learned_model(RandomForestRegressor(n_estimators=400))
        ablated.fit(trainTopics, trainQrels)
        rankers.append(full)

    # evaluate the full (4 features) model, as well as the each model containing only 3 features)
    pt.Experiment(
        rankers,
        test_topics,
        test_qrels,
        ["map"],
        names=["Full Model"]  + ["Full Minus %d" % fid for fid in range(numf) 
    )

.. autofunction:: pyterrier.ltr.keep_features()

.. autofunction:: pyterrier.ltr.feature_to_score()

.. autofunction:: pyterrier.ltr.score_to_feature()

