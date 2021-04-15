Tuning Transformer Pipelines
----------------------------

Many approaches will have parameters that require tuning. PyTerrier helps to achieve this by proving a grid 
evaluation functionality that can tune one or more parameters using a particular evaluation metric. There are
two functions which helps to achieve this:

 - `pt.GridScan()` exhaustively evaluates all possibile parameters settings and computes evaluation measures.
 - `pt.GridSearch()` applies GridScan, and determines the most effective parameter setting for a given evaluation measure.
 - `pt.KFoldGridSearch()` applies GridSearch on different folds, in order to determine the most effective parameter setting for a given 
   evaluation measure on the training topics for each fold. The results on the test topics are returned.

All of these functions are designed to have an API very similar to pt.Experiment().

Pre-requisites
==============

GridScan makes several assumption:
 - the parameters that you wish to tune are available as instance attributes within the transformers, or that the transformer responds suitably to `set_parameter()`. 
 - changing the relevant parameters has an impact upon subsequent calls to `transform()`.

Note that transformers implemented using pt.apply functions cannot address the second requirement, as any parameters are captured 
naturally within the closure, and not as instances attributes of the transformer.

Scanning and Searching API
==========================

.. autofunction:: pyterrier.GridScan()

.. autofunction:: pyterrier.GridSearch()

Examples
========

Tuning BM25
~~~~~~~~~~~

When using BatchRetrieve, the `b` parameter of the BM25 weighting model can be controled using the "c" control. 
We must give this control an initial value when contructing the BatchRetrieve instance. Thereafter, the GridSearch
parameter dictionary can be constructed by refering to the instance of transformer that has that parameter::

    BM25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c" : 0.75})
    pt.GridSearch(
        BM25,  
        {BM25 : {"c" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]}}
        train_topics, 
        train_qrels, 
        "map")

Terrier's BM25 also responds to controls named `"bm25.k_1"` and  `"bm25.k_3"`.

Tuning BM25 and RM3
~~~~~~~~~~~~~~~~~~~

The query expansion transformer in pt.rewrite have parameters controlling the number of feedback documents and expansion terms, namely:
 - fb_terms -- the number of terms to add to the query.
 - fb_docs -- the size of the pseudo-relevant set. 


A full tuning of BM25 and RM3 can be achieved as thus::

    bm25_for_qe = pt.BatchRetrieve(index, wmodel="BM25", controls={"c" : 0.75})
    rm3 = pt.rewrite.RM3(index, fb_terms=10, fb_docs=3)
    pipe_qe = bm25_for_qe >> rm3 >> bm25_for_qe

    param_map = {
            bm25_for_qe : { "c" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]},
            rm3 : { 
                "fb_terms" : list(range(1, 12, 3)), 
                "fb_docs" : list(range(2, 30, 6))
            }
    }
    pipe_qe = pt.GridSearch(pipe_qe, param_map, train_topics, train_qrels)
    pt.Experiment([pipe_qe], test_topics, test_qrels, ["map"])

Using Multiple Folds
====================

.. autofunction:: pyterrier.KFoldGridSearch()

Parallelisation
===============

GridScan, GridSearch and KFoldGridSearch can all be accelerated using parallelisation to conduct evalutions of different parameter
settings in parallel. Both accept `jobs` and `backend` kwargs, which define the number of backend processes to
conduct, and the parallelisation backend. For instance::

    pt.GridSearch(pipe_qe, param_map, train_topics, train_qrels, jobs=10)
    
This incantation will fork 10 Python processes to run the different settings in parallel. Each process will load
a new instance of any large data structures, such as Terrier indices, so your machines must have sufficient memory
to load 10 instances of the index. 

The Ray backend offers parallelisation across multiple machines. For more information, see :ref:`parallel`.