Tuning Transformer Pipelines
----------------------------

Many approaches will have parameters that require tuning. PyTerrier helps to achieve this by proving a grid 
evaluation functionality that can tune one or more parameters using a particular evaluation metric. There are
two functions which helps to achieve this:

- ``pt.GridScan()`` exhaustively evaluates all possibile parameters settings and computes evaluation measures.
- ``pt.GridSearch()`` applies GridScan, and determines the most effective parameter setting for a given evaluation measure.
- ``pt.KFoldGridSearch()`` applies GridSearch on different folds, in order to determine the most effective parameter setting for a given  evaluation measure on the training topics for each fold. The results on the test topics are returned.

All of these functions are designed to have an API very similar to pt.Experiment().

Pre-requisites
==============

GridScan makes several assumptions:
 - the parameters that you wish to tune are available as instance attributes within the transformers, or that the transformer responds suitably to ``set_parameter()``. 
 - changing the relevant parameters has an impact upon subsequent calls to ``transform()``.

Note that transformers implemented using pt.apply functions cannot address the second requirement, as any parameters are captured 
naturally within the closure, and not as instances attributes of the transformer.

Parameter Scanning and Searching API
====================================

.. autofunction:: pyterrier.GridScan()

.. autofunction:: pyterrier.GridSearch()

Examples
========

Tuning BM25
~~~~~~~~~~~

When using Retriever, the `b` parameter of the BM25 weighting model can be controled using the "bm25.b" control. 
We must give this control an initial value when contructing the Retriever instance. Thereafter, the GridSearch
parameter dictionary can be constructed by refering to the instance of transformer that has that parameter::

    BM25 = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b" : 0.75})
    pt.GridSearch(
        BM25,  
        {BM25 : {"c" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]}}
        train_topics, 
        train_qrels, 
        "map")

Terrier's BM25 also responds to controls named `"bm25.k_1"` and  `"bm25.k_3"`, such that all three controls can be tuned concurrently::

    BM25 = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b" : 0.75, "bm25.k_1": 0.75, "bm25.k_3": 0.75})
    pt.GridSearch(
        BM25,  
        {BM25: {"bm25.b"  : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],
                "bm25.k_1": [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 2],
                "bm25.k_3": [0.5, 2, 4, 6, 8, 10, 12, 14, 20]
        }}
        train_topics, 
        train_qrels, 
        "map")

Tuning BM25 and RM3
~~~~~~~~~~~~~~~~~~~

The query expansion transformer in pt.rewrite have parameters controlling the number 
of feedback documents and expansion terms, namely:

 - ``fb_terms`` -- the number of terms to add to the query.
 - ``fb_docs`` -- the size of the pseudo-relevant set. 


A full tuning of BM25 and RM3 can be achieved as thus::

    bm25_for_qe = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b" : 0.75})
    rm3 = pt.rewrite.RM3(index, fb_terms=10, fb_docs=3)
    pipe_qe = bm25_for_qe >> rm3 >> bm25_for_qe

    param_map = {
            bm25_for_qe : { "bm25.b" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]},
            rm3 : { 
                "fb_terms" : list(range(1, 12, 3)), # makes a list of 1,3,6,7,12
                "fb_docs" : list(range(2, 30, 6))   # etc.
            }
    }
    pipe_qe = pt.GridSearch(pipe_qe, param_map, train_topics, train_qrels)
    pt.Experiment([pipe_qe], test_topics, test_qrels, ["map"])

Tuning BM25F
~~~~~~~~~~~~

BM25F and PL2F are field-based weighting models which appler per-field normalisation. These have at least 
2 parameters for each field: one controlling the term frequency vs. length normalisation of that field, 
and one for controlling the importance of the per-field normalised term frequency. The general form of 
BM25F and PL2F are as follows:

.. math::

   score(d,Q) = \text{weight}(tfn)

where :math:`tfn` is defined as the weighted average of normalised lengths across each field.

.. math::

   tfn = \sum_f w_f \cdot \text{norm}(tf_f, l_f, c_f)

In the above, :math:`tf_f` and :math:`l_f` are respectively the term frequency in field :math:`f`
and the length of that field.  :math:`w_f` and :math:`c_f` are respectively the field weights and
normalisation parameter for that field.

In Terrier, for both the `BM25F <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/BM25F.html>`_ 
and `PL2F <http://terrier.org/docs/current/javadoc/org/terrier/matching/models/PL2F.html>`_ weighting 
models, the relevant configuration controls are a ``'c.'`` control for each  field (controlling normalisation) 
and a ``'w.'`` control for each field (controlling the weight). Fields are numbered, starting from 0.

The following is an example of scanning the parameters of BM25F for an index with two fields::

    from numpy import arange  # gives a list of values in an interval

    # check your index has exactly 2 fields
    assert 2 == index.getCollectionStatistics().getNumberOfFields()

    # instantiate Retriever for BM25F
    bm25f = pt.terrier.Retriever(
        index, 
        wmodel = 'BM25F', 
        controls = {'w.0' : 1, 'w.1' : 1, 'c.0' : 0.4, 'c.1' : 0.4}
    )

    # now attempt all parameter values
    pt.GridScan(
        bm25f, 
        # you can name more parameters here and their values to try
        {bm25f : {
            'w.0' : arange(0, 1.1, 0.1),
            'w.1' : arange(0, 1.1, 0.1),
            'c.0' : arange(0, 1.1, 0.1),
            'c.1' : arange(0, 1.1, 0.1),
        }},
        topics,
        qrels,
        ['map']
    )
    # GridScan returns a table of MAP values for all attempted parameter settings


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
