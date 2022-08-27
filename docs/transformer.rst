.. _pt.transformer:

PyTerrier Transformers
----------------------

PyTerrier's retrieval architecture is based on three concepts:

 - dataframes with pre-defined types (each with a minimum set of known attributes), as detailed in the data model.
 - the *transformation* of those dataframes by standard information retrieval operations, defined as transformers.
 - the compsition of transformers, supported by the operators defined on transformers.

In essence, a PyTerrier transformer is a class with a ``transform()`` method, which takes as input a dataframe, and changes it,
before returning it. 

+-------+---------+-------------+------------------+------------------------------+
+ Input | Output  | Cardinality | Example          | Concrete Transformer Example |
+=======+=========+=============+==================+==============================+
|   Q   |    Q    |   1 to 1    | Query rewriting  | `pt.rewrite.SDM()`           |
+-------+---------+-------------+------------------+------------------------------+
|   Q   |  Q x D  |   1 to N    | Retrieval        | `pt.BatchRetrieve()`         |
+-------+---------+-------------+------------------+------------------------------+
| Q x D |    Q    |   N to 1    | Query expansion  | `pt.rewrite.RM3()`           |
+-------+---------+-------------+------------------+------------------------------+
| Q x D |  Q x D  |   1 to 1    | Re-ranking       | `pt.apply.doc_score()`       |
+-------+---------+-------------+------------------+------------------------------+
| Q x D |  Q x Df |   1 to 1    | Feature scoring  | `pt.FeaturesBatchRetrieve()` |
+-------+---------+-------------+------------------+------------------------------+

Optimisation
============

Some operators applied to transformer can be optimised by the underlying search engine - for instance, cutting a ranking 
earlier. So while the following two pipelines are semantically equivalent, the latter might be more efficient::

    pipe1 = BatchRetrieve(index, "BM25") % 10
    pipe2 = pipe1.compile()

Fitting
=======
When `fit()` is called on a pipeline, all estimators (transformers that also have a ``fit()`` method, as specified by 
`Estimator`) within the pipeline are fitted, in turn. This allows one (or more) stages of learning to be 
integrated into a retrieval pipeline.  See :ref:`pyterrier.ltr` for examples.

When calling fit on a composed pipeline (i.e. one created using the ``>>`` operator), this will will call ``fit()`` on any 
estimators within that pipeline.

Transformer base classes
========================

Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class is the base class for all transformers.

.. autoclass:: pyterrier.Transformer
    :members:

Moreover, by extending Transformer, all transformer implementations gain the necessary "dunder" methods (e.g. ``__rshift__()``)
to support the transformer operators (`>>`, `+` etc). NB: This class used to be called ``pyterrier.transformer.TransformerBase``

.. _pt.transformer.estimator:

Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class exposes a ``fit()`` method that can be used for transformers that can be trained.

.. autoclass:: pyterrier.Estimator
    :members:

The ComposedPipeline implements ``fit()``, which applies the interimediate transformers on the specified training (and validation) topics, and places
the output into the ``fit()`` method of the final transformer.

Internal transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A significant number of transformers are defined in pyterrier.ops to implement operators etc. Its is not expected
to use these directly but they are documented for completeness.

+--------+------------------+---------------------------+
| Symbol | Name             | Implementing transformer  |
+========+==================+===========================+
| `>>`   | compose/then     | ComposedPipeline          |
+--------+------------------+---------------------------+
| `|`    | set-union        | SetUnionTransformer       |
+--------+------------------+---------------------------+
| `&`    | set-intersection | SetIntersectionTransformer|
+--------+------------------+---------------------------+
| `+`    | linear           | CombSumTransformer        | 
+--------+------------------+---------------------------+
| `+`    | scalar-product   | ScalarProductTransformer  | 
+--------+------------------+---------------------------+
| `%`    | rank-cutoff      | RankCutoffTransformer     |
+--------+------------------+---------------------------+
| `**`   | feature-union    | FeatureUnionPipeline      |
+--------+------------------+---------------------------+
| `^`    | concatenate      | ConcatenateTransformer    |
+--------+------------------+---------------------------+
| `~`    | cache            | ChestCacheTransformer     |
+--------+------------------+---------------------------+


.. _indexing_pipelines:

Indexing Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transformers can be chained to create indexing pipelines. The last element in the chain is assumed to be an indexer like 
IterDictIndexer - it should implement an ``index()`` method like IterDictIndexerBase. For instance::

    docs = [ {"docno" : "1", "text" : "a" } ] 
    indexer = pt.text.sliding() >> pt.IterDictIndexer()
    indexer.index(docs)

This is implemented by several methods:

 - The last stage of the pipeline should have an ``index()`` method that accepts an iterable of dictionaries
 - ComposedPipeline has a special ``index()`` method that breaks the input iterable into chunks (the size of 
   chunks can be altered by a batch_size kwarg) and passes those through the intermediate pipeline stages (i.e. all but the last).
 - In the intermediate pipeline stages, the ``transform_iter()`` method is called - by default this instantiates a DataFrame
   on batch_size records, which is passed to ``transform()``.
 - These are passed to ``index()`` of the last pipeline stage.

Writing your own transformer
============================

The first step to writing your own transformer for your own code is to consider the type of change being applied.
Several common transformations are supported through the functions in the :ref:`pyterrier.apply` module. See the 
:ref:`pyterrier.apply` documentation.

However, if your transformer has state, such as an expensive model to be loaded at startup time, you may want to 
extend ``pt.Transformer`` directly. 

Here are some hints for writing Transformers:
 - Except for an indexer, you should implement a ``transform()`` method.
 - If your approach ranks results, use ``pt.model.add_ranks()`` to add the rank column. (``pt.apply.doc_score`` will call add_ranks automatically). 
 - If your approach can be trained, your transformer should extend Estimator, and implement the ``fit()`` method.
 - If your approach is an indexer, your transformer should extend IterDictIndexerBase and implement ``index()`` method.


Mocking Transformers from DataFrames
====================================

You can make a Transformer object from dataframes. For instance, a unifom transformer will always return the input
dataframe any time ``transform()`` is called::

  df = pt.new.ranked_documents([[1,2]])
  uniformT = pt.Transformer.from_df(df, uniform=True)
  # uniformT.transform() always returns df, regardless of arguments

You can also create a Transformer object from existing results, e.g. saved on disk using ``pt.io.write_results()`` 
etc. The resulting "source transformer" will return all results by matching on the qid of the input::

  res = pt.io.read_results("/path/to/baseline.res.gz")
  baselineT = pt.Transformer.from_df(df, uniform=True)

  Q1 = pt.new.queries("test query", qid="Q1")
  resQ1 = baselineT.transform(Q1)
