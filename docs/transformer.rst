.. _pt.transformer:

PyTerrier Transformers
----------------------

PyTerrier's retrieval architecture is based on three concepts:

- dataframes with pre-defined types (each with a minimum set of known attributes), as detailed in the data model.
- the *transformation* of those dataframes by standard information retrieval operations, defined as transformers.
- the compsition of transformers, supported by the operators defined on transformers.

In essence, a PyTerrier transformer is a class that implemented one or more of two methods:

1. a ``transform()`` method, which takes as input a dataframe, and changes it, before returning it,

and/or

2.  a ``transform_iter()`` method, which takes as input a iterable of dictionaries ("iter-dict"), and changes it, before returning or yielding it.

One of these methods must be implemented. If one is implemented, the other will change the input type and call the other - for instance,
if a transformer's ``transform_iter()`` method is called, but only ``transform()`` is implemented, the iter-dict will be used to construct a
dataframe,  ``transform()`` is called, and the resulting dataframe transformed back into an iter-dict. 

Depending on the expected input and output column of a transformer, they can be described as following into different categories.

+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+
+ Input | Output  | Cardinality | Example             | Concrete Transformer Example                                                             |
+=======+=========+=============+=====================+==========================================================================================+
|   D   |    D    |   1 to 1    | Document expansion  | `Doc2Query <https://github.com/terrierteam/pyterrier_doc2query>`_                        |
+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+
|   Q   |    Q    |   1 to 1    | Query rewriting     | `pt.rewrite.SDM()`                                                                       |
+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+
|   Q   |  Q x D  |   1 to N    | Retrieval           | `pt.terrier.Retriever()`                                                                 |
+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+
| Q x D |    Q    |   N to 1    | Query expansion     | `pt.rewrite.RM3()`                                                                       |
+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+
| Q x D |  Q x D  |   1 to 1    | Re-ranking          | `pt.apply.doc_score()`                                                                   |
+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+
| Q x D |  Q x Df |   1 to 1    | Feature scoring     | `pt.terrier.FeaturesRetriever()`                                                         |
+-------+---------+-------------+---------------------+------------------------------------------------------------------------------------------+

.. hint:: 
    When writing transformers, it's a good idea to validate the inputs to make sure they contain the values you expect.
    See :ref:`pyterrier.validate` for more details.

Optimisation
============

Some operators applied to transformer can be optimised by the underlying search engine - for instance, cutting a ranking 
earlier. So while the following two pipelines are semantically equivalent, the latter might be more efficient::

    pipe1 = pt.terrier.Retrieve(index, "BM25") % 10
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
    :special-members: __call__


Default Method
,,,,,,,,,,,,,,,,

You can invoke a transformer's transfor method simply by calling the default method. If ``t`` is a transformer::

  df_in = pt.new.queries(['test query'], qid=['q1'])
  df_out = t.transform(df_in)
  df_out = t(df_in)

The default method will also detect iterable dictionaries, and pass those directly to ``transform_iter()`` 
(which usually calls ``transform()`` if ``transform_iter()`` has not been impelmented). So the following 
expression is equivalent to the examples in the previous code block, except that df_out will contain an iter-dict::

  df_out = t([{'qid' : 'q1', 'query' : 'test query'}])

This can be more succinct than creating new dataframes for testing transformer implementations.


Operator Support
,,,,,,,,,,,,,,,,

By extending Transformer, all transformer implementations gain the necessary "dunder" methods (e.g. ``__rshift__()``)
to support the transformer operators (`>>`, `+` etc). NB: This class used to be called ``pyterrier.transformer.TransformerBase``



.. _pt.transformer.estimator:

Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This base class exposes a ``fit()`` method that can be used for transformers that can be trained.

.. autoclass:: pyterrier.Estimator
    :members:

The ComposedPipeline implements ``fit()``, which applies the interimediate transformers on the specified training (and validation) topics, and places
the output into the ``fit()`` method of the final transformer.

Indexer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This base  class exposes a ``index()`` method that can be used for transformers that create an index.

.. autoclass:: pyterrier.Indexer
    :members:

The ComposedPipeline also implements ``index()``, which applies the interimediate transformers on the specified documents to be indexed, and places
the output into the ``index()`` method of the final transformer.

Internal transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A significant number of transformers are defined in pyterrier._ops to implement operators etc. Its is not expected
to use these directly but they are listed for completeness.

+--------+------------------+---------------------------+
| Symbol | Name             | Implementing transformer  |
+========+==================+===========================+
| `>>`   | compose/then     | pt._ops.Compose           |
+--------+------------------+---------------------------+
| `|`    | set-union        | pt._ops.SetUnion          |
+--------+------------------+---------------------------+
| `&`    | set-intersection | pt._ops.SetIntersection   |
+--------+------------------+---------------------------+
| `+`    | linear           | pt._ops.CombSum           | 
+--------+------------------+---------------------------+
| `+`    | scalar-product   | pt._ops.ScalarProduct     | 
+--------+------------------+---------------------------+
| `%`    | rank-cutoff      | pt._ops.RankCutoff        |
+--------+------------------+---------------------------+
| `**`   | feature-union    | pt._ops.FeatureUnion      |
+--------+------------------+---------------------------+
| `^`    | concatenate      | pt._ops.Concatenate       |
+--------+------------------+---------------------------+


.. _indexing_pipelines:

Indexing Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transformers can be chained to create indexing pipelines. The last element in the chain is assumed to be an indexer like 
IterDictIndexer - it should implement an ``index()`` method like pt.Indexer. For instance::

    docs = [ {"docno" : "1", "text" : "a" } ] 
    indexer = pt.text.sliding() >> pt.IterDictIndexer()
    indexer.index(docs)

This is implemented by several methods:

- The last stage of the pipeline should have an ``index()`` method that accepts an iterable of dictionaries

- Compose has a special ``index()`` method that breaks the input iterable into chunks (the size of 
  chunks can be altered by a batch_size kwarg) and passes those through the intermediate pipeline stages (i.e. all but the last).

- In the intermediate pipeline stages, the ``transform_iter()`` method is called - by default this instantiates a DataFrame
  on batch_size records, which is passed to ``transform()``.

- These are passed to ``index()`` of the last pipeline stage.

Writing your own transformer
============================

The first step to writing your own transformer for your own code is to consider the type of change being applied.
Several common transformations are supported through the functions in the :ref:`pyterrier.apply` module. See the 
:ref:`pyterrier.apply` documentation.

However, if your transformer has state, such as an expensive model or index data structure to be loaded at startup time, 
you may want to extend ``pt.Transformer`` directly. 

Here are some hints for writing Transformers:
 - Except for an indexer, you should implement a ``transform()`` and/or ``transform_iter()`` method.
 - If your approach ranks results, use ``pt.model.add_ranks()`` to add the rank column. (``pt.apply.doc_score`` will call add_ranks automatically). 
 - If your approach can be trained, your transformer should extend Estimator, and implement the ``fit()`` method.
 - If your approach is an indexer, your transformer should extend Indexer and implement ``index()`` method.

Optimisation of Transfomer Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``.compile()`` on a transformer or a pipeline of transformer, there is an opportunity to improve the efficiency of the pipeline,
while ensuring that the semantics remain unchanged. Implementors of a transformer wishing to support such optimisations have a number
of mechanisms open to them.

Firstly, the ``.compile()`` transformer's method can be overridden to return a new transformer instance that may be more efficient.

Secondly, the transformer can be fused with adjacent transformers. For instance, a retriever may be fused with a rank-cutoff operator,
such that the rank-cutoff is applied at retrieval rather than after. Fusion is controlled by `protocol methods <https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols>`_,
which determine how the transformer can be fused.

.. autoclass:: pyterrier.transformer.SupportsFuseRankCutoff
    :members:

.. autoclass:: pyterrier.transformer.SupportsFuseLeft
    :members:

.. autoclass:: pyterrier.transformer.SupportsFuseRight
    :members:

.. autoclass:: pyterrier.transformer.SupportsFuseFeatureUnion
    :members:

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
  baselineT = pt.Transformer.from_df(res, uniform=True)

  Q1 = pt.new.queries("test query", qid="Q1")
  resQ1 = baselineT.transform(Q1)
