PyTerrier Transformers
----------------------

PyTerrier's retrieval architecture is based on three concepts:

 - dataframes with pre-defined types (each with a minimum set of known attributes), as detailed in the data amodel.
 - the *transformation* of those dataframes by standard information retrieval operations, defined as transformers.
 - the compsition of transformers, supported by the operators defined on transformers.

In essence, a PyTerrier transformer is a class with a `transform()` method, which takes as input a dataframe, and changes it,
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
When `fit()` is called on a pipeline, all estimators (transformers that also have a `fit()` method, as specified by 
`EstimatorBase`) within the pipeline are fitted, in turn. This allows one (or more) stages of learning to be 
integrated into a retrieval pipeline.

Transformer base classes
========================

TransformerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class is the base class for all transformers.

.. autoclass:: pyterrier.transformer.TransformerBase
    :members:

Moreover, by extending TransformerBase, all transformer implementations gain the necessary "dunder" methods (e.g. `__rshift__()`)
to support the transformer operators (`>>`, `+` etc). 

EstimatorBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class exposes a `fit()` method that can be used for transformers that can be trained.

.. autoclass:: pyterrier.transformer.EstimatorBase
    :members:

Internal transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A significant number of transformers are defined in pyterrier.transformer to implement operators etc. Its is not expected
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
| `%`    | feature-union    | FeatureUnionPipeline      |
+--------+------------------+---------------------------+
| `^`    | concatenate      | ConcatenateTransformer    |
+--------+------------------+---------------------------+
| `~`    | cache            | ChestCacheTransformer     |
+--------+------------------+---------------------------+


Writing your own transformer
============================

The first step to writing your own transformer for your own code is to consider the type of change being applied.
Several common transformations are supported through the functions in the :ref:`pyterrier.apply` module. See the 
:ref:`pyterrier.apply` documentation.

However, if your transformer has state, such as an expensive model to be loaded at startup time, you may want to 
extend TransformerBase directly. 

Here are some hints for writing Transformers:

 - If you return ranked results, use `pt.model.add_ranks()` to add the rank column.
 - If your approach can be trained, you should extends EstimatorBase, and implement the `fit()` method.
