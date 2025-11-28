Troubleshooting Inspection and Verification
-------------------------------------------

Most existing pipelines will work out-of-the-box with inspection (pt.inspect) and pt.Experiment verification.

However, if your pipeline uses custom components, ``pt.Experiment`` may produce warnings.

There are two possibilities:
1. The pipeline components are inspectable, but the pipeline as whole does not pass validation.

.. code-block:: none
    /pyterrier/_evaluation/_validation.py:41: UserWarning: "Transformer XXX (AA) at position Z does not produce 
    all required columns ZZ, found only AA.


This suggests a mistake in your pipeline formulation - e.g. you are passing a Q dataframe of queries, when
ranked results (R) are expected. Such pipelines will typically produce an error when the experiment is
executed. 

2. One of the pipeline components is not inspectable.

.. code-block:: none

    /pyterrier/_evaluation/_validation.py:41: UserWarning: Transformer XXX ((AA >> BB)) at position Z failed to validate: 
    Cannot determine outputs for (AA >> BB) with inputs: ['qid', 'query'] - if your pipeline works, set validate='ignore' 
    to remove this warning, or add transform_output method to the transformers in this pipeline to clarify how it works


In both cases, if you are sure your pipeline works, as the warning suggests, you can set `validate='ignore'` 
when calling ``pt.Experiment``::
    
    pt.Experiment([AA >> BB], 
        topics,
        qrels,
        metrics=["map", "ndcg"],
        validate='ignore')

However, this can hide potential issues in your pipeline. It also means that your pipeline will not produce
useful schematics.

Visualizing the pipeline
===========================

A first step in debugging is to visualize the pipeline to see what is happening. 
You can do this by simply entering the pipeline alone in a notebook cell::

    AA >> BB

This will produce a schematic of the pipeline, showing how data flows through it.

.. image:: pipeline-not-validating.png
    :alt: Invalid pipeline schematic

By hovering your cursor over each component, you can see what inputs and outputs are expected.

.. image:: pipeline-not-validating-mouseover.png
    :alt: Invalid pipeline schematic showing data flow between components

Missing columns will be immediately viewable (its the `text` column in this case, usually resolvable
by using a ``pt.text.get_text()`` transformer).

For any transformer, inspection needs to be able to determine what inputs and output it expects.
If the transformer cannot be inspected, the schematic will show "?" for the dataframe type, and
a mouseover will produce "Unknown/incompatible columns".

.. image:: pipeline-not-inspecting-mouseover.png
    :alt: Pipeline that cannot be inspected.

We discuss how inputs and output inspection works and how to fix your transformer.

Determining inputs
=====================

Hints:

1. If your transformer accepts a DataFrame and returns a DataFrame, use pt.validate to 
    validate the input dataframe has the correct schema. This will enable inspection to determine the inputs.
    Example::

        def transform(inp : pd.DataFrame) -> pd.DataFrame:
            # expects ['qid', 'query']
            pt.validate.query_frame(input_columns, ['query'])
            ... # rest of yor transformer implementation

    TODO add link to validation documentation

2. If your transformer uses the iter-dict data types (e.g. ``transform_iter()``), then you will need to add a
    ``transform_inputs()`` method to your transformer that indicates what inputs it expects. For example::

        def transform_inputs(self) -> List[List[str]]:
            return [['qid', 'query']]

    See also:
     - :meth:`pyterrier.inspect.transformer_inputs`

    A transformer can respond to mulitple input configurations, as demonstrated in the ``List[List[str]]`` type.
    For example, a transformer that can be use on both queries and retrieved documents may have a transform_inputs
    as follows::

        def transform_inputs(self) -> List[List[str]]:
            return [
                ['qid', 'query'],
                ['qid', 'query', 'docno']
            ]

3. Apply transformers (using ``pt.apply``) can use the `required_columns` argument to indicate what inputs 
    they expect::
    
    _rewriting_fn = lambda row: ... # your rewriting function here
    p1 = pt.apply.query(_rewriting_fn, required_columns=['qid', 'query'])

    Different apply transformers have default assumptions for `required_columns`.


Determining Outputs
======================

Hints:

1. If your transformer returns a DataFrame, when provided with an empty DataFrame with the correct schema
    as input, then it should return an empty DataFrame with the correct output schema.

    Example::

        def transform(inp : pd.DataFrame) -> pd.DataFrame:
            pt.validate.query_frame(inp, ['query'])
            if not len(inp):
                return pd.DataFrame([], columns=inp.columns.tolist() + ['docno', 'score', 'rank'])
            ... # rest of yor transformer implementation

2. If your transformer uses the iter-dict data types (e.g. ``transform_iter()``), then you will need to add a
    ``transform_outputs()`` method to your transformer that indicates what outputs it produces. If you resort to
    ``transform_outputs()`` you MUST use pt.validate on the input columns.

    For example::

        def transform_outputs(self, input_columns : List[str]) -> List[str]:
            pt.validate.query_frame(input_columns, ['query'])
            return input_columns + ['docno', 'score', 'rank']
    
3. For pt.apply-based transformers, use the transform_outputs= kwarg to allow overriding of the ``transform_outputs()``

Due to risks and maintanence burden in ensuring that ``transform()`` and ``transform_outputs()`` behave identically, 
it is recommended to only implement transform_outputs when calling the transformer with an empty DataFrame 
to inspect the behavior is undesireable, e.g., if calling the transformer is expensive, even for empty inputs,
or in the case of iter-dict inputs.