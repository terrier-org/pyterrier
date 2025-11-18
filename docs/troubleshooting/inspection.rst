Troubleshooting Inspection and Verification
-------------------------------------------

Most existing pipelines will work out-of-the-box with inspection (pt.inspect) and pt.Experiment verification.

However, if your pipeline uses custom components, you will 

```
/pyterrier/_evaluation/_validation.py:41: UserWarning: Transformer XXX ((AA >> BB)) at position Z failed to validate: 
Cannot determine outputs for (AA >> BB) with inputs: ['qid', 'query'] - if your pipeline works, set validate='ignore' 
to remove this warning, or add transform_output method to the transformers in this pipeline to clarify how it works
```

As the warning suggests, you can either set `validate='ignore'` when calling ``pt.Experiment``::
    
    pt.Experiment([AA >> BB], 
        topics,
        qrels,
        metrics=["map", "ndcg"],
        validate='ignore')

However, this may hide potential issues in your pipeline. It also means that your pipeline will not produce
useful schematics.

Visualizing the pipeline
===========================

A first step in debugging is to visualize the pipeline to see what is happening. 
You can do this by simply entering the pipeline alone in a notebook cell::

    AA >> BB

This will produce a schematic of the pipeline, showing how data flows through it.

<TODO ADD EXAMPLE FIGURE>

By hovering your cursor over each component, you can see what inputs and outputs are expected.
For any transformer, inspection needs to be able to determine what inputs and output it expects.
We discuss each of these below.

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

    TOOD: add link to transform_inputs documentation

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
    

Due to risks and maintanence burden in ensuring that transform and transform_outputs behave identically, 
it is recommended to only implement transform_outputs when calling the transformer with an empty DataFrame 
to inspect the behavior is undesireable, e.g., if calling the transformer is expensive, even for empty inputs,
or in the case of iter-dict inputs.