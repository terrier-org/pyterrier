.. _pyterrier.model.DataFrameBuilder:

DataFrameBuilder
==================================================

:class:`~pyterrier.model.DataFrameBuilder` provides a simple way to progressively build a DataFrame in a
:class:`~pyterrier.Transformer`.

Usage
---------------------------------------------------------

A common pattern in `Transformer` implementation builds up an intermediate representation of the output DataFrame,
but this can be a bit clunky, as shown below:

.. code-block:: python
    :caption: Building a DataFrame without :class:`~pyterrier.model.DataFrameBuilder`

    class MyTransformer(pt.Transformer):
        def transform(self, inp: pd.DataFrame):
            result = {
                'qid': [],
                'query': [],
                'docno': [],
                'score': [],
            }
            for qid, query in zip(inp['qid'], inp['query']):
                docnos, scores = self.some_function(qid, query)
                result['qid'].append([qid] * len(docnos))
                result['query'].append([query] * len(docnos))
                result['docno'].append(docnos)
                result['score'].append(scores)
            result = pd.DataFrame({
                'qid': np.concatenate(result['qid']),
                'query': np.concatenate(result['query']),
                'docno': np.concatenate(result['docno']),
                'score': np.concatenate(result['score']),
            })
            return result

:class:`~pyterrier.model.DataFrameBuilder` simplifies the process of building a DataFrame by removing lots of
the boilerplate. It also automatically handles various types and ensures that all columns end up with the same
length. The above example can be rewritten with `pt.model.DataFrameBuilder` as follows:

.. code-block:: python
    :caption: Building a DataFrame using :class:`~pyterrier.model.DataFrameBuilder`

    class MyTransformer(pt.Transformer):
        def transform(self, inp: pd.DataFrame):
            result = pt.model.DataFrameBuilder(['qid', 'query', 'docno', 'score'])
            for qid, query in zip(inp['qid'], inp['query']):
                docnos, scores = self.some_function(qid, query)
                result.extend({
                    'qid': qid, # automatically repeats to the length of this batch
                    'query': query, # ditto
                    'docno': docnos,
                    'score': scores,
                })
            return result.to_df()


You'll often want to *extend* the set of columns passed to a transformer, rather than replacing them.
For instance, in the previous example, perhaps ``inp`` includes a ``my_special_data`` field added by
another transformer that should be passed along to the following step. If you pass the original input
frame to ``to_df``, the function will try to merge the original frames together. The columns from the
merged frame will appear before any new columns.

.. code-block:: python
    :caption: Merging the input frame's data with :class:`~pyterrier.model.DataFrameBuilder`

    class MyTransformer(pt.Transformer):
        def transform(self, inp: pd.DataFrame):
            result = pt.model.DataFrameBuilder(['docno', 'score'])
            for qid, query in zip(inp['qid'], inp['query']):
                docnos, scores = self.some_function(qid, query)
                result.extend({
                    'docno': docnos,
                    'score': scores,
                })
            return result.to_df(inp)

.. note::

    The merging functionality assumes that ``extend`` is called once per row in the original frame,
    in the same order as the original frame.

    If this is not the case, you can manually provide an ``_index`` field each time you call ``extend``.
    where ``_index`` is the integer index of the row in the original frame.


API Documentation
---------------------------------------------------------

.. autoclass:: pyterrier.model.DataFrameBuilder
    :members:
