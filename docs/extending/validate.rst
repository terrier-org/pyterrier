Input Validation
===================================

DataFrame Validation
------------------------------------

When writing a transformer, it's a good idea to check its inputs to make sure they are compatible
before you start using it. ``pt.validate`` provides functions for this.

.. code-block:: python
    :caption: DataFrame input validation in a Transformer

    def MyTransformer(pt.Transformer):
        def transform(self, inp: pd.DataFrame):
            # e.g., expects a query frame with query_vec
            pt.validate.query_frame(inp, extra_columns=['query_vec'])
            # raises an error if the specification doesn't match

=========================================================  ===============================  =======================
Function                                                   Must have column(s)              Must NOT have column(s)
=========================================================  ===============================  =======================
``pt.validate.query_frame(inp, extra_columns=...)``        qid + ``extra_columns``          docno
``pt.validate.document_frame(inp, extra_columns=...)``     docno + ``extra_columns``        qid
``pt.validate.result_frame(inp, extra_columns=...)``       qid + docno + ``extra_columns``  
``pt.validate.columns(inp, includes=..., excludes=...)``   ``includes``                     ``excludes``
=========================================================  ===============================  =======================


.. note::
    Besides providing helpful error messages to users, these methods also help perform inspection of pipelines, e.g.,
    for drawing pipeline schematic representations of pipelines and ensuring that transformers are compatible before
    running them.


Iterable validation
------------------------------------

For indexing pipelines that accept iterators, it checks the fields of the first element. You need
to first wrap `inp` in ``pt.utils.peekable()`` for this to work.

.. code-block:: python
    :caption: Iterable input validation in a Transformer

    my_iterator = [{'docno': 'doc1'}, {'docno': 'doc2'}, {'docno': 'doc3'}]
    my_iterator = pt.utils.peekable(my_iterator)
    pt.validate.columns_iter(my_iterator, includes=['docno']) # passes
    pt.validate.columns_iter(my_iterator, includes=['docno', 'toks']) # raises errors

Advanced Usage
------------------------------------------

Sometimes a transformer has multiple acceptable input specifications, e.g., if
it can act as either a retriever (with a query input) or re-ranker (with a result input).
In this case, you can specify multiple possible configurations in a ``with pt.validate.any(inpt) as v:`` block:

.. code-block:: python
    :caption: Validation with multiple acceptable input specifications

    def MyTransformer(pt.Transformer):
        def transform(self, inp: pd.DataFrame):
            # e.g., expects a query frame with query_vec
            with pt.validate.any(inp) as v:
                v.query_frame(extra_columns=['query'], mode='retrieve')
                v.result_frame(extra_columns=['query', 'text'], mode='rerank')
            # raises an error if ALL specifications do not match
            # v.mode is set to the FIRST specification that matches
            if v.mode == 'retrieve':
                ...
            if v.mode == 'rerank':
                ...

API Documentation
---------------------------------------------------------

.. autofunction:: pyterrier.validate.columns

.. autofunction:: pyterrier.validate.query_frame

.. autofunction:: pyterrier.validate.result_frame

.. autofunction:: pyterrier.validate.document_frame

.. autofunction:: pyterrier.validate.any

.. autofunction:: pyterrier.validate.columns_iter

.. autofunction:: pyterrier.validate.query_iter

.. autofunction:: pyterrier.validate.document_iter

.. autofunction:: pyterrier.validate.result_iter

.. autofunction:: pyterrier.validate.any_iter
