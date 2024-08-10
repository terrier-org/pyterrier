.. _pyterrier.apply:

pyterrier.apply - Custom Transformers
-------------------------------------

PyTerrier pipelines are easily extensible through the use of apply functions.
These are inspired by the `Pandas apply() <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html>`_
method, which allow to apply a function to each row of a dataframe. Instead, 
in PyTerrier, the apply methods allow to construct pipeline transformers 
to address common use cases by using custom functions (including Python lambda
functions) to easily transform inputs.

The table below lists the main classes of transformation in the PyTerrier data 
model, as well as the appropriate apply method to use in each case. In general,
if there is a one-to-one mapping between the input and the output, then the specific
pt.apply methods should be used (i.e. ``query()``, ``doc_score()``, ``.doc_features()``).
If the cardinality of the dataframe changes through applying the transformer, 
then ``generic()`` or ``by_query()`` must be applied.

In particular, through the use of ``pt.apply.doc_score()``, any reranking method that can be expressed
as a function of the text of the query and the text of the doucment can used as a reranker
in a PyTerrier pipeline.

Each apply method takes as input a function (e.g. a function name, or a lambda expression). 
Objects that are passed to the function vary in terms of the type of the input dataframe 
(queries or ranked documents), and also vary in terms of what should be returned by the 
function.

+-------+---------+-------------+------------------+---------------------------+----------------------+-----------------------+
+ Input | Output  | Cardinality | Example          | Example apply             | Function Input type  | Function Return type  |
+=======+=========+=============+==================+===========================+======================+=======================+
|   Q   |    Q    |   1 to 1    | Query rewriting  | `pt.apply.query()`        | row of one query     |  str                  |
+-------+---------+-------------+------------------+---------------------------+----------------------+-----------------------+
| Q x D |  Q x D  |   1 to 1    | Re-ranking       | `pt.apply.doc_score()`    | row of one document  | float                 |
+-------+---------+-------------+------------------+---------------------------+----------------------+-----------------------+
| Q x D |  Q x Df |   1 to 1    | Feature scoring  | `pt.apply.doc_features()` | row of one document  | numpy array           |
+-------+---------+-------------+------------------+---------------------------+----------------------+-----------------------+
| Q x D |    Q    |   N to 1    | Query expansion  | `pt.apply.generic()`      | entire dataframe     | entire dataframe      |
+       |         |             |                  +---------------------------+----------------------+-----------------------+
|       |         |             |                  | `pt.apply.by_query()`     | dataframe for 1 query| dataframe for 1 query |
+-------+---------+-------------+------------------+---------------------------+----------------------+-----------------------+
|   Q   |  Q x D  |   1 to N    | Retrieval        | `pt.apply.generic()`      | entire dataframe     | entire dataframe      |
+       |         |             |                  +---------------------------+----------------------+-----------------------+
|       |         |             |                  | `pt.apply.by_query()`     | dataframe for 1 query| dataframe for 1 query |
+-------+---------+-------------+------------------+---------------------------+----------------------+-----------------------+

In each case, the result from calling a pyterrier.apply method is another PyTerrier transformer 
(i.e. extends ``pt.Transformer``), which can be used for experimentation or combined with other 
PyTerrier transformers through the standard PyTerrier operators.

If `verbose=True` is passed to any pyterrier apply method (except `generic()`), then a `TQDM <https://tqdm.github.io/>`_ 
progress bar will be shown as the transformer is applied.

Example
=======

In the following, we create a document re-ranking transformer that increases the score of documents by 10% if their url attribute contains `"https:"`

    >>> df = pd.DataFrame([["q1", "d1", "https://www.example.com", 1.0, 1]], columns=["qid", "docno", "url", "score", "rank"])
    >>> df
    qid docno                      url  score  rank
    0  q1    d1  https://www.example.com    1.0     1
    >>> 
    >>> http_boost = pt.apply.doc_score(lambda row: row["score"] * 1.1 if "https:" in row["url"] else row["score"])
    >>> http_boost(df)
    qid docno                      url  score  rank
    0  q1    d1  https://www.example.com    1.1     0

We can combine this pt.apply.doc_score() transformer into as a re-ranking pipeline using the `>>` operator::

    pipeline = bm25 >> http_boost 

Further examples are shown for each apply method below.

Apply Methods
=============

.. automodule:: pyterrier.apply
    :members:

Making New Columns and Dropping Columns
=======================================

Its also possible to construct a transformer that makes a new column on a row-wise basis by directly naming the new column in pt.apply.

For instance, if the column you are creating is called rank_2, it might be created as follows::

    pipe = pt.terrier.Retriever(index) >> pt.apply.rank_2(lambda row: row["rank"] * 2)

To create a transformer that drops a column, you can instead pass `drop=True` as a kwarg::

    pipe = pt.terrier.Retriever(index, metadata=["docno", "text"] >> pt.text.scorer() >> pt.apply.text(drop=True)

