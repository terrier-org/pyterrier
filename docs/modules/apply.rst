pyterrier.apply module
--------------------------

PyTerrier pipelines are easily extensible through the use of apply functions.
These are inspired by the `Pandas apply() <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html>`_, 
which allow to apply a function to each row of a dataframe. Instead, in PyTerrier, 
apply small functions (including Python lambdas) to allow to easily construct 
pipeline transformers to address common use cases.

The table below lists the main classes of transformation in the PyTerrier data 
model, as well as the appropriate apply function to use in each case. These vary
in terms of the type of the input dataframe (queries or ranked documents), and
the cardinality change in the dataframes by application of the transformer.

+-------+---------+-------------+------------------+---------------------------+----------------------+------------------+
+ Input | Output  | Cardinality | Example          | Example apply             | Input type           | Return type      |
+=======+=========+=============+==================+===========================+======================+==================+
|   Q   |    Q    |   1 to 1    | Query rewriting  | `pt.apply.query()`        | row of one query     |  str             |
+-------+---------+-------------+------------------+---------------------------+----------------------+------------------+
| Q x D |  Q x D  |   1 to 1    | Re-ranking       | `pt.apply.doc_score()`    | row of one document  | float            |
+-------+---------+-------------+------------------+---------------------------+----------------------+------------------+
| Q x D |  Q x Df |   1 to 1    | Feature scoring  | `pt.apply.doc_features()` | row of one document  | numpy array      |
+-------+---------+-------------+------------------+---------------------------+----------------------+------------------+
| Q x D |    Q    |   N to 1    | Query expansion  | `pt.apply.generic()`      | entire dataframe     | entire dataframe |
+-------+---------+-------------+------------------+---------------------------+----------------------+------------------+
|   Q   |  Q x D  |   1 to N    | Retrieval        | `pt.apply.generic()`      | entire dataframe     | entire dataframe |
+-------+---------+-------------+------------------+---------------------------+----------------------+------------------+

In each case, the result from calling a pyterrier.apply function is another PyTerrier transformer 
(i.e. extends TransformerBase), and which can be used for experimentation or combined with other 
PyTerrier transformers through the standard PyTerrier operators.

If verbose=True is passed to any pyterrier function (except generic()), then a TQDM progress bar will be shown as the transformer is applied.

.. automodule:: pyterrier.apply
    :members:

