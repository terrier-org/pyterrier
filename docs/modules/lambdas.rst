pyterrier.lambdas module
--------------------------

PyTerrier pipelines are easily extensible through the use of lambdas. 
Lambdas refer to the use of anonymous `lambda` functions passed to the appropriate functions in the pyterrier.lambda module.
However, function references can also be passed.

In each case, the result is another PyTerrier transformer (i.e. which extends TransformerBase), and which can be used for experimentation or combined with other PyTerrier transformers through the standard PyTerrier operators.

.. automodule:: pyterrier.lambdas
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex: