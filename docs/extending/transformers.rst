Writing Custom Transformers
=====================================

.. note::
    This page is a work in progress.



Pipeline Optimization
-------------------------------------

Pipelines can be optimized using :meth:`pyterrier.Transformer.compile`. You can implement your own
optimizations by overriding this method. For instance, a pseudo-relevance feedback method that only uses the
top ``fb_docs`` documents per query can re-write itself with a preceding :class:`~pyterrier.RankCutoff` transformer,
as follows:

.. code-block:: python
    :caption: Optimizing a pseudo-relevance feedback transformer by implementing ``compile()``.

    class MyPrf(pt.Transformer):
        ...
        def compile(self) -> pt.Transformer:
            return pt.RankCutoff(self.fb_docs) >> self

Why is this helpful? :class:`~pyterrier.RankCutoff` itself implements ``optimize`` to combine ("fuse") itself with
transformers that are able to reduce computation by knowing how many documents are required by the subsequent step.
For instance, most retrievers can reduce computaional cost by reducing the top ``k`` documents retrieved per query.

This functionality is faciliated through the :class:`~pyterrier.transformer.SupportsFuseRankCutoff` protocol, which defines
the :meth:`~pyterrier.transformer.SupportsFuseRankCutoff.fuse_rank_cutoff` method. You can choose to implement this
method if your transformer can benefit from being combined with a :class:`~pyterrier.RankCutoff` transformer.

.. code-block:: python
    :caption: Implementing ``fuse_rank_cutoff`` to allow combining with ``RankCutoff``.

    class MyRetriever(pt.Transformer):
        ...
        def fuse_rank_cutoff(self, k: int) -> Optional[pt.Transformer]:
            if self.num_results > k:
                return pt.inspect.transformer_apply_attributes(self, num_results=k)

.. hint::
    :meth:`~pyterrier.inspect.transformer_apply_attributes` lets you easily construct a new transformer with some attributes
    replaced (here, ``num_results``). This can be expecially handy when your transformer has a lot of attributes.

.. caution::
    The result of fusion methods should be *functionally equivalent* to the original transformer. If the
    ``if self.num_results > k:`` condition above was not applied, it would behave differently when ``num_results<k``.


Several transformers implement ``compile`` to allow themselves to be combined ("fused") with other transformers. When
writing your own transformer, consider implementing the following protocols to allow for fusing with other transformers:

+----------------------------------------------------------------------+-----------------------------------------------------------------------------+
| If your transformer benefits from...                                 | Consider implementing...                                                    |
+======================================================================+=============================================================================+
| Returning fewer results per query                                    | :class:`~pyterrier.transformer.SupportsFuseRankCutoff.fuse_rank_cutoff`     |
+----------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Combining with a known transformer **before** it in a pipeline       | :class:`~pyterrier.transformer.SupportsFuseLeft.fuse_left`                  |
+----------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Combining with a known transformer **after** it in a pipeline        | :class:`~pyterrier.transformer.SupportsFuseRight.fuse_right`                |
+----------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Computing multiple scores/features at once (instead of individually) | :class:`~pyterrier.transformer.SupportsFuseFeatureUnion.fuse_feature_union` |
+----------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Other arbitrary optimizations                                        | :class:`~pyterrier.Transformer.compile`                                     |
+----------------------------------------------------------------------+-----------------------------------------------------------------------------+


Supporting Inspection
-------------------------------------

:ref:`pt.inspect <pyterrier.inspect>` allows users to gather information about live transformer objects, for instance
input/output specifications. Default implementations for these methods usually work well, but sometimes
you may need to override them to handle idiosyncratic cases.

You can override the behavior of the following methods by implementing python
`Protocols <https://typing.python.org/en/latest/spec/protocol.html>`__ (in these cases, it's just adding a
method with a specific signature that implements the same functionality).

+---------------------------------------------------------+--------------------------------------------------------------------+
| Override...                                             | By implementing...                                                 |
+=========================================================+====================================================================+
| :meth:`pyterrier.inspect.transformer_inputs`            | :class:`~pyterrier.inspect.HasTransformInputs.transform_inputs`    |
+---------------------------------------------------------+--------------------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_outputs`           | :class:`~pyterrier.inspect.HasTransformOutputs.transform_outputs`  |
+---------------------------------------------------------+--------------------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_attributes`        | :class:`~pyterrier.inspect.HasAttributes.attributes`               |
+---------------------------------------------------------+--------------------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_apply_attributes`  | :class:`~pyterrier.inspect.HasApplyAttributes.apply_attributes`    |
+---------------------------------------------------------+--------------------------------------------------------------------+
| :meth:`pyterrier.inspect.subtransformers`               | :class:`~pyterrier.inspect.HasSubtransformers.subtransformers`     |
+---------------------------------------------------------+--------------------------------------------------------------------+
