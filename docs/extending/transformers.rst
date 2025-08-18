Writing Custom Transformers
=====================================

.. note::
    This page is a work in progress.




Supporting Inspection
-------------------------------------

:ref:`pt.inspect <pyterrier.inspect>` allows users to gather information about live transformer objects, for instance
input/output specifications. Default implementations for these methods usually work well, but sometimes
you may need to override them to handle idiosyncratic cases.

You can override the behavior of the following methods by implementing python
`Protocols <https://typing.python.org/en/latest/spec/protocol.html>`__ (in these cases, it's just a specific
method signature).

+--------------------------------------------------------------+--------------------------------------------------------------------------+
| Override...                                                  | By implementing...                                                       |
+==============================================================+==========================================================================+
| :meth:`pyterrier.inspect.transformer_inputs`                 | :class:`pyterrier.inspect.ProvidesTransformerInputs`                     |
+--------------------------------------------------------------+--------------------------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_outputs`                | :class:`pyterrier.inspect.ProvidesTransformerOutputs`                    |
+--------------------------------------------------------------+--------------------------------------------------------------------------+
| :meth:`pyterrier.inspect.subtransformers`                    | :class:`pyterrier.inspect.ProvidesSubtransformers`                       |
+--------------------------------------------------------------+--------------------------------------------------------------------------+
