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
`Protocols <https://typing.python.org/en/latest/spec/protocol.html>`__ (in these cases, it's just adding a
method with a specific signature that implements the same functionality).

+---------------------------------------------------------+------------------------------------------------------+
| Override...                                             | By implementing...                                   |
+=========================================================+======================================================+
| :meth:`pyterrier.inspect.transformer_inputs`            | :class:`pyterrier.inspect.ProvidesTransformInputs`   |
+---------------------------------------------------------+------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_outputs`           | :class:`pyterrier.inspect.ProvidesTransformOutputs`  |
+---------------------------------------------------------+------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_attributes`        | :class:`pyterrier.inspect.ProvidesAttributes`        |
+---------------------------------------------------------+------------------------------------------------------+
| :meth:`pyterrier.inspect.transformer_apply_attributes`  | :class:`pyterrier.inspect.ProvidesApplyAttributes`   |
+---------------------------------------------------------+------------------------------------------------------+
| :meth:`pyterrier.inspect.subtransformers`               | :class:`pyterrier.inspect.ProvidesSubtransformers`   |
+---------------------------------------------------------+------------------------------------------------------+
