"""This module provides useful utility methods for getting information about PyTerrier objects.

.. note::
    This is an advanced module that is not typically used by end users.
"""
import inspect
import dataclasses
from typing import Any, Dict, List, Optional, Protocol, Type, Tuple, Union, runtime_checkable

import pandas as pd

import pyterrier as pt

__all__ = [
    'artifact_type_format',
    'transformer_inputs',
    'transformer_outputs',
    'transformer_attributes',
    'transformer_apply_attributes',
    'subtransformers',
    'InspectError',
    'TransformerAttribute',
    'HasTransformInputs',
    'HasTransformOutputs',
    'HasAttributes',
    'HasApplyAttributes',
    'HasSubtransformers',
]


class InspectError(TypeError):
    """Base exception for inspection errors."""
    pass


def artifact_type_format(
    artifact: Union[Type, 'pt.Artifact'],
    *,
    strict: bool = True,
) -> Optional[Tuple[str, str]]:
    """Returns the type and format of the specified artifact.

    These values are sourced by either the ``ARTIFACT_TYPE`` and ``ARTIFACT_FORMAT`` constants of the artifact, or (if these
    are not available) by matching on the entry points.

    Args:
        artifact: The artifact to inspect.
        strict: If True, raises an error if the artifact's type or format could not be determined.

    Returns:
        A tuple containing the artifact's type and format, or ``None`` if the type and format could not be determined
        and ``strict==False``.

    Raises:
        InspectError: If the artifact's type or format could not be determined and ``strict==True``
    """
    artifact_type, artifact_format = None, None

    # Source #1: ARTIFACT_TYPE and ARTIFACT_FORMAT constants
    if hasattr(artifact, 'ARTIFACT_TYPE') and hasattr(artifact, 'ARTIFACT_FORMAT'):
        artifact_type = artifact.ARTIFACT_TYPE
        artifact_format = artifact.ARTIFACT_FORMAT

    # Source #2: entry point name
    if artifact_type is None or artifact_format is None:
        for entry_point in pt.utils.entry_points('pyterrier.artifact'):
            if artifact.__module__.split('.')[0] != entry_point.value.split(':')[0].split('.')[0]:
                continue # only try loading entry points that share the same top-level module
            entry_point_cls = entry_point.load()
            if isinstance(artifact, type) and artifact == entry_point_cls or isinstance(artifact, entry_point_cls):
                artifact_type, artifact_format = entry_point.name.split('.', 1)
                break

    if artifact_type is None or artifact_format is None:
        if strict:
            raise InspectError(f'{artifact} does not provide type and format (either as constants or via entry point)')
        return None

    return artifact_type, artifact_format


def transformer_inputs(
    transformer: pt.Transformer,
    *,
    single: bool = False,
    strict: bool = True,
) -> Optional[Union[List[str], List[List[str]]]]:
    """Infers supported input column configurations for a transformer.

    The method tries to infer the input columns that the transformer accepts by calling it with an empty DataFrame and inspecting
    a resulting ``pt.validate.InputValidationError``. If the transformer does not raise an error, it tries to infer the input columns
    by calling it with a pre-defined set of input columns.

    To handle edge cases, you can implement the :class:`~pyterrier.inspect.HasTransformInputs` protocol, which allows you to define a custom
    ``transform_inputs`` method that returns a list of input column configurations accepted by the transformer. ``transform_inputs``
    can also be an attribute instead of a method. In this case, it be a list of lists of input columns (i.e., a list of valid
    input column configurations).

    Args:
        transformer: An instance of the transformer to inspect.
        strict: If True, raises an error if the transformer cannot be inferred or are not accepted. If False, returns
            None in these cases.
        single: If True, returns a single list of input columns. If False, returns a list of lists of possible input column configurations.

    Returns:
        A list of input columns that the transformer accepts, or a list of lists of input columns if ``single`` is False.

    Raises:
        InspectError: If the transformer cannot be inspected and ``strict==True``.
    """
    result = []
    if isinstance(transformer, HasTransformInputs):
        if not callable(transformer.transform_inputs):
            result = transformer.transform_inputs
        else:
            try:
                result = transformer.transform_inputs()
            except Exception as ex:
                if strict:
                    raise InspectError(f"Cannot determine inputs for {transformer}") from ex
                else:
                    return None
    else:
        try:
            transformer(pd.DataFrame())
        except pt.validate.InputValidationError as ex:
            result = [mode.missing_columns for mode in ex.modes]
        except Exception:
            for mode in [
                ['qid', 'query'],
                ['qid', 'query', 'docno', 'score', 'rank'],
            ]:
                try:
                    transformer(pd.DataFrame(columns=mode))
                    result.append(mode)
                except Exception:
                    continue
    if not isinstance(result, list) or len(result) == 0:
        if strict:
            raise InspectError(f"Cannot determine inputs for {transformer}")
        return None
    if not isinstance(result[0], list) or (len(result[0]) > 0 and not isinstance(result[0][0], str)):
        if strict:
            raise InspectError(f"Cannot determine inputs for {transformer}")
        return None
    if single:
        return result[0]
    return result


def transformer_outputs(
    transformer: pt.Transformer,
    input_columns: List[str],
    *,
    strict: bool = True,
) -> Optional[List[str]]:
    """Infers the output columns for a transformer based on the provided input columns.

    If the transformer implements the :class:`~pyterrier.inspect.HasTransformOutputs` protocol,
    the method calls its ``transform_outputs`` method to determine the output columns. If the transformer does not
    implement the protocol, it attempts to infer the output columns by calling the transformer with an empty DataFrame.

    Args:
        transformer: An instance of the transformer to inspect.
        input_columns: A list of the columns present in the input frame.
        strict: If True, raises an error if the transformer cannot be inferred or are not accepted. If False, returns
            None in these cases.

    Returns:
        A list of the columns present in the output for ``transformer`` given ``input_columns``.

    Raises:
        InspectError: If the transformer's outputs could not be determined and ``strict==True``.
        pt.validate.InputValidationError: If input validation fails in the transformer and ``strict==True``.
    """
    if isinstance(transformer, HasTransformOutputs):
        try:
            return transformer.transform_outputs(input_columns)
        except pt.validate.InputValidationError:
            if strict:
                raise
            else:
                return None
        except Exception as ex:
            if strict:
                raise InspectError(f"Cannot determine outputs for {transformer} with inputs: {input_columns}") from ex
            else:
                return None

    try:
        res = transformer.transform(pd.DataFrame(columns=input_columns))
        return list(res.columns)
    except pt.validate.InputValidationError:
        if strict:
            raise
        else:
            return None
    except Exception as ex:
        if strict:
            raise InspectError(f"Cannot determine outputs for {transformer} with inputs: {input_columns}") from ex
        else:
            return None


@dataclasses.dataclass
class TransformerAttribute:
    """A dataclass representing an attribute of a transformer.

    Attributes:
        name: The name of the attribute.
        value: The value of the attribute.
        init_default_value: The default value of the attribute for the ``__init__`` method (if available) or ``inspect.Parameter.empty`` if not available.
        init_parameter_kind: The kind of the parameter in the ``__init__`` method (if available) or ``None`` if not available.
    """
    name: str
    value: Any
    init_default_value: Any
    init_parameter_kind: Optional[inspect._ParameterKind]


def transformer_attributes(transformer: pt.Transformer) -> List[TransformerAttribute]:
    """Infers a list of attributes of the transformer.

    Here, an attribute is defined as any attribute of the transformer that is explicity set by the ``__init__`` method,
    either under the same name (e.g., ``self.foo = foo``) or as a private attribute (e.g., ``self._foo = foo``).

    This definition allow for a set of attributes that should describe the state of a transformer. These attributes can
    be used to reconstruct the transformer from its attributes, e.g., by calling :meth:`~pyterrier.inspect.transformer_apply_attributes`.

    To handle edge cases (e.g., where the ``__init__`` paraemters do not match the attribute names), you can implement
    the :class:`~pyterrier.inspect.HasAttributes` protocol.

    Args:
        transformer: The transformer to inspect.

    Returns:
        A list of :class:`~pyterrier.inspect.TransformerAttribute` objects representing the attributes of the transformer.

    Raises:
        InspectError: If the attributes cannot be identified from the transformer.
    """
    if isinstance(transformer, HasAttributes):
        return transformer.attributes()
    result = []
    signature = inspect.signature(transformer.__class__.__init__)
    for p in list(signature.parameters.values())[1:]: # [1:] to skip first arg ("self") which is bound to the instance.
        if hasattr(transformer, f'_{p.name}'):
            val = getattr(transformer, f'_{p.name}')
        elif hasattr(transformer, p.name):
            val = getattr(transformer, p.name)
        else:
            raise InspectError(f"Cannot identify attribute {p.name} in transformer {transformer}. Ensure that the attribute is set in the __init__ method.")
        result.append(TransformerAttribute(
            name=p.name,
            value=val,
            init_default_value=p.default,
            init_parameter_kind=p.kind,
        ))
    return result


def transformer_apply_attributes(transformer: pt.Transformer, **kwargs: Any) -> pt.Transformer:
    """Returns a new transformer instance from the provided transformer and updated attributes (as keyword arguments).

    This method uses :meth:`~pyterrier.inspect.transformer_attributes` to identify the attributes of the transformer and
    then applies the provided keyword arguments to the transformer attributes. The method then reconstructs the transformer
    by calling its ``__init__`` method with the updated attributes.

    To handle edge cases (e.g., where the ``__init__`` parameters do not match the attribute names), you can implement
    the :class:`~pyterrier.inspect.HasApplyAttributes` protocol.

    Args:
        transformer: The transformer to apply the attributes to.
        **kwargs: Keyword arguments representing the attributes to set on the transformer.

    Returns:
        A new instance of the transformer with the provided attributes applied.

    Raises:
        InspectError: If an attribute is not found in the transformer or if attributes cannot be identified from the transformer.
    """
    if isinstance(transformer, HasApplyAttributes):
        return transformer.apply_attributes(**kwargs)
    attributes = transformer_attributes(transformer)
    for attr in attributes:
        if attr.name in kwargs:
            attr.value = kwargs.pop(attr.name)
    if any(kwargs):
        raise InspectError(f"Unknown attributes {list(kwargs.keys())} for transformer {transformer}")
    init_args = []
    init_kwargs = {}
    for attr in attributes:
        if attr.init_parameter_kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            init_args.append(attr.value)
        elif attr.init_parameter_kind == inspect.Parameter.KEYWORD_ONLY:
            init_kwargs[attr.name] = attr.value
        elif attr.init_parameter_kind == inspect.Parameter.POSITIONAL_ONLY:
            init_args.append(attr.value)
        elif attr.init_parameter_kind == inspect.Parameter.VAR_POSITIONAL:
            init_args.extend(attr.value)
        elif attr.init_parameter_kind == inspect.Parameter.VAR_KEYWORD:
            init_kwargs.update(attr.value)
    return transformer.__class__(*init_args, **init_kwargs)


def subtransformers(transformer: pt.Transformer) -> Dict[str, Union[pt.Transformer, List[pt.Transformer]]]:
    """Infers a dictionary of subtransformers for the given transformer.

    A subtransformer is a transformer that is used by another transformer to complete its task. Examples
    include those used by caches (e.g., ``scorer`` in :class:`pyterrier_caching.ScorerCache`) and the list
    of transformers that are used by a :class:`pyterrier_alpha.fusion.RRFusion` transformer.

    If the transformer implements the :class:`~pyterrier.inspect.HasSubtransformers` protocol,
    the method calls its ``subtransformers`` method to retrieve the subtransformers. If the transformer does not
    implement the protocol, the method inspects the transformer to identify any attributes of a transformer that
    are instance of pt.Transformer (or list/tuple of Transformer), returning a dictionary where the keys
    where the keys are the names of the subtransformers and the values are the subtransformers themselves. If the
    transformer does not have any subtransformers, an empty dictionary is returned.

    Args:
        transformer: The transformer to inspect.

    Returns:
        A dictionary of the provided transformer's subtransformers.

    Raises:
        InspectError: If the subtransformers cannot be identified from the transformer.
    """
    if isinstance(transformer, HasSubtransformers):
        return transformer.subtransformers()
    result: Dict[str, Union[pt.Transformer, List[pt.Transformer]]] = {}
    for attr in transformer_attributes(transformer):
        if isinstance(attr.value, pt.Transformer):
            result[attr.name] = attr.value
        elif isinstance(attr.value, (list, tuple)) and all(isinstance(v, pt.Transformer) for v in attr.value):
            result[attr.name] = list(attr.value)
    return result


@runtime_checkable
class HasTransformInputs(Protocol):
    """Protocol for transformers that provide a ``transform_inputs`` method.

    ``transform_inputs`` allows for inspection of the inputs accepted by transformers without needing to run it.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a list of the input column
    configurations accepted by the transformer.

    This method need not be present in a Transformer class - it is an optional extension;
    an alternative is that the input columns are determined by calling the transformer with an empty ``DataFrame``.

    .. code-block:: python
        :caption: Example ``transform_inputs`` function, implementing :class:`~pyterrier.inspect.HasTransformInputs`.

        class MyRetriever(pt.Transformer):

            def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
                pt.validate.query_frame(inp, ['query'])
                # ... perform retrieval ...
                # return the same columns as inp plus docno, score, and rank. E.g., using DataFrameBuilder.

            def transform_inputs(self) -> List[List[str]]:
                return [['qid', 'query']]

    """
    def transform_inputs(self) -> List[List[str]]:
        """Returns a list of input columns accepted by the transformer.

        Returns:
            A list of input column configurations accepted by this transformer.
        """



@runtime_checkable
class HasTransformOutputs(Protocol):
    """Protocol for transformers that provide a ``transform_outputs`` method.

    ``transform_outputs`` allows for inspection of the outputs of transformers without needing to run it.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a
    list of the output columns present given the provided input columns or raise an ``InputValidationError``
    if the inputs are not accepted by the transformer.

    This method need not be present in a Transformer class - it is an optional extension;
    an alternative is that the output columns are determined by calling the transformer
    with an empty ``DataFrame``.

    Due to risks and maintanence burden in ensuring that ``transform`` and ``transform_outputs`` behave identically,
    it is recommended to only implement ``transform_outputs`` when calling the transformer with an empty DataFrame to
    inspect the behavior is undesireable, e.g., if calling the transformer is expensive.

    .. code-block:: python
        :caption: Example ``transform_outputs`` function, implementing :class:`~pyterrier.inspect.HasTransformOutputs`.

        class MyRetriever(pt.Transformer):

            def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
                pt.validate.query_frame(inp, ['query'])
                # ... perform retrieval ...
                # return the same columns as inp plus docno, score, and rank. E.g., using DataFrameBuilder.

            def transform_outputs(self, input_columns: List[str]) -> List[str]:
                pt.validate.query_frame(input_columns, ['query'])
                return input_columns + ['docno', 'score', 'rank']

    """
    def transform_outputs(self, input_columns: List[str]) -> List[str]:
        """Returns a list of the output columns present given the ``input_columns``.

        The method must return exactly the same output columns as ``transform`` would given the provided input
        columns. If the input columns are not accepted by the transformer, the method should raise an
        ``InputValidationError`` (e.g., through ``pt.validate``).

        Args:
            input_columns: A list of the columns present in the input frame.

        Returns:
            A list of the columns present in the output for this transformer given ``input_columns``.

        Raises:
            pt.validate.InputValidationError: If the input columns are not accepted by the transformer.
        """


@runtime_checkable
class HasAttributes(Protocol):
    """Protocol for transformers that provide an ``attributes`` method.

    ``attributes`` allows for identifying the attributes of a transformer without needing to traverse
    its attributes manually.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a
    list of :class:`~pyterrier.inspect.TransformerAttribute` objects, where each object represents an attribute of the
    transformer and corresponding metadata about how the attribute is assigned.

    This method need not be present in a Transformer class - it is an optional extension.
    """
    def attributes(self) -> List[TransformerAttribute]:
        """Returns a list of attributes of the transformer."""


@runtime_checkable
class HasApplyAttributes(Protocol):
    """Protocol for transformers that provide an ``apply_attributes`` method.

    ``apply_attributes`` returns a new transformer with updated attributes (as keyword arguments).

    This method need not be present in a Transformer class - it is an optional extension.
    """
    def apply_attributes(self, **kwargs: Any) -> pt.Transformer:
        """Returns a new transformer instance from the provided transformer and updated attributes (as keyword arguments)."""



@runtime_checkable
class HasSubtransformers(Protocol):
    """Protocol for transformers that provide a ``subtransformers`` method.

    ``subtransformers`` allows for identifying subtransformers of a transformer without needing to traverse
    its attributes manually.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a
    dict where the keys are the names of the subtransformers and the values are the subtransformers (or list
    of subtransformers) themselves.

    This method need not be present in a Transformer class - it is an optional extension. See :meth:`pyterrier.inspect.subtransformers`
    for the default implementation.
    """
    def subtransformers(self) -> Dict[str, Union[pt.Transformer, List[pt.Transformer]]]:
        """Returns a dictionary of subtransformers for the transformer.

        The method must return a dictionary where the keys are the names of the subtransformers and the values are
        the subtransformers themselves. If the transformer does not have any subtransformers, an empty dictionary
        should be returned.
        """
