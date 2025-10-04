"""This module provides useful utility methods for getting information about PyTerrier objects.

.. note::
    This is an advanced module that is not typically used by end users.
"""
import enum
import inspect
import dataclasses
from typing import Any, Dict, List, Literal, Optional, Protocol, Type, Tuple, Union, cast, overload, runtime_checkable

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

def indexer_inputs(
    indexer : pt.Indexer,
    *,
    strict : bool = True
) -> Optional[List[List[str]]]:
    """
    Infers supported input column configurations (a ``List[List[str]]``) for a pt.Indexer instance.
    Orthogonal to ``transformer_inputs``. This implementation inspects the ``index_inputs()``
    method of the indexer, as other methods of transformer inspection arent applicable to indexers.

    Args:
        indexer: An instance of the indexer to inspect.
        strict: If True, raises an error if the indexer cannot be inferred. If False, returns
            None in these cases.

    Returns:
        A list of input column configurations (``List[List[str]]``) accepted by this indexer.

    Raises:
        InspectError: If the indexer cannot be inspected and ``strict==True``.
    """
    result = indexer.index_inputs()
    if result is None or not isinstance(result[0], list) or (len(result[0]) > 0 and not isinstance(result[0][0], str)):
        if strict:
            msg = f"Cannot determine inputs for {indexer} - index_inputs() should be implemented"
            raise InspectError(msg)
        return None
    return result

def transformer_inputs(
    transformer: pt.Transformer,
    *,
    strict: bool = True,
) -> Optional[List[List[str]]]:
    """Infers supported input column configurations (a ``List[List[str]]``) for a transformer.

    The method tries to infer the input columns that the transformer accepts by calling it with an empty DataFrame and inspecting
    a resulting ``pt.validate.InputValidationError``. If the transformer does not raise an error, it tries to infer the input columns
    by calling it with a pre-defined set of input columns.

    To handle edge cases, you can implement the :class:`~pyterrier.inspect.HasTransformInputs` protocol, which allows you to define a custom
    ``transform_inputs`` method that returns a list of input column configurations accepted by the transformer. ``transform_inputs``
    can also be an attribute instead of a method. In this case, it can be a list of lists of input columns (i.e., a list of valid
    input column configurations). Note that ``transform_inputs`` is allowed to return a ``List[str]``. If this is the case, it is converted
    to a ``List[List[str]]`` automatically.

    The list of input specifications is assumed to be prioritized. For instance, schematics will show the first valid specification
    when multiple are valid for the pipeline.

    Args:
        transformer: An instance of the transformer to inspect.
        strict: If True, raises an error if the transformer cannot be inferred or are not accepted. If False, returns
            None in these cases.

    Returns:
        A list of input column configurations (``List[List[str]]``) accepted by this transformer.

    Raises:
        InspectError: If the transformer cannot be inspected and ``strict==True``.
    """
    result : List[List[str]] = []
    received = None
    if isinstance(transformer, HasTransformInputs) and transformer.transform_inputs is not None:
        ext_result : Union[List[str], List[List[str]]]
        if not callable(transformer.transform_inputs):
            ext_result = transformer.transform_inputs
            received = "transformer.transform_inputs attribute"
        else:
            try:
                ext_result = transformer.transform_inputs()
                received = "transformer.transform_inputs() method"
            except Exception as ex:
                if strict:
                    raise InspectError(f"Cannot determine inputs for {transformer}") from ex
                else:
                    return None
        if len(ext_result) > 0:
            if isinstance(ext_result[0], str):
                ext_result = cast(List[str], ext_result) # noqa: PT100 (this is typing.cast, not jinus.cast)
                result = [ext_result] # convert to a List[List[str]]
            else:
                ext_result = cast(List[List[str]], ext_result) # noqa: PT100 (this is typing.cast, not jinus.cast)
                result = ext_result
    else:
        try:
            transformer(pd.DataFrame())
            # if this succeeds without an error, the transformer accepts frames without any columns
            result = [[]]
        except pt.validate.InputValidationError as ive:
            result = [mode.missing_columns for mode in ive.modes]
            received = "validation using invocation on empty 0-cols frame"
        except Exception:
            for mode, frame_type in [
                (['qid', 'query'] , "Q"),
                (['qid', 'query', 'docno', 'score', 'rank'], "R"),
            ]:
                try:
                    transformer(pd.DataFrame(columns=mode))
                    result.append(mode)
                    received = f"validation using invocation on empty {frame_type} frame"
                except Exception:
                    continue
    if not isinstance(result, list) or len(result) == 0:
        if strict:
            msg = f"Cannot determine inputs for {transformer}"
            if received is not None:
                msg += f"received by {received}: {result}"
            else:
                msg += " - no inspections succeeded"
            raise InspectError(msg)
        return None
    if not isinstance(result[0], list) or (len(result[0]) > 0 and not isinstance(result[0][0], str)):
        if strict:
            raise InspectError(f"Cannot determine inputs for {transformer} - invalid columns specified by {received}: {result}")
        return None
    return result

@overload
def transformer_outputs(
    transformer: pt.Transformer,
    input_columns: List[str],
    *,
    strict: Literal[True] = ...,
) -> List[str]: ...

@overload
def transformer_outputs(
    transformer: pt.Transformer,
    input_columns: List[str],
    *,
    strict: Literal[False] = ...,
) -> Optional[List[str]]: ...

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
    except pt.validate.InputValidationError as ive:
        if strict:
            # add the underlying class to the IVE error, so its more clear whats not been validated
            # this improves readability for subtransformers 
            ive.args = (ive.args[0] + f" {transformer}", )
            raise ive
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
    def __init__(
        self,
        name: str,
        value: Any,
        init_default_value: Any = inspect.Parameter.empty,
        init_parameter_kind: Optional[inspect._ParameterKind] = None,
    ):
        # we need to define __init__ directly to avoid issues with sphinx thinking that self.init_parameter_kind is an alias to inspect.Parameter.empty
        self.name = name
        self.value = value
        self.init_default_value = init_default_value
        self.init_parameter_kind = init_parameter_kind

    name: str
    value: Any
    init_default_value: Any
    init_parameter_kind: Optional[inspect._ParameterKind]

    MISSING = object()


def transformer_attributes(transformer: pt.Transformer, *, strict: bool = True) -> List[TransformerAttribute]:
    """Infers a list of attributes of the transformer.

    Here, an attribute is defined as any attribute of the transformer that is explicitly set by the ``__init__`` method,
    either under the same name (e.g., ``self.foo = foo``) or as a private attribute (e.g., ``self._foo = foo``).

    This definition allows for a set of attributes that should describe the state of a transformer. These attributes can
    be used to reconstruct the transformer from its attributes, e.g., by calling :meth:`~pyterrier.inspect.transformer_apply_attributes`.

    To handle edge cases (e.g., where the ``__init__`` parameters do not match the attribute names), you can implement
    the :class:`~pyterrier.inspect.HasAttributes` protocol.

    Args:
        transformer: The transformer to inspect.
        strict: If True, raises an error if an attribute cannot be identified from the transformer. If False, the attribute's value is set to ``TransformerAttribute.MISSING`` in these cases.

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
        if p.name.startswith('_'):
            continue # Skip private constructor parameters
        if hasattr(transformer, f'_{p.name}'):
            val = getattr(transformer, f'_{p.name}')
        elif hasattr(transformer, p.name):
            val = getattr(transformer, p.name)
        else:
            if strict:
                raise InspectError(f"Cannot identify attribute {p.name} in transformer {transformer}. Ensure that the attribute is set in the __init__ method.")
            else:
                val = TransformerAttribute.MISSING # could not identify the attribute
        result.append(TransformerAttribute(
            name=p.name,
            value=val,
            init_default_value=p.default,
            init_parameter_kind=p.kind,
        ))
    return result


def transformer_apply_attributes(transformer: pt.Transformer, **kwargs: Any) -> pt.Transformer:
    """Returns a new transformer instance from the provided transformer and updated attributes (as keyword arguments).

    This method is useful for constructing new transformer with some attributes replaced. For instance, when implemeting
    methods like :meth:`~pyterrier.transformers.SupportsFuseRankCutoff.fuse_rank_cutoff`, you frequently need to replace the
    ``num_results`` attribute of a transformer with a new value while keeping the remainder of the attributes the same.

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
    missing_attributes = [attr.name for attr in attributes if attr.value is TransformerAttribute.MISSING]
    if missing_attributes:
        raise InspectError(f"Attributes {missing_attributes} for transformer {transformer} are missing.")
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
    for attr in transformer_attributes(transformer, strict=False):
        if isinstance(attr.value, pt.Transformer) and not isinstance(attr.value, pt.Artifact):
            result[attr.name] = attr.value
        elif isinstance(attr.value, (list, tuple)) and all(isinstance(v, pt.Transformer) and not isinstance(v, pt.Artifact) for v in attr.value):
            result[attr.name] = list(attr.value)
    return result

def _minimal_inputs(all_configs : List[Optional[List[List[str]]]]) -> Optional[List[List[str]]]:
    """
    Among all input configurations, find the minimal common subset(s) of attributes needed
    for success invocation.
    """
    # if any configs are unknown, the minimal inputs is unknown
    if any([config is None for config in all_configs]):
        return None
    
    # essentially a cast:
    non_optional: List[List[List[str]]] = [config for config in all_configs if config is not None]
    all_configs_sets = [ 
        set(a) for tconfig in non_optional for a in tconfig
    ]
    
    from itertools import chain, combinations
    # universe of all columns
    universe = set(chain.from_iterable(all_configs_sets))

    # test subsets in increasing size
    plausible = []
    for r in range(1, len(universe) + 1):
        for subset in combinations(universe, r):
            ssubset = set(subset)
            # Check if subset works for all objects
            if all(any(set(schema).issubset(ssubset) for schema in obj) for obj in non_optional):
                plausible.append(list(ssubset))
    return plausible

class TransformerType(enum.Flag):
    """An enum representing the type of a transformer."""
    transformer = enum.auto()
    indexer = enum.auto()


def transformer_type(transformer: pt.Transformer) -> TransformerType:
    """Returns the type of the transformer as a :class:`~pyterrier.inspect.TransformerType` flag enum.

    The type can be one of:
    - ``TransformerType.transformer``: The transformer is a :class:`~pyterrier.Transformer` but not an :class:`~pyterrier.Indexer`.
    - ``TransformerType.indexer``: The transformer is an :class:`~pyterrier.Indexer` but does not implement ``transform`` or ``transform_iter``.
    - ``TransformerType.transformer | TransformerType.indexer``: The transformer is both a :class:`~pyterrier.Transformer` and an :class:`~pyterrier.Indexer`.
    - ``TransformerType(0)``: The transformer is neither a :class:`~pyterrier.Transformer` nor an :class:`~pyterrier.Indexer`.

    Args:
        transformer: The transformer to inspect.

    Returns:
        A :class:`~pyterrier.inspect.TransformerType` flag representing the type of the transformer.
    """
    if isinstance(transformer, pt.Indexer):
        if transformer.__class__.transform != pt.Indexer.transform or transformer.__class__.transform_iter != pt.Indexer.transform_iter:
            # Indexer that also implements transform or transform_iter (or both)
            return TransformerType.transformer | TransformerType.indexer # both a Transformer and an Indexer
        else:
            # Indexer that doesn't implement transform or transform_iter
            return TransformerType.indexer # only Indexer
    if isinstance(transformer, pt.Transformer):
        return TransformerType.transformer # only Transformer
    return TransformerType(0) # neither Transformer nor Indexer


@runtime_checkable
class HasTransformInputs(Protocol):
    """Protocol for transformers that provide a ``transform_inputs`` method.

    ``transform_inputs`` allows for inspection of the inputs accepted by transformers without needing to run it.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return either:
    
    - A list of lists of input columns (i.e., a list of valid input column configurations)
    - A list of input columns (i.e., a single valid input column configuration)

    If the input columns of the transformer do not depend on the instance, ``transform_inputs`` can also be
    an attribute with a value of type ``List[str]`` or ``List[List[str]]``.

    If ``transform_inputs is None``, it is ignored.

    This method need not be present in a Transformer class - it is an optional extension;
    an alternative is that the input columns are determined by calling the transformer with an empty ``DataFrame``.

    .. code-block:: python
        :caption: Example ``transform_inputs`` function, implementing :class:`~pyterrier.inspect.HasTransformInputs`.

        class MyRetriever(pt.Transformer):

            def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
                pt.validate.query_frame(inp, ['query'])
                # ... perform retrieval ...
                # return the same columns as inp plus docno, score, and rank. E.g., using DataFrameBuilder.

            def transform_inputs(self) -> Union[List[str], List[List[str]]]:
                # NOTE: This method isn't required in this case, since inspect will be able to infer required
                # columns from pt.validate. It's just a demonstration.
                return ['qid', 'query']

    """
    def transform_inputs(self) -> Union[List[List[str]], List[str]]:
        """Returns a list of input columns accepted by the transformer.

        Returns:
            Input column configuration(s) accepted by this transformer.
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
            pt.inspect.InspectError: If the transformer is uninspectable.
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
