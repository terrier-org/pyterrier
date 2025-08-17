"""This module provides useful utility methods for getting information about PyTerrier objects.

.. note::
    This is an advanced module that is not typically used by end users.
"""

from typing import Dict, List, Optional, Protocol, Type, Tuple, Union, runtime_checkable

import pandas as pd

import pyterrier as pt


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
    single: bool = True,
    strict: bool = True,
) -> Optional[Union[List[str], List[List[str]]]]:
    """Infers supported input column configurations for a transformer.

    The method tries to infer the input columns that the transformer accepts by calling it with an empty DataFrame and inspecting
    a resulting ``pt.validate.InputValidationError``. If the transformer does not raise an error, it tries to infer the input columns
    by calling it with a pre-defined set of input columns.

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
    if isinstance(transformer, ProvidesTransformerInputs):
        try:
            result = transformer.transformer_inputs()
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
    if len(result) == 0:
        if strict:
            raise InspectError(f"Cannot determine inputs for {transformer}")
        else:
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

    If the transformer implements the :class:`~pyterrier.inspect.ProvidesTransformerOutputs` protocol,
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
        InspectError: If the artifact's type or format could not be determined and ``strict==True``.
        pt.validate.InputValidationError: If input validation fails in the trnsformer and ``strict==True``.
    """
    if isinstance(transformer, ProvidesTransformerOutputs):
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


def subtransformers(transformer: pt.Transformer) -> Dict[str, Union[pt.Transformer, List[pt.Transformer]]]:
    """Infers a dictionary of subtransformers for the given transformer.

    A subtransformer is a transformer that is used by another transformer to complete its task. Examples
    include those used by caches (e.g., ``scorer`` in :class:`pyterrier_caching.ScorerCache`) and the list
    of transformers that are used by a :class:`pyterrier_alpha.fusion.RRFusion` transformer.

    If the transformer implements the :class:`~pyterrier.inspect.ProvidesSubtransformers` protocol,
    the method calls its ``subtransformers`` method to retrieve the subtransformers. If the transformer does not
    implement the protocol, the method inspects the transformer's attributes and returns a dictionary where the keys
    where the keys are the names of the subtransformers and the values are the subtransformers themselves. If the
    transformer does not have any subtransformers, an empty dictionary is returned.

    Args:
        transformer: The transformer to inspect.

    Returns:
        A dictionary of the provided transformer's subtransformers.
    """
    if isinstance(transformer, ProvidesSubtransformers):
        return transformer.subtransformers()
    result = {}
    for attr, value in transformer.__dict__.items():
        if isinstance(value, pt.Transformer):
            result[attr] = value
        elif ((isinstance(value, list) or isinstance(value, tuple)) and
              len(value) > 0 and isinstance(value[0], pt.Transformer)):
            result[attr] = list(value)
    return result



@runtime_checkable
class ProvidesTransformerInputs(Protocol):
    """Protocol for transformers that provide a ``transformer_inputs`` method.

    ``transformer_inputs`` allows for inspection of the inputs accepted by transformers without needing to run it.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a list of the input column
    configurations accepted by the transformer.

    This method need not be present in Transformer - it is an optional extension;
    an alternative is that the input columns are determined by calling the transformer with an empty ``DataFrame``.

    .. code-block:: python
        :caption: Example ``transformer_inputs`` function, implementing :class:`~pyterrier.inspect.ProvidesTransformerInputs`.

        class MyRetriever(pt.Transformer):

            def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
                pt.validate.query_frame(inp, ['query'])
                # ... perform retrieval ...
                # return the same columns as inp plus docno, score, and rank. E.g., using DataFrameBuilder.

            def transformer_inputs(self) -> List[List[str]]:
                return [['qid', 'query']]

    """
    def transformer_inputs(self) -> List[List[str]]:
        """Returns a list of input columns accepted by the transformer.

        Returns:
            A list of input column configurations accepted by this transformer.
        """



@runtime_checkable
class ProvidesTransformerOutputs(Protocol):
    """Protocol for transformers that provide a ``transform_outputs`` method.

    ``transform_outputs`` allows for inspection of the outputs of transformers without needing to run it.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a
    list of the output columns present given the provided input columns or raise an ``InputValidationError``
    if the inputs are not accepted by the transformer.

    This method need not be present in Transformer - it is an optional extension;
    an alternative is that the output columns are determined by calling the transformer
    with an empty ``DataFrame``.

    Due to risks and maintanence burden in ensuring that ``transform`` and ``transform_outputs`` behave identically,
    it is recommended to only implement ``transform_outputs`` when calling the transformer with an empty DataFrame to
    inspect the behavior is undesireable, e.g., if calling the transformer is expensive.

    .. code-block:: python
        :caption: Example ``transform_output`` function, implementing :class:`~pyterrier.inspect.ProvidesTransformerOutputs`.

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
class ProvidesSubtransformers(Protocol):
    """Protocol for transformers that provide a ``subtransformers`` method.

    ``subtransformers`` allows for identifying subtransformers of a transformer without needing to traverse
    its attributes manually.

    When this method is present in a :class:`~pyterrier.Transformer` object, it must return a
    dict where the keys are the names of the subtransformers and the values are the subtransformers (or list
    of subtransformers) themselves.

    This method need not be present in Transformer - it is an optional extension.
    """
    def subtransformers(self) -> Dict[str, Union[pt.Transformer, List[pt.Transformer]]]:
        """Returns a dictionary of subtransformers for the transformer.

        The method must return a dictionary where the keys are the names of the subtransformers and the values are
        the subtransformers themselves. If the transformer does not have any subtransformers, an empty dictionary
        should be returned.
        """
