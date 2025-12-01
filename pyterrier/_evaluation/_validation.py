from typing import Sequence
from warnings import warn
import ir_measures
import pandas as pd
import pyterrier as pt
from pyterrier._evaluation._rendering import _convert_measures
from . import MEASURES_TYPE, SYSTEM_OR_RESULTS_TYPE, VALIDATE_TYPE


def _validate(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Sequence[str],
        validate : VALIDATE_TYPE):

    if validate == 'ignore':
        return

    # convert metrics to a list of Measure objects
    # NB: this mapping is already performed in _run_and_evaluate, but we need it here to 
    # validate the transformers; I think its inexpensive
    _metrics, rev_mapping = _convert_measures(eval_metrics)

    required_cols = set(ir_measures.run_inputs(_metrics)) # find required columns for the measures (usually: query_id, doc_id, score)
    required_cols = set(list(pt.model.from_ir_measures(required_cols))) # convert to pyterrier naming conventions (usually: qid, docno, score)

    validation_failed_pipelines = []
    invalid_pipelines = []
    wrong_outputs_pipelines = []

    for i, (name, system) in enumerate(zip(names, retr_systems)):
        friendly_name = f"Pipeline #{i}: {name} ({str(system)})" if name != str(system) else f"Pipeline #{i}: {str(system)}"
        found_cols = []
        if isinstance(system, pd.DataFrame):
            found_cols = system.columns.tolist()
            friendly_name = f"DataFrame #{i}: {found_cols}"
        elif isinstance(system, pt.Transformer):
            try:
                found_cols = pt.inspect.transformer_outputs(system, input_columns=topics.columns.tolist()) or [] # or [] will never be called, but is needed to make mypy happy
            except pt.inspect.InspectError: # when we cant tell
                validation_failed_pipelines.append(friendly_name)
                continue
            except pt.validate.InputValidationError: # (when input validation fails)
                invalid_pipelines.append(friendly_name)
                continue
        else:
            raise TypeError("Expected a list of Transformers or DataFrames, but received unexpected type %s for retrieval system at position %d" % (str(type(system)), i))

        if required_cols.difference(found_cols):
            wrong_outputs_pipelines.append(f'{friendly_name}. Produces {found_cols}')

    if validation_failed_pipelines or invalid_pipelines or wrong_outputs_pipelines:
        message = 'Experiment Pipeline Validation Report\n\n'
        if invalid_pipelines:
            message += 'The following pipelines failed validation (i.e., they are incompatible with one another or the provided topics):'
            for p in invalid_pipelines:
                message += f'\n - {p}'
            message += '\n\n'
        if wrong_outputs_pipelines:
            message += f'The following pipelines do not produce the required outputs for evaluation ({required_cols}):'
            for p in wrong_outputs_pipelines:
                message += f'\n - {p}'
            message += '\n\n'
        if validation_failed_pipelines:
            message += 'The following pipelines could not be validated (i.e., it is unclear what outputs they produce):'
            for p in validation_failed_pipelines:
                message += f'\n - {p}'
            message += "\nIf these pipelines work, set validate='ignore' to remove this warning, or make them inspectable to clarify how they work.\n\n"

        message += 'See https://pyterrier.readthedocs.io/en/latest/troubleshooting/inspection.html for more information.'

        if validate == 'error' and (invalid_pipelines or wrong_outputs_pipelines):
            raise ValueError(message)
        elif validate == 'warn':
            warn(message)
