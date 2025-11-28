import pyterrier as pt
from pyterrier._evaluation._rendering import _convert_measures
from . import MEASURES_TYPE, SYSTEM_OR_RESULTS_TYPE, VALIDATE_TYPE


import ir_measures
import pandas as pd


from typing import Sequence
from warnings import warn


def _validate(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Sequence[str],
        validate : VALIDATE_TYPE):

    if validate == 'ignore':
        return
    
    WARNING_HELP = "See https://pyterrier.readthedocs.io/en/latest/troubleshooting/inspection.html for more information."

    # convert metrics to a list of Measure objects
    # NB: this mapping is already performed in _run_and_evaluate, but we need it here to 
    # validate the transformers; I think its inexpensive
    _metrics, _ = _convert_measures(eval_metrics)

    required_cols = set(ir_measures.run_inputs(_metrics)) # find required columns for the measures (usually: query_id, doc_id, score)
    required_cols = set(list(pt.model.from_ir_measures(required_cols))) # convert to pyterrier naming conventions (usually: qid, docno, score)

    for i, (name, system) in enumerate(zip(names, retr_systems)):
        friendly_name = "Transformer %s (%s) at position %i" % (name, str(system), i) if name != str(system) else "Transformer %s at position %i" % (str(system), i)
        found_cols = []
        if isinstance(system, pd.DataFrame):
            found_cols = system.columns.tolist()
        elif isinstance(system, pt.Transformer):
            try:
                found_cols = pt.inspect.transformer_outputs(system, input_columns=topics.columns.tolist()) or [] # or [] will never be called, but is needed to make mypy happy
            except pt.inspect.InspectError as ie: # when we cant tell
                warn(
                    "%s failed to validate: %s - if your pipeline works, set validate='ignore' to remove this warning, or add transform_output method to the transformers in this pipeline to clarify how it works. %s" % 
                    (friendly_name, str(ie), WARNING_HELP))
                continue
            except pt.validate.InputValidationError as ie: # (when input validation fails)
                if validate == 'warn':
                    warn(
                        "%s failed to validate: %s. %s" % (friendly_name, str(ie), WARNING_HELP))
                    continue
                elif validate == 'error':
                    raise ValueError("%s failed to validate: %s. %s" % (friendly_name, str(ie), WARNING_HELP)) from ie
        else:
            raise TypeError("Expected a list of Transformers or DataFrames, but received unexpected type %s for retrieval system at position %d. %s" % 
                            (str(type(system)), i, WARNING_HELP))

        if required_cols.difference(found_cols):
            message = "Transformer %s (%s) at position %i does not produce all required columns %s, found only %s. %s" % (
                name, str(system), i, str(list(required_cols)), str(found_cols), WARNING_HELP)
            if validate == 'warn':
                warn(
                    message)
            elif validate == 'error':
                raise ValueError(message)