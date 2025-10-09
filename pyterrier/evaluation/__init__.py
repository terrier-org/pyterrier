import pandas as pd
from ir_measures import Measure
from typing import Literal, Union, Sequence, Callable, Tuple, IO, Dict, Any
from .. import Transformer
import types

MEASURE_TYPE=Union[str,Measure]
MEASURES_TYPE=Sequence[MEASURE_TYPE]
SAVEMODE_TYPE=Literal['reuse', 'overwrite', 'error', 'warn']
VALIDATE_TYPE = Literal['warn', 'error', 'ignore']

SYSTEM_OR_RESULTS_TYPE = Union[Transformer, pd.DataFrame]
SAVEFORMAT_TYPE = Union[Literal['trec'], types.ModuleType, Tuple[Callable[[IO], pd.DataFrame], Callable[[pd.DataFrame, IO], None]]]


NUMERIC_TYPE = Union[float,int,complex]
TEST_FN_TYPE = Callable[ [Sequence[NUMERIC_TYPE],Sequence[NUMERIC_TYPE]], Tuple[Any,NUMERIC_TYPE] ]

from ._experiment import Experiment
from ._execution import _run_and_evaluate

from ._experiment import Experiment
from ._grid import GridScan, GridSearch, KFoldGridSearch

def Evaluate(res : pd.DataFrame, qrels : pd.DataFrame, metrics : MEASURES_TYPE= ['map', 'ndcg'], perquery : bool = False) -> Dict:
    """
    Evaluate a single result dataframe with the given qrels. This method may be used as an alternative to
    ``pt.Experiment()`` for getting only the evaluation measurements given a single set of existing results.

    The PyTerrier-way is to use ``pt.Experiment()`` to evaluate a set of transformers, but this method is useful
    if you have a set of results already, and want to evaluate them without having to create a transformer pipeline.

    :param res: Either a dataframe with columns=['qid', 'docno', 'score'] or a dict {qid:{docno:score,},}
    :param qrels: Either a dataframe with columns=['qid','docno', 'label'] or a dict {qid:{docno:label,},}
    :param metrics: A list of strings specifying which evaluation metrics to use. Default=['map', 'ndcg']
    :param perquery: If true return each metric for each query, else return mean metrics. Default=False
    """
    if len(res) == 0:
        raise ValueError("No results for evaluation")

    _, rtr = _run_and_evaluate(res, None, qrels, metrics, perquery=perquery)
    return rtr

__all__ = ["Experiment", "Evaluate", "GridScan", "GridSearch", "KFoldGridSearch"]