import sys
from typing import Callable, Tuple
import platform
from functools import wraps
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as eps
import pandas as pd
from collections import defaultdict
from deprecated import deprecated
import pyterrier as pt


@deprecated(version="0.9")
def convert_qrels_to_dict(df):
    """
    Convert a qrels dataframe to dictionary for use in pytrec_eval

    Args:
        df(pandas.Dataframe): The dataframe to convert

    Returns:
        dict: {qid:{docno:label,},}
    """
    run_dict_pytrec_eval = defaultdict(dict)
    for row in df.itertuples():
        run_dict_pytrec_eval[row.qid][row.docno] = int(row.label)
    return(run_dict_pytrec_eval)


@deprecated(version="0.9")
def convert_qrels_to_dataframe(qrels_dict) -> pd.DataFrame:
    """
    Convert a qrels dictionary to a dataframe

    Args:
        qrels_dict(Dict[str, Dict[str, int]]): {qid:{docno:label,},}

    Returns:
        pd.DataFrame: columns=['qid', 'docno', 'label']
    """
    result = {'qid': [], 'docno': [], 'label': []}
    for qid in qrels_dict:
        for docno, label in qrels_dict[qid]:
            result['qid'].append(qid)
            result['docno'].append(docno)
            result['label'].append(label)

    return pd.DataFrame(result)


@deprecated(version="0.9")
def convert_res_to_dict(df):
    """
    Convert a result dataframe to dictionary for use in pytrec_eval

    Args:
        df(pandas.Dataframe): The dataframe to convert

    Returns:
        dict: {qid:{docno:score,},}
    """
    run_dict_pytrec_eval = defaultdict(dict)
    for row in df.itertuples():
        run_dict_pytrec_eval[row.qid][row.docno] = float(row.score)
    return(run_dict_pytrec_eval)


@deprecated(version="0.9", reason="Use pt.Evaluate instead")
def evaluate(res : pd.DataFrame, qrels : pd.DataFrame, metrics=['map', 'ndcg'], perquery=False):
    from .pipelines import Evaluate
    return Evaluate(res, qrels, metrics=metrics, perquery=perquery)


@deprecated(version="0.9")
def mean_of_measures(result, measures=None, num_q = None):
    from .pipelines import _mean_of_measures
    return _mean_of_measures(result, measures=measures, num_q=num_q)


def once() -> Callable:
    """
    Wraps a function that can only be called once. Subsequent calls will raise an error.
    """
    def _once(fn: Callable) -> Callable:
        called = False

        @wraps(fn)
        def _wrapper(*args, **kwargs):
            nonlocal called
            if called:
                raise ValueError(f"{fn.__name__} has already been run")
            # how to handle errors?
            res = fn(*args, **kwargs)
            called = True
            return res
        _wrapper.called = lambda: called  # type: ignore
        return _wrapper
    return _once


def entry_points(group: str) -> Tuple[EntryPoint, ...]:
    """
    A shim for Python<=3.X to support importlib.metadata.entry_points(group).
    Also ensures that no duplicates (by name) are returned.

    See <https://docs.python.org/3/library/importlib.metadata.html#entry-points> for more details.
    """
    try:
        orig_res = tuple(eps(group=group))
    except TypeError:
        orig_res = tuple(eps().get(group, tuple()))

    names = set()
    res = []
    for ep in orig_res:
        if ep.name not in names:
            res.append(ep)
            names.add(ep.name)

    return tuple(res)


def is_windows() -> bool:
    return platform.system() == 'Windows'


def noop(*args, **kwargs):
    pass


def set_tqdm(type=None):
    """
        Set the tqdm progress bar type that Pyterrier will use internally.
        Many PyTerrier transformations can be expensive to apply in some settings - users can
        view progress by using the verbose=True kwarg to many classes, such as BatchRetrieve.

        The `tqdm <https://tqdm.github.io/>`_ progress bar can be made prettier when using appropriately configured Jupyter notebook setups.
        We use this automatically when Google Colab is detected.

        Allowable options for type are:

         - `'tqdm'`: corresponds to the standard text progresss bar, ala `from tqdm import tqdm`.
         - `'notebook'`: corresponds to a notebook progress bar, ala `from tqdm.notebook import tqdm`
         - `'auto'`: allows tqdm to decide on the progress bar type, ala `from tqdm.auto import tqdm`. Note that this works fine on Google Colab, but not on Jupyter unless the `ipywidgets have been installed <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`_.
    """
    if type is None:
        if 'google.colab' in sys.modules:
            type = 'notebook'
        else:
            type = 'tqdm'
    
    if type == 'tqdm':
        from tqdm import tqdm as bartype
        pt.tqdm = bartype
    elif type == 'notebook':
        from tqdm.notebook import tqdm as bartype
        pt.tqdm = bartype
    elif type == 'auto':
        from tqdm.auto import tqdm as bartype
        pt.tqdm = bartype
    else:
        raise ValueError(f"Unknown tqdm type {type}")
    pt.tqdm.pandas()
