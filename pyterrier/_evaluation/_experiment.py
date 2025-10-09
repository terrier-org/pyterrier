import os
import pandas as pd
from typing import Union, Dict, Tuple, Sequence, Literal, Optional, overload, Any
import types

from ._execution import _run_and_evaluate, _precomputation
from ._rendering import EvaluationDataTuple, RenderFromPerQuery
from ._validation import _validate
from . import SYSTEM_OR_RESULTS_TYPE, MEASURES_TYPE, TEST_FN_TYPE, SAVEFORMAT_TYPE, SAVEMODE_TYPE, VALIDATE_TYPE
import pyterrier as pt

# perquery: bool, dataframe:true
@overload
def Experiment(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Optional[Sequence[str]] = None,
        perquery : Union[Literal[False],Literal[True]] = False,
        dataframe : Literal[True] = True,
        batch_size : Optional[int] = None,
        filter_by_qrels : bool = False,
        filter_by_topics : bool = True,
        baseline : Optional[int] = None,
        test : Union[str,TEST_FN_TYPE] = "t",
        correction : Optional[str] = None,
        correction_alpha : float = 0.05,
        highlight : Optional[str] = None,
        round : Optional[Union[int,Dict[str,int]]] = None,
        verbose : bool = False,
        validate : VALIDATE_TYPE = 'warn',
        save_dir : Optional[str] = None,
        save_mode : SAVEMODE_TYPE = 'warn',
        save_format : SAVEFORMAT_TYPE = 'trec',
        precompute_prefix : bool = False,
        **kwargs) -> pd.DataFrame:
    ...

# perquery: bool, dataframe:False
@overload
def Experiment(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Optional[Sequence[str]] = None,
        perquery : Union[Literal[False],Literal[True]] = False,
        dataframe : Literal[False] = False,
        batch_size : Optional[int] = None,
        filter_by_qrels : bool = False,
        filter_by_topics : bool = True,
        baseline : Optional[int] = None,
        test : Union[str,TEST_FN_TYPE] = "t",
        correction : Optional[str] = None,
        correction_alpha : float = 0.05,
        highlight : Optional[str] = None,
        round : Optional[Union[int,Dict[str,int]]] = None,
        verbose : bool = False,
        validate : VALIDATE_TYPE = 'warn',
        save_dir : Optional[str] = None,
        save_mode : SAVEMODE_TYPE = 'warn',
        save_format : SAVEFORMAT_TYPE = 'trec',
        precompute_prefix : bool = False,
        **kwargs) -> Dict[str,Any]:
    ...

# perquery: 'both', dataframe:True
@overload
def Experiment(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Optional[Sequence[str]] = None,
        perquery : Literal['both'] = 'both',
        dataframe : Literal[True] = True,
        batch_size : Optional[int] = None,
        filter_by_qrels : bool = False,
        filter_by_topics : bool = True,
        baseline : Optional[int] = None,
        test : Union[str,TEST_FN_TYPE] = "t",
        correction : Optional[str] = None,
        correction_alpha : float = 0.05,
        highlight : Optional[str] = None,
        round : Optional[Union[int,Dict[str,int]]] = None,
        verbose : bool = False,
        validate : VALIDATE_TYPE = 'warn',
        save_dir : Optional[str] = None,
        save_mode : SAVEMODE_TYPE = 'warn',
        save_format : SAVEFORMAT_TYPE = 'trec',
        precompute_prefix : bool = False,
        **kwargs) -> Tuple[pd.DataFrame,pd.DataFrame]:
    ...

# perquery: 'both', dataframe:False
@overload
def Experiment(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Optional[Sequence[str]] = None,
        perquery : Literal['both'] = 'both',
        dataframe : Literal[False] = False,
        batch_size : Optional[int] = None,
        filter_by_qrels : bool = False,
        filter_by_topics : bool = True,
        baseline : Optional[int] = None,
        test : Union[str,TEST_FN_TYPE] = "t",
        correction : Optional[str] = None,
        correction_alpha : float = 0.05,
        highlight : Optional[str] = None,
        round : Optional[Union[int,Dict[str,int]]] = None,
        verbose : bool = False,
        validate : VALIDATE_TYPE = 'warn',
        save_dir : Optional[str] = None,
        save_mode : SAVEMODE_TYPE = 'warn',
        save_format : SAVEFORMAT_TYPE = 'trec',
        precompute_prefix : bool = False,
        **kwargs) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    ...

def Experiment(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Optional[Sequence[str]] = None,
        perquery : Union[bool, Literal['both']] = False,
        dataframe : Union[Literal[False], Literal[True]] = True,
        batch_size : Optional[int] = None,
        filter_by_qrels : bool = False,
        filter_by_topics : bool = True,
        baseline : Optional[int] = None,
        test : Union[str,TEST_FN_TYPE] = "t",
        correction : Optional[str] = None,
        correction_alpha : float = 0.05,
        highlight : Optional[str] = None,
        round : Optional[Union[int,Dict[str,int]]] = None,
        verbose : bool = False,
        validate : VALIDATE_TYPE = 'warn',
        save_dir : Optional[str] = None,
        save_mode : SAVEMODE_TYPE = 'warn',
        save_format : SAVEFORMAT_TYPE = 'trec',
        precompute_prefix : bool = False,
        **kwargs):
    """
    Allows easy comparison of multiple retrieval transformer pipelines using a common set of topics, and
    identical evaluation measures computed using the same qrels. In essence, each transformer is applied on 
    the provided set of topics. Then the named evaluation measures are computed for each system.

    :param retr_systems: A list of transformers to evaluate. If you already have the results for one 
        (or more) of your systems, a results dataframe can also be used here. Results produced by 
        the transformers must have "qid", "docno", "score", "rank" columns.
    :param topics: Either a path to a topics file or a pandas.Dataframe with columns=['qid', 'query']
    :param qrels: Either a path to a qrels file or a pandas.Dataframe with columns=['qid','docno', 'label']   
    :param eval_metrics: Which evaluation metrics to use. E.g. ['map']
    :param names: List of names for each retrieval system when presenting the results.
        Default=None. If None: Obtains the `str()` representation of each transformer as its name.
    :param batch_size: If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
        Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
        during a run.
    :param filter_by_qrels: If True, will drop topics from the topics dataframe that have qids not appearing in the qrels dataframe. 
    :param filter_by_topics: If True, will drop topics from the qrels dataframe that have qids not appearing in the topics dataframe. 
    :param perquery: If True return each metric for each query, if False, will return mean metrics across all queries. If both, will return both averages and perquery results in a tuple. Default=False.
    :param save_dir: If set to the name of a directory, the results of each transformer will be saved in TREC-formatted results file, whose 
        filename is based on the systems names (as specified by ``names`` kwarg). If the file exists and ``save_mode`` is set to "reuse", then the file
        will be used for evaluation rather than the transformer. Default is None, such that saving and loading from files is disabled.
    :param save_mode: Defines how existing files are used when ``save_dir`` is set. If set to "reuse", then files will be preferred
        over transformers for evaluation. If set to "overwrite", existing files will be replaced. If set to "warn" or "error", the presence of any 
        existing file will cause a warning or error, respectively. Default is "warn".
    :param save_format: How are result being saved. Defaults to 'trec', which uses ``pt.io.read_results()`` and ``pt.io.write_results()`` for saving system outputs. 
        If TREC results format is insufficient, set ``save_format=pickle``. Alternatively, a tuple of read and write function can be specified, for instance, 
        ``save_format=(pandas.from_csv, pandas.DataFrame.to_csv)``, or even ``save_format=(pandas.from_parquet, pandas.DataFrame.to_parquet)``.
    :param dataframe: If True return results as a dataframe, else as a dictionary of dictionaries. Default=True.
    :param baseline: If set to the index of an item of the retr_system list, will calculate the number of queries 
        improved, degraded and the statistical significance (paired t-test p value) for each measure.
        Default=None: If None, no additional columns will be added for each measure.
    :param test: Which significance testing approach to apply. Defaults to "t". Alternatives are "wilcoxon" - not typically used for IR experiments. A Callable can also be passed - it should
        follow the specification of `scipy.stats.ttest_rel() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html>`_, 
        i.e. it expect two arrays of numbers, and return an array or tuple, of which the second value will be placed in the p-value column.
    :param correction: Whether any multiple testing correction should be applied. E.g. 'bonferroni', 'holm', 'hs' aka 'holm-sidak'. Default is None.
        Additional columns are added denoting whether the null hypothesis can be rejected, and the corrected p value. 
        See `statsmodels.stats.multitest.multipletests() <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html#statsmodels.stats.multitest.multipletests>`_
        for more information about available testing correction.
    :param correction_alpha: What alpha value for multiple testing correction. Default is 0.05.
    :param highlight: If `highlight="bold"`, highlights in bold the best measure value in each column; 
        if `highlight="color"` or `"colour"`, then the cell with the highest metric value will have a green background.
    :param round: How many decimal places to round each measure value to. This can also be a dictionary mapping measure name to number of decimal places.
        Default is None, which is no rounding.
    :param precompute_prefix: If set to True, then pt.Experiment will look for a common prefix on all input pipelines, and execute that common prefix pipeline only once. 
        This functionality assumes that the intermidiate results of the common prefix can fit in memory. Set to False by default.
    :param verbose: If True, a tqdm progress bar is shown as systems (or systems*batches if batch_size is set) are executed. Default=False.
    :param validate: If set to value other than 'ignore', each transformer is validated against the topics dataframe, to ensure that it produces the expected output columns.
        ``pt.inspect.transformer_outputs()`` is used to determine the output columns. If 'warn', then transformers whose output columns don't match the columns required 
        by the specified evaluation measures will product warnings; If 'error', then an error is produced. If a transformer cannot be inspected, a warning is produced.

    :return: A Dataframe/dict with each retrieval system with each metric evaluated, or alternatively a tuple with averages and perquery results. 
    """
    
    if not isinstance(retr_systems, list):
        raise TypeError("Expected list of transformers for retr_systems, instead received %s" % str(type(retr_systems)))

    if len(kwargs):
        raise TypeError("Unknown kwargs: %s" % (str(list(kwargs.keys()))))

    if baseline is not None:
        assert int(baseline) >= 0 and int(baseline) < len(retr_systems)
        assert not perquery

    if isinstance(topics, str):
        if os.path.isfile(topics):
            topics = pt.io.read_topics(topics)
    if isinstance(qrels, str):
        if os.path.isfile(qrels):
            qrels = pt.io.read_qrels(qrels)

    if round is not None:
        if isinstance(round, int):
            assert round >= 0, "round argument should be integer >= 0, not %s" % str(round)
        elif isinstance(round, dict):
            assert not perquery, "Sorry, per-measure rounding only support when reporting means" 
            for k,v in round.items():
                assert isinstance(v, int) and v >= 0, "rounding number for measure %s should be integer >= 0, not %s" % (k, str(v))
        else:
            raise ValueError("Argument round should be an integer or a dictionary")

    if correction is not None and baseline is None:
        raise ValueError("Requested multiple testing correction, but no baseline was specified.")

    # drop queries not appear in the qrels
    if filter_by_qrels:
        # the commented variant would drop queries not having any RELEVANT labels
        # topics = topics.merge(qrels[qrels["label"] > 0][["qid"]].drop_duplicates())        
        topics = topics.merge(qrels[["qid"]].drop_duplicates())
        if len(topics) == 0:
            raise ValueError('There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False.')

    # drop qrels not appear in the topics
    if filter_by_topics:
        qrels = qrels.merge(topics[["qid"]].drop_duplicates())
        if len(qrels) == 0:
            raise ValueError('There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False.')

    from scipy import stats
    test_fn : TEST_FN_TYPE
    if test == "t":
        test_fn = stats.ttest_rel
    elif test == "wilcoxon":
        test_fn = stats.wilcoxon
    else:
        assert not isinstance(test, str), "Unknown test function name %s" % test
        test_fn = test
    
    # obtain system names if not specified
    if names is None:
        names = [str(system) for system in retr_systems]
    elif len(names) != len(retr_systems):
        raise ValueError("names should be the same length as retr_systems")

    # validate save_dir and resulting filenames
    if save_dir is not None:
        if not os.path.exists(save_dir):
            raise ValueError("save_dir %s does not exist" % save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError("save_dir %s is not a directory" % save_dir)
        from ..io import ok_filename
        for n in names:
            if not ok_filename(n):
                raise ValueError("Name contains bad characters and save_dir is set, name is %s" % n)
        if len(set(names)) < len(names):
            raise ValueError("save_dir is set, but names are not unique. Use names= to set unique names")

    all_topic_qids = topics["qid"].values

    mrt_needed = False
    if "mrt" in eval_metrics:
        mrt_needed = True
        eval_metrics = list(eval_metrics).copy()
        eval_metrics.remove("mrt")

    # validate the transformers produce the expected columns
    _validate(retr_systems, topics, eval_metrics, names, validate)

    # split the transformers into a common prefix and individual suffixes, improved efficiency
    precompute_time, execution_topics, execution_retr_systems = _precomputation(retr_systems, topics, precompute_prefix, verbose, batch_size)

    # progress bar construction
    tqdm_args={
        'disable' : not verbose,
        'unit' : 'system',
        'total' : len(retr_systems),
        'desc' : 'pt.Experiment'
    }

    if batch_size is not None:
        import math
        tqdm_args['unit'] = 'batches'
        # round number of batches up for each system
        tqdm_args['total'] = math.ceil((len(topics) / batch_size)) * len(retr_systems)

    renderer = RenderFromPerQuery(names, 
                                  baseline=baseline, 
                                  test_fn=test_fn, 
                                  correction=correction, 
                                  correction_alpha=correction_alpha, 
                                  round=round, 
                                  precompute_time=precompute_time)
    with pt.tqdm(**tqdm_args) as pbar:
        # run and evaluate each system
        for sysid, (name, system) in enumerate(zip(names, execution_retr_systems)):
            save_file = None
            if save_dir is not None:
                if save_format == 'trec':
                    save_ext = 'res.gz'
                elif isinstance(save_format, types.ModuleType):
                    save_ext = 'mod'
                elif isinstance(save_format, tuple):
                    save_ext = 'custom'
                else:
                    raise ValueError("Unrecognised save_mode %s" % str(save_format)) 
                save_file = os.path.join(save_dir, "%s.%s" % (name, save_ext))

            time, evalMeasuresDict = _run_and_evaluate(
                system, execution_topics, qrels, eval_metrics, 
                perquery=True, 
                batch_size=batch_size, 
                backfill_qids=all_topic_qids if perquery else None,
                save_file=save_file,
                save_mode=save_mode,
                save_format=save_format,
                pbar=pbar)
            renderer.add_metrics(sysid, evalMeasuresDict, time)

    if not perquery:
        return renderer.averages(dataframe=dataframe, highlight=highlight, mrt_needed=mrt_needed)
    
    perquery_results = renderer.perquery(dataframe=dataframe)
    if perquery == 'both':
        from typing import reveal_type
        average_results = renderer.averages(dataframe=dataframe, highlight=highlight)

        if dataframe:
            from typing import cast as tcast
            average_results_df = tcast(pd.DataFrame, average_results)
            perquery_results_df = tcast(pd.DataFrame, perquery_results)

            average_results_df.style.set_caption("Averages")
            perquery_results_df.style.set_caption("Per Query")
            return EvaluationDataTuple(average_results, perquery_results)
        return (average_results, perquery_results)
    return perquery_results
