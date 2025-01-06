from warnings import warn
import os
import sys
import pandas as pd
import numpy as np
from typing import Callable, Iterator, Union, Dict, List, Tuple, Sequence, Any, Literal, Optional, overload
import types
from . import Transformer
from .model import coerce_dataframe_types
from ._ops import Compose
import ir_measures
import tqdm as tqdm_module
from ir_measures import Measure, Metric
import pyterrier as pt
MEASURE_TYPE=Union[str,Measure]
MEASURES_TYPE=Sequence[MEASURE_TYPE]
SAVEMODE_TYPE=Literal['reuse', 'overwrite', 'error', 'warn']

SYSTEM_OR_RESULTS_TYPE = Union[Transformer, pd.DataFrame]
SAVEFORMAT_TYPE = Union[Literal['trec'], types.ModuleType]

def _bold_cols(data : pd.Series, col_type):
    if data.name not in col_type:
        return [''] * len(data)
    
    colormax_attr = 'font-weight: bold'
    colormaxlast_attr = 'font-weight: bold'
    if col_type[data.name] == "+":  
        max_value = data.max()
    else:
        max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

def _color_cols(data : pd.Series, col_type, 
                       colormax='antiquewhite', colormaxlast='lightgreen', 
                       colormin='antiquewhite', colorminlast='lightgreen' ):
    if data.name not in col_type:
      return [''] * len(data)
    
    if col_type[data.name] == "+":
      colormax_attr = f'background-color: {colormax}'
      colormaxlast_attr = f'background-color: {colormaxlast}'
      max_value = data.max()
    else:
      colormax_attr = f'background-color: {colormin}'
      colormaxlast_attr = f'background-color: {colorminlast}'
      max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

_irmeasures_columns = {
    'qid' : 'query_id',
    'docno' : 'doc_id'
}

def _mean_of_measures(result, measures=None, num_q = None):
        if len(result) == 0:
            raise ValueError("No measures received - perhaps qrels and topics had no results in common")
        measures_sum = {}
        mean_dict = {}
        if measures is None:
            measures = list(next(iter(result.values())).keys())
        measures_remove = ["runid"]
        for m in measures_remove:
            if m in measures:
                measures.remove(m)
        measures_no_mean = set(["num_q", "num_rel", "num_ret", "num_rel_ret"])
        for val in result.values():
            for measure in measures:
                measure_val = val[measure]
                measures_sum[measure] = measures_sum.get(measure, 0.0) + measure_val
        if num_q is None:
            num_q = len(result.values())
        for measure, value in measures_sum.items():
            mean_dict[measure] = value / (1 if measure in measures_no_mean else num_q)
        return mean_dict

def _convert_measures(metrics : MEASURES_TYPE) -> Tuple[Sequence[Measure], Dict[Measure,str]]:
    from ir_measures import parse_trec_measure
    rtr = []
    rev_mapping = {}
    for m in metrics:
        if isinstance(m, Measure):
            rtr.append(m)
            continue
        elif isinstance(m, str):
            measures = parse_trec_measure(m)
            if len(measures) == 1:
                metric = measures[0]
                rtr.append(metric)
                rev_mapping[metric] = m
            elif len(measures) > 1:
                #m is family nickname, e.g. 'official;
                rtr.extend(measures)
            else:
                raise KeyError("Could not convert measure %s" % m)
        else:
            raise KeyError("Unknown measure %s of type %s" % (str(m), str(type(m))))
    assert len(rtr) > 0, "No measures were found in %s" % (str(metrics))
    return rtr, rev_mapping

#list(iter_calc([ir_measures.AP], qrels, run))
#[Metric(query_id='Q0', measure=AP, value=1.0), Metric(query_id='Q1', measure=AP, value=1.0)]
def _ir_measures_to_dict(
        seq : Iterator[Metric], 
        measures: Sequence[Measure],
        rev_mapping : Dict[Measure,str], 
        num_q : int,
        perquery : bool = True,
        backfill_qids : Optional[Sequence[str]] = None) -> Union[ Dict[str, Dict[str, float]], Dict[str, float]]:
    from collections import defaultdict
    if perquery:
        # qid -> measure -> value
        rtr_perquery : Dict[str, Dict[str, float]] = defaultdict(dict)
        for metric in seq:
            measure = metric.measure
            measure_name = rev_mapping.get(measure, str(measure))
            rtr_perquery[metric.query_id][measure_name] = metric.value
        # When reporting per-query results, it can desirable to show something for topics that were executed
        # do not have corresponding qrels. If the caller passes in backfill_qids, we'll ensure that these
        # qids are present, and if not add placeholders with NaN values for all measures.
        if backfill_qids is not None:
            backfill_count = 0
            for qid in backfill_qids:
                if qid not in rtr_perquery:
                    backfill_count += 1
                    for m in measures:
                        rtr_perquery[qid][rev_mapping.get(m, str(m))] = float('NaN')
            if backfill_count > 0:
                warn(f'{backfill_count} topic(s) not found in qrels. Scores for these topics are given as NaN and should not contribute to averages.')
        return rtr_perquery
    assert backfill_qids is None, "backfill_qids only supported when perquery=True"
    
    metric_agg = {rev_mapping.get(m, str(m)): m.aggregator() for m in measures}
    for metric in seq:
        measure_name = rev_mapping.get(metric.measure, str(metric.measure))
        metric_agg[measure_name].add(metric.value)
    
    rtr_aggregated : Dict[str,float] = {} # measure -> value
    for m_name in metric_agg:
        rtr_aggregated[m_name] = metric_agg[m_name].result()
    return rtr_aggregated

def _common_prefix(pipes: List[List[Transformer]]) -> Tuple[List[Transformer], List[List[Transformer]]]:
    # finds the common prefix within a list of transformers
    assert len(pipes) > 1, "pipes must contain at least two lists"
    common_prefix = []
    for stage in zip(*pipes):
        elements = set(stage) # uses Transformer.__equals__ 
        if len(elements) == 1:
            common_prefix.append(next(iter(elements)))
        else:
            break
    suffixes = [p[len(common_prefix):] for p in pipes]
    return common_prefix, suffixes

def _identifyCommon(pipes : List[Union[pt.Transformer, pd.DataFrame]]) -> Tuple[Optional[pt.Transformer], List[pt.Transformer]]:
    # constructs a common prefix pipeline across a list of pipelines, along with various suffices. 
    # pt.Transformer.identity() is used for a no-op suffix
    
    # no precomputation for single-system case
    if len(pipes) == 1:
        return None, pipes
    pipe_lists: List[List[pt.Transformer]] = []
    for p in pipes:
        # no optimisation possible for experiments involving dataframes as systems
        if isinstance(p, pd.DataFrame):
            return None, pipes
        if isinstance(p, Compose):
            pipe_lists.append(list(p._transformers))
        else:
            if not isinstance(p, pt.Transformer):
                raise ValueError("pt.Experiment has systems that are not either DataFrames or Transformers")
            pipe_lists.append([p])
    
    common_prefix, suffices  = _common_prefix(pipe_lists)

    if len(common_prefix) == 0:
        # no common prefix, return existing pipelines as-is
        return None, pipes
    
    def _construct(inp: List[pt.Transformer]) -> pt.Transformer:
        # use identify as a no-op
        if len(inp) == 0:
            return pt.Transformer.identity() 
        # use transformer itself
        if len(inp) == 1:
            return inp[0]
        # more than 1, compose...
        return Compose(*inp)

    return (
        _construct(common_prefix), # prefix common to all 
        [ _construct(remainder) for remainder in suffices ] # individual suffices
    )

def _precomputation(
        retr_systems : List[SYSTEM_OR_RESULTS_TYPE], 
        topics : pd.DataFrame, 
        precompute_prefix : bool, 
        verbose : bool, 
        batch_size : Optional[int] = None
        ) -> Tuple[float, pd.DataFrame, List[SYSTEM_OR_RESULTS_TYPE]]:
    
    # this method identifies any common prefix to the pipelines in retr_systems,
    # and if precompute_prefix=True, then it computes the (partial) results of topics
    # on that prefix, and returns that, which is used later for evaluating the remainder
    # of each pipeline.

    tqdm_args_precompute: Dict[str, Any] = {
        'disable' : not verbose,
    }
    
    common_pipe, execution_retr_systems = _identifyCommon(retr_systems)
    precompute_time = 0.
    if precompute_prefix and common_pipe is not None: 
        print("Precomputing results of %d topics on shared pipeline component %s" % (len(topics), str(common_pipe)), file=sys.stderr)

        tqdm_args_precompute['desc'] = "pt.Experiment precomputation"
        from timeit import default_timer as timer
        starttime = timer()
        if batch_size is not None:
            
            warn("precompute_prefix with batch_size is very experimental. Please report any problems")
            import math
            tqdm_args_precompute['unit'] = 'batches'
            # round number of batches up for each system
            tqdm_args_precompute['total'] = math.ceil((len(topics) / batch_size))
            with pt.tqdm(**tqdm_args_precompute) as pbar:
                precompute_results : List[pd.DataFrame] = []
                for r in common_pipe.transform_gen(topics, batch_size=batch_size):
                    assert isinstance(r, pd.DataFrame) # keep mypy happy
                    precompute_results.append(r)
                    pbar.update(1)
                execution_topics = pd.concat(precompute_results)
        
        else: # no batching  
            tqdm_args_precompute['total'] = 1 
            tqdm_args_precompute['unit'] = "prefix pipeline"
            with pt.tqdm(**tqdm_args_precompute) as pbar:
                execution_topics = common_pipe(topics)
                pbar.update(1)
        
        endtime = timer()
        precompute_time = float(endtime - starttime) * 1000.

    elif precompute_prefix and common_pipe is None:
        warn('precompute_prefix was True for pt.Experiment, but no common pipeline prefix was found among %d pipelines' % len(retr_systems))
        execution_retr_systems = retr_systems
        execution_topics = topics
    
    else: # precomputation not requested
        execution_retr_systems = retr_systems
        execution_topics = topics
    
    return precompute_time, execution_topics, execution_retr_systems

def _run_and_evaluate(
        system : SYSTEM_OR_RESULTS_TYPE, 
        topics : Optional[pd.DataFrame], 
        qrels: pd.DataFrame, 
        metrics : MEASURES_TYPE, 
        pbar : Optional[tqdm_module.tqdm] = None,
        save_mode : Optional[SAVEMODE_TYPE] = None,
        save_file : Optional[str] = None,
        save_format : SAVEFORMAT_TYPE = 'trec',
        perquery : bool = False,
        batch_size : Optional[int] = None,
        backfill_qids : Optional[Sequence[str]] = None):
    
    from .io import read_results, write_results

    if pbar is None:
        pbar = pt.tqdm(disable=True)

    metrics, rev_mapping = _convert_measures(metrics)
    qrels = qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
    from timeit import default_timer as timer
    runtime : float = 0.
    num_q = qrels['query_id'].nunique()
    if save_file is not None and os.path.exists(save_file):
        if save_mode == 'reuse':
            if save_format == 'trec':
                system = read_results(save_file)
            elif isinstance(save_format, types.ModuleType):
                with pt.io.autoopen(save_file, 'rb') as fin:
                    system = save_format.load(fin)
            elif isinstance(save_format, tuple) and len(save_format) == 2:
                with pt.io.autoopen(save_file, 'rb') as fin:
                    system = save_format[0](fin)
            else:
                raise ValueError("Unknown save_format %s" % str(save_format))  
        elif save_mode == 'overwrite':
            os.remove(save_file)
        elif save_mode == 'warn':
            warn(("save_dir is set, but the file '%s' already exists. If you are aware of are happy to reuse this " % save_file)+
                             "file to speed up evaluation, set save_mode='reuse'; if you want to overwrite it, set save_mode='overwrite'."+
                             " To make this condition an error, use save_mode='error'.")
        elif save_mode == 'error':
            raise ValueError(("save_dir is set, but the file '%s' already exists. If you are aware of are happy to reuse this " % save_file)+
                             "file to speed up evaluation, set save_mode='reuse'; if you want to overwrite it, set save_mode='overwrite'."+
                              "To make this condition a warning, use save_mode='warn'.")
        else:
            raise ValueError("Unknown save_mode argument '%s', valid options are 'error', 'warn', 'reuse' or 'overwrite'." % save_mode)

    res : pd.DataFrame
    # if its a DataFrame, use it as the results
    if isinstance(system, pd.DataFrame):
        res = system
        res = coerce_dataframe_types(res)
        if len(res) == 0:
            if topics is None:
                raise ValueError("No topics specified, and no results in dataframe")
            else:
                raise ValueError("%d topics, but no results in dataframe" % len(topics))
        evalMeasuresDict = _ir_measures_to_dict(
            ir_measures.iter_calc(metrics, qrels, res.rename(columns=_irmeasures_columns)), 
            metrics,
            rev_mapping,
            num_q,
            perquery,
            backfill_qids)
        pbar.update()

    elif batch_size is None:

        assert topics is not None, "topics must be specified"
        #transformer, evaluate all queries at once
            
        starttime = timer()
        res = system.transform(topics)
        endtime = timer()
        runtime =  float(endtime - starttime) * 1000.

        # write results to save_file; we can be sure this file does not exist
        if save_file is not None:
            if save_format == 'trec':
                write_results(res, save_file)
            elif isinstance(save_format, types.ModuleType):
                with pt.io.autoopen(save_file, 'wb') as fout:
                    save_format.dump(res, fout)
            elif isinstance(save_format, tuple) and len(save_format) == 2:
                with pt.io.autoopen(save_file, 'wb') as fout:
                    save_format[1](res, fout)
            else:
                raise ValueError("Unknown save_format %s" % str(save_format))    
        res = coerce_dataframe_types(res)

        if len(res) == 0:
            raise ValueError("%d topics, but no results received from %s" % (len(topics), str(system)) )

        evalMeasuresDict = _ir_measures_to_dict(
            ir_measures.iter_calc(metrics, qrels, res.rename(columns=_irmeasures_columns)), 
            metrics,
            rev_mapping,
            num_q,
            perquery,
            backfill_qids)
        pbar.update()
    else:
        assert topics is not None, "topics must be specified"
        if save_file is not None:
            # only 
            assert save_format == 'trec', 'save_format=%s is not supported when save_dir is enabled and batch_size is not None' % str(save_format)
        
        #transformer, evaluate queries in batches
        assert batch_size > 0
        starttime = timer()
        evalMeasuresDict = {}
        remaining_qrel_qids = set(qrels.query_id)
        try:
            batch_topics : pd.DataFrame
            for i, (res, batch_topics) in enumerate( system.transform_gen(topics, batch_size=batch_size, output_topics=True)):
                if len(res) == 0:
                    raise ValueError("batch of %d topics, but no results received in batch %d from %s" % (len(batch_topics), i, str(system) ) )
                endtime = timer()
                runtime += (endtime - starttime) * 1000.

                # write results to save_file; we will append for subsequent batches
                if save_file is not None:
                    write_results(res, save_file, append=True)

                res = coerce_dataframe_types(res)
                batch_qids = set(batch_topics.qid)
                batch_qrels = qrels[qrels.query_id.isin(batch_qids)] # filter qrels down to just the qids that appear in this batch
                remaining_qrel_qids.difference_update(batch_qids)
                batch_backfill = [qid for qid in backfill_qids if qid in batch_qids] if backfill_qids is not None else None
                evalMeasuresDict.update(_ir_measures_to_dict(
                    ir_measures.iter_calc(metrics, batch_qrels, res.rename(columns=_irmeasures_columns)),
                    metrics,
                    rev_mapping,
                    num_q,
                    perquery=True,
                    backfill_qids=batch_backfill))
                pbar.update()
                starttime = timer()
        except:
            # if an error is thrown, we need to clean up our existing file
            if save_file is not None and os.path.exists(save_file):
                os.remove(save_file)
            raise
        if remaining_qrel_qids:
            # there are some qids in the qrels that were not in the topics. Get the default values for these and update evalMeasuresDict
            missing_qrels = qrels[qrels.query_id.isin(remaining_qrel_qids)]
            empty_res = pd.DataFrame([], columns=['query_id', 'doc_id', 'score'])
            evalMeasuresDict.update(_ir_measures_to_dict(
                ir_measures.iter_calc(metrics, missing_qrels, empty_res),
                metrics,
                rev_mapping,
                num_q,
                perquery=True))
        if not perquery:
            # aggregate measures if not in per query mode
            aggregators = {rev_mapping.get(m, str(m)): m.aggregator() for m in metrics}
            q : str
            for q in evalMeasuresDict:
                for metric in metrics:
                    s_metric = rev_mapping.get(metric, str(metric))
                    aggregators[s_metric].add(evalMeasuresDict[q][s_metric])
            evalMeasuresDict = {m: agg.result() for m, agg in aggregators.items()}
    return (runtime, evalMeasuresDict)

NUMERIC_TYPE = Union[float,int,complex]
TEST_FN_TYPE = Callable[ [Sequence[NUMERIC_TYPE],Sequence[NUMERIC_TYPE]], Tuple[Any,NUMERIC_TYPE] ]

def Experiment(
        retr_systems : Sequence[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        eval_metrics : MEASURES_TYPE,
        names : Optional[Sequence[str]] = None,
        perquery : bool = False,
        dataframe : bool = True,
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
        save_dir : Optional[str] = None,
        save_mode : SAVEMODE_TYPE = 'warn',
        save_format : SAVEFORMAT_TYPE = 'trec',
        precompute_prefix : bool = False,
        **kwargs):
    """
    Allows easy comparison of multiple retrieval transformer pipelines using a common set of topics, and
    identical evaluation measures computed using the same qrels. In essence, each transformer is applied on 
    the provided set of topics. Then the named evaluation measures are computed for each system.

    Args:
        retr_systems(list): A list of transformers to evaluate. If you already have the results for one 
            (or more) of your systems, a results dataframe can also be used here. Results produced by 
            the transformers must have "qid", "docno", "score", "rank" columns.
        topics: Either a path to a topics file or a pandas.Dataframe with columns=['qid', 'query']
        qrels: Either a path to a qrels file or a pandas.Dataframe with columns=['qid','docno', 'label']   
        eval_metrics(list): Which evaluation metrics to use. E.g. ['map']
        names(list): List of names for each retrieval system when presenting the results.
            Default=None. If None: Obtains the `str()` representation of each transformer as its name.
        batch_size(int): If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
            Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
            during a run.
        filter_by_qrels(bool): If True, will drop topics from the topics dataframe that have qids not appearing in the qrels dataframe. 
        filter_by_topics(bool): If True, will drop topics from the qrels dataframe that have qids not appearing in the topics dataframe. 
        perquery(bool): If True return each metric for each query, else return mean metrics across all queries. Default=False.
        save_dir(str): If set to the name of a directory, the results of each transformer will be saved in TREC-formatted results file, whose 
            filename is based on the systems names (as specified by ``names`` kwarg). If the file exists and ``save_mode`` is set to "reuse", then the file
            will be used for evaluation rather than the transformer. Default is None, such that saving and loading from files is disabled.
        save_mode(str): Defines how existing files are used when ``save_dir`` is set. If set to "reuse", then files will be preferred
            over transformers for evaluation. If set to "overwrite", existing files will be replaced. If set to "warn" or "error", the presence of any 
            existing file will cause a warning or error, respectively. Default is "warn".
        save_format(str): How are result being saved. Defaults to 'trec', which uses ``pt.io.read_results()`` and ``pt.io.write_results()`` for saving system outputs. 
            If TREC results format is insufficient, set ``save_format=pickle``. Alternatively, a tuple of read and write function can be specified, for instance, 
            ``save_format=(pandas.from_csv, pandas.DataFrame.to_csv)``, or even ``save_format=(pandas.from_parquet, pandas.DataFrame.to_parquet)``.
        dataframe(bool): If True return results as a dataframe, else as a dictionary of dictionaries. Default=True.
        baseline(int): If set to the index of an item of the retr_system list, will calculate the number of queries 
            improved, degraded and the statistical significance (paired t-test p value) for each measure.
            Default=None: If None, no additional columns will be added for each measure.
        test(string): Which significance testing approach to apply. Defaults to "t". Alternatives are "wilcoxon" - not typically used for IR experiments. A Callable can also be passed - it should
            follow the specification of `scipy.stats.ttest_rel() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html>`_, 
            i.e. it expect two arrays of numbers, and return an array or tuple, of which the second value will be placed in the p-value column.
        correction(string): Whether any multiple testing correction should be applied. E.g. 'bonferroni', 'holm', 'hs' aka 'holm-sidak'. Default is None.
            Additional columns are added denoting whether the null hypothesis can be rejected, and the corrected p value. 
            See `statsmodels.stats.multitest.multipletests() <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html#statsmodels.stats.multitest.multipletests>`_
            for more information about available testing correction.
        correction_alpha(float): What alpha value for multiple testing correction. Default is 0.05.
        highlight(str): If `highlight="bold"`, highlights in bold the best measure value in each column; 
            if `highlight="color"` or `"colour"`, then the cell with the highest metric value will have a green background.
        round(int): How many decimal places to round each measure value to. This can also be a dictionary mapping measure name to number of decimal places.
            Default is None, which is no rounding.
        precompute_prefix(bool): If set to True, then pt.Experiment will look for a common prefix on all input pipelines, and execute that common prefix pipeline only once. 
            This functionality assumes that the intermidiate results of the common prefix can fit in memory. Set to False by default.
        verbose(bool): If True, a tqdm progress bar is shown as systems (or systems*batches if batch_size is set) are executed. Default=False.

    Returns:
        A Dataframe with each retrieval system with each metric evaluated.
    """
    
    
    # map to the old signature of Experiment
    warn_old_sig=False
    if isinstance(retr_systems, pd.DataFrame) and isinstance(topics, list):
        tmp = topics
        topics = retr_systems
        retr_systems = tmp
        warn_old_sig = True
    if isinstance(eval_metrics, pd.DataFrame) and isinstance(qrels, list):
        tmp = eval_metrics
        eval_metrics = qrels
        qrels = tmp
        warn_old_sig = True
    if warn_old_sig:
        warn(
            "Signature of Experiment() is now (retr_systems, topics, qrels, eval_metrics), please update your code", DeprecationWarning, 2)

    if not isinstance(retr_systems, list):
        raise TypeError("Expected list of transformers for retr_systems, instead received %s" % str(type(retr_systems)))

    if 'drop_unused' in kwargs:
        filter_by_qrels = kwargs.pop('drop_unused')
        warn(
            'drop_unused is deprecated; use filter_by_qrels instead', DeprecationWarning)
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

    def _apply_round(measure, value):
        import builtins
        if round is not None and isinstance(round, int):
            value = builtins.round(value, round)
        if round is not None and isinstance(round, dict) and measure in round:
            value = builtins.round(value, round[measure])
        return value

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
        from .io import ok_filename
        for n in names:
            if not ok_filename(n):
                raise ValueError("Name contains bad characters and save_dir is set, name is %s" % n)
        if len(set(names)) < len(names):
            raise ValueError("save_dir is set, but names are not unique. Use names= to set unique names")

    all_topic_qids = topics["qid"].values

    evalsRows=[]
    evalDict={}
    evalDictsPerQ=[]
    actual_metric_names=[]
    mrt_needed = False
    if "mrt" in eval_metrics:
        mrt_needed = True
        eval_metrics = list(eval_metrics).copy()
        eval_metrics.remove("mrt")

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

    with pt.tqdm(**tqdm_args) as pbar:
        # run and evaluate each system
        for name, system in zip(names, execution_retr_systems):
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
                perquery=perquery or baseline is not None, 
                batch_size=batch_size, 
                backfill_qids=all_topic_qids if perquery else None,
                save_file=save_file,
                save_mode=save_mode,
                save_format=save_format,
                pbar=pbar)

            if baseline is not None:
                evalDictsPerQ.append(evalMeasuresDict)
                evalMeasuresDict = _mean_of_measures(evalMeasuresDict)

            if perquery:
                for qid in evalMeasuresDict:
                    for measurename in evalMeasuresDict[qid]:
                        evalsRows.append([
                            name, 
                            qid, 
                            measurename, 
                            _apply_round(
                                measurename, 
                                evalMeasuresDict[qid][measurename]
                            ) 
                        ])
                evalDict[name] = evalMeasuresDict
            else:
                if mrt_needed:
                    time += precompute_time
                    evalMeasuresDict["mrt"] = time / float(len(all_topic_qids))
                actual_metric_names = list(evalMeasuresDict.keys())
                # gather mean values, applying rounding if necessary
                evalMeasures=[ _apply_round(m, evalMeasuresDict[m]) for m in actual_metric_names]

                evalsRows.append([name]+evalMeasures)
                evalDict[name] = evalMeasures
    

    if dataframe:
        if perquery:
            df = pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"]).sort_values(['name', 'qid'])
            if round is not None and isinstance(round, int):
                df["value"] = df["value"].round(round)
            return df

        highlight_cols = { m : "+"  for m in actual_metric_names }
        if mrt_needed:
            highlight_cols["mrt"] = "-"

        p_col_names : List[str] = []
        if baseline is not None:
            assert len(evalDictsPerQ) == len(retr_systems)
            baselinePerQuery={}
            per_q_metrics = actual_metric_names.copy()
            if mrt_needed:
                per_q_metrics.remove("mrt")

            for m in per_q_metrics:
                baselinePerQuery[m] = np.array([ evalDictsPerQ[baseline][q][m] for q in evalDictsPerQ[baseline] ])

            for i in range(0, len(retr_systems)):
                additionals=[]
                if i == baseline:
                    additionals = [None] * (3*len(per_q_metrics))
                else:
                    for m in per_q_metrics:
                        # we iterate through queries based on the baseline, in case run has different order
                        perQuery = np.array( [ evalDictsPerQ[i][q][m] for q in evalDictsPerQ[baseline] ])
                        delta_plus = (perQuery > baselinePerQuery[m]).sum()
                        delta_minus = (perQuery < baselinePerQuery[m]).sum()
                        p = test_fn(perQuery, baselinePerQuery[m])[1]
                        additionals.extend([delta_plus, delta_minus, p])
                evalsRows[i].extend(additionals)
            delta_names=[]
            for m in per_q_metrics:
                delta_names.append("%s +" % m)
                highlight_cols["%s +" % m] = "+"
                delta_names.append("%s -" % m)
                highlight_cols["%s -" % m] = "-"
                pcol = "%s p-value" % m
                delta_names.append(pcol)
                p_col_names.append(pcol)
            actual_metric_names.extend(delta_names)

        df = pd.DataFrame(evalsRows, columns=["name"] + actual_metric_names)

        # multiple testing correction. This adds two new columns for each measure experience statistical significance testing        
        if baseline is not None and correction is not None:
            import statsmodels.stats.multitest # type: ignore
            for pcol in p_col_names:
                pcol_reject = pcol.replace("p-value", "reject")
                pcol_corrected = pcol + " corrected"                
                reject, corrected, _, _ = statsmodels.stats.multitest.multipletests(df[pcol].drop(df.index[baseline]), alpha=correction_alpha, method=correction)
                insert_pos : int = df.columns.get_loc(pcol)
                # add reject/corrected values for the baseline
                reject = np.insert(reject, baseline, False)
                corrected = np.insert(corrected, baseline, np.nan)
                # add extra columns, put place directly after the p-value column
                df.insert(insert_pos+1, pcol_reject, reject)
                df.insert(insert_pos+2, pcol_corrected, corrected)
        
        if highlight == "color" or highlight == "colour" :
            df = df.style.apply(_color_cols, axis=0, col_type=highlight_cols) # type: ignore
        elif highlight == "bold":
            df = df.style.apply(_bold_cols, axis=0, col_type=highlight_cols) # type: ignore
            
        return df 
    return evalDict

TRANSFORMER_PARAMETER_VALUE_TYPE = Union[str,float,int,str]
GRID_SCAN_PARAM_SETTING = Tuple[
            Transformer, 
            str, 
            TRANSFORMER_PARAMETER_VALUE_TYPE
        ]
GRID_SEARCH_RETURN_TYPE_SETTING = Tuple[
    float, 
    List[GRID_SCAN_PARAM_SETTING]
]

GRID_SEARCH_RETURN_TYPE_BOTH = Tuple[
    Transformer,
    float, 
    List[GRID_SCAN_PARAM_SETTING]
]

def _save_state(param_dict):
    rtr = []
    for tran, param_set in param_dict.items():
        for param_name in param_set:
            rtr.append((tran, param_name, tran.get_parameter(param_name)))
    return rtr

def _restore_state(param_state):
    for (tran, param_name, param_value) in param_state:
        tran.set_parameter(param_name, param_value)

def Evaluate(res : pd.DataFrame, qrels : pd.DataFrame, metrics=['map', 'ndcg'], perquery=False) -> Dict:
    """
    Evaluate a single result dataframe with the given qrels. This method may be used as an alternative to
    ``pt.Experiment()`` for getting only the evaluation measurements given a single set of existing results.

    Args:
        res: Either a dataframe with columns=['qid', 'docno', 'score'] or a dict {qid:{docno:score,},}
        qrels: Either a dataframe with columns=['qid','docno', 'label'] or a dict {qid:{docno:label,},}
        metrics(list): A list of strings specifying which evaluation metrics to use. Default=['map', 'ndcg']
        perquery(bool): If true return each metric for each query, else return mean metrics. Default=False
    """
    if len(res) == 0:
        raise ValueError("No results for evaluation")

    _, rtr = _run_and_evaluate(res, None, qrels, metrics, perquery=perquery)
    return rtr

def KFoldGridSearch( 
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics_list : List[pd.DataFrame],
        qrels : Union[pd.DataFrame,List[pd.DataFrame]],
        metric : MEASURE_TYPE = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size : Optional[int] = None) -> Tuple[pd.DataFrame, GRID_SEARCH_RETURN_TYPE_SETTING]:
    """
    Applies a GridSearch using different folds. It returns the *results* of the 
    tuned transformer pipeline on the test topics. The number of topics dataframes passed
    to topics_list defines the number of folds. For each fold, all but one of the dataframes
    is used as training, and the remainder used for testing. 

    The state of the transformers in the pipeline is restored after the KFoldGridSearch has
    been executed.

    Args:
        pipeline(Transformer): a transformer or pipeline to tune
        params(dict): a two-level dictionary, mapping transformer to param name to a list of values
        topics_list(List[DataFrame]): a *list* of topics dataframes to tune upon
        qrels(DataFrame or List[DataFrame]): qrels to tune upon. A single dataframe, or a list for each fold.       
        metric(str): name of the metric on which to determine the most effective setting. Defaults to "map".
        batch_size(int): If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
            Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
            during a run. Default is None.
        jobs(int): Number of parallel jobs to run. Default is 1, which means sequentially.
        backend(str): Parallelisation backend to use. Defaults to "joblib". 
        verbose(bool): whether to display progress bars or not

    Returns:
    A tuple containing, firstly, the results of pipeline on the test topics after tuning, and secondly, a list of the best parameter settings for each fold.

    Consider tuning PL2 where folds of queries are pre-determined::

        pl2 = pt.terrier.Retriever(index, wmodel="PL2", controls={'c' : 1})
        tuned_pl2, _ = pt.KFoldGridSearch(
            pl2, 
            {pl2 : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            [topicsf1, topicsf2],
            qrels,
            ["map"]
        )
        pt.Experiment([pl2, tuned_pl2], all_topics, qrels, ["map"])

    As 2 splits are defined, PL2 is first tuned on topicsf1 and tested on topicsf2, then 
    trained on topicsf2 and tested on topicsf1. The results dataframe of PL2 after tuning of the c
    parameter are returned by the KFoldGridSearch, and can be used directly in a pt.Experiment().
    """
    
    import pandas as pd
    num_folds = len(topics_list)
    if isinstance(qrels, pd.DataFrame):
        qrels = [qrels] * num_folds    
    
    FOLDS=list(range(0, num_folds))
    results : List[pd.DataFrame] = []
    settings=[]

    # save state
    initial_state = _save_state(params)

    for fold in FOLDS:
        print("Fold %d" % (fold+1))

        train_indx = FOLDS.copy()
        train_indx.remove(fold)
        train_topics = pd.concat([topics_list[offset] for offset in train_indx])
        train_qrels = pd.concat([qrels[offset] for offset in train_indx])
        test_topics = topics_list[fold]
        #test_qrels arent needed
        #test_qrels = qrels[fold]
        
        # safety - give the GridSearch a stable initial setting
        _restore_state(initial_state)

        optPipe: Transformer
        max_measure: float
        max_setting: List[GRID_SCAN_PARAM_SETTING]
        optPipe, max_measure, max_setting = GridSearch(
            pipeline,
            params,
            train_topics,
            train_qrels,
            metric,
            jobs=jobs,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            return_type="both")
        results.append(optPipe.transform(test_topics))
        settings.append(max_setting)
    
    # restore state
    _restore_state(initial_state)
    
    return (pd.concat(results), settings)

@overload
def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE,
        jobs : int,
        backend: str,
        verbose: bool ,
        batch_size : Optional[int],
        return_type : Literal['opt_pipeline'],
    ) -> Transformer: ...

@overload
def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE,
        jobs : int,
        backend: str,
        verbose: bool ,
        batch_size : Optional[int],
        return_type : Literal['best_setting'],
    ) -> GRID_SEARCH_RETURN_TYPE_SETTING: ...

@overload
def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE,
        jobs : int,
        backend: str,
        verbose: bool ,
        batch_size : Optional[int],
        return_type : Literal['both'],
    ) -> GRID_SEARCH_RETURN_TYPE_BOTH: ...

def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size : Optional[int] = None,
        return_type : Literal['opt_pipeline', 'best_setting', 'both'] = "opt_pipeline"
    ) -> Union[Transformer,GRID_SEARCH_RETURN_TYPE_SETTING,GRID_SEARCH_RETURN_TYPE_BOTH]:
    """
    GridSearch is essentially, an argmax GridScan(), i.e. it returns an instance of the pipeline to tune
    with the best parameter settings among params, that were found that were obtained using the specified
    topics and qrels, and for the specified measure.

    Args:
        pipeline(Transformer): a transformer or pipeline to tune
        params(dict): a two-level dictionary, mapping transformer to param name to a list of values
        topics(DataFrame): topics to tune upon
        qrels(DataFrame): qrels to tune upon       
        metric(str): name of the metric on which to determine the most effective setting. Defaults to "map".
        batch_size(int): If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
            Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
            during a run. Default is None.
        jobs(int): Number of parallel jobs to run. Default is 1, which means sequentially.
        backend(str): Parallelisation backend to use. Defaults to "joblib". 
        verbose(bool): whether to display progress bars or not
        return_type(str): whether to return the same transformer with optimal pipeline setting, and/or a setting of the
            higher metric value, and the resulting transformers and settings.
    """
    # save state
    initial_state = _save_state(params)

    if isinstance(metric, list):
        raise KeyError("GridSearch can only maximise ONE metric, but you passed a list (%s)." % str(metric))

    grid_outcomes = GridScan(
        pipeline, 
        params, 
        topics, 
        qrels, 
        [metric], 
        jobs, 
        backend, 
        verbose, 
        batch_size, 
        dataframe=False)
    assert not isinstance(grid_outcomes, pd.DataFrame)

    assert len(grid_outcomes) > 0, "GridScan returned 0 rows"
    max_measure = grid_outcomes[0][1][metric]
    max_setting = grid_outcomes[0][0]
    for setting, measures in grid_outcomes: # TODO what is the type of this iteration?
        if measures[metric] > max_measure:
            max_measure = measures[metric]
            max_setting = setting
    print("Best %s is %f" % (metric, max_measure))
    print("Best setting is %s" % str(["%s %s=%s" % (str(t), k, v) for t, k, v in max_setting]))

    if return_type == "opt_pipeline":
        for tran, param, value in max_setting:
            tran.set_parameter(param, value)
        return pipeline
    if return_type == "best_setting":
        _restore_state(initial_state)
        return max_measure, max_setting
    if return_type == "both":
        for tran, param, value in max_setting:
            tran.set_parameter(param, value)
        return (pipeline, max_measure, max_setting)
    raise ValueError("Unknown return_type option %s" % return_type)

def GridScan(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metrics : Union[MEASURE_TYPE,MEASURES_TYPE] = ["map"],
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None,
        dataframe = True,
    ) -> Union[pd.DataFrame, List [ Tuple [ List[ GRID_SCAN_PARAM_SETTING ], Dict[Union[str, Measure] ,float]  ]  ] ]:
    """
    GridScan applies a set of named parameters on a given pipeline and evaluates the outcome. The topics and qrels 
    must be specified. The trec_eval measure names can be optionally specified.
    The transformers being tuned, and their respective parameters are named in the param_dict. The parameter being
    varied must be changable using the :func:`set_parameter()` method. This means instance variables,
    as well as controls in the case of Retriever.

    Args:
        pipeline(Transformer): a transformer or pipeline
        params(dict): a two-level dictionary, mapping transformer to param name to a list of values
        topics(DataFrame): topics to tune upon
        qrels(DataFrame): qrels to tune upon       
        metrics(List[str]): name of the metrics to report for each setting. Defaults to ["map"].
        batch_size(int): If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
            Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
            during a run. Default is None.
        jobs(int): Number of parallel jobs to run. Default is 1, which means sequentially.
        backend(str): Parallelisation backend to use. Defaults to "joblib". 
        verbose(bool): whether to display progress bars or not
        dataframe(bool): return a dataframe or a list
    Returns:
        A dataframe showing the effectiveness of all evaluated settings, if dataframe=True
        A list of settings and resulting evaluation measures, if dataframe=False
    Raises:
        ValueError: if a specified transformer does not have such a parameter

    Example::

        # graph how PL2's c parameter affects MAP
        pl2 = pt.terrier.Retriever(index, wmodel="PL2", controls={'c' : 1})
        rtr = pt.GridScan(
            pl2, 
            {pl2 : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            topics,
            qrels,
            ["map"]
        )
        import matplotlib.pyplot as plt
        plt.plot(rtr["tran_0_c"], rtr["map"])
        plt.xlabel("PL2's c value")
        plt.ylabel("MAP")
        plt.show()

    """
    import itertools

    if verbose and jobs > 1:
        from warnings import warn
        warn("Cannot provide progress on parallel job")
    if isinstance(metrics, str):
        metrics = [metrics]

    # Store the all parameter names and candidate values into a dictionary, keyed by a tuple of the transformer and the parameter name
    # such as {(Retriever, 'wmodel'): ['BM25', 'PL2'], (Retriever, 'c'): [0.1, 0.2, 0.3], (Bla, 'lr'): [0.001, 0.01, 0.1]}
    candi_dict: Dict[Tuple[Transformer, str], List[TRANSFORMER_PARAMETER_VALUE_TYPE]] = {}
    for tran, param_set in params.items():
        for param_name, values in param_set.items():
            candi_dict[ (tran, param_name) ] = values
    if len(candi_dict) == 0:
        raise ValueError("No parameters specified to optimise")
    for tran, param in candi_dict:
        try:
            tran.get_parameter(param)
        except Exception:
            raise ValueError("Transformer %s does not expose a parameter named %s" % (str(tran), param))
    
    keys, vals = zip(*candi_dict.items())
    combinations = list(itertools.product(*vals))
    assert len(combinations) > 0, "No combinations selected"

    def _evaluate_one_setting(keys, values):
        #'params' is every combination of candidates
        params = dict(zip(keys, values))
        parameter_list = []
        
        # Set the parameter value in the corresponding transformer of the pipeline
        for (tran, param_name), value in params.items():
            tran.set_parameter(param_name, value)
            # such as (Retriever, 'wmodel', 'BM25')
            parameter_list.append( (tran, param_name, value) )
            
        time, eval_scores = _run_and_evaluate(pipeline, topics, qrels, metrics, perquery=False, batch_size=batch_size)
        return parameter_list, eval_scores

    def _evaluate_several_settings(inputs : List[Tuple]):
        return [_evaluate_one_setting(k,v) for k, v in inputs]

    eval_list = []
    #for each combination of parameter values
    if jobs == 1:
        for v in pt.tqdm(combinations, total=len(combinations), desc="GridScan", mininterval=0.3) if verbose else combinations:
            parameter_list, eval_scores = _evaluate_one_setting(keys, v)
            eval_list.append( (parameter_list, eval_scores) )
    else:
        import itertools
        import more_itertools
        from .parallel import parallel_lambda
        all_inputs = [(keys, values) for values in combinations]

        # how many jobs to distribute this to
        num_batches = int(len(combinations)/jobs) if len(combinations) >= jobs else len(combinations)

        # built the batches to distribute
        batched_inputs = list(more_itertools.chunked(all_inputs, num_batches))
        assert len(batched_inputs) > 0, "No inputs identified for parallel_lambda"
        eval_list = parallel_lambda(_evaluate_several_settings, batched_inputs, jobs, backend=backend)
        eval_list =  list(itertools.chain(*eval_list))
        assert len(eval_list) > 0, "parallel_lambda returned 0 rows" 
    
    # resulting eval_list has the form [ 
    #   ( [(BR, 'wmodel', 'BM25'), (BR, 'c', 0.2)]  ,   {"map" : 0.2654} )
    # ]
    # ie, a list of possible settings, combined with measure values    
    if not dataframe:
        return eval_list

    rtr=[]
    for setting, measures in eval_list:
        row={}
        for i, (tran, param, value) in enumerate(setting):
            row["tran_%d" % i]  = tran
            row["tran_%d_%s" % (i,param) ]  = value
        row.update(measures)
        rtr.append(row)
    # resulting dataframe looks like:
    #    tran_0  tran_0_c       map
    #0  BR(PL2)     0.1  0.104820
    #1  BR(PL2)     1.0  0.189274
    #2  BR(PL2)     5.0  0.230838
    return pd.DataFrame(rtr)


class PerQueryMaxMinScoreTransformer(Transformer):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res
