from collections import defaultdict
from warnings import warn
import os
import pandas as pd
import numpy as np
from typing import Callable, Union, Dict, List, Tuple, Sequence, Any
from .utils import Utils
from .transformer import TransformerBase, EstimatorBase
from .model import add_ranks, coerce_dataframe_types
import deprecation
import ir_measures
from ir_measures.measures import BaseMeasure 
MEASURE_TYPE=Union[str,BaseMeasure]
MEASURES_TYPE=Sequence[MEASURE_TYPE]


SYSTEM_OR_RESULTS_TYPE = Union[TransformerBase, pd.DataFrame]

def _bold_cols(data, col_type):
    if not data.name in col_type:
        return [''] * len(data)
    
    colormax_attr = f'font-weight: bold'
    colormaxlast_attr = f'font-weight: bold'
    if col_type[data.name] == "+":  
        max_value = data.max()
    else:
        max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

def _color_cols(data, col_type, 
                       colormax='antiquewhite', colormaxlast='lightgreen', 
                       colormin='antiquewhite', colorminlast='lightgreen' ):
    if not data.name in col_type:
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

def _convert_measures(metrics : MEASURES_TYPE) -> Tuple[Sequence[BaseMeasure], Dict[BaseMeasure,str]]:
    from ir_measures import parse_trec_measure
    rtr = []
    rev_mapping = {}
    for m in metrics:
        if isinstance(m, BaseMeasure):
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
        seq : Sequence, 
        metrics: Sequence[BaseMeasure],
        rev_mapping : Dict[BaseMeasure,str], 
        num_q : int,
        perquery : bool = True,
        backfill_qids : Sequence[str] = None):
    from collections import defaultdict
    if perquery:
        # qid -> measure -> value
        rtr=defaultdict(dict)
        for m in seq:
            metric = m.measure
            metric = rev_mapping.get(metric, str(metric))
            rtr[m.query_id][metric] = m.value
        # When reporting per-query results, it can desirable to show something for topics that were executed
        # do not have corresponding qrels. If the caller passes in backfill_qids, we'll ensure that these
        # qids are present, and if not add placeholders with NaN values for all measures.
        if backfill_qids is not None:
            backfill_count = 0
            for qid in backfill_qids:
                if qid not in rtr:
                    backfill_count += 1
                    for metric in metrics:
                        rtr[qid][rev_mapping.get(metric, str(metric))] = float('NaN')
            if backfill_count > 0:
                warn(f'{backfill_count} topic(s) not found in qrels. Scores for these topics are given as NaN and should not contribute to averages.')
        return rtr
    assert backfill_qids is None, "backfill_qids only supported when perquery=True"
    # measure -> value
    rtr = {rev_mapping.get(m, str(m)): m.aggregator() for m in metrics}
    for m in seq:
        metric = m.measure
        metric = rev_mapping.get(metric, str(metric))
        rtr[metric].add(m.value)
    for m in rtr:
        rtr[m] = rtr[m].result()
    return rtr

def _run_and_evaluate(
        system : SYSTEM_OR_RESULTS_TYPE, 
        topics : pd.DataFrame, 
        qrels: pd.DataFrame, 
        metrics : MEASURES_TYPE, 
        pbar = None,
        save_mode = None,
        save_file = None,
        perquery : bool = False,
        batch_size = None,
        backfill_qids : Sequence[str] = None):
    
    from .io import read_results, write_results

    if pbar is None:
        from . import tqdm
        pbar = tqdm(disable=True)

    metrics, rev_mapping = _convert_measures(metrics)
    qrels = qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
    from timeit import default_timer as timer
    runtime = 0
    num_q = qrels['query_id'].nunique()
    if save_file is not None and os.path.exists(save_file):
        if save_mode == "reuse":
            system = read_results(save_file)
        elif save_mode == "overwrite":
            os.remove(save_file)
        else:
            raise ValueError("Unknown save_file argument '%s', valid options are 'reuse' or 'overwrite'" % save_mode)

    # if its a DataFrame, use it as the results
    if isinstance(system, pd.DataFrame):
        res = system
        res = coerce_dataframe_types(res)
        if len(res) == 0:
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
        #transformer, evaluate all queries at once
            
        starttime = timer()
        res = system.transform(topics)
        endtime = timer()
        runtime =  (endtime - starttime) * 1000.

        # write results to save_file; we can be sure this file does not exist
        if save_file is not None:
            write_results(res, save_file)

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
        #transformer, evaluate queries in batches
        assert batch_size > 0
        starttime = timer()
        evalMeasuresDict = {}
        remaining_qrel_qids = set(qrels.query_id)
        try:
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
        names : Sequence[str] = None,
        perquery : bool = False,
        dataframe : bool =True,
        batch_size : int = None,
        filter_by_qrels : bool = False,
        filter_by_topics : bool = True,
        baseline : int = None,
        test : Union[str,TEST_FN_TYPE] = "t",
        correction : str = None,
        correction_alpha : float = 0.05,
        highlight : str = None,
        round : Union[int,Dict[str,int]] = None,
        verbose : bool = False,
        save_dir : str = None,
        save_mode : str = 'reuse',
        **kwargs):
    """
    Allows easy comparison of multiple retrieval transformer pipelines using a common set of topics, and
    identical evaluation measures computed using the same qrels. In essence, each transformer is applied on 
    the provided set of topics. Then the named trec_eval evaluation measures are computed 
    (using `pt.Utils.evaluate()`) for each system.

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
            over transformers for evaluation. If set to "overwrite", existing files will be replaced. Default is "reuse".
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
        warn("Signature of Experiment() is now (retr_systems, topics, qrels, eval_metrics), please update your code", DeprecationWarning, 2)

    if 'drop_unused' in kwargs:
        filter_by_qrels = kwargs.pop('drop_unused')
        warn('drop_unused is deprecated; use filter_by_qrels instead', DeprecationWarning)
    if len(kwargs):
        raise TypeError("Unknown kwargs: %s" % (str(list(kwargs.keys()))))

    if baseline is not None:
        assert int(baseline) >= 0 and int(baseline) < len(retr_systems)
        assert not perquery

    if isinstance(topics, str):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if isinstance(qrels, str):
        if os.path.isfile(qrels):
            qrels = Utils.parse_qrels(qrels)

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
    if test == "t":
        test = stats.ttest_rel
    if test == "wilcoxon":
        test = stats.wilcoxon
    
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
        eval_metrics.remove("mrt")

    # progress bar construction
    from . import tqdm
    tqdm_args={
        'disable' : not verbose,
        'unit' : 'system',
        'total' : len(retr_systems),
        'desc' : 'pt.Experiment'
    }
    if batch_size is not None:
        tqdm_args['unit'] = 'batches'
        tqdm_args['total'] = (len(topics) / batch_size) * len(retr_systems)

    with tqdm(**tqdm_args) as pbar:
        # run and evaluate each system
        for name, system in zip(names, retr_systems):
            save_file = None
            if save_dir is not None:
                save_file = os.path.join(save_dir, "%s.res.gz" % name)

            time, evalMeasuresDict = _run_and_evaluate(
                system, topics, qrels, eval_metrics, 
                perquery=perquery or baseline is not None, 
                batch_size=batch_size, 
                backfill_qids=all_topic_qids if perquery else None,
                save_file=save_file,
                save_mode=save_mode,
                pbar=pbar)

            if baseline is not None:
                evalDictsPerQ.append(evalMeasuresDict)
                evalMeasuresDict = Utils.mean_of_measures(evalMeasuresDict)

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
                import builtins
                if mrt_needed:
                    evalMeasuresDict["mrt"] = time / float(len(all_topic_qids))
                actual_metric_names = list(evalMeasuresDict.keys())
                # gather mean values, applying rounding if necessary
                evalMeasures=[ _apply_round(m, evalMeasuresDict[m]) for m in actual_metric_names]

                evalsRows.append([name]+evalMeasures)
                evalDict[name] = evalMeasures
    

    if dataframe:
        if perquery:
            df = pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"]).sort_values(['name', 'qid'])
            if round is not None:
                df["value"] = df["value"].round(round)
            return df

        highlight_cols = { m : "+"  for m in actual_metric_names }
        if mrt_needed:
            highlight_cols["mrt"] = "-"

        p_col_names=[]
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
                        p = test(perQuery, baselinePerQuery[m])[1]
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
            import statsmodels.stats.multitest
            for pcol in p_col_names:
                pcol_reject = pcol.replace("p-value", "reject")
                pcol_corrected = pcol + " corrected"                
                reject, corrected, _, _ = statsmodels.stats.multitest.multipletests(df[pcol], alpha=correction_alpha, method=correction)
                insert_pos = df.columns.get_loc(pcol)
                # add extra columns, put place directly after the p-value column
                df.insert(insert_pos+1, pcol_reject, reject)
                df.insert(insert_pos+2, pcol_corrected, corrected)
        
        if highlight == "color" or highlight == "colour" :
            df = df.style.apply(_color_cols, axis=0, col_type=highlight_cols)
        elif highlight == "bold":
            df = df.style.apply(_bold_cols, axis=0, col_type=highlight_cols)
            
        return df 
    return evalDict

TRANSFORMER_PARAMETER_VALUE_TYPE = Union[str,float,int,str]
GRID_SCAN_PARAM_SETTING = Tuple[
            TransformerBase, 
            str, 
            TRANSFORMER_PARAMETER_VALUE_TYPE
        ]
GRID_SEARCH_RETURN_TYPE_SETTING = Tuple[
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


def KFoldGridSearch( 
        pipeline : TransformerBase,
        params : Dict[TransformerBase,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics_list : List[pd.DataFrame],
        qrels : Union[pd.DataFrame,List[pd.DataFrame]],
        metric : MEASURE_TYPE = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None) -> Tuple[pd.DataFrame, GRID_SEARCH_RETURN_TYPE_SETTING]:
    """
    Applies a GridSearch using different folds. It returns the *results* of the 
    tuned transformer pipeline on the test topics. The number of topics dataframes passed
    to topics_list defines the number of folds. For each fold, all but one of the dataframes
    is used as training, and the remainder used for testing. 

    The state of the transformers in the pipeline is restored after the KFoldGridSearch has
    been executed.

    Args:
        pipeline(TransformerBase): a transformer or pipeline to tune
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

        pl2 = pt.BatchRetrieve(index, wmodel="PL2", controls={'c' : 1})
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
    results=[]
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

def GridSearch(
        pipeline : TransformerBase,
        params : Dict[TransformerBase,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None,
        return_type : str = "opt_pipeline"
    ) -> Union[TransformerBase,GRID_SEARCH_RETURN_TYPE_SETTING]:
    """
    GridSearch is essentially, an argmax GridScan(), i.e. it returns an instance of the pipeline to tune
    with the best parameter settings among params, that were found that were obtained using the specified
    topics and qrels, and for the specified measure.

    Args:
        pipeline(TransformerBase): a transformer or pipeline to tune
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

    assert len(grid_outcomes) > 0, "GridScan returned 0 rows"
    max_measure = grid_outcomes[0][1][metric]
    max_setting = grid_outcomes[0][0]
    for setting, measures in grid_outcomes:
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


def GridScan(
        pipeline : TransformerBase,
        params : Dict[TransformerBase,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metrics : Union[MEASURE_TYPE,MEASURES_TYPE] = ["map"],
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None,
        dataframe = True,
    ) -> Union[pd.DataFrame, List [ Tuple [ List[ GRID_SCAN_PARAM_SETTING ] , Dict[str,float]  ]  ] ]:
    """
    GridScan applies a set of named parameters on a given pipeline and evaluates the outcome. The topics and qrels 
    must be specified. The trec_eval measure names can be optionally specified.
    The transformers being tuned, and their respective parameters are named in the param_dict. The parameter being
    varied must be changable using the :func:`set_parameter()` method. This means instance variables,
    as well as controls in the case of BatchRetrieve.

    Args:
        pipeline(TransformerBase): a transformer or pipeline
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
        pl2 = pt.BatchRetrieve(index, wmodel="PL2", controls={'c' : 1})
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
    from . import Utils, tqdm

    if verbose and jobs > 1:
        from warnings import warn
        warn("Cannot provide progress on parallel job")
    if isinstance(metrics, str):
        metrics = [metrics]

    # Store the all parameter names and candidate values into a dictionary, keyed by a tuple of the transformer and the parameter name
    # such as {(BatchRetrieve, 'wmodel'): ['BM25', 'PL2'], (BatchRetrieve, 'c'): [0.1, 0.2, 0.3], (Bla, 'lr'): [0.001, 0.01, 0.1]}
    candi_dict={}
    for tran, param_set in params.items():
        for param_name, values in param_set.items():
            candi_dict[ (tran, param_name) ] = values
    #candi_dict = { : params[tran][param_name] for tran in params for param_name in params[tran]}
    if len(candi_dict) == 0:
        raise ValueError("No parameters specified to optimise")
    for tran, param in candi_dict:
        try:
            tran.get_parameter(param)
        except Exception as e:
            raise ValueError("Transformer %s does not expose a parameter named %s" % (str(tran), param))
    
    keys,values = zip(*candi_dict.items())
    combinations = list(itertools.product(*values))
    assert len(combinations) > 0, "No combinations selected"

    def _evaluate_one_setting(keys, values):
        #'params' is every combination of candidates
        params = dict(zip(keys, values))
        parameter_list = []
        
        # Set the parameter value in the corresponding transformer of the pipeline
        for (tran, param_name), value in params.items():
            tran.set_parameter(param_name, value)
            # such as (BatchRetrieve, 'wmodel', 'BM25')
            parameter_list.append( (tran, param_name, value) )
            
        time, eval_scores = _run_and_evaluate(pipeline, topics, qrels, metrics, perquery=False, batch_size=batch_size)
        return parameter_list, eval_scores

    def _evaluate_several_settings(inputs : List[Tuple]):
        return [_evaluate_one_setting(k,v) for k, v in inputs]

    eval_list = []
    #for each combination of parameter values
    if jobs == 1:
        for v in tqdm(combinations, total=len(combinations), desc="GridScan", mininterval=0.3) if verbose else combinations:
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


class PerQueryMaxMinScoreTransformer(TransformerBase):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res
