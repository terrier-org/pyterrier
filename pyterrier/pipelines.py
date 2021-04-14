from warnings import warn
import os
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
from .utils import Utils
from .transformer import TransformerBase, EstimatorBase
from .model import add_ranks
import deprecation

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


def _run_and_evaluate(
        system : Union[TransformerBase, pd.DataFrame], 
        topics : pd.DataFrame, 
        qrels_dict, 
        metrics : List[str], 
        perquery : bool = False,
        batch_size = None):
    
    from timeit import default_timer as timer
    runtime = 0
    # if its a DataFrame, use it as the results
    if isinstance(system, pd.DataFrame):
        res = system
        evalMeasuresDict = Utils.evaluate(res, qrels_dict, metrics=metrics, perquery=perquery)

    elif batch_size is None:
        #transformer, evaluate all queries at once
            
        starttime = timer()
        res = system.transform(topics)
        endtime = timer()
        runtime =  (endtime - starttime) * 1000.
        evalMeasuresDict = Utils.evaluate(res, qrels_dict, metrics=metrics, perquery=perquery)
    else:
        #transformer, evaluate queries in batches
        assert batch_size > 0
        starttime = timer()
        results=[]
        evalMeasuresDict={}
        for res in system.transform_gen(topics, batch_size=batch_size):
            endtime = timer()
            runtime += (endtime - starttime) * 1000.
            localEvalDict = Utils.evaluate(res, qrels_dict, metrics=metrics, perquery=False)
            evalMeasuresDict.update(localEvalDict)
            starttime = timer()
        if not perquery:
            evalMeasuresDict = Utils.mean_of_measures(evalMeasuresDict)
    return (runtime, evalMeasuresDict)


def Experiment(retr_systems, topics, qrels, eval_metrics, 
        names=None, 
        perquery=False, 
        dataframe=True, 
        batch_size=None, 
        drop_unused=False, 
        baseline=None, 
        correction=None, 
        correction_alpha=0.05, 
        highlight=None, 
        round=None):
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
        drop_unused(bool): If True, will drop topics from the topics dataframe that have qids not appearing in the qrels dataframe. 
        perquery(bool): If True return each metric for each query, else return mean metrics across all queries. Default=False.
        dataframe(bool): If True return results as a dataframe. Else as a dictionary of dictionaries. Default=True.
        baseline(int): If set to the index of an item of the retr_system list, will calculate the number of queries 
            improved, degraded and the statistical significance (paired t-test p value) for each measure.
            Default=None: If None, no additional columns will be added for each measure.
        correction(string): Whether any multiple testing correction should be applied. E.g. 'bonferroni', 'holm', 'hs' aka 'holm-sidak'. Default is None.
            Additional columns are added denoting whether the null hypothesis can be rejected, and the corrected p value. 
            See `statsmodels.stats.multitest.multipletests() <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html#statsmodels.stats.multitest.multipletests>`_
            for more information about available testing correction.
        correction_alpha(float): What alpha value for multiple testing correction Default is 0.05.
        highlight(str): If `highlight="bold"`, highlights in bold the best measure value in each column; 
            if `highlight="color"` or `"colour"`, then the cell with the highest metric value will have a green background.
        round(int): How many decimal places to round each measure value to. This can be a dictionary mapping measure name to number of decimal places.
            Default is None, which is no rounding.

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
    if drop_unused:
        # the commented variant would drop queries not having any RELEVANT labels
        # topics = topics.merge(qrels[qrels["label"] > 0][["qid"]].drop_duplicates())        
        topics = topics.merge(qrels[["qid"]].drop_duplicates())

    # obtain system names if not specified
    if names is None:
        names = [str(system) for system in retr_systems]
    elif len(names) != len(retr_systems):
        raise ValueError("names should be the same length as retr_systems")

    qrels_dict = Utils.convert_qrels_to_dict(qrels)
    all_qids = topics["qid"].values

    evalsRows=[]
    evalDict={}
    evalDictsPerQ=[]
    actual_metric_names=[]
    mrt_needed = False
    if "mrt" in eval_metrics:
        mrt_needed = True
        eval_metrics.remove("mrt")
    
    # run and evaluate each system
    for name,system in zip(names, retr_systems):
        time, evalMeasuresDict = _run_and_evaluate(system, topics, qrels_dict, eval_metrics, perquery=perquery or baseline is not None)
        
        if perquery or baseline is not None:
            # this ensures that all queries are present in various dictionaries
            # its equivalent to "trec_eval -c"
            (evalMeasuresDict, missing) = Utils.ensure(evalMeasuresDict, eval_metrics, all_qids)
            if missing > 0:
                warn("%s was missing %d queries, expected %d" % (name, missing, len(all_qids) ))

        if baseline is not None:
            evalDictsPerQ.append(evalMeasuresDict)
            evalMeasuresDict = Utils.mean_of_measures(evalMeasuresDict)

        if mrt_needed:
            evalMeasuresDict["mrt"] = time / float(len(all_qids))

        if perquery:
            for qid in all_qids:
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
            actual_metric_names = list(evalMeasuresDict.keys())
            # gather mean values, applying rounding if necessary
            evalMeasures=[ _apply_round(m, evalMeasuresDict[m]) for m in actual_metric_names]

            evalsRows.append([name]+evalMeasures)
            evalDict[name] = evalMeasures

    if dataframe:
        if perquery:
            df = pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"])
            if round is not None:
                df["value"] = df["value"].round(round)
            return df

        highlight_cols = { m : "+"  for m in actual_metric_names }
        if mrt_needed:
            highlight_cols["mrt"] = "-"

        p_col_names=[]
        if baseline is not None:
            assert len(evalDictsPerQ) == len(retr_systems)
            from scipy import stats
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
                        p = stats.ttest_rel(perQuery, baselinePerQuery[m])[1]
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

def GridSearch(
        pipeline : TransformerBase,
        params : Dict[TransformerBase,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : str = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None,
        return_type : str = "opt_pipeline"
    ) -> Union[TransformerBase,GRID_SEARCH_RETURN_TYPE_SETTING]:
    """
    GridSearch. Essentially, an argmax GridScan() 
    """
    
    grid_outcomes = GridScan(pipeline, params, topics, qrels, [metric], jobs, backend, verbose, batch_size)
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
        return tran
    if return_type == "best_setting":
        return max_measure, max_setting

def GridScan(
        pipeline : TransformerBase,
        params : Dict[TransformerBase,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metrics : List[str] = ["map"],
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None
    ) -> List [ Tuple [ List[ GRID_SCAN_PARAM_SETTING ] , Dict[str,float]  ]  ]:
    """
    GridScan applies a set of named parameters on a given pipeline and evaluates the outcome. The topics and qrels 
    must be specified. The trec_eval measure names can be optionally specified.
    The transformers being tuned, and their respective parameters are named in the param_dict. The parameter being
    varied must be changable using the :func:`set_parameter()` method. This means instance variables,
    as well as controls in the case of BatchRetrieve.

    Args:
        - pipeline(TransformerBase): a transformer or pipeline
        - params(dict): a two-level dictionary, mapping transformer id to param name to a list of values
        - topics(DataFrame): topics to tune upon
        - qrels(DataFrame): qrels to tune upon       
        - metrics(List[str]): name of the metrics to report for each setting. Defaults to ["map"].
        - verbose(bool): whether to display progress bars or not 
    Returns:
        - A dataframe showing the best settings
    Raises:
        - ValueError: if a specified transformer does not have such a parameter    
    """
    import itertools
    from . import Utils, tqdm

    if verbose and jobs > 1:
        from warnings import warn
        warn("Cannot provide progress on parallel job")

    # Store the all parameter names and candidate values into a dictionary, keyed by a tuple of the transformer and the parameter name
    # such as {(BatchRetrieve, 'wmodel'): ['BM25', 'PL2'], (BatchRetrieve, 'c'): [0.1, 0.2, 0.3], (Bla, 'lr'): [0.001, 0.01, 0.1]}
    candi_dict = { (tran, param_name) : params[tran][param_name] for tran in params for param_name in params[tran]}
    if len(candi_dict) == 0:
        raise ValueError("No parameters specified to optimise")
    for tran, param in candi_dict:
        try:
            tran.get_parameter(param)
        except Exception as e:
            raise ValueError("Transformer %s does not expose a parameter named %s" % (str(tran), param))
    
    keys,values = zip(*candi_dict.items())
    combinations = list(itertools.product(*values))

    # use this for repeated evaluation
    qrels_dict = Utils.convert_qrels_to_dict(qrels)

    def _evaluate_one_setting(keys, values):
        #'params' is every combination of candidates
        params = dict(zip(keys, values))
        parameter_list = []
        
        # Set the parameter value in the corresponding transformer of the pipeline
        for (tran, param_name), value in params.items():
            tran.set_parameter(param_name, value)
            # such as (BatchRetrieve, 'wmodel', 'BM25')
            parameter_list.append( (tran, param_name, value) )
            
        _run_and_evaluate(pipeline, topics, qrels_dict, metrics, perquery=False, batch_size=batch_size)
        # using topics and evaluation
        res = pipeline.transform(topics)
        eval_scores = Utils.evaluate(res, qrels, metrics=metrics, perquery=False)
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
        all_inputs = [(keys, values) for values in combinations]
        batched_inputs = more_itertools.chunked(all_inputs, int(len(combinations)/jobs))
        from .parallel import parallel_lambda
        eval_list = parallel_lambda(_evaluate_several_settings, batched_inputs, jobs, backend=backend)
        eval_list =  list(itertools.chain(*eval_list))
    
    # eval_list has the form [ 
    #   ( [(BR, 'wmodel', 'BM25'), (BR, 'c', 0.2)]  ,   {"map" : 0.2654} )
    # ]
    # ie, a list of possible settings, combined with measure values    
    return eval_list

from .ltr import RegressionTransformer, LTRTransformer
@deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.ltr.apply_learned_model(learner, form='regression')")
class LTR_pipeline(RegressionTransformer):
    pass

@deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.ltr.apply_learned_model(learner, form='ltr')")
class XGBoostLTR_pipeline(LTRTransformer):
    pass


class PerQueryMaxMinScoreTransformer(TransformerBase):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res
