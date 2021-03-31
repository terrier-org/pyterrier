from warnings import warn
import os
import pandas as pd
import numpy as np
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

def Experiment(retr_systems, topics, qrels, eval_metrics, names=None, perquery=False, dataframe=True, baseline=None, correction=None, correction_alpha=0.05, highlight=None, round=None):
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
        perquery(bool): If true return each metric for each query, else return mean metrics across all queries. Default=False.
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
    from timeit import default_timer as timer

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

    results = []
    times=[]
    neednames = names is None
    if neednames:
        names = []
    elif len(names) != len(retr_systems):
        raise ValueError("names should be the same length as retr_systems")
    for system in retr_systems:
        # if its a DataFrame, use it as the results
        if isinstance(system, pd.DataFrame):
            results.append(system)
            times.append(0)
        else:
            starttime = timer()            
            results.append(system.transform(topics))
            endtime = timer()
            times.append( (endtime - starttime) * 1000.)
            
        if neednames:
            names.append(str(system))

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
    for name,res,time in zip(names, results, times):
        evalMeasuresDict = Utils.evaluate(res, qrels_dict, metrics=eval_metrics, perquery=perquery or baseline is not None)
        
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

class ParallelTransformer(TransformerBase):

    def __init__(self, parent, n_jobs, backend='joblib', **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.n_jobs = n_jobs
        self.backend = backend
        if self.backend not in ["joblib", "ray"]:
            raise ValueError("Backend of %s unknown, only 'joblib' or 'ray' supported.")

    def _transform_joblib(self, splits):
        def with_initializer(p, f_init):
            # Overwrite initializer hook in the Loky ProcessPoolExecutor
            # https://github.com/tomMoral/loky/blob/f4739e123acb711781e46581d5ed31ed8201c7a9/loky/process_executor.py#L850
            hasattr(p._backend, '_workers') or p.__enter__()
            origin_init = p._backend._workers._initializer
            def new_init():
                origin_init()
                f_init()
            p._backend._workers._initializer = new_init if callable(origin_init) else f_init
            return p

        import pyterrier as pt
        from joblib import Parallel, delayed
        with Parallel(n_jobs=self.n_jobs) as parallel:
            results = with_initializer(parallel, lambda: pt.init(**pt.init_args))(delayed(self.parent)(topics) for topics in splits)
            return pd.concat(results)
        
    def _transform_ray(self, splits):
        from ray.util.multiprocessing import Pool
        import pyterrier as pt
        with Pool(self.self.n_jobs, lambda: pt.init(**pt.init_args)) as pool:
            results = pool.map(lambda topics : self.parent(topics), splits)
            return pd.concat(results)

    def transform(self, topics_and_res):
        #TODO group by qid.
        
        def chunks(df, n):
            """Yield successive n-sized chunks from df."""
            for i in range(0, len(df), n):
                yield df.iloc[ i: min(len(df),i + n)]
        
        from math import ceil
        splits = list( chunks(topics_and_res, ceil(len(topics_and_res)/self.n_jobs)))
        
        if self.backend == 'joblib':
            return self._transform_joblib(self, splits)
        if self.backend == 'ray':
            return self._transform_ray(self, splits)