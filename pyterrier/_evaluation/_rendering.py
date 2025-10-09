import numpy as np
import pandas as pd
from collections import namedtuple
from . import MEASURES_TYPE
from ir_measures import Measure
from typing import Sequence, Tuple, Dict, Union, List, Optional, overload, Literal, Any

class EvaluationDataTuple(namedtuple('EvaluationDataTuple', ['averages', 'perquery'])):
    averages : pd.DataFrame
    perquery : pd.DataFrame

    def _repr_html_(self, *a, **kw):
        return self.averages._repr_html_(*a, *kw) + self.perquery._repr_html_(*a, *kw)

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

  

class RenderFromPerQuery():
    """
    Responsible for calculating means of measures, applying significance testing, rounding and constructing dataframes
    """

    def __init__(self, systems, baseline=None, test_fn=None, correction=None, correction_alpha : float = 0.05, round=None, precompute_time=0):
        self.systems = systems
        self.baseline = baseline
        self.test_fn = test_fn
        self.correction = correction
        self.correction_alpha = correction_alpha
        self.round = round
        self.precompute_time = precompute_time
        # sysid -> qid -> measure -> value
        self.systemEvalDictsPerQ : Dict[int,Dict[str,Dict[str,float]]] = {}
        # sysid -> mrt
        self.mrts : Dict[int,float] = {}

    def add_metrics(self, sysid : int, evalRows : Dict[str, Dict[str, float]], mrt : float):
        if sysid >= len(self.systems):
            raise KeyError()
        self.systemEvalDictsPerQ[sysid] = evalRows
        self.mrts[sysid] = mrt
    
    @overload
    def averages(self, dataframe : Literal[True] = True, highlight : Optional[str] = None, mrt_needed : bool = False) -> pd.DataFrame: ...
    @overload
    def averages(self, dataframe : Literal[False], highlight : Optional[str] = None, mrt_needed : bool = False) -> Dict[str,Any]: ...

    def averages(self, dataframe : Union[Literal[True], Literal[False]] = True, highlight : Optional[str] = None, mrt_needed : bool = False) -> Union[Dict[str,Any], pd.DataFrame]:

        assert len(self.systemEvalDictsPerQ) == len(self.systems), "evaluation has not finished"
        baseline = self.baseline
        
        # this is needed for dataframe return
        evalMeasuresDicts = {}
        # one list of column values, one row for each system
        evalsRows = []
        # calculate means for all systems
        for i in range(len(self.systems)):
            evalMeasuresDict = _mean_of_measures(self.systemEvalDictsPerQ[i])
            if mrt_needed:
                evalMeasuresDict["mrt"] = self.precompute_time + self.mrts[i]
            actual_metric_names = list(evalMeasuresDict.keys())
            evalMeasures=[ self._apply_round(m, evalMeasuresDict[m]) for m in actual_metric_names]
            evalsRows.append([self.systems[i]]+evalMeasures) #
            evalMeasuresDicts[self.systems[i]] = evalMeasuresDict
        
        if not dataframe:
            return evalMeasuresDicts
        del(evalMeasuresDicts)

        highlight_cols = { m : "+"  for m in actual_metric_names }
        p_col_names : List[str] = []
        
        if baseline is not None:
            baselinePerQuery={}
            
            per_q_metrics = actual_metric_names.copy()
            if mrt_needed:
                per_q_metrics.remove("mrt")

            for m in per_q_metrics:
                baselinePerQuery[m] = np.array([ self.systemEvalDictsPerQ[baseline][q][m] for q in self.systemEvalDictsPerQ[baseline] ])

            for i in range(len(self.systems)):
                additionals: List[Optional[Union[float, int, complex]]] = []
                if i == baseline:
                    additionals = [None] * (3*len(per_q_metrics))
                else:
                    for m in per_q_metrics:
                        # we iterate through queries based on the baseline, in case run has different order
                        perQuery = np.array( [ self.systemEvalDictsPerQ[i][q][m] for q in self.systemEvalDictsPerQ[baseline] ])
                        delta_plus = (perQuery > baselinePerQuery[m]).sum()
                        delta_minus = (perQuery < baselinePerQuery[m]).sum()
                        p = self.test_fn(perQuery, baselinePerQuery[m])[1] # type: ignore[arg-type]
                        additionals.extend([delta_plus, delta_minus, p])
                evalsRows[i].extend(additionals)

            additional_col_names=[]
            for m in per_q_metrics:
                additional_col_names.append("%s +" % m)
                highlight_cols["%s +" % m] = "+"
                additional_col_names.append("%s -" % m)
                highlight_cols["%s -" % m] = "-"
                pcol = "%s p-value" % m
                additional_col_names.append(pcol)
                p_col_names.append(pcol)
            actual_metric_names.extend(additional_col_names)

        # its easier to build the dataframe, then apply the correction
        df = pd.DataFrame(evalsRows, columns=["name"] + actual_metric_names)
        
        # multiple testing correction. This adds two new columns for each measure experiencing statistical significance testing        
        if self.baseline is not None and self.correction is not None:
            import statsmodels.stats.multitest # type: ignore
            for pcol in p_col_names:
                pcol_reject = pcol.replace("p-value", "reject")
                pcol_corrected = pcol + " corrected"                
                reject, corrected, _, _ = statsmodels.stats.multitest.multipletests(df[pcol].drop(df.index[baseline]), alpha=self.correction_alpha, method=self.correction)
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

    def _apply_round(self, measure, value):
        import builtins
        round = self.round
        if round is not None and isinstance(round, int):
            value = builtins.round(value, round)
        if round is not None and isinstance(round, dict) and measure in round:
            value = builtins.round(value, round[measure])
        return value
    
    @overload
    def perquery(self, dataframe: Literal[True] = True) -> pd.DataFrame: ...
    @overload
    def perquery(self, dataframe: Literal[False]) -> Dict[int,Dict[str, Dict[str,float]]]: ...

    def perquery(self, dataframe : Union[Literal[True], Literal[False]] = True) -> Union[Dict[int,Dict[str, Dict[str,float]]], pd.DataFrame]:
        """
        Return per-query results.
        DF has columns ``["name", "qid", "measure", "value"]``.
        """
        if not dataframe:
            return self.systemEvalDictsPerQ

        evalsRows = []
        for sysid in self.systemEvalDictsPerQ:
            evalMeasuresDict = self.systemEvalDictsPerQ[sysid]
            for qid in evalMeasuresDict:
                for measurename in evalMeasuresDict[qid]:
                    evalsRows.append([
                        self.systems[sysid], 
                        qid, 
                        measurename, 
                        self._apply_round(
                            measurename, 
                            evalMeasuresDict[qid][measurename]
                        )
                    ])
        
        perquery_df = pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"]).sort_values(['name', 'qid'])
        if self.round is not None and isinstance(self.round, int):
            perquery_df["value"] = perquery_df["value"].round(self.round)
        return perquery_df

