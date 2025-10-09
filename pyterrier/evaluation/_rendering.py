import pandas as pd


from collections import namedtuple
from . import MEASURES_TYPE
from ir_measures import Measure
from typing import Sequence, Tuple, Dict

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
