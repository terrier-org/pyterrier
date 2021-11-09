import pandas as pd
import pytrec_eval
from collections import defaultdict
import os
import deprecation


class Utils:


    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def evaluate(res, qrels, metrics=['map', 'ndcg'], perquery=False):
        """
        Evaluate the result dataframe with the given qrels

        Args:
            res: Either a dataframe with columns=['qid', 'docno', 'score'] or a dict {qid:{docno:score,},}
            qrels: Either a dataframe with columns=['qid','docno', 'label'] or a dict {qid:{docno:label,},}
            metrics(list): A list of strings specifying which evaluation metrics to use. Default=['map', 'ndcg']
            perquery(bool): If true return each metric for each query, else return mean metrics. Default=False
        """
        from .io import coerce_dataframe
        if not isinstance(res, dict):
            res = coerce_dataframe(res)
        if isinstance(res, pd.DataFrame):
            batch_retrieve_results_dict = Utils.convert_res_to_dict(res)
        else:
            batch_retrieve_results_dict = res

        if isinstance(qrels, pd.DataFrame):
            qrels_df = qrels
        else:
            qrels_df = Utils.convert_qrels_to_dataframe(qrels)
        if len(batch_retrieve_results_dict) == 0:
            raise ValueError("No results for evaluation")

        from .pipelines import _run_and_evaluate
        _, rtr = _run_and_evaluate(res, None, qrels_df, metrics, perquery=perquery)
        return rtr

    @staticmethod
    def mean_of_measures(result, measures=None, num_q = None):
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
