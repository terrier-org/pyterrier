import pandas as pd
from collections import defaultdict
from deprecated import deprecated


class Utils:
    # THIS CLASS WILL BE REMOVED IN A FUTURE RELEASE

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    @deprecated(version="0.9", reason="Use pt.Evaluate instead")
    def evaluate(res : pd.DataFrame, qrels : pd.DataFrame, metrics=['map', 'ndcg'], perquery=False):
        from .pipelines import Evaluate
        return Evaluate(res, qrels, metrics=metrics, perquery=perquery)

    @staticmethod
    @deprecated(version="0.9")
    def mean_of_measures(result, measures=None, num_q = None):
        from .pipelines import _mean_of_measures
        return _mean_of_measures(result, measures=measures, num_q=num_q)
