from warnings import warn
import os
import pandas as pd
import numpy as np
from .utils import Utils
from .transformer import TransformerBase, EstimatorBase

#def Experiment(topics, retr_systems, eval_metrics, qrels, names=None, perquery=False, dataframe=True):
def Experiment(retr_systems, topics, qrels, eval_metrics, names=None, perquery=False, dataframe=True):
    """
    Cornac style experiment. Combines retrieval and evaluation.
    Allows easy comparison of multiple retrieval systems with different properties and controls.

    Args:
        topics: Either a path to a topics file or a pandas.Dataframe with columns=['qid', 'query']
        retr_systems(list): A list of BatchRetrieve objects to compare
        eval_metrics(list): Which evaluation metrics to use. E.g. ['map']
        qrels: Either a path to a qrels file or a pandas.Dataframe with columns=['qid','docno', 'label']
        names(list)=List of names for each retrieval system when presenting the results.
            Default=None. If None: Use names of weighting models for each retrieval system.
        perquery(bool): If true return each metric for each query, else return mean metrics. Default=False.
        dataframe(bool): If True return results as a dataframe. Else as a dictionary of dictionaries. Default=True.

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
    

    if isinstance(topics, str):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if isinstance(qrels, str):
        if os.path.isfile(qrels):
            qrels = Utils.parse_qrels(qrels)

    results = []
    neednames = names is None
    if neednames:
        names = []
    for system in retr_systems:
        results.append(system.transform(topics))
        if neednames:
            names.append(str(system))

    evalsRows=[]
    evalDict={}
    actual_metric_names=[]
    for name,res in zip(names,results):
        evalMeasuresDict = Utils.evaluate(res,qrels, metrics=eval_metrics, perquery=perquery)
        if perquery:
            for qid in evalMeasuresDict:
                for measurename in evalMeasuresDict[qid]:
                    evalsRows.append([name, qid, measurename,  evalMeasuresDict[qid][measurename]])
            evalDict[name] = evalMeasuresDict
        else:
            actual_metric_names = list(evalMeasuresDict.keys())
            evalMeasures = [evalMeasuresDict[m] for m in actual_metric_names]
            evalsRows.append([name]+evalMeasures)
            evalDict[name] = evalMeasures
    if dataframe:
        if perquery:
            return pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"])
        return pd.DataFrame(evalsRows, columns=["name"] + actual_metric_names)
    return evalDict
    # evals = {}

    # for weight, res in zip(names, results):
    #     evals[weight] = Utils.evaluate(res, qrels, metrics=eval_metrics, perquery=perquery)
    # if dataframe:
    #     evals = pd.DataFrame.from_dict(evals, orient='index')
    #return evals

class LTR_pipeline(EstimatorBase):
    """
    This class simplifies the use of Scikit-learn's techniques for learning-to-rank.
    """
    def __init__(self, LTR, *args, **kwargs):
        """
        Init method

        Args:
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
        """
        super().__init__(*args, **kwargs)
        # if isinstance(model, str):
        #     from .batchretrieve import FeaturesBatchRetrieve
        #     self.feat_retrieve = FeaturesBatchRetrieve(index, features)
        #     self.feat_retrieve.setControl('wmodel', model)
        # else:
        #     self.feat_retrieve = model
        # self.qrels = qrels
        self.LTR = LTR

    def fit(self, topics_and_results_Train, qrelsTrain, topics_and_results_Valid=None, qrelsValid=None):
        """
        Trains the model with the given topics.

        Args:
            topicsTrain(DataFrame): A dataframe with the topics to train the model
        """
        if len(topics_and_results_Train) == 0:
            raise ValueError("No topics to fit to")
        if 'features' not in topics_and_results_Train.columns:
            raise ValueError("No features column retrieved")
        train_DF = topics_and_results_Train.merge(qrelsTrain, on=['qid', 'docno'], how='left').fillna(0)
        self.LTR.fit(np.stack(train_DF["features"].values), train_DF["label"].values)
        return self

    def transform(self, test_DF):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

class XGBoostLTR_pipeline(LTR_pipeline):
    """
    This class simplifies the use of XGBoost's techniques for learning-to-rank.
    """

    def transform(self, topics_and_docs_Test):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = topics_and_docs_Test
        # xgb is more sensitive about the type of the values.
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

    def fit(self, topics_and_results_Train, qrelsTrain, topics_and_results_Valid, qrelsValid):
        """
        Trains the model with the given training and validation topics.

        Args:
            topics_and_results_Train(DataFrame): A dataframe with the topics and results to train the model
            topics_and_results_Valid(DataFrame): A dataframe with the topics and results for validation
        """
        if len(topics_and_results_Train) == 0:
            raise ValueError("No training topics to fit to")
        if len(topics_and_results_Valid) == 0:
            raise ValueError("No training topics to fit to")

        if 'features' not in topics_and_results_Train.columns:
            raise ValueError("No features column retrieved in training")
        if 'features' not in topics_and_results_Valid.columns:
            raise ValueError("No features column retrieved in validation")

        tr_res = topics_and_results_Train.merge(qrelsTrain, on=['qid', 'docno'], how='left').fillna(0)
        va_res = topics_and_results_Valid.merge(qrelsValid, on=['qid', 'docno'], how='left').fillna(0)

        self.LTR.fit(
            np.stack(tr_res["features"].values),
            tr_res["label"].values, tr_res.groupby(["qid"]).count()["docno"].values,
            eval_set=[(np.stack(va_res["features"].values), va_res["label"].values)],
            eval_group=[va_res.groupby(["qid"]).count()["docno"].values]
        )

class PerQueryMaxMinScoreTransformer(TransformerBase):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        #topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res
