import os
import pandas as pd
import numpy as np
from .utils import Utils

def Experiment(topics,retr_systems,eval_metrics,qrels, names=None, perquery=False, dataframe=True):
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
    if type(topics)==type(""):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if type(qrels)==type(""):
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
    evals={}

    for weight,res in zip(names,results):
        evals[weight]=Utils.evaluate(res,qrels, metrics=eval_metrics, perquery=perquery)
    if dataframe:
        evals = pd.DataFrame.from_dict(evals, orient='index')
    return evals

class LambdaPipeline():
    """
    This class allows pipelines components to be written as functions or lambdas

    :Example:
    >>> #this pipeline would remove all but the first two documents from a result set
    >>> lp = LambdaPipeline(lambda res : res[res["rank"] < 2])

    """

    def __init__(self, lambdaFn):
        self.fn = lambdaFn

    def transform(self, inputRes):
        fn = self.fn
        return fn(inputRes)

class ComposedPipeline():
    """ 
    This class allows pipeline components to be chained together.

    :Example:

    >>> comp = ComposedPipeline([ DPH_br, LambdaPipeline(lambda res : res[res["rank"] < 2])])
    >>> OR
    >>>  # we can even use lambdas as transformers
    >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
    
    """
    def __init__(self, models=[]):
        import types
        self.models = list( map(lambda x : LambdaPipeline(x) if callable(x) else x, models) )
    
    def transform(self, topics):
        for m in self.models:
            topics = m.transform(topics)
        return topics

class LTR_pipeline():
    """
    This class simplifies the use of Scikit-learn's techniques for learning-to-rank.
    """
    def __init__(self, index, model, features, qrels, LTR):
        """
        Init method

        Args:
            index: The index which to query.
                Can be an Indexer object(Can be parent Indexer or any of its child classes)
                or a string with the path to the index_dir/data.properties
            model(str): The weighting model to use. E.g. "PL2", OR another pipeline
            features(list): A list of the feature names to use
            qrels(DataFrame): Dataframe with columns=['qid','docno', 'label']
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
        """
        if isinstance(model, str):
            from .batchretrieve import FeaturesBatchRetrieve
            self.feat_retrieve = FeaturesBatchRetrieve(index, features)
            self.feat_retrieve.setControl('wmodel', model)
        else:
            self.feat_retrieve = model
        self.qrels = qrels
        self.LTR = LTR

    def fit(self, topicsTrain):
        """
        Trains the model with the given topics.

        Args:
            topicsTrain(DataFrame): A dataframe with the topics to train the model
        """
        if len(topicsTrain) == 0:
            raise ValueError("No topics to fit to")
        train_DF = self.feat_retrieve.transform(topicsTrain)
        if not 'features' in train_DF.columns:
            raise ValueError("No features column retrieved")
        train_DF = train_DF.merge(self.qrels, on=['qid','docno'], how='left').fillna(0)
        self.LTR.fit(np.stack(train_DF["features"].values), train_DF["label"].values)

    def transform(self, topicsTest):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = self.feat_retrieve.transform(topicsTest)
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

class XGBoostLTR_pipeline(LTR_pipeline):
    """
    This class simplifies the use of XGBoost's techniques for learning-to-rank.
    """

    def __init__(self, index, model, features, qrels, LTR, validqrels):
        """
        Init method

        Args:
            index: The index which to query.
                Can be an Indexer object(Can be parent Indexer or any of its child classes)
                or a string with the path to the index_dir/data.properties
            model(str): The weighting model to use. E.g. "PL2"
            features(list): A list of the feature names to use
            qrels(DataFrame): Dataframe with columns=['qid','docno', 'label']
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
            validqrels(DataFrame): The qrels which to use for the validation.
        """
        super().__init__(index,model,features, qrels, LTR)
        self.validqrels = validqrels

    def transform(self, topicsTest):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = self.feat_retrieve.transform(topicsTest)
        #xgb is more sensitive about the type of the values.
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

    def fit(self, topicsTrain, topicsValid):
        """
        Trains the model with the given training and validation topics.

        Args:
            topicsTrain(DataFrame): A dataframe with the topics to train the model
            topicsValid(DataFrame): A dataframe with the topics for validation
        """
        if len(topicsTrain) == 0:
            raise ValueError("No training topics to fit to")
        if len(topicsValid) == 0:
            raise ValueError("No training topics to fit to")

        tr_res = self.feat_retrieve.transform(topicsTrain)
        va_res = self.feat_retrieve.transform(topicsValid)
        if not 'features' in tr_res.columns:
            raise ValueError("No features column retrieved in training")
        if not 'features' in va_res.columns:
            raise ValueError("No features column retrieved in validation")

        tr_res = tr_res.merge(self.qrels, on=['qid','docno'], how='left').fillna(0)
        va_res = va_res.merge(self.validqrels, on=['qid','docno'], how='left').fillna(0)

        self.LTR.fit(
            np.stack(tr_res["features"].values),
            tr_res["label"].values, tr_res.groupby(["qid"]).count()["docno"].values,
            eval_set=[(np.stack(va_res["features"].values),
            va_res["label"].values)],
            eval_group=[va_res.groupby(["qid"]).count()["docno"].values]
        )
