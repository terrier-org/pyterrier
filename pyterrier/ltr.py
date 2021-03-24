

from .transformer import TransformerBase, EstimatorBase
from .apply import doc_score, doc_features
from .model import add_ranks
from typing import Sequence, Union
import numpy as np

FeatureList = Union[Sequence[int], int]


class AblateFeatures(TransformerBase):
    
    def __init__(self, fids: FeatureList):
        self.fids = fids if isinstance(fids, list) else [fids]
        self.null = 0
        
    def transform(self, topics_and_res):
        
        def _reset(row):
            fvalues = row["features"].copy() 
            for findex in self.fids:
                fvalues[findex] = self.null
            return fvalues
        
        assert "features" in topics_and_res.columns
        topics_and_res = topics_and_res.copy()
        topics_and_res["features"] = topics_and_res.apply(_reset, axis=1)
        return topics_and_res

class KeepFeatures(TransformerBase):
    
    def __init__(self, fids : FeatureList):
        self.fids = fids if isinstance(fids, list) else [fids]
        self.null = 0
        
    def transform(self, topics_and_res):
        
        assert "features" in topics_and_res.columns
        topics_and_res = topics_and_res.copy()
        topics_and_res["features"] = topics_and_res.apply(lambda row: row["features"][self.fids], axis=1)
        return topics_and_res

class RegressionTransformer(EstimatorBase):
    """
    This class simplifies the use of Scikit-learn's techniques for learning-to-rank.
    """
    def __init__(self, learner, *args, fit_kwargs={}, **kwargs):
        """
        Init method

        Args:
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
            fit_kwargs: A dictionary containing additional arguments that can be passed to LTR's fit() method.  
        """
        self.fit_kwargs = fit_kwargs
        super().__init__(*args, **kwargs)
        self.learner = learner
        self.num_f = None

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
        kwargs = self.fit_kwargs
        self.learner.fit(np.stack(train_DF["features"].values), train_DF["label"].values, **kwargs)
        self.num_f = train_DF.iloc[0].features.shape[0]
        return self

    def transform(self, test_DF):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = test_DF.copy()

        # check for change in number of features
        found_numf = test_DF.iloc[0].features.shape[0]
        if self.num_f is not None:
            if found_numf != self.num_f:
                raise ValueError("Expected %d features, but found %d features" % (self.num_f, found_numf))
        if hasattr(self.learner, 'feature_importances_'):
            if len(self.learner.feature_importances_) != found_numf:
                raise ValueError("Expected %d features, but found %d features" % (len(self.learner.feature_importances_), found_numf))

        test_DF["score"] = self.learner.predict(np.stack(test_DF["features"].values))
        return add_ranks(test_DF)

class LTRTransformer(RegressionTransformer):
    """
    This class simplifies the use of LightGBM and XGBoost for learning-to-rank.
    """

    def fit(self, topics_and_results_Train, qrelsTrain, topics_and_results_Valid, qrelsValid):
        """
        Trains the model with the given training and validation topics.

        Args:
            topics_and_results_Train(DataFrame): A dataframe with the topics and results to train the model
            qrelsTrain(DataFrame): A dataframe containing the qrels for the training topics
            topics_and_results_Valid(DataFrame): A dataframe with the topics and results for validation
            qrelsValid(DataFrame): A dataframe containing the qrels for the validation topics
            
        """
        if topics_and_results_Train is None or len(topics_and_results_Train) == 0:
            raise ValueError("No training results to fit to")
        if topics_and_results_Valid is None or len(topics_and_results_Valid) == 0:
            raise ValueError("No validation results to fit to")

        if 'features' not in topics_and_results_Train.columns:
            raise ValueError("No features column retrieved in training")
        if 'features' not in topics_and_results_Valid.columns:
            raise ValueError("No features column retrieved in validation")

        tr_res = topics_and_results_Train.merge(qrelsTrain, on=['qid', 'docno'], how='left').fillna(0)
        va_res = topics_and_results_Valid.merge(qrelsValid, on=['qid', 'docno'], how='left').fillna(0)

        kwargs = self.fit_kwargs
        self.learner.fit(
            np.stack(tr_res["features"].values), tr_res["label"].values, 
            group=tr_res.groupby(["qid"]).count()["docno"].values, # we name group here for lightgbm compat. 
            eval_set=[(np.stack(va_res["features"].values), va_res["label"].values)],
            eval_group=[va_res.groupby(["qid"]).count()["docno"].values],
            **kwargs
        )
        self.num_f = tr_res.iloc[0].features.shape[0]

class FastRankEstimator(EstimatorBase):
    """
    This class simplifies the use of FastRank's techniques for learning-to-rank.
    """
    def __init__(self, learner, *args, **kwargs):
        """
        Init method

        Args:
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
            fit_kwargs: A dictionary containing additional arguments that can be passed to LTR's fit() method.  
        """
        super().__init__(*args, **kwargs)
        self.learner = learner
        self.model = None
        self.num_f = None

    def _make_dataset(self, test_DF, add_labels = False):
        
        from collections import defaultdict
        from itertools import count
        from fastrank import CDataset
        qid_map = defaultdict(count().__next__)
        features = np.stack(test_DF["features"].values).astype('float32')
        qids = test_DF["qid"].apply(lambda qid : qid_map[qid]).values
        if add_labels:
            y = test_DF["label"].values
        else:
            y = np.zeros(len(test_DF))
        dataset = CDataset.from_numpy(features, y, qids)
        return dataset

    def fit(self, topics_and_results_Train, qrelsTrain, topics_and_results_Valid=None, qrelsValid=None):
        if topics_and_results_Train is None or len(topics_and_results_Train) == 0:
            raise ValueError("No training results to fit to")

        if 'features' not in topics_and_results_Train.columns:
            raise ValueError("No features column retrieved in training")

        tr_res = topics_and_results_Train.merge(qrelsTrain, on=['qid', 'docno'], how='left').fillna(0)
        dataset = self._make_dataset(tr_res, add_labels=True)
        self.num_f = dataset.num_features()
        self.model = dataset.train_model(self.learner)

    def transform(self, topics_and_docs_Test):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        if self.model is None:
            raise ValueError("fit() must be called first")
        test_DF = topics_and_docs_Test.copy()
        dataset = self._make_dataset(test_DF, add_labels=False)

        # check for change in number of features
        found_numf = dataset.num_features()
        if self.num_f is not None and found_numf != self.num_f:
            raise ValueError("Expected %d features, but found %d features" % (self.num_f, found_numf))
        if hasattr(self.learner, 'feature_importances_'):
            if len(self.learner.feature_importances_) != found_numf:
                raise ValueError("Expected %d features, but found %d features" % (len(self.learner.feature_importances_), found_numf))
        
        rtr = dataset.predict_scores(self.model)
        scores = [rtr[i] for i in range(len(rtr))]
        test_DF["score"] = scores
        return add_ranks(test_DF)

def ablate_features(fids : FeatureList) -> TransformerBase:
    """
        Ablates features (sets feature value to 0) from a pipeline. This is useful for 
        performing feature ablation studies, whereby a feature is removed from the pipeline
        before learning. 

        Args: 
            fids: one or a list of integers corresponding to features indices to be removed
    """
    return AblateFeatures(fids)

def keep_features(fids : FeatureList) -> TransformerBase:
    """
        Reduces the features in a pipeline to only those mentioned. This is useful for 
        performing feature ablation studies, whereby only some features are kept 
        (and other removed) from a pipeline before learning occurs. 

        Args: 
            fids: one or a list of integers corresponding to the features indice to be kept
    """
    return KeepFeatures(fids)

def feature_to_score(fid : int) -> TransformerBase:
    """
        Applies a specified feature for ranking. Useful for evaluating which of a number of 
        pre-computed features are useful for ranking. 

        Args: 
            fid: a single feature id that should be kept
    """
    return doc_score(lambda row : row["features"][fid])

def apply_learned_model(learner, form : str = 'regression', **kwargs) -> TransformerBase:
    """
        Results in a transformer that can take in documents that have a "features" column,
        and pass that to the specified learner via its transform() function, to obtain the
        documents' "score" column. Learners should follow the sklearn's general pattern
        with a fit() method (
        c.f. an sklearn `Estimator <https://scikit-learn.org/stable/glossary.html#term-estimator>`_)
        and a `predict() <https://scikit-learn.org/stable/glossary.html#term-predict>`_ method.

        xgBoost and LightGBM are also supported through the use of `type='ltr'` kwarg.

        Args: 
            learner: an sklearn-compatible estimator
            form(str): either 'regression' or 'ltr'        
    """
    if form == 'ltr':
        return LTRTransformer(learner, **kwargs)
    if form == 'fastrank':
        return FastRankEstimator(learner, **kwargs)
    return RegressionTransformer(learner, **kwargs)

def score_to_feature() -> TransformerBase:
    """
        Takes the document's "score" from the score attribute, and uses it as a single feature. 
        In particular, a feature union operator does not use any score of the documents in the
        candidate set as a ranking feaure. Using the resulting transformer within a feature-union
        means that an additional ranking feature is added to the "feature" column.

        Example::

            cands = pt.BatchRetrieve(index, wmodel="BM25")
            bm25f = pt.BatchRetrieve(index, wmodel="BM25F")
            pl2f = pt.BatchRetrieve(index, wmodel="PL2F")
            
            two_features = cands >> (bm25f  **  pl2f)
            three_features = cands >> (bm25f  **  pl2f ** pt.ltr.score_to_feature())  

    """
    return doc_features(lambda row : np.array(row["score"]))