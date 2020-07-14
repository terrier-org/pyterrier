import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

class TestLTRPipeline(BaseTestCase):

    def test_xgltr_pipeline(self):
        import xgboost as xgb

        xgparams = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'gamma': 1.0, 'min_child_weight': 0.1,
            'max_depth': 6,
            'verbose': 2,
            'random_state': 42
        }

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        pipeline = pt.FeaturesBatchRetrieve(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >> \
            pt.XGBoostLTR_pipeline(xgb.sklearn.XGBRanker(**xgparams))
        
        pipeline.fit(topics, qrels, topics, qrels)
        pt.Utils.evaluate(
            pipeline.transform(topics),
            qrels
        )

    def test_ltr_pipeline(self):
        from sklearn.ensemble import RandomForestClassifier

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        pipeline = pt.FeaturesBatchRetrieve(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >> \
            pt.LTR_pipeline(RandomForestClassifier())
        
        pipeline.fit(topics, qrels)
        pt.Utils.evaluate(
            pipeline.transform(topics),
            qrels,
        )

if __name__ == "__main__":
    unittest.main()
