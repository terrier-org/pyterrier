import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

class TestLTRPipeline(BaseTestCase):

    def test_xgltr_pipeline(self):
        import xgboost as xgb

        params = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1, # 0.05, # was 0.1
            'gamma': 1.0, 'min_child_weight': 0.1,
            'max_depth': 6,
            'verbose': 2,
            'random_state': 42
        }

        topics = pt.Utils.parse_trec_topics_file(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.Utils.parse_qrels(self.here + "/fixtures/vaswani_npl/qrels")
        pipeline = pt.XGBoostLTR_pipeline(
            self.here + "/fixtures/index/data.properties",
            "DPH",
            ["WMODEL:PL2", "WMODEL:BM25"],
            qrels,
            xgb.sklearn.XGBRanker(**params),
            qrels)

        pipeline.fit(topics, topics)
        # metrics = pt.Utils.evaluate(
        pt.Utils.evaluate(
            pipeline.transform(topics),
            qrels,
        )

    def test_ltr_pipeline(self):
        from sklearn.ensemble import RandomForestClassifier

        topics = pt.Utils.parse_trec_topics_file(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.Utils.parse_qrels(self.here + "/fixtures/vaswani_npl/qrels")
        pipeline = pt.LTR_pipeline(
            self.here + "/fixtures/index/data.properties",
            "DPH",
            ["WMODEL:PL2", "WMODEL:BM25"],
            qrels,
            RandomForestClassifier())

        pipeline.fit(topics)
        # metrics = pt.Utils.evaluate(
        pt.Utils.evaluate(
            pipeline.transform(topics),
            qrels,
        )

if __name__ == "__main__":
    unittest.main()
