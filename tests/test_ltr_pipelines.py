import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

class TestLTRPipeline(BaseTestCase):

    def test_fastrank(self):
        import fastrank
        train_request = fastrank.TrainRequest.coordinate_ascent()
        params = train_request.params
        params.init_random = True
        params.normalize = True
        params.seed = 1234567

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        pipeline = pt.FeaturesBatchRetrieve(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >> \
            pt.ltr.apply_learned_model(train_request, form="fastrank")
        
        pipeline.fit(topics, qrels, topics, qrels)
        pt.Utils.evaluate(
            pipeline.transform(topics),
            qrels
        )

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
            pt.ltr.apply_learned_model(xgb.sklearn.XGBRanker(**xgparams), form="ltr")
        
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
            pt.ltr.apply_learned_model(RandomForestClassifier())
        
        pipeline.fit(topics, qrels)
        pt.Utils.evaluate(
            pipeline.transform(topics),
            qrels,
        )

    def test_ltr_pipeline_feature_change(self):
        from sklearn.ensemble import RandomForestClassifier

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        rf = RandomForestClassifier()

        pipeline = pt.FeaturesBatchRetrieve(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >> \
            pt.ltr.apply_learned_model(rf)
        
        pipeline.fit(topics, qrels)
        pipeline.transform(topics)

        pipeline2 = pt.FeaturesBatchRetrieve(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25", "WMODEL:Dl"], controls={"wmodel" : "DPH"}) >> \
            pt.ltr.apply_learned_model(rf)
        with self.assertRaises(ValueError):
            pipeline2.transform(topics)

if __name__ == "__main__":
    unittest.main()
