import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

class TestLTRPipeline(BaseTestCase):

    def test_fastrank(self):
        try:
            import fastrank
        except:
            self.skipTest("Fastrank not installed")
        train_request = fastrank.TrainRequest.coordinate_ascent()
        params = train_request.params
        params.init_random = True
        params.normalize = True
        params.seed = 1234567

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        pipeline = pt.terrier.FeaturesRetriever(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >> \
            pt.ltr.apply_learned_model(train_request, form="fastrank")
        
        pipeline.fit(topics, qrels, topics, qrels)
        pt.Evaluate(
            pipeline.transform(topics),
            qrels
        )

    def test_xgltr_pipeline(self):
        try:
            import xgboost as xgb
        except:
            self.skipTest("xgboost not installed")

        xgparams = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'gamma': 1.0, 'min_child_weight': 0.1,
            'max_depth': 6,
            'random_state': 42
        }

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        features = pt.terrier.FeaturesRetriever(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"})
        pipeline = features >> pt.ltr.apply_learned_model(xgb.sklearn.XGBRanker(**xgparams), form="ltr")
        
        pipeline.fit(topics, qrels, topics, qrels)
        pt.Evaluate(
            pipeline.transform(topics),
            qrels
        )
  
    def test_ltr_mocked(self):

        class MockLTR():
            def fit(self, 
                    features_tr, labels_tr, 
                    group=None, eval_set=None, eval_group=None):
                self.features_tr = features_tr
                self.labels_tr = labels_tr
                self.group = group
                self.eval_set = eval_set
                self.eval_group = eval_group
        
        import pandas as pd, numpy as np
        res = pd.DataFrame([ 
                ["q2", "b", np.array([2])], # feature value encodes the number of docs for this query
                ["q2", "a", np.array([2])], 
                ["q1", "a", np.array([1])] 
            ], columns=["qid", "docno", "features"])
        qrels = pd.DataFrame([ ["q2", "a", 1], ["q1", "a", 1] ], columns=["qid", "docno", "label"])

        mockltr = MockLTR()
        mock_pipeline = pt.ltr.apply_learned_model(mockltr, form="ltr")
        mock_pipeline.fit(res, qrels, res, qrels)

        # check feature shapes
        self.assertEqual(3, mockltr.features_tr.shape[0])
        self.assertEqual(1, mockltr.features_tr.shape[1])
        # check label shape
        self.assertEqual(3, mockltr.labels_tr.shape[0])
        # check label counts
        self.assertEqual(2, np.count_nonzero(mockltr.labels_tr == 1))
        self.assertEqual(1, np.count_nonzero(mockltr.labels_tr == 0))

        # idea here is to check ordering between group and features information
        for i in range(2): # 2 queries
            self.assertEqual(mockltr.features_tr[i, 0], mockltr.group[i], "at query index %d" % i)

    def test_skltr_pipeline(self):
        from sklearn.ensemble import RandomForestClassifier

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        pipeline = ( pt.terrier.FeaturesRetriever(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >>
            pt.ltr.apply_learned_model(RandomForestClassifier())
        )
        
        pipeline.fit(topics, qrels)
        pt.Evaluate(
            pipeline.transform(topics),
            qrels
        )

    def test_ltr_pipeline_feature_change(self):
        from sklearn.ensemble import RandomForestClassifier

        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query_light.trec").head(5)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")

        rf = RandomForestClassifier()

        pipeline = pt.terrier.FeaturesRetriever(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25"], controls={"wmodel" : "DPH"}) >> \
            pt.ltr.apply_learned_model(rf)
        
        pipeline.fit(topics, qrels)
        pipeline.transform(topics)

        pipeline2 = pt.terrier.FeaturesRetriever(self.here + "/fixtures/index/data.properties", ["WMODEL:PL2", "WMODEL:BM25", "WMODEL:Dl"], controls={"wmodel" : "DPH"}) >> \
            pt.ltr.apply_learned_model(rf)
        with self.assertRaises(ValueError):
            pipeline2.transform(topics)

if __name__ == "__main__":
    unittest.main()
