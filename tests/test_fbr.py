import pandas as pd
import unittest, math, os, ast, statistics
import pyterrier as pt




class TestFeaturesBatchRetrieve(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFeaturesBatchRetrieve, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init()
        self.here=os.path.dirname(os.path.realpath(__file__))

    def test_fbr_ltr(self):
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here+"/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"])
        topics = pt.Utils.parse_trec_topics_file(self.here+"/../vaswani_npl/query-text.trec").head(3)
        qrels = pt.Utils.parse_qrels(self.here+"/../vaswani_npl/qrels")
        res = retr.transform(topics)
        res = res.merge(qrels, on=['qid','docno'], how='left').fillna(0)
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        print(res.dtypes)
        RandomForestClassifier(n_estimators=10).fit(np.stack(res["features"]), res["label"])


    def test_fbr(self):
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here+"/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"])
        input=pd.DataFrame([["1", "Stability"]],columns=['qid','query'])
        result = retr.transform(input)
        self.assertTrue("qid" in result.columns)
        self.assertTrue("docno" in result.columns)
        self.assertTrue("score" in result.columns)
        self.assertTrue("features" in result.columns)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result.iloc[0]["features"].size, 1)

        retrBasic = pt.BatchRetrieve(indexref)
        if "matching" in retrBasic.controls:
            self.assertNotEqual(retrBasic.controls["matching"], "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull")


