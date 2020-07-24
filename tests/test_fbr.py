import pandas as pd
import unittest
import os
import pyterrier as pt
from .base import BaseTestCase

class TestFeaturesBatchRetrieve(BaseTestCase):

    def test_fbr_reranking(self):
        if not pt.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        # this test examines the use of ScoringMatchingWithFat 
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 2 documents
        firstpass = pt.BatchRetrieve(indexref, wmodel="BM25") % 2
        pipe = firstpass >> pt.FeaturesBatchRetrieve(indexref, features=["WMODEL:DPH", "WMODEL:PL2"])
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
        result = pipe.transform(input)
        self.assertTrue("qid" in result.columns)
        self.assertTrue("docno" in result.columns)
        self.assertTrue("score" in result.columns)
        self.assertTrue("features" in result.columns)
        self.assertEqual(3, len(result))
        self.assertEqual(result.iloc[0]["features"].size, 2)

        pipe_simple = firstpass >> (pt.BatchRetrieve(indexref, wmodel="DPH") ** pt.BatchRetrieve(indexref, wmodel="PL2"))
        result2 = pipe.transform(input)
        import numpy as np
        f1 = np.stack(result["features"].values)
        f2 = np.stack(result2["features"].values)
        self.assertTrue( np.array_equal(f1, f2))

    def test_fbr_ltr(self):
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"])
        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query-text.trec").head(3)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")
        res = retr.transform(topics)
        res = res.merge(qrels, on=['qid', 'docno'], how='left').fillna(0)
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        #print(res.dtypes)
        RandomForestClassifier(n_estimators=10).fit(np.stack(res["features"]), res["label"])

    def test_fbr(self):
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"])
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
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

if __name__ == "__main__":
    unittest.main()
