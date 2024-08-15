import pandas as pd
import unittest
import os
import pyterrier as pt
from .base import BaseTestCase
import warnings

class TestFeaturesBatchRetrieve(BaseTestCase):

    def __init__(self, *args):
        super().__init__(*args)
        self.method = 'fat'

    def test_compile_to_fbr(self):
        indexref = pt.IndexRef.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 2 documents
        firstpass = pt.BatchRetrieve(indexref, wmodel="BM25")
        pipe_f_fbr = firstpass >> pt.FeaturesBatchRetrieve(indexref, features=["WMODEL:DPH", "WMODEL:PL2"])
        pipe_fbr = pt.FeaturesBatchRetrieve(indexref, wmodel="BM25", features=["WMODEL:DPH", "WMODEL:PL2"])
        pipe_raw = firstpass >> ( pt.BatchRetrieve(indexref, wmodel="DPH") ** pt.BatchRetrieve(indexref, wmodel="PL2") )
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
        res1 = (pipe_f_fbr %2)(input)
        res2 = (pipe_fbr % 2)(input)
        res3 = (pipe_raw % 2)(input)
        compiled = (pipe_raw % 2).compile()
        print(repr(compiled))
        res4 = compiled(input)
        

    def test_fbr_reranking2(self):
        if not pt.terrier.check_version("5.4"):
            self.skipTest("Requires Terrier 5.4")
        # this test examines the use of ScoringMatchingWithFat, using a particular case known to with Terrier 5.3 
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 3 documents
        firstpass = pt.BatchRetrieve(indexref, wmodel="BM25") % 3
        pipe1 = firstpass >> pt.FeaturesBatchRetrieve(indexref, features=["WMODEL:PL2", "WMODEL:Tf"], method=self.method)
        pipe2 = firstpass >> pt.BatchRetrieve(indexref, wmodel="PL2")
        pipe3 = firstpass >> pt.BatchRetrieve(indexref, wmodel="Tf")
        
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
        result0 = firstpass.transform(input)
        result1 = pipe1.transform(input)
        result2 = pipe2.transform(input)
        result3 = pipe3.transform(input)

        print(result1)
        print(result3)
        #BM25 score
        result0_map = { row.docno : row.score for row in result0.itertuples() }
        result1S_map = { row.docno : row.score for row in result1.itertuples() }
        #Tf score
        result12F_map = { row.docno : row.features[1] for row in result1.itertuples() }
        result3_map = { row.docno : row.score for row in result3.itertuples() }
        #PL2 score
        result1F_map = { row.docno : row.features[0] for row in result1.itertuples() }
        result2_map = { row.docno : row.score for row in result2.itertuples() }

        

        # check features scores
        # NB: places can go no less than 4, as two documents have similar PL2 scores
        for rank, row in enumerate(result0.itertuples()):
            docno = row.docno
            # check that score is unchanged
            self.assertAlmostEqual(result1S_map[docno], result0_map[docno], msg="input score mismatch at rank %d for docno %s" % (rank, docno), places=4)
            #  check that Tf feature score is correct
            self.assertAlmostEqual(result12F_map[docno], result3_map[docno], msg="Tf feature score mismatch at rank %d for docno %s" % (rank, docno))
            #  check that feature score is correct
            self.assertAlmostEqual(result1F_map[docno], result2_map[docno], msg="PL2 feature score mismatch at rank %d for docno %s" % (rank, docno), places=4)

    def test_fbr_reranking2(self):
        if not pt.terrier.check_version("5.4"):
            self.skipTest("Requires Terrier 5.4")
        # this test examines the use of ScoringMatchingWithFat, using a particular case known to with Terrier 5.3 
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 3 documents
        firstpass = pt.BatchRetrieve(indexref, wmodel="BM25") % 3
        pipe1 = firstpass >> pt.FeaturesBatchRetrieve(indexref, features=["WMODEL:PL2"], method=self.method)
        pipe2 = firstpass >> pt.BatchRetrieve(indexref, wmodel="PL2")
        
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
        result0 = firstpass.transform(input)
        result1 = pipe1.transform(input)
        result2 = pipe2.transform(input)

        result1["feature0"] = result1.apply(lambda row: row["features"][0], axis=1)
        #BM25 score
        result0_map = { row.docno : row.score for row in result0.itertuples() }
        result1S_map = { row.docno : row.score for row in result1.itertuples() }
        #PL2 score
        result1F_map = { row.docno : row.feature0 for row in result1.itertuples() }
        result2_map = { row.docno : row.score for row in result2.itertuples() }

        print(result1F_map)
        print(result2_map)
        

        # check features scores
        # NB: places can go no less than 4, as two documents have similar PL2 scores
        for rank, row in enumerate(result0.itertuples()):
            docno = row.docno
            # check that score is unchanged
            self.assertAlmostEqual(result1S_map[docno], result0_map[docno], msg="input score mismatch at rank %d for docno %s" % (rank, docno), places=4)
            #  check that feature score is correct
            self.assertAlmostEqual(result1F_map[docno], result2_map[docno], msg="feature score mismatch at rank %d for docno %s" % (rank, docno), places=4)

    def test_fbr_ltr(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"], wmodel="DPH", method=self.method)
        topics = pt.io.read_topics(self.here + "/fixtures/vaswani_npl/query-text.trec").head(3)
        qrels = pt.io.read_qrels(self.here + "/fixtures/vaswani_npl/qrels")
        res = retr.transform(topics)
        res = res.merge(qrels, on=['qid', 'docno'], how='left').fillna(0)
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        #print(res.dtypes)
        RandomForestClassifier(n_estimators=10).fit(np.stack(res["features"]), res["label"])

    def test_fbr(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"], wmodel="DPH", method=self.method)
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
        result = retr.transform(input)
        self.assertTrue("qid" in result.columns)
        self.assertTrue("docno" in result.columns)
        self.assertTrue("score" in result.columns)
        self.assertTrue("features" in result.columns)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result.iloc[0]["features"].size, 1)

        input = pd.DataFrame([["1", "Stability", "u1"]], columns=['qid', 'query', 'username'])
        result = retr.transform(input)
        self.assertIn("username", result.columns)

        retrBasic = pt.BatchRetrieve(indexref)
        if "matching" in retrBasic.controls:
            self.assertNotEqual(retrBasic.controls["matching"], "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull")

    def test_fbr_example(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        index = pt.IndexFactory.of(indexref)
        # this ranker will make the candidate set of documents for each query
        BM25 = pt.BatchRetrieve(index, wmodel="BM25")

        # these rankers we will use to re-rank the BM25 results
        Tf =  pt.BatchRetrieve(index, wmodel="Tf")
        Dl =  pt.BatchRetrieve(index, wmodel="Dl")
        PL2 =  pt.BatchRetrieve(index, wmodel="PL2")

        pipe =  (BM25 %2) >> (Tf ** Dl ** PL2)
        resultP = pipe.search("chemical")
        
        variants = [
            pt.FeaturesBatchRetrieve(indexref, ["WMODEL:Tf", "WMODEL:Dl", "WMODEL:PL2"], wmodel="BM25", method=self.method) % 2,
            BM25 %2 >> pt.FeaturesBatchRetrieve(indexref, ["WMODEL:Tf", "WMODEL:Dl", "WMODEL:PL2"], method=self.method) 
        ]

        for fbr in variants:
            resultF = fbr.search("chemical")
            print(resultP)
            print(resultF)
            
            pd.set_option('display.max_columns', None)

            self.assertEqual(resultP.iloc[0].docno, resultF.iloc[0].docno)
            self.assertEqual(resultP.iloc[0].score, resultF.iloc[0].score)
            self.assertEqual(resultP.iloc[0].features[0], resultF.iloc[0].features[0])
            self.assertEqual(resultP.iloc[0].features[1], resultF.iloc[0].features[1])
            self.assertEqual(resultP.iloc[0].features[2], resultF.iloc[0].features[2])

            pipeCompiled = pipe.compile()
            resultC = pipeCompiled.search("chemical")
            self.assertEqual(resultP.iloc[0].docno, resultC.iloc[0].docno)
            self.assertEqual(resultP.iloc[0].score, resultC.iloc[0].score)
            self.assertEqual(resultP.iloc[0].features[0], resultC.iloc[0].features[0])
            self.assertEqual(resultP.iloc[0].features[1], resultC.iloc[0].features[1])
            self.assertEqual(resultP.iloc[0].features[2], resultC.iloc[0].features[2])

    def test_fbr_empty(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"], wmodel="DPH")
        input = pd.DataFrame([["1", ""]], columns=['qid', 'query'])
        with warnings.catch_warnings(record=True) as w:
            result = retr.transform(input)
            assert "Skipping empty query" in str(w[-1].message)
        self.assertTrue(len(result) == 0)

if __name__ == "__main__":
    unittest.main()
