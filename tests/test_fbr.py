import pandas as pd
import unittest
import os
import pyterrier as pt
from .base import BaseTestCase
import warnings

class TestFeaturesBatchRetrieve(BaseTestCase):

    def test_compile_to_fbr(self):
        indexref = pt.IndexRef.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 2 documents
        firstpass = pt.terrier.Retriever(indexref, wmodel="BM25")
        pipe_f_fbr = firstpass >> pt.terrier.FeaturesRetriever(indexref, features=["WMODEL:DPH", "WMODEL:PL2"])
        pipe_fbr = pt.terrier.FeaturesRetriever(indexref, wmodel="BM25", features=["WMODEL:DPH", "WMODEL:PL2"])
        pipe_raw = firstpass >> ( pt.terrier.Retriever(indexref, wmodel="DPH") ** pt.terrier.Retriever(indexref, wmodel="PL2") )
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])
        res1 = (pipe_f_fbr %2)(input)
        res2 = (pipe_fbr % 2)(input)
        res3 = (pipe_raw % 2)(input)
        compiled = (pipe_raw % 2).compile()
        print(repr(compiled))
        res4 = compiled(input)
        

    def test_fbr_reranking(self):
        if not pt.terrier.check_version("5.4"):
            self.skipTest("Requires Terrier 5.4")
        # this test examines the use of ScoringMatchingWithFat 
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 2 documents
        firstpass = pt.terrier.Retriever(indexref, wmodel="BM25") % 2
        pipe = firstpass >> pt.terrier.FeaturesRetriever(indexref, features=["WMODEL:DPH", "WMODEL:PL2"])
        input = pd.DataFrame([["1", "Stability"]], columns=['qid', 'query'])

        result0 = firstpass(input)
        result0_map = { row.docno : row.score for row in result0.itertuples() }

        result = pipe.transform(input)
        self.assertTrue("qid" in result.columns)
        self.assertTrue("docno" in result.columns)
        self.assertTrue("score" in result.columns)
        self.assertTrue("features" in result.columns)
        self.assertEqual(2, len(result))
        self.assertEqual(result.iloc[0]["features"].size, 2)
        
        result1S_map = { row.docno : row.score for row in result.itertuples() }
        self.assertEqual(result0_map, result1S_map)

        pipe_simple = firstpass >> (pt.terrier.Retriever(indexref, wmodel="DPH") ** pt.terrier.Retriever(indexref, wmodel="PL2"))
        result2 = pipe.transform(input)
        import numpy as np
        f1 = np.stack(result["features"].values)
        f2 = np.stack(result2["features"].values)
        self.assertTrue( np.array_equal(f1, f2))

        result2S_map = { row.docno : row.score for row in result2.itertuples() }
        self.assertEqual(result0_map, result2S_map)

        result1F0_map = { row.docno : row.features[0] for row in result.itertuples() }
        result2F0_map = { row.docno : row.features[0] for row in result2.itertuples() }
        result1F1_map = { row.docno : row.features[1] for row in result.itertuples() }
        result2F1_map = { row.docno : row.features[1] for row in result2.itertuples() }

        self.assertEqual(result1F0_map, result2F0_map)
        self.assertEqual(result1F1_map, result2F1_map)

    def test_fbr_reranking2(self):
        if not pt.terrier.check_version("5.4"):
            self.skipTest("Requires Terrier 5.4")
        # this test examines the use of ScoringMatchingWithFat, using a particular case known to with Terrier 5.3 
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        # we only want a candidate set of 3 documents
        firstpass = pt.terrier.Retriever(indexref, wmodel="BM25") % 3
        pipe1 = firstpass >> pt.terrier.FeaturesRetriever(indexref, features=["WMODEL:PL2"])
        pipe2 = firstpass >> pt.terrier.Retriever(indexref, wmodel="PL2")
        
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
        retr = pt.terrier.FeaturesRetriever(indexref, ["WMODEL:PL2"], wmodel="DPH")
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

        for name, retr in [
                ("FeaturesRetriever", pt.terrier.FeaturesRetriever(indexref, ["WMODEL:PL2"], wmodel="DPH")), 
                ("FeaturesBatchRetrieve", pt.FeaturesBatchRetrieve(indexref, ["WMODEL:PL2"], wmodel="DPH"))
            ]:
            self.subTest(name)
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

            retrBasic = pt.terrier.Retriever(indexref)
            if "matching" in retrBasic.controls:
                self.assertNotEqual(retrBasic.controls["matching"], "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull")

    def test_fbr_example(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        index = pt.IndexFactory.of(indexref)
        # this ranker will make the candidate set of documents for each query
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")

        # these rankers we will use to re-rank the BM25 results
        TF_IDF =  pt.terrier.Retriever(index, wmodel="Dl")
        PL2 =  pt.terrier.Retriever(index, wmodel="PL2")

        pipe =  (BM25 %2) >> (TF_IDF ** PL2)
        fbr = pt.terrier.FeaturesRetriever(indexref, ["WMODEL:Dl", "WMODEL:PL2"], wmodel="BM25") % 2
        resultP = pipe.search("chemical")
        resultF = fbr.search("chemical")
        pd.set_option('display.max_columns', None)

        self.assertEqual(resultP.iloc[0].docno, resultF.iloc[0].docno)
        self.assertEqual(resultP.iloc[0].score, resultF.iloc[0].score)
        self.assertEqual(resultP.iloc[0].features[0], resultF.iloc[0].features[0])
        self.assertEqual(resultP.iloc[0].features[1], resultF.iloc[0].features[1])

        pipeCompiled = pipe.compile()
        resultC = pipeCompiled.search("chemical")
        self.assertEqual(resultP.iloc[0].docno, resultC.iloc[0].docno)
        self.assertEqual(resultP.iloc[0].score, resultC.iloc[0].score)
        self.assertEqual(resultP.iloc[0].features[0], resultC.iloc[0].features[0])
        self.assertEqual(resultP.iloc[0].features[1], resultC.iloc[0].features[1])

    def test_fbr_empty(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.terrier.FeaturesRetriever(indexref, ["WMODEL:PL2"], wmodel="DPH")
        input = pd.DataFrame([["1", ""]], columns=['qid', 'query'])
        with warnings.catch_warnings(record=True) as w:
            result = retr.transform(input)
            assert "Skipping empty query" in str(w[-1].message)
        self.assertTrue(len(result) == 0)

if __name__ == "__main__":
    unittest.main()
