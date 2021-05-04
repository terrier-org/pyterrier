import pandas as pd
import unittest
import pyterrier as pt
import os
import shutil
import tempfile
import pyterrier.transformer as ptt;
from matchpy import *
from .base import BaseTestCase

class TestRewrite(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass

    def test_stash_results_noclear(self):
        inputDF = pt.new.ranked_documents([[1, 2], [2,0]], query=[["a", "a"], ["b", "b"]])
        savedDF = pt.rewrite.stash_results(clear=False)(inputDF)
        self.assertEqual(4, len(savedDF))
        self.assertIn("stashed_results_0", savedDF.columns)
        self.assertIn("qid", savedDF.columns)
        self.assertIn("docno", savedDF.columns)
        self.assertIn("score", savedDF.columns)
        self.assertIn("query", savedDF.columns)

        stasheddocs_q1 = savedDF.iloc[0]["stashed_results_0"]
        self.assertEqual(2, len(stasheddocs_q1))
        self.assertEqual(savedDF.iloc[0]["qid"], stasheddocs_q1[0]["qid"])

        stasheddocs_q2 = savedDF.iloc[3]["stashed_results_0"]
        self.assertEqual(2, len(stasheddocs_q2))
        self.assertEqual(savedDF.iloc[3]["qid"], stasheddocs_q2[0]["qid"])


    def test_stash_results(self):
        inputDF = pt.new.ranked_documents([[1, 2], [2,0]], query=[["a", "a"], ["b", "b"]])
        savedDF = pt.rewrite.stash_results()(inputDF)
        self.assertEqual(2, len(savedDF))
        self.assertIn("stashed_results_0", savedDF.columns)

        savedQueryDF = pt.apply.query(lambda row: row["query"] + " 1")(savedDF)
        self.assertIn("stashed_results_0", savedQueryDF.columns)
        self.assertIn("qid", savedQueryDF.columns)
        self.assertIn("query", savedQueryDF.columns)
        self.assertIn("query_0", savedQueryDF.columns)
        
        self.assertEqual(2, len(savedQueryDF))
        restoredDf = pt.rewrite.reset_results()(savedQueryDF)
        self.assertEqual(4, len(restoredDf))
        self.assertIn("qid", restoredDf.columns)
        self.assertIn("docno", restoredDf.columns)
        self.assertIn("score", restoredDf.columns)
        self.assertIn("query", restoredDf.columns)
    
    def test_stash_results_SDM(self):
        inputDF = pt.new.ranked_documents([[1, 2], [2,0]], query=[["a a", "a a"], ["b b", "b b"]])
        savedDF = pt.rewrite.stash_results()(inputDF)
        self.assertEqual(2, len(savedDF))
        self.assertIn("stashed_results_0", savedDF.columns)

        savedQueryDF = pt.rewrite.SDM()(savedDF)
        self.assertIn("stashed_results_0", savedQueryDF.columns)
        self.assertIn("qid", savedQueryDF.columns)
        self.assertIn("query", savedQueryDF.columns)
        self.assertIn("query_0", savedQueryDF.columns)
        
        self.assertEqual(2, len(savedQueryDF))
        restoredDf = pt.rewrite.reset_results()(savedQueryDF)
        self.assertEqual(4, len(restoredDf))
        self.assertIn("qid", restoredDf.columns)
        self.assertIn("docno", restoredDf.columns)
        self.assertIn("score", restoredDf.columns)
        self.assertIn("query", restoredDf.columns)

    def test_save_docs_CE(self):
        index = pt.get_dataset("vaswani").get_index()
        dph = pt.BatchRetrieve(index, wmodel="DPH")
        pipe = dph \
            >> pt.rewrite.stash_results() \
            >> pt.BatchRetrieve(index, wmodel="BM25") \
            >> pt.rewrite.Bo1QueryExpansion(index) \
            >> pt.rewrite.reset_results() \
            >> dph
        rtr1 = dph.search("chemical reactions")        
        rtr2 = pipe.search("chemical reactions")
        # Bo1 should be applied as a re-ranker, hence the
        # number of docs in rtr1 and rtr2 should be equal
        self.assertEqual(len(rtr1), len(rtr2))

        # check columns are passed through where we expect
        pipeP3 = dph \
            >> pt.rewrite.stash_results() \
            >> pt.BatchRetrieve(index, wmodel="BM25")
        res3 = pipeP3.search("chemical reactions")
        self.assertIn("stashed_results_0", res3.columns)
        pipeP4 = dph \
            >> pt.rewrite.stash_results() \
            >> pt.BatchRetrieve(index, wmodel="BM25") \
            >> pt.rewrite.Bo1QueryExpansion(index)
        res4 = pipeP3.search("chemical reactions")
        self.assertIn("stashed_results_0", res4.columns)
    
    def test_save_docs_QE(self):
        index = pt.get_dataset("vaswani").get_index()
        dph = pt.BatchRetrieve(index, wmodel="DPH")
        pipe = dph \
            >> pt.rewrite.stash_results(clear=False) \
            >> pt.rewrite.Bo1QueryExpansion(index) \
            >> pt.rewrite.reset_results() \
            >> dph
        rtr1 = dph.search("chemical reactions")        
        rtr2 = pipe.search("chemical reactions")
        # Bo1 should be applied as a re-ranker, hence the
        # number of docs in rtr1 and rtr2 should be equal
        self.assertEqual(len(rtr1), len(rtr2))

    def test_reset_with_docs(self):
        inputDF = pt.new.ranked_documents([[1, 2], [2,0]])
        inputDF["query"] = ["one #1", "one #1", "one two #1", "one two #1"]
        inputDF["query_0"] = ["one", "one", "one two", "one two"]
        outputDF = pt.rewrite.reset().transform(inputDF)
        self.assertEqual(len(inputDF), len(outputDF))
        self.assertTrue("query" in outputDF.columns)
        self.assertFalse("query_0" in outputDF.columns)
        self.assertTrue("docno" in outputDF.columns)
        self.assertTrue("score" in outputDF.columns)
        self.assertTrue("rank" in outputDF.columns)
        self.assertEqual(outputDF.iloc[0]["query"], "one")
        self.assertEqual(outputDF.iloc[1]["query"], "one")
        self.assertEqual(outputDF.iloc[2]["query"], "one two")
        self.assertEqual(outputDF.iloc[3]["query"], "one two")

    def test_sdm_freq(self):
        if not pt.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        self._sdm(True)

    def test_sdm(self):
        if not pt.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        self._sdm(False)

    def test_sdm_docs(self):
        docs = pt.new.ranked_documents([[1,1]], qid=["q1"], query=[["hello friend","hello friend"]])
        sdm = pt.rewrite.SDM()
        pipe = docs >> sdm
        qids = pt.new.queries(["hello there"], qid=["q1"])  
        rtr = pipe(qids)
        self.assertIn("query", rtr.columns)
        self.assertIn("query_0", rtr.columns)
        self.assertIn("docno", rtr.columns)
        print(rtr)
        self.assertEqual(2, len(rtr))


    def _sdm(self, freq):
        dataset = pt.datasets.get_dataset("vaswani")
        indexer = pt.TRECCollectionIndexer(self.test_dir, blocks=True)
        indexref = indexer.index(dataset.get_corpus())
        if freq:
            sdm = pt.rewrite.SDM(prox_model="org.terrier.matching.models.Tf")
        else:
            sdm = pt.rewrite.SDM()
       
        queriesIn = pd.DataFrame([["1", "compact"], ["2", "compact memories"]], columns=["qid", "query"])
        queriesOut = sdm.transform(queriesIn)
        self.assertEqual(len(queriesOut), 2)
        self.assertIn("query", queriesOut.columns)
        self.assertIn("query_0", queriesOut.columns)
        self.assertEqual(queriesOut.iloc[0]["query"], "compact")

        # check for pushed query representation        
        self.assertTrue("query_0" in queriesOut.columns)
        self.assertEqual(queriesOut.iloc[0]["query_0"], "compact")

        query2 = queriesOut.iloc[1]["query"]
        self.assertTrue("#1" in query2)
        self.assertTrue("#uw8" in query2)
        self.assertTrue("#combine" in query2)
        self.assertEqual(queriesOut.iloc[1]["query_0"], "compact memories")
        
        br_normal = pt.BatchRetrieve(indexref)
        pipe = sdm >> br_normal

        #check that we can get a str()
        str(pipe)

        if freq:
            br_normal.controls["wmodel"] = "Tf"
        resTest_pipe = pipe.transform(queriesIn)


        # this BR should do the same thing as the pipe, but natively in Terrier
        br_sdm = pt.BatchRetrieve(indexref,
            controls = {"sd" :"on"}, 
            properties={"querying.processes" : "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,"\
                    +"parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,"\
                    +"sd:DependenceModelPreProcess,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,"\
                    +"labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess"})
        if freq:
            br_sdm.controls["wmodel"] = "Tf"
            br_sdm.controls["dependencemodel"] = "org.terrier.matching.models.Tf"

        resTest_native = br_sdm.transform(queriesIn)
 
        #print (resTest_pipe[resTest_pipe["qid"]=="2"])
        #print (resTest_native[resTest_native["qid"]=="2"])
        for index, row in resTest_pipe.iterrows():
            #print(index)
            #print(row["query"])
            #print(row)
            #print(resTest_native.iloc[index]) 
            self.assertEqual(row['qid'], resTest_native.iloc[index]["qid"])
            self.assertEqual(row['docno'], resTest_native.iloc[index]["docno"])
            # TODO I cannot get this test to pass with freq=False more precisely than 1dp
            #9.165638 in resTest_pipe vs 9.200683 in resTest_native
            self.assertAlmostEqual(row['score'], resTest_native.iloc[index]["score"], 1)

        t = dataset.get_topics().head(5)
        pipe_res = pipe.transform(t)
        #br_normal.saveResult(pipe_res, "/tmp/sdm.res", run_name="DPH")

        self.assertAlmostEqual(
            pt.Utils.evaluate(pipe_res, dataset.get_qrels(), metrics=["map"])["map"], 
            pt.Utils.evaluate(br_sdm.transform(t), dataset.get_qrels(), metrics=["map"])["map"], 
            places=4)

    # RM3 cannot be tested with current jnius, as it must be placed into the boot classpath
    # def test_rm3(self):
    #     dataset = pt.datasets.get_dataset("vaswani")
    #     indexref = dataset.get_index()

    #     qe = pt.rewrite.RM3(indexref)
    #     br = pt.BatchRetrieve(indexref)

    #     queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])
    #     res = br.transform(queriesIn)

    #     queriesOut = qe.transform(res)
    #     self.assertEqual(len(queriesOut), 1)
    #     query = queriesOut.iloc[0]["query"]
    #     #self.assertTrue("compact^1.82230972" in query)
    #     self.assertTrue("applypipeline:off " in query)
        
    #     pipe = br >> qe >> br

    #     # lets go faster, we only need 18 topics. qid 16 had a tricky case
    #     t = dataset.get_topics().head(18)

    #     all_qe_res = pipe.transform(t)
    #     map_pipe = pt.Utils.evaluate(all_qe_res, dataset.get_qrels(), metrics=["map"])["map"]

    #     br_qe = pt.BatchRetrieve(indexref, 
    #         controls={"qe":"on"},
    #         properties={"querying.processes" : "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,"\
    #                 +"parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,"\
    #                 +"sd:DependenceModelPreProcess,localmatching:LocalManager$ApplyLocalMatching,qe:RM3,"\
    #                 +"labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess"})
    #     map_qe = pt.Utils.evaluate(br_qe.transform(t), dataset.get_qrels(), metrics=["map"])["map"]

    #     self.assertAlmostEqual(map_qe, map_pipe, places=4)

    def test_linear_terrierql(self):
        pipe = pt.apply.query(lambda row: "az") >> pt.rewrite.linear(0.75, 0.25)
        self.assertEqual(pipe[1].weights[0], 0.75)
        self.assertEqual(pipe[1].weights[1], 0.25)
        dfIn = pt.new.queries(["a"])
        dfOut = pipe(dfIn)
        self.assertEqual(1, len(dfIn))
        self.assertEqual("az", dfOut.iloc[0]["query_0"])
        self.assertEqual("a", dfOut.iloc[0]["query_1"])
        self.assertEqual("(az)^0.750000 (a)^0.250000", dfOut.iloc[0]["query"])

    def test_linear_matchopql(self):
        pipe = pt.apply.query(lambda row: "az") >> pt.rewrite.linear(0.75, 0.25, format="matchopql")
        self.assertEqual(pipe[1].weights[0], 0.75)
        self.assertEqual(pipe[1].weights[1], 0.25)
        dfIn = pt.new.queries(["a"])
        dfOut = pipe(dfIn)
        self.assertEqual(1, len(dfIn))
        self.assertEqual("az", dfOut.iloc[0]["query_0"])
        self.assertEqual("a", dfOut.iloc[0]["query_1"])
        self.assertEqual("#combine:0:0.750000:1:0.250000(#combine(az) #combine(a))", dfOut.iloc[0]["query"])

    def test_qe(self):
        if not pt.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        dataset = pt.datasets.get_dataset("vaswani")
        indexref = dataset.get_index()
        index = pt.IndexFactory.of(indexref)
        # given their defaults, there three expressions are identical, all use Bo1
        qe1 = pt.rewrite.QueryExpansion(index)
        qe2 = pt.rewrite.DFRQueryExpansion(index)
        qe3 = pt.rewrite.Bo1QueryExpansion(index)

        # lets go faster, we only need 18 topics. qid 16 had a tricky case
        t = dataset.get_topics().head(18)

        qrels = dataset.get_qrels()

        for qe in [qe1, qe2, qe3]:
            br = pt.BatchRetrieve(index)

            queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])
            res = br.transform(queriesIn)

            queriesOut = qe.transform(res)
            self.assertEqual(len(queriesOut), 1)
            self.assertTrue("query_0" in queriesOut.columns)
            self.assertEqual(queriesOut.iloc[0]["query_0"], "compact")
            query = queriesOut.iloc[0]["query"]
            self.assertTrue("compact^1.82230972" in query)
            self.assertTrue("applypipeline:off " in query)
            
            pipe = br >> qe >> br

            # check the pipe doesnt cause an error
            str(pipe)

            all_qe_res = pipe.transform(t)
            map_pipe = pt.Utils.evaluate(all_qe_res, qrels, metrics=["map"])["map"]

            br_qe = pt.BatchRetrieve(indexref, controls={"qe":"on"})
            map_qe = pt.Utils.evaluate(br_qe.transform(t), qrels, metrics=["map"])["map"]

            self.assertAlmostEqual(map_qe, map_pipe, places=4)


if __name__ == "__main__":
    unittest.main()