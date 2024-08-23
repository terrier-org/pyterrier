import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import warnings

def parse_res_file(filename):
    results = []
    with open(filename, 'r') as file:
        for line in file:
            split_line = line.strip("\n").split(" ")
            results.append([split_line[0], split_line[2], float(split_line[4])])
    return results

def parse_query_result(filename):
    results = []
    with open(filename, 'rt') as file:
        for line in file:
            split_line = line.strip("\n").split(" ")
            results.append([split_line[1], float(split_line[2])])
    return results

class TestBatchRetrieve(BaseTestCase):

    def test_candidate_set_one_doc(self):
        if not pt.terrier.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        indexloc = self.here + "/fixtures/index/data.properties"
        # docid 50 == docno 51
        input_set = pd.DataFrame([["q1", "light", 50]], columns=["qid", "query", "docid"])
        retr = pt.terrier.Retriever(indexloc)
        
        # this test the implementation of __call__() redirecting to transform()
        for result in [retr.transform(input_set), retr(input_set)]:
            result = retr.transform(input_set)
            self.assertTrue("qid" in result.columns)
            self.assertTrue("docno" in result.columns)
            self.assertTrue("score" in result.columns)
            self.assertTrue("rank" in result.columns)
            self.assertEqual(1, len(result))
            row = result.iloc[0]
            self.assertEqual("q1", row["qid"])
            self.assertEqual("51", row["docno"])
            self.assertEqual(pt.model.FIRST_RANK, row["rank"])
            self.assertTrue(row["score"] > 0)

    def test_br_cutoff(self):
        indexloc = self.here + "/fixtures/index/data.properties"
        
        input_set = pd.DataFrame([
                    ["q1", "chemical"],
                ],
            columns=["qid", "query"])
        retr = pt.terrier.Retriever(indexloc) % 10
        result = retr.transform(input_set)
        self.assertEqual(10, len(result))

    def test_br_query_toks(self):
        indexloc = self.here + "/fixtures/index/data.properties"
        
        retr = pt.terrier.Retriever(indexloc)
        query_terrier = 'applytermpipeline:off chemic^2 reaction^0.5'
        result_terrier = retr.search(query_terrier)

        query_matchop = '#combine:0=2:1=0.5(chemic reaction)'
        result_matchop = retr.search(query_matchop)

        query_toks = { 'chemic' : 2, 'reaction' : 0.5}
        result_toks = retr.transform(pd.DataFrame([['1', query_toks]], columns=['qid', 'query_toks']))
        
        self.assertEqual(len(result_terrier), len(result_matchop))
        self.assertEqual(len(result_terrier), len(result_toks))
        from pandas.testing import assert_frame_equal
        assert_frame_equal(result_terrier[["qid", "docno", "score", "rank"]], result_matchop[["qid", "docno", "score", "rank"]])
        assert_frame_equal(result_terrier[["qid", "docno", "score", "rank"]], result_toks[["qid", "docno", "score", "rank"]])

    def test_br_cutoff_stability(self):
        indexloc = self.here + "/fixtures/index/data.properties"
        input_set = pd.DataFrame([
                    ["q1", "chemical"],
                ],
            columns=["qid", "query"])
        br_cut_3 = pt.terrier.Retriever(indexloc, wmodel='Tf') % 3
        br_3 = pt.terrier.Retriever(indexloc, wmodel='Tf', num_results=3)
        #br = pt.terrier.Retriever(indexloc, wmodel='Tf')
        #print(br.transform(input_set))

        result_cut = br_cut_3.transform(input_set)
        result_tr = br_3.transform(input_set)
        print("Rank cutoff operator")
        print(result_cut.docno)
        print("terrier cutoff")
        print(result_tr.docno)        
        pd.testing.assert_series_equal(result_cut.docno, result_tr.docno)
    
    def test_br_col_passthrough(self):
        indexloc = self.here + "/fixtures/index/data.properties"
        
        input_set = pd.DataFrame([
                    ["q1", "chemical", "u1"],
                ],
            columns=["qid", "query", "username"])
        retr = pt.terrier.Retriever(indexloc) % 10
        result = retr.transform(input_set)
        self.assertIn("username", result.columns)

    def test_br_mem(self):
        indexloc = self.here + "/fixtures/index/data.properties"
        memindex = pt.IndexFactory.of(indexloc, memory=True)
        pindex = pt.java.cast("org.terrier.structures.IndexOnDisk", memindex)
        self.assertEqual("fileinmem", pindex.getIndexProperty("index.lexicon.data-source", "notfound"))
        retr = pt.terrier.Retriever(memindex)
        retr.search("chemical reactions")

    def test_br_empty(self):
        indexloc = self.here + "/fixtures/index/data.properties"
        
        input_set = pd.DataFrame([
                    ["q1", ""],
                ],
            columns=["qid", "query"])
        retr = pt.terrier.Retriever(indexloc)
        with warnings.catch_warnings(record=True) as w:
            result = retr.transform(input_set)
            assert "Skipping empty query" in str(w[-1].message)
        self.assertEqual(0, len(result))

    def test_candidate_set_two_doc(self):
        if not pt.terrier.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")

        indexloc = self.here + "/fixtures/index/data.properties"
        # docid 50 == docno 51
        # docid 66 == docno 67

        input_set = pd.DataFrame([
                    ["q1", "light", 50],
                    ["q1", None, 66]
                ],
            columns=["qid", "query", "docid"])
        retr = pt.terrier.Retriever(indexloc)
        result = retr.transform(input_set)
        self.assertTrue("qid" in result.columns)
        self.assertTrue("docno" in result.columns)
        self.assertTrue("score" in result.columns)
        self.assertEqual(2, len(result))

    # this also tests different index-like inputs, namely:
    # a String index location, an IndexRef, and an Index
    def test_one_term_query_correct_docid_score_rank(self):

        indexloc = self.here + "/fixtures/index/data.properties"
        jindexref = pt.IndexRef.of(indexloc)
        jindex = pt.IndexFactory.of(jindexref)
        for indexSrc in (indexloc, jindexref, jindex):
            retr = pt.terrier.Retriever(indexSrc)
            result = retr.search("light")
            exp_result = parse_query_result(os.path.dirname(
                os.path.realpath(__file__)) + "/fixtures/light_results")
            i=0
            for index, row in result.iterrows():
                self.assertEqual(row['docno'], exp_result[index][0])
                self.assertAlmostEqual(row['score'], exp_result[index][1])
                self.assertEqual(pt.model.FIRST_RANK + i, row["rank"])
                i+=1

        jindex.close()

    def test_two_term_query_correct_qid_docid_score(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.terrier.Retriever(indexref)
        input = pd.DataFrame([["1", "Stability"], ["2", "Generator"]], columns=['qid', 'query'])
        result = retr.transform(input)
        exp_result = parse_res_file(os.path.dirname(
            os.path.realpath(__file__)) + "/fixtures/two_queries_result")
        for index, row in result.iterrows():
            self.assertEqual(row['qid'], exp_result[index][0])
            self.assertEqual(row['docno'], exp_result[index][1])
            self.assertAlmostEqual(row['score'], exp_result[index][2])

        input = pd.DataFrame([[1, "Stability"], [2, "Generator"]], columns=['qid', 'query'])
        result = retr.transform(input)
        exp_result = parse_res_file(os.path.dirname(
            os.path.realpath(__file__)) + "/fixtures/two_queries_result")
        for index, row in result.iterrows():
            self.assertEqual(str(row['qid']), exp_result[index][0])
            self.assertEqual(row['docno'], exp_result[index][1])
            self.assertAlmostEqual(row['score'], exp_result[index][2])

    def test_num_results(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here+"/fixtures/index/data.properties")
        retr = pt.terrier.Retriever(indexref, num_results=10)
        input=pd.DataFrame([["1", "Stability"]],columns=['qid','query'])
        result = retr.transform(input)
        self.assertEqual(len(result), 10)

        if not pt.terrier.check_version("5.5"):
            return

        retr = pt.terrier.Retriever(indexref, num_results=1001)        
        result = retr.search("results")
        self.assertEqual(len(result), 1001)

    def test_num_manual_wmodel(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        Tf = pt.java.autoclass("org.terrier.matching.models.Tf")()
        indexref = JIR.of(self.here+"/fixtures/index/data.properties")
        from jnius import JavaException
        try:
            retr = pt.terrier.Retriever(indexref, wmodel=Tf)
            input=pd.DataFrame([["1", "Stability"]],columns=['qid','query'])
            result = retr.transform(input)
        except JavaException as ja:
            print(ja.stacktrace)
            raise ja

    def test_num_python_wmodel(self):
        indexref = self.here+"/fixtures/index/data.properties"
        Tf = lambda keyFreq, posting, entryStats, collStats: posting.getFrequency()
        retr = pt.terrier.Retriever(indexref, wmodel=Tf)
        input=pd.DataFrame([["1", "Stability"]],columns=['qid','query'])
        result = retr.transform(input)

    def test_threading_manualref(self):
        
        if not pt.terrier.check_version("5.5"):
            self.skipTest("Requires Terrier 5.5")

        topics = pt.get_dataset("vaswani").get_topics().head(8)

        #this test ensures that we operate when the indexref is specified to be concurrent 
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of("concurrent:" + self.here+"/fixtures/index/data.properties")
        retr = pt.terrier.Retriever(indexref, threads=4)
        result = retr.transform(topics)

        #check that use of a callback model works under threading
        Tf = lambda keyFreq, posting, entryStats, collStats: posting.getFrequency()
        retr = pt.terrier.Retriever(indexref, threads=4, wmodel=Tf)
        result = retr.transform(topics)

    def test_threading_selfupgrade(self):
        if not pt.terrier.check_version("5.5"):
            self.skipTest("Requires Terrier 5.5")

        topics = pt.get_dataset("vaswani").get_topics().head(10)

        #this test ensures we can upgrade the indexref to be concurrent
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here+"/fixtures/index/data.properties")
        retr = pt.terrier.Retriever(indexref, threads=5)
        result = retr.transform(topics)

    def test_terrier_retrieve_alias(self):
        # based off test_candidate_set_one_doc
        if not pt.terrier.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        indexloc = self.here + "/fixtures/index/data.properties"
        # docid 50 == docno 51
        input_set = pd.DataFrame([["q1", "light", 50]], columns=["qid", "query", "docid"])
        for name, retr in [ ("TerrierRetrieve", pt.TerrierRetrieve(indexloc)), ("BatchRetrieve", pt.BatchRetrieve(indexloc)) ]:
            self.subTest(name)
            # this test the implementation of __call__() redirecting to transform()
            for result in [retr.transform(input_set), retr(input_set)]:
                result = retr.transform(input_set)
                self.assertTrue("qid" in result.columns)
                self.assertTrue("docno" in result.columns)
                self.assertTrue("score" in result.columns)
                self.assertTrue("rank" in result.columns)
                self.assertEqual(1, len(result))
                row = result.iloc[0]
                self.assertEqual("q1", row["qid"])
                self.assertEqual("51", row["docno"])
                self.assertEqual(pt.model.FIRST_RANK, row["rank"])
                self.assertTrue(row["score"] > 0)


if __name__ == "__main__":
    unittest.main()
