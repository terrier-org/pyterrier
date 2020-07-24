import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

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

    def test_form_dataframe_with_string(self):
        input = "light"
        exp_result = pd.DataFrame([["1", "light"]], columns=['qid', 'query'])
        result = pt.Utils.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_form_dataframe_with_list(self):
        input = ["light", "mathematical", "electronic"]
        exp_result = pd.DataFrame([["1", "light"], ["2", "mathematical"], ["3", "electronic"]], columns=['qid', 'query'])
        result = pt.Utils.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_form_dataframe_throws_assertion_error(self):
        input = ("light", "mathematical", 25)
        self.assertRaises(AssertionError, pt.Utils.form_dataframe, input)

    def test_form_dataframe_with_tuple(self):
        input = ("light", "mathematical", "electronic")
        exp_result = pd.DataFrame([["1", "light"], ["2", "mathematical"], ["3", "electronic"]], columns=['qid', 'query'])
        result = pt.Utils.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_candidate_set_one_doc(self):
        if not pt.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")
        indexloc = self.here + "/fixtures/index/data.properties"
        # docid 50 == docno 51
        input_set = pd.DataFrame([["q1", "light", 50]], columns=["qid", "query", "docid"])
        retr = pt.BatchRetrieve(indexloc)
        
        # this test the implementation of __call__() redirecting to transform()
        for result in [retr.transform(input_set), retr(input_set)]:
            result = retr.transform(input_set)
            self.assertTrue("qid" in result.columns)
            self.assertTrue("docno" in result.columns)
            self.assertTrue("score" in result.columns)
            self.assertEqual(1, len(result))
            row = result.iloc[0]
            self.assertEqual("q1", row["qid"])
            self.assertEqual("51", row["docno"])
            self.assertTrue(row["score"] > 0)

    def test_candidate_set_two_doc(self):
        if not pt.check_version("5.3"):
            self.skipTest("Requires Terrier 5.3")

        indexloc = self.here + "/fixtures/index/data.properties"
        # docid 50 == docno 51
        # docid 66 == docno 67

        input_set = pd.DataFrame([
                    ["q1", "light", 50],
                    ["q1", None, 66]
                ],
            columns=["qid", "query", "docid"])
        retr = pt.BatchRetrieve(indexloc)
        result = retr.transform(input_set)
        self.assertTrue("qid" in result.columns)
        self.assertTrue("docno" in result.columns)
        self.assertTrue("score" in result.columns)
        self.assertEqual(2, len(result))

    # this also tests different index-like inputs, namely:
    # a String index location, an IndexRef, and an Index
    def test_one_term_query_correct_docid_and_score(self):

        indexloc = self.here + "/fixtures/index/data.properties"
        jindexref = pt.IndexRef.of(indexloc)
        jindex = pt.IndexFactory.of(jindexref)
        for indexSrc in (indexloc, jindexref, jindex):
            retr = pt.BatchRetrieve(indexSrc)
            result = retr.transform("light")
            exp_result = parse_query_result(os.path.dirname(
                os.path.realpath(__file__)) + "/fixtures/light_results")
            for index, row in result.iterrows():
                self.assertEqual(row['docno'], exp_result[index][0])
                self.assertAlmostEqual(row['score'], exp_result[index][1])
        jindex.close()

    def test_two_term_query_correct_qid_docid_score(self):
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.BatchRetrieve(indexref)
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
        JIR = pt.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here+"/fixtures/index/data.properties")
        retr = pt.BatchRetrieve(indexref, num_results=10)
        input=pd.DataFrame([["1", "Stability"]],columns=['qid','query'])
        result = retr.transform(input)
        self.assertEqual(len(result), 10)

if __name__ == "__main__":
    unittest.main()
