from .base import BaseTestCase
import pandas as pd
from pyterrier.model import add_ranks, FIRST_RANK, coerce_queries_dataframe

class TestModel(BaseTestCase):

    def test_rank_one_query(self):
        df = pd.DataFrame([["q1", "doc1", 5], ["q1", "doc2", 5]], columns=["qid", "docno", "score"])
        df = add_ranks(df)
        self.assertTrue("rank" in df.columns)
        # check that first item is rank 1
        self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
        # check that ties are resolved by keeping the same order.
        # trec_eval instead breaks ties on ascending docno
        self.assertEqual(df.iloc[1]["rank"], FIRST_RANK+1)

    def test_rank_two_queries(self):
        df = pd.DataFrame([["q1", "doc1", 5], ["q1", "doc2", 4], ["q2", "doc1", 4]], columns=["qid", "docno", "score"])
        df = add_ranks(df)
        self.assertTrue("rank" in df.columns)
        self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
        self.assertEqual(df.iloc[1]["rank"], FIRST_RANK+1)
        self.assertEqual(df.iloc[2]["rank"], FIRST_RANK)

    def test_coerce_dataframe_with_string(self):
        input = "light"
        exp_result = pd.DataFrame([["1", "light"]], columns=['qid', 'query'])
        result = coerce_queries_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_coerce_dataframe_with_list(self):
        input = ["light", "mathematical", "electronic"]
        exp_result = pd.DataFrame([["1", "light"], ["2", "mathematical"], ["3", "electronic"]], columns=['qid', 'query'])
        result = coerce_queries_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_coerce_dataframe_throws_assertion_error(self):
        input = ("light", "mathematical", 25)
        self.assertRaises(AssertionError, coerce_queries_dataframe, input)

        input = {"25" : "mathematical"}
        self.assertRaises(ValueError, coerce_queries_dataframe, input)

        input = None
        self.assertRaises(ValueError, coerce_queries_dataframe, input)

    def test_coerce_dataframe_with_tuple(self):
        input = ("light", "mathematical", "electronic")
        exp_result = pd.DataFrame([["1", "light"], ["2", "mathematical"], ["3", "electronic"]], columns=['qid', 'query'])
        result = coerce_queries_dataframe(input)
        self.assertTrue(exp_result.equals(result))