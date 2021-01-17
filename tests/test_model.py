from .base import BaseTestCase
import pandas as pd
from pyterrier.model import add_ranks, FIRST_RANK, coerce_queries_dataframe
import pyterrier as pt
class TestModel(BaseTestCase):

    def test_push_query(self):
        df = pt.new.queries(["q1", "q2"])
        self.assertEqual(2, len(df))

        df2 = pt.model.push_queries(df, keep_original=False)
        self.assertTrue("query_0" in df2.columns)
        self.assertFalse("query" in df2.columns)
        self.assertEqual("q1", df2.iloc[0]["query_0"])
        self.assertEqual("q2", df2.iloc[1]["query_0"])

        df2 = pt.model.push_queries(df, keep_original=True)
        for col in ["query", "query_0"]:
            self.assertTrue(col in df2.columns)
            self.assertEqual("q1", df2.iloc[0][col])
            self.assertEqual("q2", df2.iloc[1][col])
        
        df3 = pt.model.push_queries(df2, keep_original=True)
        for col in ["query", "query_0", "query_1"]:
            self.assertTrue(col in df3.columns)
            self.assertEqual("q1", df3.iloc[0][col])
            self.assertEqual("q2", df3.iloc[1][col])
    
    def test_pop_query(self):
        df = pt.new.queries(["q1", "q2"])
        self.assertEqual(2, len(df))

        df2 = pt.model.push_queries(df, keep_original=False)
        self.assertTrue("query_0" in df2.columns)
        self.assertFalse("query" in df2.columns)
        df2["query"] = ["a", "b"]
        self.assertTrue("query" in df2.columns)
        self.assertEqual("q1", df2.iloc[0]["query_0"])
        self.assertEqual("q2", df2.iloc[1]["query_0"])

        df3 = pt.model.pop_queries(df2)
        self.assertFalse("query_1" in df3.columns)
        self.assertTrue("query" in df3.columns)
        # check that we dont have duplicated query column
        self.assertEqual(2, len(df3.columns))
        self.assertEqual("q1", df3.iloc[0]["query"])
        self.assertEqual("q2", df3.iloc[1]["query"])


    def test_rank_zero_query(self):
        df = pd.DataFrame([], columns=["qid", "docno", "score"])
        df = add_ranks(df)
        self.assertTrue("rank" in df.columns)

    def test_rank_one_query(self):
        df = pd.DataFrame([["q1", "doc1", 5], ["q1", "doc2", 5]], columns=["qid", "docno", "score"])
        df = add_ranks(df)
        self.assertTrue("rank" in df.columns)
        # check that first item is rank 1
        self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
        # check that ties are resolved by keeping the same order.
        # trec_eval instead breaks ties on ascending docno
        self.assertEqual(df.iloc[1]["rank"], FIRST_RANK+1)

    def test_rank_one_query_sort(self):
        import pyterrier as pt
        sort_status = pt.model.STRICT_SORT
        pt.model.STRICT_SORT = True
        df = pd.DataFrame([["q1", "doc1", 4], ["q1", "doc2", 5]], columns=["qid", "docno", "score"])
        df = add_ranks(df)
        print(df)
        self.assertTrue("rank" in df.columns)
        # check that first item is rank 1
        self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
        self.assertEqual(df.iloc[0]["docno"], "doc2")
        pt.model.STRICT_SORT = sort_status

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