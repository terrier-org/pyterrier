from .base import BaseTestCase
import pandas as pd
from pyterrier.model import add_ranks, FIRST_RANK, coerce_queries_dataframe, coerce_dataframe_types, split_df
import pyterrier as pt
class TestModel(BaseTestCase):

    def test_R_to_Q(self):
        df1 = pt.new.ranked_documents([[1, 2], [2,0]], query=[["a a", "a a"], ["b b", "b b"]])
        df2 = pt.model.ranked_documents_to_queries(df1)
        self.assertEqual(set(['qid', 'query']), set(df2.columns))
        df1 = pt.new.ranked_documents([[1, 2], [2,0]], query=[["a a", "a a"], ["b b", "b b"]], stashed_results_0=[[1,1],[1,1,]])
        df2 = pt.model.ranked_documents_to_queries(df1)
        self.assertEqual(set(['qid', 'query', 'stashed_results_0']), set(df2.columns))        

    def test_doc_cols(self):
        df = pt.new.queries(["q1", "q2"])
        self.assertListEqual(['qid'], pt.model.document_columns(df))
        df2 = pt.new.ranked_documents([[1]], query=[["hello"]])
        self.assertEqual(set(['qid', 'docno', 'rank', 'score']), set(pt.model.document_columns(df2)))
        df2 = pt.new.ranked_documents([[1]], text=[["a"]], query=[["hello"]])
        self.assertEqual(set(['qid', 'docno', 'rank', 'score', "text"]), set(pt.model.document_columns(df2)))

    def test_query_cols(self):
        df = pt.new.queries(["q1", "q2"])
        self.assertEqual(2, len(df))
        df2 = pt.model.push_queries(df, keep_original=True)
        query_cols = pt.model.query_columns(df2)
        for col in ["qid", "query", "query_0"]:
            self.assertTrue(col in query_cols)

        df3 = pt.new.ranked_documents([[1]], query=[["hello"]])
        query_cols = pt.model.query_columns(df3)
        for col in ["qid", "query"]:
            self.assertTrue(col in query_cols)
        for col in ["docno", "score", "rank"]:
            self.assertFalse(col in query_cols)

    def test_push_query(self):
        df = pt.new.queries(["q1", "q2"])
        self.assertEqual(2, len(df))

        df2 = pt.model.push_queries(df, keep_original=False)
        self.assertTrue("query_0" in df2.columns)
        self.assertFalse("query" in df2.columns)
        self.assertEqual("q1", df2.iloc[0]["query_0"])
        self.assertEqual("q2", df2.iloc[1]["query_0"])

        df_empty = df.head(0)
        df2_empty = pt.model.push_queries(df_empty, keep_original=False)
        self.assertTrue("query_0" in df2_empty.columns)
        self.assertFalse("query" in df2_empty.columns)

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
        for sq in [True, False]:
            df = pd.DataFrame([["q1", "doc1", 5], ["q1", "doc2", 5]], columns=["qid", "docno", "score"])
            df = add_ranks(df, single_query=True)
            self.assertTrue("rank" in df.columns)
            # check that first item is rank 1
            self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
            # check that ties are resolved by keeping the same order.
            # trec_eval instead breaks ties on ascending docno
            self.assertEqual(df.iloc[1]["rank"], FIRST_RANK+1)

    def test_rank_one_query_neg(self):
        for sq in [True, False]:
            df = pd.DataFrame([["q1", "doc1", -4], ["q1", "doc2", -5]], columns=["qid", "docno", "score"])
            df = add_ranks(df, single_query=sq)
            df = df.sort_values("rank", ascending=True)
            self.assertTrue("rank" in df.columns)
            # check that first item is rank 1
            self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
            self.assertEqual(df.iloc[0]["docno"], "doc1")

            df = pd.DataFrame([["q1", "doc2", -5], ["q1", "doc1", -4]], columns=["qid", "docno", "score"])
            df = add_ranks(df, single_query=sq)
            df = df.sort_values("rank", ascending=True)
            self.assertTrue("rank" in df.columns)
            # check that first item is rank 1
            self.assertEqual(df.iloc[0]["rank"], FIRST_RANK)
            self.assertEqual(df.iloc[0]["docno"], "doc1")
        
    def test_rank_one_query_sort(self):
        import pyterrier as pt
        sort_status = pt.model.STRICT_SORT
        pt.model.STRICT_SORT = True
        for sq in [True, False]:
            df = pd.DataFrame([["q1", "doc1", 4], ["q1", "doc2", 5]], columns=["qid", "docno", "score"])
            df = add_ranks(df, single_query=sq)
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

    def test_coerce_dataframe_types(self):
        with self.subTest('typical'):
            input = pd.DataFrame([[1, 'query', 5, '1.3']], columns=['qid', 'query', 'docno', 'score'])
            exp_result = pd.DataFrame([['1', 'query', '5', 1.3]], columns=['qid', 'query', 'docno', 'score'])
            self.assertFalse(input.equals(exp_result))
            result = coerce_dataframe_types(input)
            pd.testing.assert_frame_equal(result, exp_result)
        with self.subTest('missing column'):
            input = pd.DataFrame([['query', 5, '1.3']], columns=['query', 'docno', 'score'])
            exp_result = pd.DataFrame([['query', '5', 1.3]], columns=['query', 'docno', 'score'])
            self.assertFalse(input.equals(exp_result))
            result = coerce_dataframe_types(input)
            pd.testing.assert_frame_equal(result, exp_result)
        with self.subTest('score not parsable as float'):
            input = pd.DataFrame([[1, 'query', 5, 'A']], columns=['qid', 'query', 'docno', 'score'])
            self.assertFalse(input.equals(exp_result))
            with self.assertRaises(ValueError):
                result = coerce_dataframe_types(input)

    def test_coerce_dataframe_types_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("No torch installed")

        with self.subTest('score as integer'):
            input = pd.DataFrame([[1, 'query', 5, 1]], columns=['qid', 'query', 'docno', 'score'])
            exp_result = pd.DataFrame([['1', 'query', '5', 1.]], columns=['qid', 'query', 'docno', 'score'])
            self.assertFalse(input.equals(exp_result))
            result = coerce_dataframe_types(input)
            pd.testing.assert_frame_equal(result, exp_result)
        with self.subTest('score as torch type'):
            import torch
            input = pd.DataFrame([[1, 'query', 5, torch.tensor(1.3)]], columns=['qid', 'query', 'docno', 'score'])
            exp_result = pd.DataFrame([['1', 'query', '5', 1.3]], columns=['qid', 'query', 'docno', 'score'])
            self.assertFalse(input.equals(exp_result))
            result = coerce_dataframe_types(input)
            pd.testing.assert_frame_equal(result, exp_result)
        
    def test_split_Q(self):
        df = pt.new.queries(["a", "b", "c"])
        dfs = split_df(df, 2)
        self.assertEqual(2, len(dfs))
        self.assertEqual(2, len(dfs[0]))
        self.assertEqual(1, len(dfs[1]))

    def test_split_R(self):
        df = pt.new.ranked_documents([[2,1], [2]])
        dfs = split_df(df, 2)
        self.assertEqual(2, len(dfs))
        self.assertEqual(2, len(dfs[0]))
        self.assertEqual(1, len(dfs[1]))

        df = pt.new.ranked_documents([[2,1], [2,1], [2,1]])

        dfs = split_df(df, 2)
        self.assertEqual(2, len(dfs))
        self.assertEqual(4, len(dfs[0]))
        self.assertEqual(2, len(dfs[1]))
        
