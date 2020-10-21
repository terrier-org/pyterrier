from .base import BaseTestCase
import pandas as pd
from pyterrier.model import add_ranks, FIRST_RANK

class TestBatchRetrieve(BaseTestCase):

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