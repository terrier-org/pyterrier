from .base import BaseTestCase
import pandas as pd
import pyterrier as pt

class TestModel(BaseTestCase):

    def test_empty(self):
        df = pt.new.empty_Q()
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)

    def test_newR1(self):
        df = pt.new.ranked_documents([[1]])
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertEqual("1", df.iloc[0]["qid"])
        self.assertEqual("d1", df.iloc[0]["docno"])
        self.assertEqual(1, df.iloc[0]["score"])
        self.assertEqual(pt.model.FIRST_RANK, df.iloc[0]["rank"])

        df = pt.new.ranked_documents([[1]], docno=[["d100"]])
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertEqual("1", df.iloc[0]["qid"])
        self.assertEqual("d100", df.iloc[0]["docno"])
        self.assertEqual(1, df.iloc[0]["score"])
        self.assertEqual(pt.model.FIRST_RANK, df.iloc[0]["rank"])

        df = pt.new.ranked_documents([[2,1], [2,1], [2,1]])
        self.assertEqual(6, len(df))
        qids=["a", "b", "c"]
        df = pt.new.ranked_documents([[2,1], [2,1], [2,1]], qid=qids)
        self.assertEqual(6, len(df))
        offset=0
        for qid in qids:
            for i in [0,1]:
                self.assertEqual(df.iloc[offset]["qid"], qid)
                offset +=1
 
    def test_newR1_with_qid(self):
        df = pt.new.ranked_documents([[1]], qid=["q1"])
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertEqual("q1", df.iloc[0]["qid"])
        self.assertEqual("d1", df.iloc[0]["docno"])
        self.assertEqual(1, df.iloc[0]["score"])
        self.assertEqual(pt.model.FIRST_RANK, df.iloc[0]["rank"])

        df = pt.new.ranked_documents([[1]], docno=[["d100"]], qid=["q1"])
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertEqual("q1", df.iloc[0]["qid"])
        self.assertEqual("d100", df.iloc[0]["docno"])
        self.assertEqual(1, df.iloc[0]["score"])
        self.assertEqual(pt.model.FIRST_RANK, df.iloc[0]["rank"])

    def test_newR1_with_docid(self):
        df = pt.new.ranked_documents([[1]], qid=["q1"], docid=[[0]])
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertEqual("q1", df.iloc[0]["qid"])
        self.assertEqual("d1", df.iloc[0]["docno"])
        self.assertEqual(0, df.iloc[0]["docid"])
        self.assertEqual(1, df.iloc[0]["score"])
        self.assertEqual(pt.model.FIRST_RANK, df.iloc[0]["rank"])

    def test_new_1query_no_qid(self):
        df = pt.new.queries("a")
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertEqual("1", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])
        
    def test_new_1query_w_qid(self):
        df = pt.new.queries("a", "2")
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertEqual("2", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])

    def test_new_2query_w_qid(self):
        df = pt.new.queries(["a", "b"], ["2", "1"])
        self.assertEqual(2, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertEqual("2", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])
        self.assertEqual("1", df.iloc[1]["qid"])
        self.assertEqual("b", df.iloc[1]["query"])

    def test_new_2query_w_qid_hist(self):
        df = pt.new.queries(["a", "b"], ["2", "1"], hist=["cc", "dd"])
        self.assertEqual(2, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertTrue("hist" in df.columns)
        self.assertEqual("2", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])
        self.assertEqual("1", df.iloc[1]["qid"])
        self.assertEqual("b", df.iloc[1]["query"])
        self.assertEqual("cc", df.iloc[0]["hist"])
        self.assertEqual("dd", df.iloc[1]["hist"])
        