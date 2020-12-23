from .base import BaseTestCase
import pandas as pd
import pyterrier as pt

class TestModel(BaseTestCase):

    def test_new_1query_no_qid(self):
        df = pt.new.query("a")
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertEqual("1", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])
        
    def test_new_1query_w_qid(self):
        df = pt.new.query("a", "2")
        self.assertEqual(1, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertEqual("2", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])

    def test_new_2query_w_qid(self):
        df = pt.new.query(["a", "b"], ["2", "1"])
        self.assertEqual(2, len(df))
        self.assertTrue("qid" in df.columns)
        self.assertTrue("query" in df.columns)
        self.assertEqual("2", df.iloc[0]["qid"])
        self.assertEqual("a", df.iloc[0]["query"])
        self.assertEqual("1", df.iloc[1]["qid"])
        self.assertEqual("b", df.iloc[1]["query"])

    def test_new_2query_w_qid_hist(self):
        df = pt.new.query(["a", "b"], ["2", "1"], hist=["cc", "dd"])
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
        