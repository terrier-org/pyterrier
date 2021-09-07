
from pyterrier.transformer import TransformerBase
import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import tempfile
import shutil
import os

class TestCache(BaseTestCase):

    def test_drop_columns(self):
        from pyterrier.transformer import TransformerBase
        testDF = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla"])
        p = pt.apply.Bla(drop=True)
        self.assertTrue(isinstance(p, TransformerBase))
        rtr = p(testDF)
        self.assertTrue("Bla" not in rtr.columns)

    def test_make_columns(self):
        from pyterrier.transformer import TransformerBase
        testDF = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla"])
        p = pt.apply.BlaB(lambda row: row["Bla"] * 2)
        self.assertTrue(isinstance(p, TransformerBase))
        rtr = p(testDF)
        self.assertTrue("BlaB" in rtr.columns)
        self.assertEqual(rtr.iloc[0]["BlaB"], 2)
        emptyQs = pt.new.empty_Q()
        rtr = p(emptyQs)
        self.assertTrue("BlaB" in rtr.columns)

    def test_rename_columns(self):
        from pyterrier.transformer import TransformerBase
        testDF = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla"])
        p = pt.apply.rename({'Bla' : "Bla2"})
        self.assertTrue(isinstance(p, TransformerBase))
        rtr = p(testDF)
        self.assertTrue("Bla2" in rtr.columns)
        self.assertFalse("Bla" in rtr.columns)

    def test_query_apply(self):
        stops=set(["and", "the"])
        origquery="the bear and the wolf"
        p = pt.apply.query(
                lambda q : " ".join([t for t in q["query"].split(" ") if not t in stops ])
            )
        testDF = pd.DataFrame([["q1", origquery]], columns=["qid", "query"])
        rtr = p(testDF)
        print(rtr)
        self.assertEqual(rtr.iloc[0]["query"], "bear wolf")
        self.assertEqual(rtr.iloc[0]["query_0"], origquery)

    def test_by_query_apply(self):
        inputDf = pt.new.ranked_documents([[1], [2]], qid=["1", "2"])
        def _inc_score(res):
            if len(res) == 0:
                return res
            res = res.copy()
            res["score"] = res["score"] + int(res.iloc[0]["qid"])
            return res
        p = pt.apply.by_query(_inc_score)
        outputDf = p(inputDf)
        self.assertEqual(outputDf.iloc[0]["qid"], "1")
        self.assertEqual(outputDf.iloc[0]["score"], 2)
        self.assertEqual(outputDf.iloc[1]["qid"], "2")
        self.assertEqual(outputDf.iloc[1]["score"], 4)

        outputDfEmpty = p(inputDf.head(0))

    def test_docscore_apply(self):
        p = pt.apply.doc_score(lambda doc_row: len(doc_row["text"]))
        testDF = pd.DataFrame([["q1", "hello", "d1", "aa"]], columns=["qid", "query", "docno", "text"])
        rtr = p(testDF)
        self.assertEqual(rtr.iloc[0]["score"], 2.0)
        self.assertEqual(rtr.iloc[0]["rank"], pt.model.FIRST_RANK)

    def test_docfeatures_apply(self):
        import numpy as np
        p = pt.apply.doc_features(lambda doc_row: np.array([0,1]) )
        testDF = pd.DataFrame([["q1", "hello", "d1", "aa"]], columns=["qid", "query", "docno", "text"])
        rtr = p(testDF)
        self.assertTrue(np.array_equal(rtr.iloc[0]["features"], np.array([0,1])))


