
import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import tempfile
import shutil
import os

class TestCache(BaseTestCase):

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


