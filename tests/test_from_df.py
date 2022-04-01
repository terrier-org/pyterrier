import pandas as pd
import pyterrier as pt
from .base import *

class TestFromDf(BaseTestCase):

    def test_1(self):
        queries = pt.new.queries(["this is a query"])
        docs = pt.new.ranked_documents([[1, 2]], qid=["1"])
        rtr = pt.transformer.SourceTransformer(docs).transform(queries)
        self.assertTrue("query" in rtr.columns)
        self.assertTrue("score" in rtr.columns)
        self.assertTrue("docno" in rtr.columns)
        self.assertEqual(2, len(rtr))

    def test_query_cols(self):
        queries = pt.new.queries(["this is a query"], query_embs=[[1,1]])
        self.assertTrue("query_embs" in queries.columns)
        docs = pt.new.ranked_documents([[1, 2]], qid=["1"])
        rtr = pt.transformer.SourceTransformer(docs).transform(queries)
        self.assertTrue("query" in rtr.columns)
        self.assertTrue("score" in rtr.columns)
        self.assertTrue("docno" in rtr.columns)
        self.assertTrue("query_embs" in rtr.columns)
        self.assertEqual(2, len(rtr))
