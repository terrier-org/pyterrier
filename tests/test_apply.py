import numpy as np
import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import tempfile
import shutil
import os

class TestApply(BaseTestCase):

    def test_drop_columns(self):
        from pyterrier.transformer import Transformer
        testDF = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla"])
        p = pt.apply.Bla(drop=True)
        self.assertTrue(isinstance(p, Transformer))
        rtr = p(testDF)
        self.assertTrue("Bla" not in rtr.columns)

        rtr = p(testDF.to_dict(orient='records'))
        self.assertTrue("Bla" not in rtr[0])

    def test_index_apply(self):
        def _counter(iter_dict):
            count = 0
            for d in iter_dict:
                count += 1
            return count
        indexer = pt.apply.indexer(_counter)
        rtr = indexer.index([ {'docno' : 'd1'}, {'docno' : 'd2'}])
        self.assertEqual(2, rtr)

    def test_make_columns(self):
        from pyterrier.transformer import Transformer
        testDF = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla"])
        p = pt.apply.BlaB(lambda row: row["Bla"] * 2)
        self.assertTrue(isinstance(p, Transformer))
        rtr = p(testDF)
        self.assertTrue("BlaB" in rtr.columns)
        self.assertEqual(rtr.iloc[0]["BlaB"], 2)
        emptyQs = pt.new.empty_Q()
        rtr = p(emptyQs)
        self.assertTrue("BlaB" in rtr.columns)

    def test_rename_columns(self):
        from pyterrier.transformer import Transformer
        testDF = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla"])
        p = pt.apply.rename({'Bla' : "Bla2"})
        self.assertTrue(isinstance(p, Transformer))
        for input in [testDF, testDF.to_dict(orient='records')]:
            rtr = p(input)
            rtr = pd.DataFrame(list(rtr)) if not isinstance(rtr, pd.DataFrame) else rtr  # recover DF for easier testing
            self.assertTrue("Bla2" in rtr.columns)
            self.assertFalse("Bla" in rtr.columns)

        # test inspection
        self.assertEqual([["Bla"]], list(pt.inspect.transformer_inputs(p)))
        self.assertEqual(["qid", "query", "Bla2"], pt.inspect.transformer_outputs(p, ["qid", "query", "Bla"]))
        
        testDF2 = pd.DataFrame([["q1", "the bear and the wolf", 1]], columns=["qid", "query", "Bla2"])
        for input in [testDF2, testDF2.to_dict(orient='records')]:
            with self.assertRaises(KeyError):
                rtr = p(input)
        p_ignore = pt.apply.rename({'Bla' : "Bla2"}, errors='ignore')
        rtr = p_ignore(testDF2)

    def test_query_apply(self):
        stops=set(["and", "the"])
        origquery="the bear and the wolf"
        p = pt.apply.query(
                lambda q : " ".join([t for t in q["query"].split(" ") if not t in stops ])
            )
        testDF = pd.DataFrame([["q1", origquery]], columns=["qid", "query"])
        rtr = p(testDF)
        self.assertEqual(rtr.iloc[0]["query"], "bear wolf")
        self.assertEqual(rtr.iloc[0]["query_0"], origquery)

        testDF2 = pd.DataFrame([["q1"]], columns=["qid"])
        rtrDR2 = pt.apply.query(lambda row : row["qid"], required_columns=['qid'])(testDF2)
        self.assertEqual(rtrDR2.iloc[0]["query"], "q1")

    def test_query_apply_error(self):
        origquery="the bear and the wolf"
        testDF = pd.DataFrame([["q1", origquery]], columns=["qid", "query"])
        p = pt.apply.query(lambda q : q) # should thrown an error, as pt.apply.query should return a string, not a row
        with self.assertRaises(TypeError) as te:
            p(testDF)
        self.assertTrue("Could not coerce return from pt.apply.query function into a list of strings" in str(te.exception))

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

        p2 = pt.apply.by_query(lambda x: x)
        with self.assertRaisesRegex(ValueError, 'score column not present'):
            p2(pt.new.queries(['query 1', 'query 2']))

        p3 = pt.apply.by_query(lambda x: 5/0, add_ranks=False)
        with self.assertRaisesRegex(Exception, 'for qid 1'):
            p3(pt.new.queries(['query 1', 'query 2']))

    def test_by_query_apply_iter(self):
        inputDf = pt.new.ranked_documents([[1], [2]], qid=["1", "2"])
        def _inc_score(res):
            for row in res:
                row["score"] = row["score"] + int(row["qid"])
                yield row

        def _inc_score_list(res):
            rtr = []
            for row in res:
                row["score"] = row["score"] + int(row["qid"])
                rtr.append(row)
            return rtr
    
        for name, t in [
            ("gen", pt.apply.by_query(_inc_score, iter=True)),
            ("gen batch", pt.apply.by_query(_inc_score, iter=True, batch_size=1)),
            ("list", pt.apply.by_query(_inc_score_list, iter=True)),
            ("list batch", pt.apply.by_query(_inc_score_list, iter=True, batch_size=1))]:
            with self.subTest(name):
            
                outputDf = t(inputDf)
                self.assertEqual(outputDf.iloc[0]["qid"], "1")
                self.assertEqual(outputDf.iloc[0]["score"], 2)
                self.assertEqual(outputDf.iloc[1]["qid"], "2")
                self.assertEqual(outputDf.iloc[1]["score"], 4)

                # check no error on empty input
                outputDfEmpty = t(inputDf.head(0))

    def test_by_query_apply_batch(self):
        # same as test_by_query_apply, but batch_size is set.
        inputDf = pt.new.ranked_documents([[1], [2]], qid=["1", "2"])
        def _inc_score(res):
            if len(res) == 0:
                return res
            res = res.copy()
            res["score"] = res["score"] + int(res.iloc[0]["qid"])
            return res
        p = pt.apply.by_query(_inc_score, batch_size=1)
        outputDf = p(inputDf)
        self.assertEqual(outputDf.iloc[0]["qid"], "1")
        self.assertEqual(outputDf.iloc[0]["score"], 2)
        self.assertEqual(outputDf.iloc[1]["qid"], "2")
        self.assertEqual(outputDf.iloc[1]["score"], 4)

        outputDfEmpty = p(inputDf.head(0))

    def test_generic(self):
        inputDf = pt.new.ranked_documents([[1], [2]], qid=["1", "2"])
        def _fn1(df):
            df = df.copy()
            df["score"] = df["score"] * 2
            return df
        for i, t in enumerate([
            pt.apply.generic(_fn1),
            pt.apply.generic(_fn1, batch_size=1)
        ]):
            outputDf = t(inputDf)
            self.assertEqual(2, len(outputDf))
            self.assertEqual(outputDf.iloc[0]["qid"], "1")
            self.assertEqual(outputDf.iloc[0]["score"], 2)
            self.assertEqual(outputDf.iloc[1]["qid"], "2")
            self.assertEqual(outputDf.iloc[1]["score"], 4)
        
        def _fn2(df):
            df = df.copy()
            df["score"] = len(df)
            return df
        t1 = pt.apply.generic(_fn2)
        t2 = pt.apply.generic(_fn2, batch_size=1)
        outputDf1 = t1(inputDf)
        outputDf2 = t2(inputDf)
        self.assertEqual(2, len(outputDf1))
        self.assertEqual(2, len(outputDf2))
        # batch is the entire dataframe, ie 2 rows
        self.assertEqual(2, outputDf1.iloc[0]["score"])
        # batch is a one row dataframe
        self.assertEqual(1, outputDf2.iloc[0]["score"])
    
    def test_generic_iter(self):
        def _fn1_list(iter):
            rtr = []
            for i in iter:
                i["score"] *= 2
                rtr.append(i)
            return rtr
        
        def _fn1_gen(iter):
           for i in iter:
                i["score"] *= 2
                yield i

        for name, t in [
            ("list, no batch", pt.apply.generic(_fn1_list, iter=True)),
            ("list, batch", pt.apply.generic(_fn1_list, batch_size=1, iter=True)),
            ("gen, no batch", pt.apply.generic(_fn1_gen, iter=True)),
            ("gen, batch", pt.apply.generic(_fn1_gen, batch_size=1, iter=True)),
        ]:
            inputDf = pt.new.ranked_documents([[1], [2]], qid=["1", "2"])
            with self.subTest(name):
                outputDf1 = t(inputDf)
                self.assertEqual(2, len(outputDf1))
                self.assertEqual(2, outputDf1.iloc[0]["score"])
                self.assertEqual(4, outputDf1.iloc[1]["score"])

                outputDf1 = pd.DataFrame(t(inputDf.to_dict(orient='records')))
                self.assertEqual(2, len(outputDf1))
                self.assertEqual(2, outputDf1.iloc[0]["score"])
                self.assertEqual(4, outputDf1.iloc[1]["score"])

    
    def test_docscore_apply(self):
        p = pt.apply.doc_score(lambda doc_row: len(doc_row["text"]))
        testDF = pd.DataFrame([["q1", "hello", "d1", "aa"]], columns=["qid", "query", "docno", "text"])
        for input in [testDF, testDF.to_dict(orient='records')]:
            rtr = p(testDF)
            rtr = pd.DataFrame(list(rtr)) if not isinstance(rtr, pd.DataFrame) else rtr  # recover DF for easier testing
            self.assertEqual(rtr.iloc[0]["score"], 2.0)
            self.assertEqual(rtr.iloc[0]["rank"], pt.model.FIRST_RANK)
        
        rtr2 = p(testDF.head(0))
        self.assertTrue("rank" in rtr2.columns)
        self.assertTrue("score" in rtr2.columns)
        self.assertEqual('float64', rtr2['score'].dtype)

        rtr2 = p([])

    def test_docscore_batch(self):
        p = pt.apply.doc_score(lambda df: df["text"].str.len(), batch_size=2)
        testDF = pd.DataFrame([["q1", "hello", "d1", "aa"]], columns=["qid", "query", "docno", "text"])
        rtr = p(testDF)
        self.assertEqual(rtr.iloc[0]["score"], 2.0)
        self.assertEqual(rtr.iloc[0]["rank"], pt.model.FIRST_RANK)

        rtr2 = p(testDF.head(0))
        self.assertTrue("rank" in rtr2.columns)
        self.assertTrue("score" in rtr2.columns)
        self.assertEqual('float64', rtr2['score'].dtype)

    def test_docfeatures_apply(self):
        import numpy as np
        p = pt.apply.doc_features(lambda doc_row: np.array([0,1]) )
        testDF = pd.DataFrame([["q1", "hello", "d1", "aa"]], columns=["qid", "query", "docno", "text"])
        for input in [testDF, testDF.to_dict(orient='records')]:
            rtr = p(input)
            rtr = pd.DataFrame(list(rtr)) if not isinstance(rtr, pd.DataFrame) else rtr # recover DF for easier testing
            self.assertTrue(np.array_equal(rtr.iloc[0]["features"], np.array([0,1])))



@pt.testing.transformer_test_class
def test_apply_doc_features():
    return pt.apply.doc_features(lambda row: np.array([0,1]))


@pt.testing.transformer_test_class
def test_apply_doc_score():
    return pt.apply.doc_score(lambda row: 5)
