import pandas as pd
import pyterrier as pt
import os
import unittest
import warnings
from .base import BaseTestCase

class TestExperiment(BaseTestCase):

    def test_differing_queries(self):
        topics = pd.DataFrame([["q1", "q1"], ["q2", "q1"] ], columns=["qid", "query"])
        res1 = pd.DataFrame([["q1", "d1", 1.0]], columns=["qid", "docno", "score"])
        res2 = pd.DataFrame([["q1", "d1", 1.0], ["q2", "d1", 2.0] ], columns=["qid", "docno", "score"])
        qrels = pd.DataFrame([["q1", "d1", 1], ["q2", "d1", 1] ], columns=["qid", "docno", "label"])
        from pyterrier.transformer import UniformTransformer
        with warnings.catch_warnings(record=True) as w:
            pt.pipelines.Experiment(
                [UniformTransformer(res1), UniformTransformer(res2)],
                topics,
                qrels,
                ["map"],
                baseline=0)
            self.assertTrue("missing" in str(w[-1].message))


    def test_differing_order(self):
        topics = pd.DataFrame([["q1", "q1"], ["q2", "q1"] ], columns=["qid", "query"])
        res1 = pd.DataFrame([ ["q2", "d1", 2.0], ["q1", "d1", 1.0],], columns=["qid", "docno", "score"])
        res2 = pd.DataFrame([["q1", "d1", 1.0], ["q2", "d1", 2.0] ], columns=["qid", "docno", "score"])
        qrels = pd.DataFrame([["q1", "d1", 1], ["q2", "d3", 1] ], columns=["qid", "docno", "label"])
        from pyterrier.transformer import UniformTransformer
        measures = pt.pipelines.Experiment(
                [UniformTransformer(res1), UniformTransformer(res2)],
                topics,
                qrels,
                ["map"],
                baseline=0)
        self.assertEqual(measures.iloc[0]["map"], 0.5)
        self.assertEqual(measures.iloc[1]["map"], 0.5)
        self.assertEqual(measures.iloc[1]["map +"], 0)
        self.assertEqual(measures.iloc[1]["map -"], 0)

    def test_one_row(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.pipelines.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q"])
        print(rtr)
        self.assertEqual(10, rtr.iloc[0]["num_q"])
        
        rtr = pt.pipelines.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], dataframe=False)
        print(rtr)

        with warnings.catch_warnings(record=True) as w:
            rtr = pt.pipelines.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels(), dataframe=False)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "Signature" in str(w[-1].message)


    def test_perquery(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.pipelines.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], perquery=True)
        print(rtr)

        rtr = pt.pipelines.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], perquery=True, dataframe=False)
        print(rtr)

    def test_baseline(self):
        dataset = pt.get_dataset("vaswani")
        df = pt.Experiment(
            [pt.BatchRetrieve(dataset.get_index(), wmodel="BM25"), pt.BatchRetrieve(dataset.get_index(), wmodel="DPH")], 
            dataset.get_topics().head(10), 
            dataset.get_qrels(),
            eval_metrics=["map", "ndcg"], 
            baseline=0)
        self.assertTrue("map +" in df.columns)
        self.assertTrue("map -" in df.columns)
        self.assertTrue("map p-value" in df.columns)