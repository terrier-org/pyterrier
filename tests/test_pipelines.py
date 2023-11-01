import pandas as pd
import unittest
import pyterrier as pt
from .base import BaseTestCase
from matchpy import *

class TestOperators(BaseTestCase):

    def test_evaluate(self):
        input_qrels = pd.DataFrame([["1", "12", 1], ["1", "26", 1], ["1", "5", 1], ["1", "6", 1], ["2", "12", 1], ["2", "13", 1], ["2", "7", 1], ["2", "17", 1]], columns=["qid", "docno", "label"])
        input_res = pd.DataFrame([["1", "12", 3.917300970672472], ["1", "17", 3.912008156607317], ["1", "5", 3.895776784815295], ["1", "6", 1.6976053561565434], ["1", "11394", 1.419217511596875],
                                  ["2", "12", 3.352655284198764], ["2", "13", 3.3410694508732677], ["2", "7", 3.32843147860022], ["2", "15", 3.210614190096991], ["2", "17", 1.3688610792424558], ["2", "25", 1.2673250497019404]],
                                 columns=['qid', 'docno', 'score'])
        exp_result = [0.6042, 0.9500]
        result = pt.Evaluate(input_res, input_qrels, perquery=True)
        self.assertAlmostEqual(sum(exp_result) / len(exp_result), 0.7771, places=4)
        for i, item in enumerate(exp_result):
            self.assertAlmostEqual(result[str(i + 1)]["map"], item, places=4)

        result = pt.Evaluate(input_res, input_qrels, metrics=["iprec_at_recall"])
        self.assertTrue("IPrec@0.1" in result)
    
    def test_evaluate_ndcg_cut(self):
        input_qrels = pd.DataFrame([["1", "12", 1], ["1", "26", 1], ["1", "5", 1], ["1", "6", 1], ["2", "12", 1], ["2", "13", 1], ["2", "7", 1], ["2", "17", 1]], columns=["qid", "docno", "label"])
        input_res = pd.DataFrame([["1", "12", 3.917300970672472], ["1", "17", 3.912008156607317], ["1", "5", 3.895776784815295], ["1", "6", 1.6976053561565434], ["1", "11394", 1.419217511596875],
                                  ["2", "12", 3.352655284198764], ["2", "13", 3.3410694508732677], ["2", "7", 3.32843147860022], ["2", "15", 3.210614190096991], ["2", "17", 1.3688610792424558], ["2", "25", 1.2673250497019404]],
                                 columns=['qid', 'docno', 'score'])
        exp_result = [0.6042, 0.9500]
        result = pt.Evaluate(input_res, input_qrels, metrics=["ndcg_cut_5"])
        self.assertTrue("ndcg_cut_5" in result)
        self.assertFalse("ndcg_cut_10" in result)

        result = pt.Evaluate(input_res, input_qrels, perquery=True, metrics=["ndcg_cut_5"])
        self.assertTrue("ndcg_cut_5" in result["1"])
        self.assertFalse("ndcg_cut_10" in result["1"])

    def test_maxmin_normalisation(self):
        import pyterrier.pipelines as ptp

        df = pd.DataFrame([
            ["q1", "doc1", 10], ["q1", "doc2", 2], ["q2", "doc1", 1], ["q3", "doc1", 0], ["q3", "doc2", 0]], columns=["qid", "docno", "score"])
        mock_input = pt.Transformer.from_df(df, uniform=True)
        pipe = mock_input >> ptp.PerQueryMaxMinScoreTransformer()
        rtr = pipe.transform(None)
        self.assertTrue("qid" in rtr.columns)
        self.assertTrue("docno" in rtr.columns)
        self.assertTrue("score" in rtr.columns)
        thedict = rtr.set_index(['qid', 'docno']).to_dict()['score']
        print(thedict)
        self.assertEqual(1, thedict[("q1", "doc1")])
        self.assertEqual(0, thedict[("q1", "doc2")])
        self.assertEqual(0, thedict[("q2", "doc1")])
        self.assertEqual(0, thedict[("q3", "doc1")])
        self.assertEqual(0, thedict[("q3", "doc2")])
