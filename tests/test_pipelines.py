import pandas as pd
import unittest
import pyterrier as pt
from .base import BaseTestCase
from matchpy import *

class TestOperators(BaseTestCase):

    def test_maxmin_normalisation(self):
        import pyterrier.transformer as ptt;
        import pyterrier.pipelines as ptp;

        df = pd.DataFrame([
            ["q1", "doc1", 10], ["q1", "doc2", 2], ["q2", "doc1", 1], ["q3", "doc1", 0], ["q3", "doc2", 0]], columns=["qid", "docno", "score"])
        mock_input = ptt.UniformTransformer(df)
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
        

