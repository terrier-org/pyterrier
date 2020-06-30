import pandas as pd
import unittest
import pyterrier as pt
from .base import BaseTestCase

class TestScoring(BaseTestCase):

    def test_scoring_manual(self):
        input = pd.DataFrame([["q1", "fox", "d1", "all the fox were fox"]], columns=["qid", "query", "docno", "body"])
        from pyterrier.batchretrieve import TextScorer
        scorer = TextScorer(wmodel="Tf")
        rtr = scorer(input)
        self.assertEqual(1, len(rtr))
        self.assertTrue("score" in rtr.columns)
        self.assertEqual(2, rtr.iloc[0]["score"])
        
        