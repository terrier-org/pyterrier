import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

class TestBatchRetrieve(BaseTestCase):

    def test_one_row(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.pipelines.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels())
        print(rtr)
