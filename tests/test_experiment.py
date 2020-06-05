import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase

class TestExperiment(BaseTestCase):

    def test_one_row(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.pipelines.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels())
        print(rtr)

        rtr = pt.pipelines.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels(), dataframe=False)
        print(rtr)

    def test_perquery(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.pipelines.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels(), perquery=True)
        print(rtr)

        rtr = pt.pipelines.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels(), perquery=True, dataframe=False)
        print(rtr)
