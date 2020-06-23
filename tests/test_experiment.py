import pandas as pd
import pyterrier as pt
import os
import unittest
import warnings
from .base import BaseTestCase

class TestExperiment(BaseTestCase):

    def test_one_row(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.pipelines.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"])
        print(rtr)

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
