import pandas as pd
import pyterrier as pt
import os
import unittest
import warnings
import math
from pyterrier.measures import *
from .base import BaseTestCase
import pytest

class TestExperiment(BaseTestCase):
    def test_common(self):
        bm25 = pt.terrier.Retriever.from_dataset('vaswani', 'terrier_stemmed', wmodel='BM25')
        pipeA = bm25 %3
        pipeB = bm25 %10
        import pyterrier.pipelines
        common, suffices = pyterrier.pipelines._identifyCommon([pipeA, pipeB])
        self.assertEqual(bm25, common)
        self.assertIsInstance(suffices[0], pt.RankCutoff)
        self.assertEqual(3, suffices[0].k)
        self.assertIsInstance(suffices[1], pt.RankCutoff)
        self.assertEqual(10, suffices[1].k)

        common, suffices = pyterrier.pipelines._identifyCommon([bm25, pipeB])
        self.assertEqual(bm25, common)
        self.assertIsInstance(suffices[0], pt.transformer.IdentityTransformer)
        self.assertIsInstance(suffices[1], pt.RankCutoff)
        self.assertEqual(10, suffices[1].k)
    
    def test_precompute_experiment(self):
        bm25 = pt.terrier.Retriever.from_dataset('vaswani', 'terrier_stemmed', wmodel='BM25')
        pipeB = bm25 %10
        df1 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'])
        df2 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], precompute_shared=True)
        pd.testing.assert_frame_equal(df1, df2)

        df3 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], precompute_shared=True, batch_size=4)
        pd.testing.assert_frame_equal(df1, df3)
