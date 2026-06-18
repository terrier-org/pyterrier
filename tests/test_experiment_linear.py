import pandas as pd
import pyterrier as pt
import os
import pytest
from .base_experiment import TestExperimentBase
import ir_measures


class TestExperimentLinear(TestExperimentBase):
    """Test suite for PyTerrier Experiment functionality specific to linear execution plans."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_exp_kwargs = {'plan': 'linear'}

    def test_precomp_common(self):
        """Test the _identifyCommon utility for detecting common pipeline prefixes."""
        bm25 = pt.terrier.Retriever(self._vaswani_index(), wmodel='BM25')
        pipeA = bm25 % 3
        pipeB = bm25 % 10
        import pyterrier._evaluation._execution
        common, suffices = pyterrier._evaluation._exec_linear._identifyCommon([pipeA, pipeB])
        self.assertEqual(bm25, common)
        self.assertIsInstance(suffices[0], pt.RankCutoff)
        self.assertEqual(3, suffices[0].k)
        self.assertIsInstance(suffices[1], pt.RankCutoff)
        self.assertEqual(10, suffices[1].k)

        common, suffices = pyterrier._evaluation._exec_linear._identifyCommon([bm25, pipeB])
        self.assertEqual(bm25, common)
        self.assertIsInstance(suffices[0], pt.transformer.IdentityTransformer)
        self.assertIsInstance(suffices[1], pt.RankCutoff)
        self.assertEqual(10, suffices[1].k)
    
    def test_precompute_experiment(self):
        """Test precomputation of common pipeline prefix for experiment optimization."""
        bm25 = pt.terrier.Retriever(self._vaswani_index(), wmodel='BM25')
        pipeB = bm25 % 10
        df1 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], **self.pt_exp_kwargs)
        df2 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], precompute_prefix=True, **self.pt_exp_kwargs)
        pd.testing.assert_frame_equal(df1, df2)

        df3 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], precompute_prefix=True, batch_size=4, **self.pt_exp_kwargs)
        pd.testing.assert_frame_equal(df1, df3)
