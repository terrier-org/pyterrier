import pyterrier as pt
import unittest
from .base import BaseTestCase
import os
import pandas as pd
from pytest import warns

class TestTransformer(BaseTestCase):

    def test_call(self):
        inputDocs = pt.new.ranked_documents([[2, 1], [2]], qid=["q100", "q10"])
        t = pt.Transformer.from_df(inputDocs)
        self.assertEqual(2, len(t(pt.new.queries(['a'], qid=['q100']))))
        self.assertEqual(1, len(t(pt.new.queries(['a'], qid=['q10']))))
        self.assertEqual(2, len(t([{'qid' : 'q100'}])))
        self.assertEqual(1, len(t([{'qid' : 'q10'}])))

    def test_is_transformer(self):
        class MyTransformer1(pt.Transformer):
            pass
        class MyTransformer2(pt.transformer.TransformerBase):
            pass
        class MyTransformer3(pt.Indexer):
            pass
        class MyTransformer3a(pt.transformer.IterDictIndexerBase):
            pass
        class MyTransformer4a(pt.Estimator):
            pass
        class MyTransformer4(pt.transformer.EstimatorBase):
            pass

        # check normal API
        for T in [MyTransformer1, MyTransformer3, MyTransformer4a]:
            self.assertTrue(pt.transformer.is_transformer(T()))

        # check deprecated API
        for T in [MyTransformer2, MyTransformer3a, MyTransformer4]:
            with warns(DeprecationWarning, match='instead of'):
                instance = T()
            self.assertTrue(pt.transformer.is_transformer(instance))
