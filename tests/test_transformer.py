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
        class IncompleteTransformer(pt.Transformer):
            pass # doesnt implement .transform() or .transform_iter()
        class MyTransformer1(pt.Transformer):
            def transform(self, df):
                pass
        class MyTransformer1a(pt.Transformer):
            def transform_iter(self, df):
                pass
        class MyTransformer2(pt.transformer.TransformerBase):
            def transform(self, df):
                pass
        class MyTransformer3(pt.Indexer):
            pass # indexers dont need a transform
        class MyTransformer3a(pt.transformer.IterDictIndexerBase):
            pass # indexers dont need a transform
        class MyTransformer4a(pt.Estimator):
            def transform(self, df):
                pass
        class MyTransformer4(pt.transformer.EstimatorBase):
            def transform(self, df):
                pass

        with self.assertRaises(NotImplementedError):
            IncompleteTransformer()

        # check normal API
        for T in [MyTransformer1, MyTransformer1a, MyTransformer3, MyTransformer4a]:
            self.assertTrue(pt.transformer.is_transformer(T()))

        # check deprecated API
        for T in [MyTransformer2, MyTransformer3a, MyTransformer4]:
            with warns(DeprecationWarning, match='instead of'):
                instance = T()
            self.assertTrue(pt.transformer.is_transformer(instance))
