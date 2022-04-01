import pyterrier as pt
import unittest
from .base import BaseTestCase
import os
import pandas as pd

class TestTransformer(BaseTestCase):

    def test_is_transformer(self):
        class MyTransformer1(pt.Transformer):
            pass
        class MyTransformer2(pt.transformer.TransformerBase):
            pass
        class MyTransformer3(pt.transformer.IterDictIndexerBase):
            pass
        class MyTransformer4(pt.Estimator):
            pass
        for T in [MyTransformer1, MyTransformer2, MyTransformer3, MyTransformer4]:
            self.assertTrue(pt.transformer.is_transformer(T()))
