import pandas as pd
import unittest
import pyterrier as pt
import warnings
from .base import BaseTestCase
from pytest import warns

class TestInspect(BaseTestCase):

    def test_rename(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'context': ['context1', 'context2']
        })
        br0 = pt.Transformer.from_df(df[['qid', 'query']], uniform=True)
        br1 = pt.Transformer.from_df(df, uniform=True)

        cols = pt.inspect.transformer_outputs(br1 >> pt.apply.rename({'context' : 'prompt'}), ["qid", "query", "context"])
        self.assertEqual(cols, ['qid', 'query', 'prompt'])

        with self.assertRaises(pt.validate.InputValidationError):
            pt.inspect.transformer_outputs(br0 >> pt.apply.rename({'context' : 'prompt'}), ["qid", "query"])

    def test_rename_nocols(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'context': ['context1', 'context2']
        })
        br0 = pt.Transformer.from_df(df[['qid', 'query']], uniform=True)
        br1 = pt.Transformer.from_df(df, uniform=True)

        def _rename_context(df):
            if len(df) == 0:
                raise ValueError("Empty DataFrame")
            return df.rename(columns={'context' : 'prompt'})

        with self.assertRaises(pt.inspect.InspectError):
            pt.inspect.transformer_outputs(br1 >> pt.apply.generic(_rename_context), ["qid", "query", "context"])

        with self.assertRaises(pt.inspect.InspectError):
            pt.inspect.transformer_outputs(br0 >> pt.apply.generic(_rename_context), ["qid", "query"])

if __name__ == "__main__":
    unittest.main()
        
