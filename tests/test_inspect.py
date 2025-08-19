import pandas as pd
import unittest
import pyterrier as pt
import warnings
from .base import BaseTestCase
from pytest import warns

class TestInspect(BaseTestCase):

    def test_terrier_retriever(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.terrier.Retriever(indexref)

        # as a retriever
        rank_cols = pt.inspect.transformer_outputs(retr, ["qid", "query"])
        example_res = retr.search("what are chemical reactions")
        self.assertEqual(rank_cols, example_res.columns.tolist())

        # as a reranker
        rerank_cols = pt.inspect.transformer_outputs(retr, rank_cols)
        example_res = retr.transform(example_res)
        self.assertEqual(rerank_cols, example_res.columns.tolist())

        # with prf
        rm3 = pt.terrier.rewrite.RM3(indexref)
        prf_cols = pt.inspect.transformer_outputs(rm3, rank_cols)
        example_res = rm3.transform(example_res)
        self.assertEqual(prf_cols, example_res.columns.tolist())

        # as a feature pipeline
        feat_pipe = retr >> (retr ** retr)
        feat_cols = pt.inspect.transformer_outputs(feat_pipe, ["qid", "query"])
        example_res = feat_pipe.search("what are chemical reactions")
        self.assertEqual(feat_cols, example_res.columns.tolist())

        # with a text loader:
        textl = pt.text.get_text(pt.get_dataset('irds:vaswani'), "text")
        textl_pipe = retr >> textl
        text_cols = pt.inspect.transformer_outputs(textl_pipe, ["qid", "query"])
        self.assertEqual(rank_cols + ["text"], text_cols)

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
        
