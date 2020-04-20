import pandas as pd
import unittest
import pyterrier as pt

import pyterrier.transformer as ptt;

class TestOperators(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestOperators, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init()

    def test_then(self):
        def rewrite(topics):
            for index, row in topics.iterrows():
                row["query"] = row["query"] + " test"
            return topics
        fn1 = lambda topics : rewrite(topics)
        fn2 = lambda topics : rewrite(topics)
        sequence = ptt.LambdaPipeline(fn1) >> ptt.LambdaPipeline(fn2)

        self.assertTrue(isinstance(sequence, ptt.TransformerBase))
        input = pd.DataFrame([["q1", "hello"]], columns=["qid", "query"])
        output = sequence.transform(input)
        self.assertEqual(1, len(output))
        self.assertEqual("q1", output.iloc[0]["qid"])
        self.assertEqual("hello test test", output.iloc[0]["query"])

    def test_mul(self):
        mock = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        for comb in [mock * 10, 10 * mock]:
            rtr = comb.transform(None)
            self.assertEqual(1, len(rtr))
            self.assertEqual("q1", rtr.iloc[0]["qid"])
            self.assertEqual("doc1", rtr.iloc[0]["docno"])
            self.assertEqual(50, rtr.iloc[0]["score"])
    
    def test_plus(self):
        mock1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        mock2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))

        combined = mock1 + mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(1, len(rtr))
        self.assertEqual("q1", rtr.iloc[0]["qid"])
        self.assertEqual("doc1", rtr.iloc[0]["docno"])
        self.assertEqual(15, rtr.iloc[0]["score"])

    def test_union(self):
        mock1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        mock2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc2", 10]], columns=["qid", "docno", "score"]))

        combined = mock1 | mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(2, len(rtr))
        self.assertTrue("q1" in rtr["qid"].values)
        self.assertTrue("doc1" in rtr["docno"].values)
        self.assertTrue("doc2" in rtr["docno"].values)

    def test_intersect(self):
        mock1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        mock2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc2", 10], ["q1", "doc1", 10]], columns=["qid", "docno", "score"]))

        combined = mock1 & mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(1, len(rtr))
        self.assertTrue("q1" in rtr["qid"].values)
        self.assertTrue("doc1" in rtr["docno"].values)
        self.assertFalse("doc2" in rtr["docno"].values)
        

if __name__ == "__main__":
    unittest.main()
        