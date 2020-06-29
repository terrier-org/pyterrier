import pandas as pd
import unittest
import pyterrier as pt

import pyterrier.transformer as ptt;
from matchpy import *

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
        sequence1 = ptt.LambdaPipeline(fn1) >> ptt.LambdaPipeline(fn2)
        sequence2 = ptt.LambdaPipeline(fn1) >> fn2
        sequence3 = ptt.LambdaPipeline(fn1) >> rewrite
        sequence4 = fn1 >> ptt.LambdaPipeline(fn2)
        sequence5 = rewrite >> ptt.LambdaPipeline(fn2)
        
        for sequence in [sequence1, sequence2, sequence3, sequence4, sequence5]:
            self.assertTrue(isinstance(sequence, ptt.TransformerBase))
            #check we can access items
            self.assertEqual(2, len(sequence))
            self.assertTrue(sequence[0], ptt.TransformerBase)
            self.assertTrue(sequence[1], ptt.TransformerBase)            
            input = pd.DataFrame([["q1", "hello"]], columns=["qid", "query"])
            output = sequence.transform(input)
            self.assertEqual(1, len(output))
            self.assertEqual("q1", output.iloc[0]["qid"])
            self.assertEqual("hello test test", output.iloc[0]["query"])



    def test_then_multi(self):
        mock1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        mock2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))
        mock3 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))
        mock4 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))
        
        combined12 = mock1 >> mock2
        combined23 = mock2 >> mock3 
        combined123_a = combined12 >> mock3
        combined123_b = mock1 >> mock2 >> mock3
        combined123_c = mock2 >> combined23

        combined123_a_C = combined123_a.compile()
        combined123_b_C = combined123_b.compile()
        combined123_c_C = combined123_c.compile()


        self.assertEqual(2, len(combined12.models))
        self.assertEqual(2, len(combined23.models))
        self.assertEqual(2, len(combined12.models))
        self.assertEqual(2, len(combined23.models))

        for C in [combined123_a_C, combined123_b_C, combined123_c_C]:
            self.assertEqual(3, len(C.models))
            self.assertEqual("ComposedPipeline(UniformTransformer(), UniformTransformer(), UniformTransformer())",
                C.__repr__())
        
        # finally check recursive application
        C4 = (mock1 >> mock2 >> mock3 >> mock4).compile()
        self.assertEqual(
            "ComposedPipeline(UniformTransformer(), UniformTransformer(), UniformTransformer(), UniformTransformer())", 
            C4.__repr__())
        self.assertEqual(4, len(C4.models))


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

    def test_rank_cutoff(self):
        mock1 = ptt.UniformTransformer( pd.DataFrame([["q1", "d2", 1, 5.1], ["q1", "d3", 2, 5.1]], columns=["qid", "docno", "rank", "score"]))
        cutpipe = mock1 % 1
        rtr = cutpipe.transform(None)
        self.assertEqual(1, len(rtr))
        
    def test_concatenate(self):
        import numpy as np
        mock1 = ptt.UniformTransformer( pd.DataFrame([["q1", "d2", 2, 4.9, np.array([1,2])], ["q1", "d3", 1, 5.1, np.array([1,2])]], columns=["qid", "docno", "rank", "score", "bla"]))
        mock2 = ptt.UniformTransformer( pd.DataFrame([["q1", "d1", 1, 4.9, np.array([1,1])], ["q1", "d3", 2, 5.1, np.array([1,2])]], columns=["qid", "docno", "rank", "score", "bla"]))

        cutpipe = mock1 ^ mock2
        rtr = cutpipe.transform(None)
        self.assertEqual(3, len(rtr))
        row0 = rtr.iloc[0] 
        self.assertEqual("d3", row0["docno"])
        self.assertEqual(5.1, row0["score"])
        row1 = rtr.iloc[1] 
        self.assertEqual("d2", row1["docno"])
        self.assertEqual(4.9, row1["score"])
        row2 = rtr.iloc[2] 
        self.assertEqual("d1", row2["docno"])
        self.assertEqual(4.9-0.0001, row2["score"])


    def test_plus_multi_rewrite(self):
        mock1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        mock2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))
        mock3 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 15]], columns=["qid", "docno", "score"]))

        combined = mock1 + mock2 + mock3
        for pipe in [combined, combined.compile()]:

            # we dont need an input, as both Identity transformers will return anyway
            rtr = pipe.transform(None)

            self.assertEqual(1, len(rtr))
            self.assertEqual("q1", rtr.iloc[0]["qid"])
            self.assertEqual("doc1", rtr.iloc[0]["docno"])
            self.assertEqual(30, rtr.iloc[0]["score"])


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

    def test_feature_union_multi_actual(self):
        dataset = pt.get_dataset("vaswani")
        index = dataset.get_index()
        BM25 = pt.BatchRetrieve(index, wmodel="BM25")
        TF_IDF = pt.BatchRetrieve(index, wmodel="TF_IDF")
        PL2 = pt.BatchRetrieve(index, wmodel="PL2")
        pipe = BM25 >> (pt.transformer.IdentityTransformer() ** TF_IDF ** PL2)

        def _check(expression):
            self.assertEqual(2, len(expression))
            self.assertEqual(2, len(expression[1]))
            print("funion outer %d" % id(expression[1]))
            print("funion inner %d" % id(expression[1][1]))
            
            #print(repr(pipe))
            res = expression.transform(dataset.get_topics().head(2))
            self.assertTrue("features" in res.columns)
            self.assertFalse("features_x" in res.columns)
            self.assertFalse("features_y" in res.columns)
            print(res.iloc[0]["features"])
            self.assertEqual(3, len(res.iloc[0]["features"]))
        _check(pipe)


    def test_feature_union_multi(self):
        mock0 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 0]], columns=["qid", "docno", "score"]))

        mock1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        mock2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))
        mock3 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 15]], columns=["qid", "docno", "score"]))

        mock12a = mock1 ** mock2
        mock123a = mock1 ** mock2 ** mock3
        mock123b = mock12a ** mock3
        mock123a_manual = ptt.FeatureUnionPipeline(
                ptt.FeatureUnionPipeline(mock1, mock2),
                mock3
        )
        mock123b_manual = ptt.FeatureUnionPipeline(
                mock1,
                ptt.FeatureUnionPipeline(mock2, mock3),
        )
        
        self.assertEqual(2, len(mock12a.models))
        self.assertEqual(2, len(mock12a.models))
        ptt.setup_rewrites()

        mock123_simple = mock123a.compile()
        self.assertIsNotNone(mock123_simple)
        self.assertEqual(
            "FeatureUnionPipeline(UniformTransformer(), UniformTransformer(), UniformTransformer())", 
            mock123_simple.__repr__())
        #
        #
        self.assertEqual(3, len(mock123_simple.models))

        def _test_expression(expression):
            # we dont need an input, as both Identity transformers will return anyway
            rtr = (mock0 >> expression).transform(None)
            self.assertIsNotNone(rtr)
            self.assertEqual(1, len(rtr))
            self.assertTrue("qid" in rtr.columns)
            self.assertTrue("docno" in rtr.columns)
            self.assertFalse("features_x" in rtr.columns)
            self.assertFalse("features_y" in rtr.columns)
            #self.assertTrue("score" in rtr.columns)
            self.assertTrue("features" in rtr.columns)
            self.assertTrue("q1" in rtr["qid"].values)
            self.assertTrue("doc1" in rtr["docno"].values)
            import numpy as np
            self.assertTrue( np.array_equal(np.array([5,10,15]), rtr.iloc[0]["features"]))

        _test_expression(mock123_simple)
        _test_expression(mock123a)
        _test_expression(mock123b)

    def test_feature_union(self): 
        mock_input = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]))
        
        mock_f1 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]))
        mock_f2 = ptt.UniformTransformer(pd.DataFrame([["q1", "doc1", 50]], columns=["qid", "docno", "score"]))

        def _test_expression(pipeline):
            # check access to the objects
            self.assertEqual(2, len(pipeline))
            self.assertEqual(2, len(pipeline[1]))
            
            # we dont need an input, as both Uniform transformers will return anyway
            rtr = pipeline.transform(None)
            self.assertEqual(1, len(rtr))
            self.assertTrue("qid" in rtr.columns)
            self.assertTrue("docno" in rtr.columns)
            #self.assertTrue("score" in rtr.columns)
            self.assertTrue("features" in rtr.columns)
            self.assertFalse("features_x" in rtr.columns)
            self.assertFalse("features_y" in rtr.columns)
            self.assertTrue("q1" in rtr["qid"].values)
            self.assertTrue("doc1" in rtr["docno"].values)
            import numpy as np
            self.assertTrue( np.array_equal(np.array([10,50]), rtr.iloc[0]["features"]))

        # test using direct instantiation, as well as using the ** operator
        _test_expression(mock_input >> ptt.FeatureUnionPipeline(mock_f1, mock_f2))
        _test_expression(mock_input >> mock_f1 ** mock_f2)       

if __name__ == "__main__":
    unittest.main()
        