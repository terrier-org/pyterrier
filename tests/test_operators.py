import pandas as pd
import unittest
import pyterrier as pt
import warnings
from matchpy import *
from .base import BaseTestCase
from pytest import warns

class TestOperators(BaseTestCase):

    def test_then_dataframe(self):
        #this test is we can have DataFrame >> SOMETHINGELSE

        topicsSource = pd.DataFrame([["1", "AA"]], columns=["qid", "query"])

        def rewrite(topics):
            for index, row in topics.iterrows():
                row["query"] = row["query"] + " test"
            return topics
        fn1 = lambda topics : rewrite(topics)
        import pyterrier.transformer as ptt
        
        topics = pd.DataFrame([["1", "A"]], columns=["qid", "query"])
        rtr = pt.Transformer.from_df(topicsSource)(topics)
        self.assertEqual(1, len(rtr))
        self.assertTrue("query" in rtr.columns)
        self.assertTrue("qid" in rtr.columns)
        self.assertEqual(2, len(rtr.columns))  
        self.assertEqual("AA", rtr.iloc[0]["query"])

        sequence1 = pt.Transformer.from_df(topicsSource) >> pt.apply.generic(fn1)
        rtr = sequence1(topics)
        self.assertTrue("query" in rtr.columns)
        self.assertTrue("qid" in rtr.columns)
        self.assertEqual(2, len(rtr.columns))        
        self.assertEqual(1, len(rtr))
        self.assertEqual("AA test", rtr.iloc[0]["query"])

    def test_then(self):
        
        def rewrite(topics):
            for index, row in topics.iterrows():
                row["query"] = row["query"] + " test"
            return topics
        fn1 = lambda topics : rewrite(topics)
        fn2 = lambda topics : rewrite(topics)
        import pyterrier.apply_base as ptt
        sequence1 = ptt.ApplyGenericTransformer(fn1) >> ptt.ApplyGenericTransformer(fn2)
        with warns(DeprecationWarning, match='Coercion of a'):
            sequence2 = ptt.ApplyGenericTransformer(fn1) >> fn2
        with warns(DeprecationWarning, match='Coercion of a'):
            sequence3 = ptt.ApplyGenericTransformer(fn1) >> rewrite
        with warns(DeprecationWarning, match='Coercion of a'):
            sequence4 = fn1 >> ptt.ApplyGenericTransformer(fn2)
        with warns(DeprecationWarning, match='Coercion of a'):
            sequence5 = rewrite >> ptt.ApplyGenericTransformer(fn2)
        
        for sequence in [sequence1, sequence2, sequence3, sequence4, sequence5]:
            self.assertTrue(isinstance(sequence, ptt.Transformer))
            #check we can access items
            self.assertEqual(2, len(sequence))
            self.assertTrue(sequence[0], ptt.Transformer)
            self.assertTrue(sequence[1], ptt.Transformer)            
            input = pd.DataFrame([["q1", "hello"]], columns=["qid", "query"])
            output = sequence.transform(input)
            self.assertEqual(1, len(output))
            self.assertEqual("q1", output.iloc[0]["qid"])
            self.assertEqual("hello test test", output.iloc[0]["query"])



    def test_then_multi(self):
        import pyterrier.transformer as ptt
        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]), uniform=True)
        mock3 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]), uniform=True)
        mock4 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]), uniform=True)
        
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

        import pyterrier.transformer as ptt
        mock = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]), uniform=True)
        for comb in [mock * 10, 10 * mock]:
            rtr = comb.transform(None)
            self.assertEqual(1, len(rtr))
            self.assertEqual("q1", rtr.iloc[0]["qid"])
            self.assertEqual("doc1", rtr.iloc[0]["docno"])
            self.assertEqual(50, rtr.iloc[0]["score"])

        import pyterrier.transformer as ptt
        from pyterrier.model import add_ranks
        mock = pt.Transformer.from_df(add_ranks(pd.DataFrame([["q1", "doc1", 5], ["q1", "doc2", 10]], columns=["qid", "docno", "score"]), single_query=True), uniform=True)
        rtr = mock.search("bla", qid="q1")
        self.assertEqual(2, len(rtr))
        self.assertEqual("q1", rtr.iloc[0]["qid"])
        self.assertEqual("doc2", rtr.iloc[0]["docno"])
        self.assertEqual(pt.model.FIRST_RANK, rtr.iloc[0]["rank"])

        rtr = (-1 * mock).search("bla", qid="q1")
        self.assertEqual(2, len(rtr))
        self.assertEqual("q1", rtr.iloc[0]["qid"])
        self.assertEqual("doc1", rtr.iloc[0]["docno"])
        self.assertEqual(pt.model.FIRST_RANK, rtr.iloc[0]["rank"])
    
    def test_plus(self):
        import pyterrier.transformer as ptt
        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]), uniform=True)

        combined = mock1 + mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(1, len(rtr))
        self.assertEqual("q1", rtr.iloc[0]["qid"])
        self.assertEqual("doc1", rtr.iloc[0]["docno"])
        self.assertEqual(15, rtr.iloc[0]["score"])

    def test_plus_notoverlap(self):
        import pyterrier.transformer as ptt
        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc2", 10]], columns=["qid", "docno", "score"]), uniform=True)

        combined = mock1 + mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(2, len(rtr))
        self.assertEqual("q1", rtr.iloc[0]["qid"])
        self.assertEqual("doc1", rtr.iloc[0]["docno"])
        self.assertEqual(5, rtr.iloc[0]["score"])
        self.assertEqual("doc2", rtr.iloc[1]["docno"])
        self.assertEqual(10, rtr.iloc[1]["score"])

    def test_plus_more_cols(self):
        from pyterrier.model import add_ranks
        mock1 = pt.Transformer.from_df(add_ranks(pd.DataFrame([["q1", "a query", "doc1", 5]], columns=["qid", "query", "docno", "score"]), single_query=True), uniform=True)
        mock2 = pt.Transformer.from_df(add_ranks(pd.DataFrame([["q1", "a query", "doc1", 10]], columns=["qid", "query", "docno", "score"]), single_query=True), uniform=True)

        combined = mock1 + mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(1, len(rtr))
        self.assertEqual("q1", rtr.iloc[0]["qid"])
        self.assertEqual("doc1", rtr.iloc[0]["docno"])
        self.assertEqual(15, rtr.iloc[0]["score"])
        bad_columns = ["rank_x", "rank_y", "rank_r", "query_x", "query_y", "query_R", "score_x", "score_y", "score_r"]
        for bad in bad_columns:
            self.assertFalse(bad in rtr.columns, "column %s in returned dataframe" % bad)

    def test_rank_cutoff(self):
        mock1 = pt.Transformer.from_df( pd.DataFrame([["q1", "d2", 0, 5.1], ["q1", "d3", 1, 5.1]], columns=["qid", "docno", "rank", "score"]), uniform=True)
        cutpipe = mock1 % 1
        rtr = cutpipe.transform(None)
        self.assertEqual(1, len(rtr))
        
    def test_concatenate(self):
        import numpy as np
        mock1 = pt.Transformer.from_df( pd.DataFrame([["q1", "d2", 2, 4.9, np.array([1,2])], ["q1", "d3", 1, 5.1, np.array([1,2])]], columns=["qid", "docno", "rank", "score", "bla"]), uniform=True)
        mock2 = pt.Transformer.from_df( pd.DataFrame([["q1", "d1", 1, 4.9, np.array([1,1])], ["q1", "d3", 2, 5.1, np.array([1,2])]], columns=["qid", "docno", "rank", "score", "bla"]), uniform=True)

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
        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]), uniform=True)
        mock3 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 15]], columns=["qid", "docno", "score"]), uniform=True)

        combined = mock1 + mock2 + mock3
        for pipe in [combined, combined.compile()]:

            # we dont need an input, as both Identity transformers will return anyway
            rtr = pipe.transform(None)

            self.assertEqual(1, len(rtr))
            self.assertEqual("q1", rtr.iloc[0]["qid"])
            self.assertEqual("doc1", rtr.iloc[0]["docno"])
            self.assertEqual(30, rtr.iloc[0]["score"])


    def test_union(self):
        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "q1texta", "doc1", 5, "body text"], ["q1", "q1texta", "doc3", 5, "body text"]], columns=["qid", "query", "docno", "score", "body"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "q1textb", "doc2", 10, "body text" ]], [["q1", "q1textb", "doc3", 10, "body text"]], columns=["qid", "query", "docno", "score", "body"]), uniform=True)

        combined = mock1 | mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(3, len(rtr))
        self.assertTrue("q1" in rtr["qid"].values)
        self.assertTrue("doc1" in rtr["docno"].values)
        self.assertTrue("doc2" in rtr["docno"].values)
        # in case we have different values for query for the same (qid, docno), we use only the first one
        self.assertTrue("q1texta" in rtr["query"].values)
        self.assertTrue("q1textb" in rtr[rtr.docno == "doc2"]["query"].values) 
        self.assertTrue("q1textb" not in rtr[rtr.docno == "doc3"]["query"].values) 

        print(rtr.columns)
        for col in ["qid", "query", "docno", "body"]:
            self.assertTrue(col in rtr.columns, "%s not found in cols" % col)
        for col in ["rank", "score", 'query_y', 'score_y', 'body_y']:
            self.assertFalse(col in rtr.columns, "%s found in cols" % col)            

    def test_intersect(self):
        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "q1texta", "doc1", 5, "body text"]], columns=["qid", "query", "docno", "score", "body"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "q1textb", "doc2", 10, "body text"], ["q1", "q1textb", "doc1", 10, "body text"]], columns=["qid", "query", "docno", "score", "body"]), uniform=True)

        combined = mock1 & mock2
        # we dont need an input, as both Identity transformers will return anyway
        rtr = combined.transform(None)

        self.assertEqual(1, len(rtr))
        self.assertTrue("q1" in rtr["qid"].values)
        self.assertTrue("doc1" in rtr["docno"].values)
        self.assertFalse("doc2" in rtr["docno"].values)
        # in case we have different values for query for the same (qid, docno), we use the left one
        self.assertTrue("q1texta" in rtr["query"].values)
        self.assertTrue("q1textb" not in rtr["query"].values)

        for col in ["qid", "query", "docno", "body"]:
            self.assertTrue(col in rtr.columns, "%s not found in cols" % col)

        for col in ["rank", "score", 'query_y', 'score_y', 'body_y']:
            self.assertFalse(col in rtr.columns, "%s found in cols" % col)        
        
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
        import pyterrier.ops as pto
        import pyterrier.batchretrieve
        mock0 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 0], ["q1", "doc2", 0]], columns=["qid", "docno", "score"]), uniform=True)

        mock1 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 5], ["q1", "doc2", 0]], columns=["qid", "docno", "score"]), uniform=True)
        mock2 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10], ["q1", "doc2", 0]], columns=["qid", "docno", "score"]), uniform=True)
        mock3 = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 15], ["q1", "doc2", 0]], columns=["qid", "docno", "score"]), uniform=True)

        mock3_empty = pt.Transformer.from_df(pd.DataFrame([], columns=["qid", "docno", "score"]), uniform=True)
        mock2_partial = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 10]], columns=["qid", "docno", "score"]), uniform=True)
        mock3_partial = pt.Transformer.from_df(pd.DataFrame([["q1", "doc1", 15]], columns=["qid", "docno", "score"]), uniform=True)


        mock12a = mock1 ** mock2
        mock123a = mock1 ** mock2 ** mock3
        mock123b = mock12a ** mock3
        mock123a_manual = pto.FeatureUnionPipeline(
                pto.FeatureUnionPipeline(mock1, mock2),
                mock3
        )
        mock123b_manual = pto.FeatureUnionPipeline(
                mock1,
                pto.FeatureUnionPipeline(mock2, mock3),
        )
        mock123e = pto.FeatureUnionPipeline(
                mock1,
                pto.FeatureUnionPipeline(mock2, mock3_empty),
        )

        mock12e3 = pto.FeatureUnionPipeline(
                mock1,
                pto.FeatureUnionPipeline(mock3_empty, mock3),
        )

        mock123p = pto.FeatureUnionPipeline(
                mock1,
                pto.FeatureUnionPipeline(mock2, mock3_partial),
        )

        mock12p3 = pto.FeatureUnionPipeline(
                mock1,
                pto.FeatureUnionPipeline(mock2_partial, mock3),
        )
        
        
        self.assertEqual(2, len(mock12a.models))
        self.assertEqual(2, len(mock12a.models))
        pyterrier.batchretrieve.setup_rewrites()

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
            #print(rtr)
            self.assertIsNotNone(rtr)
            self.assertEqual(2, len(rtr))
            self.assertTrue("qid" in rtr.columns)
            self.assertTrue("docno" in rtr.columns)
            self.assertFalse("features_x" in rtr.columns)
            self.assertFalse("features_y" in rtr.columns)
            self.assertTrue("features" in rtr.columns)
            self.assertTrue("q1" in rtr["qid"].values)
            self.assertTrue("doc1" in rtr["docno"].values)
            import numpy as np
            self.assertTrue( np.allclose(np.array([5,10,15]), rtr.iloc[0]["features"]))

        _test_expression(mock123_simple)
        _test_expression(mock123a)
        _test_expression(mock123b)
        _test_expression(mock123b)
        with self.assertRaises(ValueError):
            _test_expression(mock123e)
        with self.assertRaises(ValueError):
            _test_expression(mock12e3)
        
        with warnings.catch_warnings(record=True) as w:
            _test_expression(mock123p)
            assert "Got number of results" in str(w[-1].message)
        
        with warnings.catch_warnings(record=True) as w:
            _test_expression(mock12p3)
            assert "Got number of results" in str(w[-1].message)

    def test_feature_union_empty(self):
        mock_input = pt.Transformer.from_df(pd.DataFrame([["q1", "a query", "doc1", 5]], columns=["qid", "query", "docno", "score"]))
        mock_f1 = pt.Transformer.from_df(pd.DataFrame([["q1", "a query", "doc1", 10]], columns=["qid", "query", "docno", "score"]))
        mock_f2 = pt.Transformer.from_df(pd.DataFrame([["q1", "a query", "doc1", 50]], columns=["qid", "query", "docno", "score"]))

        pipe = mock_input >> (mock_f1 ** mock_f2)
        rtr = pipe.search('another query', qid='q2')
        self.assertEqual(0, len(rtr))
        self.assertIn('score', rtr.columns)
        self.assertIn('features', rtr.columns)

    def test_feature_union(self): 
        import pyterrier.ops as ptt
        mock_input = pt.Transformer.from_df(pd.DataFrame([["q1", "a query", "doc1", 5]], columns=["qid", "query", "docno", "score"]), uniform=True)
        
        mock_f1 = pt.Transformer.from_df(pd.DataFrame([["q1", "a query", "doc1", 10]], columns=["qid", "query", "docno", "score"]), uniform=True)
        mock_f2 = pt.Transformer.from_df(pd.DataFrame([["q1", "a query", "doc1", 50]], columns=["qid", "query", "docno", "score"]), uniform=True)

        def _test_expression(pipeline):
            # check access to the objects
            self.assertEqual(2, len(pipeline))
            self.assertEqual(2, len(pipeline[1]))
            
            # we dont need an input, as both Uniform transformers will return anyway
            for name, input in [
                ('none', None),
                ('query df', pt.new.queries(['a query'], qid=['q1'])),
                ('empty query df', pt.new.queries(['a query'], qid=['q1']).head(0))
            ]:
                with self.subTest(name):
                    rtr = pipeline.transform(input)
                    self.assertEqual(1, len(rtr))
                    self.assertTrue("qid" in rtr.columns)
                    self.assertTrue("docno" in rtr.columns)
                    #self.assertTrue("score" in rtr.columns)
                    self.assertTrue("features" in rtr.columns)

                    bad_columns = ["rank_x", "rank_y", "rank_r", "query_x", "query_y", "query_R", "score_x", "score_y", "score_r", "features_x", "features_y"]
                    for bad in bad_columns:
                        self.assertFalse(bad in rtr.columns, "column %s in returned dataframe" % bad)

                    self.assertTrue("q1" in rtr["qid"].values)
                    self.assertTrue("doc1" in rtr["docno"].values)
                    import numpy as np
                    self.assertTrue( np.array_equal(np.array([10,50]), rtr.iloc[0]["features"]))

        # test using direct instantiation, as well as using the ** operator
        _test_expression(mock_input >> ptt.FeatureUnionPipeline(mock_f1, mock_f2))
        _test_expression(mock_input >> mock_f1 ** mock_f2)       

if __name__ == "__main__":
    unittest.main()
        
