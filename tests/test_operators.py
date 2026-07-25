import pandas as pd
import unittest
import pyterrier as pt
import warnings
from .base import BaseTestCase
from pytest import warns

class TestOperators(BaseTestCase):

    def test_then_dataframe(self):
        #this test is we can have DataFrame >> SOMETHINGELSE

        topicsSource = pd.DataFrame([["1", "AA"]], columns=["qid", "query"])

        def rewrite(topics):
            topics = topics.copy()
            topics["query"] = topics["query"].apply(lambda x: f"{x} test")
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
            topics = topics.copy()
            topics["query"] = topics["query"].apply(lambda x: f"{x} test")
            return topics
        fn1 = lambda topics : rewrite(topics)
        fn2 = lambda topics : rewrite(topics)
        import pyterrier.apply_base as ptt
        sequence1 = ptt.ApplyGenericTransformer(fn1) >> ptt.ApplyGenericTransformer(fn2)

        
        for sequence in [sequence1]:
            self.assertTrue(isinstance(sequence, pt.Transformer))
            #check we can access items
            self.assertEqual(2, len(sequence))
            self.assertTrue(sequence[0], pt.Transformer)
            self.assertTrue(sequence[1], pt.Transformer)
            input = pd.DataFrame([["q1", "hello"]], columns=["qid", "query"])
            output = sequence.transform(input)
            self.assertEqual(1, len(output))
            self.assertEqual("q1", output.iloc[0]["qid"])
            self.assertEqual("hello test test", output.iloc[0]["query"])
            # now test transform_iter pathway via __call__
            output = sequence(input.to_dict(orient='records'))
            self.assertIsInstance(output, list)
            output = pd.DataFrame(output)
            self.assertEqual(1, len(output))
            self.assertEqual("q1", output.iloc[0]["qid"])
            self.assertEqual("hello test test", output.iloc[0]["query"])

    def test_compile_fuse_right(self):
        from typing import Optional

        class B(pt.Transformer):
            def transform(self, df):
                pass

        class C(pt.Transformer):
            def transform(self, df):
                pass

        class A(pt.Transformer):
            def transform(self, df):
                pass
            def fuse_right(self, right: pt.Transformer) -> Optional[pt.Transformer]:
                if isinstance(right, B):
                    return C()
                return None

        test1 = A()
        self.assertIsInstance(test1.compile(), A)

        test2 = A() >> A()
        test2_c = test2.compile()
        self.assertEqual(2, len(test2_c))
        self.assertIsInstance(test2_c[0], A)
        self.assertIsInstance(test2_c[1], A)

        test3 = A() >> B()
        test3_c = test3.compile()
        self.assertIsInstance(test3_c, C)

    def test_compile_fuse_left(self):
        from typing import Optional

        class B(pt.Transformer):
            def transform(self, df):
                pass

            def fuse_left(self, left: pt.Transformer) -> Optional[pt.Transformer]:
                if isinstance(left, A):
                    return C()
                return None

        class C(pt.Transformer):
            def transform(self, df):
                pass

        class A(pt.Transformer):
            def transform(self, df):
                pass

        test1 = B()
        self.assertIsInstance(test1.compile(), B)

        test2 = B() >> B()
        test2_c = test2.compile()
        self.assertEqual(2, len(test2_c))
        self.assertIsInstance(test2_c[0], B)
        self.assertIsInstance(test2_c[1], B)

        test3 = A() >> B()
        test3_c = test3.compile()
        self.assertIsInstance(test3_c, C)

    def test_compile(self):
        class PRF(pt.Transformer):
            def __init__(self, k=10):
                self.k = k
            def transform(self, df):
                pass
            def compile(self):
                # TODO: this shouldnt be so hacky
                import pyterrier._ops
                return pyterrier._ops.RankCutoff(self.k) >> self
            def __repr__(self):
                return f'PRF(k={self.k})'
                
        class Retr(pt.Transformer):
            def __init__(self, k=1000):
                self.k = k
            def fuse_rank_cutoff(self, k):
                return Retr(k=k)
            def transform(self, df):
                pass
            def __repr__(self):
                return f'Retr(k={self.k})'

        class Extractor(pt.Transformer):
            def transform(self, df):
                pass
            def fuse_rank_cutoff(self, k):
                # TODO: this shouldnt be so hacky
                import pyterrier._ops
                return pyterrier._ops.RankCutoff(k) >> self
            def __repr__(self):
                return 'Extr'

        # check that the rank cutoff fusion works.
        pipe1 = Retr() % 3
        pipe1c = pipe1.compile()
        self.assertEqual(3, pipe1c.k)
        self.assertIsInstance(pipe1c, Retr)

        # now check that we can propagate k=3 (from PRF.compile()) all the way back to Retr
        slow_pipe = Retr() >> Extractor() >> PRF(k=3) >> Retr()
        fast_pipe = slow_pipe.compile()
        # the resulting pipe should be:
        # Retr(3) >> Extr() >> PRF(k=3) >> Retr(1000)
        self.assertEqual(4, len(fast_pipe))
        self.assertEqual(Retr, type(fast_pipe[0]))
        self.assertEqual(3, fast_pipe[0].k)
        self.assertEqual(Extractor, type(fast_pipe[1]))
        self.assertEqual(PRF, type(fast_pipe[2]))
        self.assertEqual(3, fast_pipe[2].k)
        self.assertEqual(Retr, type(fast_pipe[3]))
        self.assertEqual(1000, fast_pipe[3].k)

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


        self.assertEqual(2, len(combined12))
        self.assertEqual(2, len(combined23))
        self.assertEqual(2, len(combined12))
        self.assertEqual(2, len(combined23))

        for C in [combined123_a_C, combined123_b_C, combined123_c_C]:
            self.assertEqual(3, len(C))
            self.assertEqual("(UniformTransformer() >> UniformTransformer() >> UniformTransformer())", repr(C))
        
        # finally check recursive application
        C4 = (mock1 >> mock2 >> mock3 >> mock4).compile()
        self.assertEqual("(UniformTransformer() >> UniformTransformer() >> UniformTransformer() >> UniformTransformer())", repr(C4))
        self.assertEqual(4, len(C4))


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

    def test_kneedle(self):
        import numpy as np

        self.assertEqual(
            [1201, 1401, 1601, 1801, 2001, 2201, 2401, 2601, 3001],
            pt.Kneedle()._check_positions(3001),
        )
        results = pd.DataFrame({
            'qid': ['q1'] * 1401,
            'docno': [f'd{i}' for i in range(1401)],
            'rank': range(1401),
            'label': [1] * 150 + [0] * 1251,
        })
        output = pt.Kneedle()(results)
        self.assertEqual(1201, len(output))
        self.assertEqual(1200, output['rank'].max())
        self.assertEqual(150, output['label'].sum())

        labels = np.r_[np.ones(150), np.zeros(1251)]
        self.assertEqual(1201, pt.Kneedle()._stop_position(labels))
        self.assertEqual(1201, pt.Kneedle()._stop_position(np.r_[labels, np.ones(400)]))

    def test_tar_stopping_rules(self):
        import numpy as np

        labels = np.r_[np.ones(101), np.zeros(700)]
        self.assertEqual(401, pt.FixedRound(2)._stop_position(labels))
        self.assertEqual(201, pt.BatchPrecision()._stop_position(np.r_[1, np.zeros(200)]))
        self.assertEqual(2401, pt.Rule2399()._stop_position(np.zeros(2401)))
        self.assertEqual(601, pt.ReviewHalf(1000)._stop_position(labels))
        self.assertEqual(201, pt.Budget(1000)._stop_position(np.r_[np.ones(150), np.zeros(51)]))
        self.assertEqual(801, pt.Budget(1000)._stop_position(np.zeros(801)))
        self.assertEqual(401, pt.CMHHeuristic(.8, 1000)._stop_position(np.r_[1, np.ones(100), np.zeros(300)]))

    def test_poisson_point_stopping(self):
        import numpy as np

        rule = pt.PoissonPoint(100, initial_fraction=.2, check_fraction=.2, initial_min_relevant=1)
        self.assertEqual([20, 40, 60, 80, 100], rule._sample_sizes())
        labels = np.r_[np.ones(10), np.zeros(90)]
        self.assertEqual(20, rule._stop_position(labels))
        self.assertEqual(20, rule._stop_position(np.r_[labels[:20], np.ones(80)]))
        self.assertEqual(6, rule._poisson_upper_bound(3, .95, 100))

    def test_control_set_stopping_rules(self):
        results = pd.DataFrame({
            'qid': ['q1'] * 60,
            'docno': [f'd{i}' for i in range(60)],
            'rank': range(60),
            'label': [0] * 60,
        })
        target_control = pd.DataFrame({
            'qid': ['q1'] * 10,
            'docno': [f'd{i}' for i in range(4, 50, 5)],
            'label': [1] * 10,
        })
        target = pt.TargetRecapture(target_control)
        self.assertEqual(10, target.target_size)
        self.assertEqual(10, target.control_cost('q1'))
        self.assertEqual(50, len(target(results)))

        qbcb_control = pd.DataFrame({
            'qid': ['q1'] * 30,
            'docno': [f'd{i}' for i in range(30)],
            'label': [1] * 30,
        })
        qbcb = pt.QBCB(qbcb_control)
        self.assertEqual(14, qbcb._required_control_rank(14))
        self.assertEqual(21, qbcb._required_control_rank(22))
        self.assertEqual(28, qbcb._required_control_rank(30))
        self.assertEqual(28, len(qbcb(results)))
        self.assertEqual(28, len(qbcb(results.assign(label=1))))

    def test_grlstop_reward(self):
        rule = pt.GRLStop(.8, n_windows=4, total_timesteps=1)
        self.assertEqual(.5, rule._reward(0, 1))
        self.assertEqual(-.5, rule._reward(2, 1))

    def test_grlstop_train_and_replay(self):
        try:
            import gymnasium  # noqa: F401
            import stable_baselines3  # noqa: F401
        except ImportError:
            self.skipTest('GRLStop optional dependencies are not installed')
        import numpy as np

        training = pd.DataFrame({
            'qid': ['q1'] * 8,
            'docno': [f'd{i}' for i in range(8)],
            'rank': range(8),
            'features': [np.array([i, 1.]) for i in range(8)],
            'label': [1, 0, 0, 1, 0, 0, 1, 0],
        })
        testing = training.assign(features=[np.array([i, 1., 0.]) for i in range(8)])
        rule = pt.GRLStop(.8, n_windows=4, total_timesteps=16, n_steps=4, n_epochs=1)
        trajectory = rule._trajectories(training)[0][1]
        environment = rule._environment(trajectory, .8)()
        environment.reset()
        _, reward, _, _, _ = environment.step(0)
        self.assertEqual(rule._reward(0, rule._target_position(trajectory, .8)), reward)
        replay = rule.fit(training)(testing)
        self.assertEqual({'q1'}, set(replay['qid']))
        self.assertLessEqual(len(replay), len(testing))
        self.assertGreater(len(replay), 0)
        
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
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        PL2 = pt.terrier.Retriever(index, wmodel="PL2")
        expression = BM25 >> (pt.transformer.IdentityTransformer() ** TF_IDF ** PL2)

        self.assertEqual(2, len(expression))
        self.assertEqual(3, len(expression[1]))

        res = expression.transform(dataset.get_topics().head(2))
        self.assertTrue("features" in res.columns)
        self.assertFalse("features_x" in res.columns)
        self.assertFalse("features_y" in res.columns)
        print(res.iloc[0]["features"])
        self.assertEqual(3, len(res.iloc[0]["features"]))


    def test_feature_union_multi(self):
        import pyterrier._ops as pto
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
        mock123a_manual = pto.FeatureUnion(
                pto.FeatureUnion(mock1, mock2),
                mock3
        )
        mock123b_manual = pto.FeatureUnion(
                mock1,
                pto.FeatureUnion(mock2, mock3),
        )
        mock123e = pto.FeatureUnion(
                mock1,
                pto.FeatureUnion(mock2, mock3_empty),
        )

        mock12e3 = pto.FeatureUnion(
                mock1,
                pto.FeatureUnion(mock3_empty, mock3),
        )

        mock123p = pto.FeatureUnion(
                mock1,
                pto.FeatureUnion(mock2, mock3_partial),
        )

        mock12p3 = pto.FeatureUnion(
                mock1,
                pto.FeatureUnion(mock2_partial, mock3),
        )
        
        
        self.assertEqual(2, len(mock12a))
        self.assertEqual(2, len(mock12a))

        mock123_simple = mock123a.compile()
        self.assertIsNotNone(mock123_simple)
        self.assertEqual("(UniformTransformer() ** UniformTransformer() ** UniformTransformer())", repr(mock123_simple))
        #
        #
        self.assertEqual(3, len(mock123_simple))

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
        import pyterrier._ops as ptt
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
        _test_expression(mock_input >> ptt.FeatureUnion(mock_f1, mock_f2))
        _test_expression(mock_input >> mock_f1 ** mock_f2)  

    def _assertInNoOrder(self, member, container):
        container_set = [set(item) for item in container]
        self.assertIn(set(member), container_set)     

    def test_funion_inspection(self):
        dataset = pt.get_dataset("vaswani")
        index = dataset.get_index()
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        pipe = (BM25 ** BM25)
        self._assertInNoOrder(['qid', 'docno', 'query'], pt.inspect.transformer_inputs(pipe)) 


if __name__ == "__main__":
    unittest.main()
        
