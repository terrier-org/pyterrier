import pandas as pd
import pyterrier as pt
import os
import unittest
import warnings
from .base import BaseTestCase

class TestExperiment(BaseTestCase):

    def test_irm_APrel2(self):
        topics = pd.DataFrame([["q1", "q1"], ["q2", "q1"] ], columns=["qid", "query"])
        res1 = pd.DataFrame([["q1", "d1", 1.0], ["q2", "d1", 2.0] ], columns=["qid", "docno", "score"])
        qrels = pd.DataFrame([["q1", "d1", 1], ["q2", "d1", 2] ], columns=["qid", "docno", "label"])
        df = pt.Experiment(
                [res1],
                topics,
                qrels,
                [
                    pt.measures.AP(rel=2),
                ])
        self.assertEqual(0.5, df.iloc[0]["AP(rel=2)"])
        df = pt.Experiment(
                [res1],
                topics,
                qrels,
                [
                    pt.measures.AP,
                    pt.measures.AP(rel=2),
                    pt.measures.P(rel=2)@1
                ])
        print(df.columns)
        self.assertEqual(1, df.iloc[0]["AP"])
        self.assertEqual(0.5, df.iloc[0]["AP(rel=2)"])
        self.assertEqual(0.5, df.iloc[0]["P(rel=2)@1"])
        df = pt.Experiment(
                [res1],
                topics,
                qrels,
                [
                    pt.measures.AP,
                    pt.measures.AP(rel=2),
                    pt.measures.P(rel=2)@1
                ],
                perquery=True)
        self.assertEqual(1, df[(df.measure == "AP") & (df.qid == "q1")].value.iloc[0])
        self.assertEqual(1, df[(df.measure == "AP") & (df.qid == "q2")].value.iloc[0])
        self.assertEqual(0, df[(df.measure == "AP(rel=2)") & (df.qid == "q1")].value.iloc[0])
        self.assertEqual(1, df[(df.measure == "AP(rel=2)") & (df.qid == "q2")].value.iloc[0])
        self.assertEqual(0, df[(df.measure == "P(rel=2)@1") & (df.qid == "q1")].value.iloc[0])
        self.assertEqual(1, df[(df.measure == "P(rel=2)@1") & (df.qid == "q2")].value.iloc[0])

    def test_differing_queries(self):
        topics = pd.DataFrame([["q1", "q1"], ["q2", "q1"] ], columns=["qid", "query"])
        res1 = pd.DataFrame([["q1", "d1", 1.0, 0]], columns=["qid", "docno", "score", "rank"])
        res2 = pd.DataFrame([["q1", "d1", 1.0, 0], ["q2", "d1", 2.0, 0] ], columns=["qid", "docno", "score", "rank"])
        qrels = pd.DataFrame([["q1", "d1", 1], ["q2", "d1", 1] ], columns=["qid", "docno", "label"])
        from pyterrier.transformer import UniformTransformer
        with warnings.catch_warnings(record=True) as w:
            pt.Experiment(
                [UniformTransformer(res1), UniformTransformer(res2)],
                topics,
                qrels,
                ["map"],
                baseline=0)
            self.assertTrue("missing" in str(w[-1].message))

    def test_mrt(self):
        brs = [
            pt.BatchRetrieve(pt.datasets.get_dataset("vaswani").get_index(), wmodel="DPH"), 
            pt.BatchRetrieve(pt.datasets.get_dataset("vaswani").get_index(), wmodel="BM25")
        ]
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels =  pt.datasets.get_dataset("vaswani").get_qrels()
        pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"])
        pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], highlight="color")
        pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], baseline=0, highlight="color")

    def test_various_metrics(self):
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        res = [
            pt.BatchRetrieve(pt.datasets.get_dataset("vaswani").get_index(), wmodel="DPH")(topics), 
            pt.BatchRetrieve(pt.datasets.get_dataset("vaswani").get_index(), wmodel="BM25")(topics)
        ]
        
        qrels =  pt.datasets.get_dataset("vaswani").get_qrels()
        # what we ask for -> what we should get as a metric
        family2measure = {
            'ndcg_cut_5' : 'ndcg_cut_5',
            'P' : "P@5",
            'P_5' : "P_5",
            "iprec_at_recall" : "IPrec@0.0",
            "official" : "AP",
            "set" : "SetP",
            "recall" : "R@5",
            "recall_1000" : "recall_1000"
        }
        # what we ask for -> what we should NOT get
        family2black = {
            'ndcg_cut_5' : 'ndcg_cut_10',
            'P_5' : "P_100",
            "recall_1000" : "recall_5"
        }
        for m in family2measure:
            df1 = pt.Experiment(res, topics, qrels, eval_metrics=[m])
            df2 = pt.Experiment(res, topics, qrels, eval_metrics=[m], baseline=0)
            df3 = pt.Experiment(res, topics, qrels, eval_metrics=[m], perquery=True)
            self.assertIn(family2measure[m], df1.columns)
            self.assertIn(family2measure[m], df2.columns)
            self.assertTrue(len(df3[df3["measure"] == family2measure[m]])>0)

            # check that we dont get back measures that we did NOT ask for
            if m in family2black:
                self.assertNotIn(family2black[m], df1.columns)
                self.assertNotIn(family2black[m], df2.columns)
                self.assertTrue(len(df3[df3["measure"] == family2black[m]])==0)

    def test_differing_order(self):
        topics = pd.DataFrame([["q1", "q1"], ["q2", "q1"] ], columns=["qid", "query"])
        res1 = pd.DataFrame([ ["q2", "d1", 2.0, 0], ["q1", "d1", 1.0, 1],], columns=["qid", "docno", "score", "rank"])
        res2 = pd.DataFrame([["q1", "d1", 1.0, 1], ["q2", "d1", 2.0, 0] ], columns=["qid", "docno", "score", "rank"])
        qrels = pd.DataFrame([["q1", "d1", 1], ["q2", "d3", 1] ], columns=["qid", "docno", "label"])
        from pyterrier.transformer import UniformTransformer
        measures = pt.Experiment(
                [UniformTransformer(res1), UniformTransformer(res2)],
                topics,
                qrels,
                ["map"],
                baseline=0)
        self.assertEqual(measures.iloc[0]["map"], 0.5)
        self.assertEqual(measures.iloc[1]["map"], 0.5)
        self.assertEqual(measures.iloc[1]["map +"], 0)
        self.assertEqual(measures.iloc[1]["map -"], 0)

    def test_one_row(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q"])
        self.assertEqual(10, rtr.iloc[0]["num_q"])
        
        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], dataframe=False)
        
        with warnings.catch_warnings(record=True) as w:
            rtr = pt.Experiment(vaswani.get_topics().head(10), [br], ["map", "ndcg"], vaswani.get_qrels(), dataframe=False)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "Signature" in str(w[-1].message)

    def test_one_row_round(self):
        import pyterrier as pt
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q", pt.measures.NDCG@5], round=2)
        self.assertEqual(str(rtr.iloc[0]["map"]), "0.31")
        self.assertEqual(str(rtr.iloc[0]["nDCG@5"]), "0.46")

        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q", pt.measures.NDCG@5], round={"nDCG@5" : 1})
        self.assertEqual(str(rtr.iloc[0]["nDCG@5"]), "0.5")

    def test_batching(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr1 = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q", "mrt"])
        rtr2 = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q", "mrt"], batch_size=2)
        self.assertTrue("mrt" in rtr1.columns)
        self.assertTrue("mrt" in rtr2.columns)
        rtr1.drop(columns=["mrt"], inplace=True)
        rtr2.drop(columns=["mrt"], inplace=True)
        pd.testing.assert_frame_equal(rtr1, rtr2)
        
        rtr1 = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q"], perquery=True)
        rtr2 = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q"], batch_size=2, perquery=True)
        pd.testing.assert_frame_equal(rtr1, rtr2)

    def test_perquery(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], perquery=True)
        print(rtr)

        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], perquery=True, dataframe=False)
        print(rtr)

    def test_perquery_round(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg"], perquery=True, round=2)
        self.assertEqual(str(rtr.iloc[0]["value"]), "0.36")

    def test_baseline_and_tests(self):
        dataset = pt.get_dataset("vaswani")
        numt=10
        res1 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")(dataset.get_topics().head(numt))
        res2 = pt.BatchRetrieve(dataset.get_index(), wmodel="DPH")(dataset.get_topics().head(numt))

        # t-test
        df = pt.Experiment(
            [res1, res2], 
            dataset.get_topics().head(numt), 
            dataset.get_qrels(),
            eval_metrics=["map", "ndcg"], 
            baseline=0)
        self.assertTrue("map +" in df.columns)
        self.assertTrue("map -" in df.columns)
        self.assertTrue("map p-value" in df.columns)

        # wilcoxon signed-rank test
        df = pt.Experiment(
            [res1, res2], 
            dataset.get_topics().head(numt), 
            dataset.get_qrels(),
            eval_metrics=["map", "ndcg"], 
            test='wilcoxon', 
            baseline=0)
        self.assertTrue("map +" in df.columns)
        self.assertTrue("map -" in df.columns)
        self.assertTrue("map p-value" in df.columns)


        # user-specified TOST
        # TOST will omit warnings here, due to low numbers of topics
        import statsmodels.stats.weightstats
        fn = lambda X,Y: (0, statsmodels.stats.weightstats.ttost_ind(X, Y, -0.01, 0.01)[0])
        
        #This filter doesnt work
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            df = pt.Experiment(
                [res1, res2], 
                dataset.get_topics().head(numt), 
                dataset.get_qrels(),
                eval_metrics=["map", "ndcg"], 
                test=fn,
                baseline=0)
            print(w)
        self.assertTrue("map +" in df.columns)
        self.assertTrue("map -" in df.columns)
        self.assertTrue("map p-value" in df.columns)
        

    def test_baseline_correction_userdefined_test(self):
        dataset = pt.get_dataset("vaswani")
        res1 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")(dataset.get_topics().head(10))
        res2 = pt.BatchRetrieve(dataset.get_index(), wmodel="DPH")(dataset.get_topics().head(10))
        # TOST will omit warnings here, due to low numbers of topics
        import statsmodels.stats.weightstats
        fn = lambda X,Y: (0, statsmodels.stats.weightstats.ttost_ind(X, Y, -0.01, 0.01)[0])
        for corr in ['hs', 'bonferroni', 'holm-sidak']:            
            df = pt.Experiment(
                [res1, res2], 
                dataset.get_topics().head(10), 
                dataset.get_qrels(),
                eval_metrics=["map", "ndcg"], 
                baseline=0, correction='hs', test=fn)
            self.assertTrue("map +" in df.columns)
            self.assertTrue("map -" in df.columns)
            self.assertTrue("map p-value" in df.columns)
            self.assertTrue("map p-value corrected" in df.columns)
            self.assertTrue("map reject" in df.columns)

    def test_baseline_corrected(self):
        dataset = pt.get_dataset("vaswani")
        res1 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")(dataset.get_topics().head(10))
        res2 = pt.BatchRetrieve(dataset.get_index(), wmodel="DPH")(dataset.get_topics().head(10))
        for corr in ['hs', 'bonferroni', 'holm-sidak']:            
            df = pt.Experiment(
                [res1, res2], 
                dataset.get_topics().head(10), 
                dataset.get_qrels(),
                eval_metrics=["map", "ndcg"], 
                baseline=0, correction='hs')
            self.assertTrue("map +" in df.columns)
            self.assertTrue("map -" in df.columns)
            self.assertTrue("map p-value" in df.columns)
            self.assertTrue("map p-value corrected" in df.columns)
            self.assertTrue("map reject" in df.columns)
