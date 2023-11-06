import pandas as pd
import pyterrier as pt
import os
import unittest
import warnings
import math
from pyterrier.measures import *
from .base import TempDirTestCase

class TestExperiment(TempDirTestCase):

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
        dataset = pt.get_dataset("vaswani")
        bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")
        qrels = pd.DataFrame({
            'qid':   ["1",     "1",    "2"],
            'docno': ["10703", "1056", "9374"],
            'label': [1,       1,      1],
        })
        topics = pd.DataFrame({
            'qid':   ["1",        "2",         "3"],
            'query': ["chemical", "reactions", "reaction"]
        })
        test_cases = [
            # perfect topic/qrel overlap, filter_by_qrels and filter_by_topics have no effect
            (['1', '2'], ['1', '2'], True,  True,   {'1': 1.0, '2': 0.5}),
            (['1', '2'], ['1', '2'], True,  False,  {'1': 1.0, '2': 0.5}),
            (['1', '2'], ['1', '2'], False, True,   {'1': 1.0, '2': 0.5}),
            (['1', '2'], ['1', '2'], False, False,  {'1': 1.0, '2': 0.5}),
            # qid=2 missing from topics; qid=2 should only be included if filter_by_topics=False, filter_by_qrels has no effect
            (['1', '2'], ['1'], True,  True,   {'1': 1.0}),
            (['1', '2'], ['1'], True,  False,  {'1': 1.0, '2': 0.}),
            (['1', '2'], ['1'], False, True,   {'1': 1.0}),
            (['1', '2'], ['1'], False, False,  {'1': 1.0, '2': 0.}),
            # qid=2 missing from qrels; qid=2 should never be included in the results, '2' should be NaN if filter_by_qrels=False
            (['1'], ['1', '2'], True,  True,   {'1': 1.0}),
            (['1'], ['1', '2'], True,  False,  {'1': 1.0}),
            (['1'], ['1', '2'], False, True,   {'1': 1.0, '2': float('NaN')}),
            (['1'], ['1', '2'], False, False,  {'1': 1.0, '2': float('NaN')}),
            # qid=3 missing from qrels and qid=1 is missing from the topics; qid=1 should only be included if filter_by_topics=False
            (['1', '2'], ['2', '3'], True,  True,   {'2': 0.5}),
            (['1', '2'], ['2', '3'], True,  False,  {'1': 0.0, '2': 0.5}),
            (['1', '2'], ['2', '3'], False, True,   {'2': 0.5, '3': float('NaN')}),
            (['1', '2'], ['2', '3'], False, False,  {'1': 0.0, '2': 0.5, '3': float('NaN')}),
            # no qid overlap between topics and qrels; should throw exception if filter_by_topics=True
            (['1'], ['3'], True,  True,   ValueError('There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False.')),
            (['1'], ['3'], True,  False,  ValueError('There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False.')),
            (['1'], ['3'], False, True,   ValueError('There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False.')),
            (['1'], ['3'], False, False,  {'1': 0.0, '3': float('NaN')}),
        ]
        for qrel_qids, topic_qids, filter_by_qrels, filter_by_topics, result in test_cases:
            for batch_size in [1]:
                with self.subTest(f'qrel_qids={qrel_qids} topic_qids={topic_qids} filter_by_qrels={filter_by_qrels} filter_by_topics={filter_by_topics} batch_size={batch_size}'):
                    if isinstance(result, ValueError):
                        with self.assertRaises(ValueError) as context:
                            pt.Experiment([bm25], topics[topics.qid.isin(topic_qids)], qrels[qrels.qid.isin(qrel_qids)], [P@2, 'P', 'mrt'], filter_by_qrels=filter_by_qrels, filter_by_topics=filter_by_topics, perquery=True, batch_size=batch_size)
                        self.assertEqual(context.exception.args, result.args)
                        with self.assertRaises(ValueError) as context:
                            pt.Experiment([bm25], topics[topics.qid.isin(topic_qids)], qrels[qrels.qid.isin(qrel_qids)], [P@2, 'P', 'mrt'], filter_by_qrels=filter_by_qrels, filter_by_topics=filter_by_topics, perquery=False, batch_size=batch_size)
                        self.assertEqual(context.exception.args, result.args)
                    else:
                        with warnings.catch_warnings(record=True) as w:
                            res = pt.Experiment([bm25], topics[topics.qid.isin(topic_qids)], qrels[qrels.qid.isin(qrel_qids)], [P@2, 'P', 'mrt'], filter_by_qrels=filter_by_qrels, filter_by_topics=filter_by_topics, perquery=True, batch_size=batch_size)
                        if any(math.isnan(v) for v in result.values()):
                            self.assertEqual(len(w), 1)
                            self.assertEqual(w[0].message.args[0], f'1 topic(s) not found in qrels. Scores for these topics are given as NaN and should not contribute to averages.')
                        else:
                            self.assertEqual(len(w), 0)
                        res = res[res['measure'] == 'P@2'].drop(columns=['name', 'measure'])
                        expected_res = pd.DataFrame([{'qid': qid, 'value': val} for qid, val in result.items()])
                        pd.testing.assert_frame_equal(res.reset_index(drop=True), expected_res.reset_index(drop=True))
                        res = pt.Experiment([bm25], topics[topics.qid.isin(topic_qids)], qrels[qrels.qid.isin(qrel_qids)], [P@2, 'P', 'mrt'], filter_by_qrels=filter_by_qrels, filter_by_topics=filter_by_topics, perquery=False, batch_size=batch_size)
                        num_result = {k: v for k, v in result.items() if not math.isnan(v)}
                        self.assertEqual(res.loc[0, 'P@2'], sum(num_result.values())/len(num_result))

    def test_wrong(self):
        brs = [
            pt.BatchRetrieve(pt.datasets.get_dataset("vaswani").get_index(), wmodel="DPH"), 
            pt.BatchRetrieve(pt.datasets.get_dataset("vaswani").get_index(), wmodel="BM25")
        ]
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels =  pt.datasets.get_dataset("vaswani").get_qrels()
        with self.assertRaises(TypeError):
            pt.Experiment(brs, topics, qrels, eval_metrics=["map"], filter_qrels=True)
        

    def test_mrt(self):
        index = pt.datasets.get_dataset("vaswani").get_index()
        brs = [
            pt.BatchRetrieve(index, wmodel="DPH"), 
            pt.BatchRetrieve(index, wmodel="BM25")
        ]
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels =  pt.datasets.get_dataset("vaswani").get_qrels()
        measures = ["map", "mrt"]
        pt.Experiment(brs, topics, qrels, eval_metrics=measures)
        self.assertTrue("mrt" in measures)
        pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], highlight="color")
        pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], baseline=0, highlight="color")

    def test_save(self):
        index = pt.datasets.get_dataset("vaswani").get_index()
        brs = [
            pt.BatchRetrieve(index, wmodel="DPH"), 
            pt.BatchRetrieve(index, wmodel="BM25")
        ]
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels =  pt.datasets.get_dataset("vaswani").get_qrels()
        df1 = pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir)
        # check save_dir files are there
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "BR(DPH).res.gz")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "BR(BM25).res.gz")))
        with self.assertRaises(ValueError):
            # reuse only kicks in when save_mode is set.
            df2 = pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir)
        df2 = pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir, save_mode='reuse')
        # a successful experiment using save_dir should be faster
        self.assertTrue(df2.iloc[0]["mrt"] < df1.iloc[0]["mrt"])
        
    def test_empty(self):
        df1 = pt.new.ranked_documents([[1]]).head(0)
        t1 = pt.Transformer.from_df(df1)

        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels =  pt.datasets.get_dataset("vaswani").get_qrels()
        with self.assertRaises(ValueError):
            pt.Experiment([df1], topics, qrels, eval_metrics=["map", "mrt"])
        with self.assertRaises(ValueError):
            pt.Experiment([t1], topics, qrels, eval_metrics=["map", "mrt"])
        with self.assertRaises(ValueError):
            pt.Experiment([t1], topics, qrels, eval_metrics=["map", "mrt"], batch_size=2)
        

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
        res1 = pd.DataFrame([ ["q2", "d1", 2.0], ["q1", "d1", 1.0],], columns=["qid", "docno", "score"])
        res2 = pd.DataFrame([["q1", "d1", 1.0], ["q2", "d1", 2.0] ], columns=["qid", "docno", "score"])
        qrels = pd.DataFrame([["q1", "d1", 1], ["q2", "d3", 1] ], columns=["qid", "docno", "label"])
        measures = pt.Experiment(
                [pt.Transformer.from_df(res1, uniform=True), pt.Transformer.from_df(res2, uniform=True)],
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

    def test_bad_measure(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index())
        with self.assertRaises(KeyError):
            pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), [map])

    def test_baseline_and_tests(self):
        dataset = pt.get_dataset("vaswani")
        numt=10
        res1 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")(dataset.get_topics().head(numt))
        res2 = pt.BatchRetrieve(dataset.get_index(), wmodel="DPH")(dataset.get_topics().head(numt))

        # t-test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
