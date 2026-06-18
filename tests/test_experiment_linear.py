import pandas as pd
import pyterrier as pt
import os
import pytest
from .test_experiment_base import TestExperimentBase
import ir_measures


class TestExperimentLinear(TestExperimentBase):
    """Test suite for PyTerrier Experiment functionality specific to linear execution plans."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_exp_kwargs = {'plan': 'linear'}

    def test_precomp_common(self):
        """Test the _identifyCommon utility for detecting common pipeline prefixes."""
        bm25 = pt.terrier.Retriever(self._vaswani_index(), wmodel='BM25')
        pipeA = bm25 % 3
        pipeB = bm25 % 10
        import pyterrier._evaluation._execution
        common, suffices = pyterrier._evaluation._exec_linear._identifyCommon([pipeA, pipeB])
        self.assertEqual(bm25, common)
        self.assertIsInstance(suffices[0], pt.RankCutoff)
        self.assertEqual(3, suffices[0].k)
        self.assertIsInstance(suffices[1], pt.RankCutoff)
        self.assertEqual(10, suffices[1].k)

        common, suffices = pyterrier._evaluation._exec_linear._identifyCommon([bm25, pipeB])
        self.assertEqual(bm25, common)
        self.assertIsInstance(suffices[0], pt.transformer.IdentityTransformer)
        self.assertIsInstance(suffices[1], pt.RankCutoff)
        self.assertEqual(10, suffices[1].k)
    
    def test_precompute_experiment(self):
        """Test precomputation of common pipeline prefix for experiment optimization."""
        bm25 = pt.terrier.Retriever(self._vaswani_index(), wmodel='BM25')
        pipeB = bm25 % 10
        df1 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], **self.pt_exp_kwargs)
        df2 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], precompute_prefix=True, **self.pt_exp_kwargs)
        pd.testing.assert_frame_equal(df1, df2)

        df3 = pt.Experiment([bm25, pipeB], pt.get_dataset('vaswani').get_topics().head(10), pt.get_dataset('vaswani').get_qrels(), eval_metrics=['map'], precompute_prefix=True, batch_size=4, **self.pt_exp_kwargs)
        pd.testing.assert_frame_equal(df1, df3)

    def test_save_invalid_format(self):
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels = pt.datasets.get_dataset("vaswani").get_qrels()
        t = pt.Transformer.from_df(pd.DataFrame([{'qid': topics.iloc[0].qid, 'query': 'hello', "context": "here"}]))
        with self.assertRaises(TypeError) as te:
            pt.Experiment([t], topics, qrels, ['map'], save_dir=self.test_dir, names=['t'], **self.pt_exp_kwargs)
            self.assertIn("missing ['docno', 'score', 'rank']" in te.msg)

        dummy_measure = ir_measures.define_byquery(lambda a, b: 0)
        import pickle
        pt.Experiment(
            [t],
            topics,
            qrels,
            [dummy_measure],
            save_dir=self.test_dir,
            names=['t'],
            save_format=pickle,
            **self.pt_exp_kwargs)

    def test_save(self):
        index = self._vaswani_index()
        brs = [
            pt.terrier.Retriever(index, wmodel="DPH"),
            pt.terrier.Retriever(index, wmodel="BM25")
        ]
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels = pt.datasets.get_dataset("vaswani").get_qrels()

        import pickle
        for name, format, ext in [
                ('trec', 'trec', 'res.gz'),
                ('pkl_manual', pickle, 'mod'),
                ('pandas', (pd.read_csv, pd.DataFrame.to_csv), 'custom')
            ]:
            with self.subTest(name):
                df1 = pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir, save_format=format, **self.pt_exp_kwargs)
                print("\n".join(os.listdir(self.test_dir)))
                self.assertTrue(os.path.exists(os.path.join(self.test_dir, "TerrierRetr(DPH)." + ext)), os.path.join(self.test_dir, "TerrierRetr(DPH)." + ext) + " not found")
                self.assertTrue(os.path.exists(os.path.join(self.test_dir, "TerrierRetr(BM25)." + ext)), os.path.join(self.test_dir, "TerrierRetr(BM25)." + ext) + " not found")

                with pytest.warns(UserWarning):
                    pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir, save_format=format, **self.pt_exp_kwargs)

                with self.assertRaises(ValueError):
                    pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir, save_mode='error', save_format=format, **self.pt_exp_kwargs)

                df2 = pt.Experiment(brs, topics, qrels, eval_metrics=["map", "mrt"], save_dir=self.test_dir, save_mode='reuse', save_format=format, **self.pt_exp_kwargs)
                self.assertTrue(df2.iloc[0]["mrt"] < df1.iloc[0]["mrt"])

    def test_save_csv(self):
        index = self._vaswani_index()
        brs = [
            pt.terrier.Retriever(index, wmodel="DPH"),
            pt.terrier.Retriever(index, wmodel="BM25")
        ]
        topics = pt.datasets.get_dataset("vaswani").get_topics().head(10)
        qrels = pt.datasets.get_dataset("vaswani").get_qrels()

        pt.Experiment(brs, topics, qrels, eval_metrics=["map"], save_dir=self.test_dir, names=["DPH", "BM25"], **self.pt_exp_kwargs)

        aggregated_path = os.path.join(self.test_dir, "aggregated.csv")
        perquery_path = os.path.join(self.test_dir, "perquery.csv")

        self.assertTrue(os.path.exists(aggregated_path), "aggregated.csv not found")
        self.assertTrue(os.path.exists(perquery_path), "perquery.csv not found")

        agg_df = pd.read_csv(aggregated_path)
        self.assertEqual(2, len(agg_df), "aggregated.csv should have one row per system")
        self.assertIn("name", agg_df.columns)
        self.assertIn("map", agg_df.columns)
        self.assertEqual({"DPH", "BM25"}, set(agg_df["name"].tolist()))

        pq_df = pd.read_csv(perquery_path)
        self.assertIn("name", pq_df.columns)
        self.assertIn("qid", pq_df.columns)
        self.assertIn("measure", pq_df.columns)
        self.assertIn("value", pq_df.columns)
        self.assertEqual(20, len(pq_df))

        pl2 = pt.terrier.Retriever(index, wmodel="PL2")
        pt.Experiment([pl2], topics, qrels, eval_metrics=["map"], save_dir=self.test_dir, names=["PL2"], **self.pt_exp_kwargs)

        agg_df2 = pd.read_csv(aggregated_path)
        self.assertEqual(3, len(agg_df2), "aggregated.csv should retain rows from previous runs")
        self.assertEqual({"DPH", "BM25", "PL2"}, set(agg_df2["name"].tolist()))

        pq_df2 = pd.read_csv(perquery_path)
        self.assertEqual(30, len(pq_df2))
        self.assertEqual({"DPH", "BM25", "PL2"}, set(pq_df2["name"].tolist()))

        pt.Experiment([brs[1]], topics, qrels, eval_metrics=["map"], save_dir=self.test_dir, names=["BM25"], save_mode="overwrite", **self.pt_exp_kwargs)

        agg_df3 = pd.read_csv(aggregated_path)
        self.assertEqual(3, len(agg_df3), "re-running BM25 must not create duplicate rows")
        self.assertEqual({"DPH", "BM25", "PL2"}, set(agg_df3["name"].tolist()))
