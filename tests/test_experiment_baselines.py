"""CPU-only Experiment baseline tests. No JVM, index, or dataset download."""
import unittest

import pandas as pd
import pyterrier as pt


def _synthetic_baseline_frames():
    topics = pd.DataFrame({
        "qid": ["q1", "q2", "q3", "q4"],
        "query": ["one", "two", "three", "four"],
    })
    qrels = pd.DataFrame({
        "qid": ["q1", "q2", "q3", "q4"],
        "docno": ["rel", "rel", "rel", "rel"],
        "label": [1, 1, 1, 1],
    })
    # AP is 1 when the relevant doc is retrieved, else 0 (one relevant per query).
    system_0 = pd.DataFrame({
        "qid": ["q1", "q2", "q3", "q4"],
        "docno": ["rel", "rel", "rel", "rel"],
        "score": [1.0, 1.0, 1.0, 1.0],
        "rank": [0, 0, 0, 0],
    })
    system_1 = pd.DataFrame({
        "qid": ["q1", "q2", "q3", "q4"],
        "docno": ["rel", "rel", "other", "other"],
        "score": [1.0, 1.0, 1.0, 1.0],
        "rank": [0, 0, 0, 0],
    })
    system_2 = pd.DataFrame({
        "qid": ["q1", "q2", "q3", "q4"],
        "docno": ["rel", "other", "other", "other"],
        "score": [1.0, 1.0, 1.0, 1.0],
        "rank": [0, 0, 0, 0],
    })
    return topics, qrels, [system_0, system_1, system_2]


class TestExperimentBaselines(unittest.TestCase):

    def test_experiment_single_baseline_default_columns(self):
        topics, qrels, runs = _synthetic_baseline_frames()
        df = pt.Experiment(
            runs, topics, qrels, ["map"], names=["system_0", "system_1", "system_2"],
            baseline=0)
        self.assertIn("map +", df.columns)
        self.assertIn("map -", df.columns)
        self.assertIn("map p-value", df.columns)
        self.assertNotIn("map + (vs system_0)", df.columns)
        self.assertTrue(pd.isna(df.iloc[0]["map +"]))
        self.assertEqual(0, df.iloc[1]["map +"])
        self.assertEqual(2, df.iloc[1]["map -"])

        df_list = pt.Experiment(
            runs, topics, qrels, ["map"], names=["system_0", "system_1", "system_2"],
            baseline=[0])
        self.assertIn("map +", df_list.columns)
        self.assertNotIn("map + (vs system_0)", df_list.columns)

    def test_experiment_multiple_baselines_int(self):
        topics, qrels, runs = _synthetic_baseline_frames()
        df = pt.Experiment(
            runs, topics, qrels, ["map"], names=["system_0", "system_1", "system_2"],
            baseline=[0, 1])
        self.assertIn("map + (vs system_0)", df.columns)
        self.assertIn("map - (vs system_0)", df.columns)
        self.assertIn("map p-value (vs system_0)", df.columns)
        self.assertIn("map + (vs system_1)", df.columns)
        self.assertIn("map - (vs system_1)", df.columns)
        self.assertIn("map p-value (vs system_1)", df.columns)
        self.assertNotIn("map +", df.columns)
        self.assertTrue(pd.isna(df.iloc[0]["map + (vs system_0)"]))
        self.assertTrue(pd.isna(df.iloc[1]["map + (vs system_1)"]))
        self.assertEqual(0, df.iloc[1]["map + (vs system_0)"])
        self.assertEqual(2, df.iloc[1]["map - (vs system_0)"])

    def test_experiment_multiple_baselines_named(self):
        topics, qrels, runs = _synthetic_baseline_frames()
        systems = {"system_0": runs[0], "system_1": runs[1], "system_2": runs[2]}
        df = pt.Experiment(
            systems, topics, qrels, ["map"],
            baseline=["system_0", "system_1"])
        self.assertIn("map p-value (vs system_0)", df.columns)
        self.assertIn("map p-value (vs system_1)", df.columns)
        self.assertEqual(["system_0", "system_1", "system_2"], df["name"].tolist())

    def test_experiment_multiple_baselines_correction(self):
        topics, qrels, runs = _synthetic_baseline_frames()
        df = pt.Experiment(
            runs, topics, qrels, ["map"], names=["system_0", "system_1", "system_2"],
            baseline=[0, 1], correction="bonferroni")
        self.assertIn("map p-value (vs system_0)", df.columns)
        self.assertIn("map reject (vs system_0)", df.columns)
        self.assertIn("map p-value (vs system_0) corrected", df.columns)
        self.assertIn("map reject (vs system_1)", df.columns)
        self.assertIn("map p-value (vs system_1) corrected", df.columns)
        compared = df["map p-value (vs system_0) corrected"].drop(df.index[0])
        self.assertFalse(compared.isna().all())

    def test_experiment_render_multi_baseline_correction_family(self):
        from scipy import stats

        from pyterrier._evaluation._rendering import RenderFromPerQuery

        r = RenderFromPerQuery(
            ["system_0", "system_1", "system_2"],
            baseline=[0, 1],
            test_fn=stats.ttest_rel,
            correction="bonferroni",
        )
        r.add_metrics(0, {"q1": {"map": 1.0}, "q2": {"map": 1.0}, "q3": {"map": 1.0}, "q4": {"map": 1.0}}, 0.0)
        r.add_metrics(1, {"q1": {"map": 1.0}, "q2": {"map": 1.0}, "q3": {"map": 0.0}, "q4": {"map": 0.0}}, 0.0)
        r.add_metrics(2, {"q1": {"map": 1.0}, "q2": {"map": 0.0}, "q3": {"map": 0.0}, "q4": {"map": 0.0}}, 0.0)
        df = r.averages()
        self.assertIn("map p-value (vs system_0)", df.columns)
        self.assertIn("map p-value (vs system_0) corrected", df.columns)
        self.assertIn("map reject (vs system_1)", df.columns)

    def test_experiment_resolve_baseline_errors(self):
        topics, qrels, runs = _synthetic_baseline_frames()
        with self.assertRaises(ValueError):
            pt.Experiment(runs, topics, qrels, ["map"], names=["a", "b", "c"], baseline=[])
        with self.assertRaises(ValueError):
            pt.Experiment(runs, topics, qrels, ["map"], names=["a", "b", "c"], baseline=[0, 0])
        with self.assertRaises(ValueError):
            pt.Experiment(runs, topics, qrels, ["map"], names=["a", "b", "c"], baseline=5)
        with self.assertRaises(ValueError):
            pt.Experiment(
                {"a": runs[0], "b": runs[1]}, topics, qrels, ["map"],
                baseline=["a", "missing"])
