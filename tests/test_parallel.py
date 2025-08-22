from .base import BaseTestCase, parallel_test
import pyterrier as pt
class TestParallel(BaseTestCase):

    def skip_py_311_or_newer(self):
        import sys
        if sys.version >= (3,11):
            self.skipTest("Problems with recent python")

    @parallel_test
    def test_parallel_joblib_experiment(self):
        self.skip_windows()
        self.skip_py_311_or_newer()
        dataset = pt.get_dataset("vaswani")
        br = pt.terrier.Retriever(dataset.get_index())
        df = pt.Experiment(
            [br, br.parallel(3)],
            dataset.get_topics(),
            dataset.get_qrels(),
            ["map", "mrt"]
        )
        self.assertEqual(df.iloc[0]["map"], df.iloc[1]["map"])

    @parallel_test
    def test_parallel_joblib_experiment_br_callback(self):
        self.skip_windows()
        self.skip_py_311_or_newer()
        dataset = pt.get_dataset("vaswani")
        Tf = lambda keyFreq, posting, entryStats, collStats: posting.getFrequency()
        br = pt.terrier.Retriever(dataset.get_index(), wmodel=Tf)
        df = pt.Experiment(
            [br, br.parallel(3)],
            dataset.get_topics().head(4),
            dataset.get_qrels(),
            ["map", "mrt"]
        )
        self.assertEqual(df.iloc[0]["map"], df.iloc[1]["map"])

    @parallel_test
    def test_parallel_joblib_ops(self):
        self.skip_windows()
        self.skip_py_311_or_newer()
        dataset = pt.get_dataset("vaswani")
        topics = dataset.get_topics().head(3)
        dph = pt.terrier.Retriever(dataset.get_index())
        tf = pt.terrier.Retriever(dataset.get_index(), wmodel="Tf")
        for pipe in [
            dph,
            dph % 10,
            dph >> tf,
            dph + tf,
            pt.apply.query(lambda row: row["query"] + " chemical") >> dph
        ]:
            res1 = pipe(topics)
            res2 = pipe.parallel(3)(topics)
            self.assertEqual(len(res1), len(res2))
            
