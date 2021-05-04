from .base import BaseTestCase
import pyterrier as pt
class TestParallel(BaseTestCase):

    def skip_windows(self):
        if TestParallel.is_windows():
            self.skipTest("Test disabled on Windows")

    @staticmethod
    def is_windows() -> bool:
        import platform
        return platform.system() == 'Windows'

    def test_parallel_joblib_experiment(self):
        self.skip_windows()
        dataset = pt.get_dataset("vaswani")
        br = pt.BatchRetrieve(dataset.get_index())
        df = pt.Experiment(
            [br, br.parallel(3)],
            dataset.get_topics(),
            dataset.get_qrels(),
            ["map", "mrt"]
        )
        self.assertEqual(df.iloc[0]["map"], df.iloc[1]["map"])

    def test_parallel_joblib_ops(self):
        self.skip_windows()
        dataset = pt.get_dataset("vaswani")
        topics = dataset.get_topics().head(3)
        dph = pt.BatchRetrieve(dataset.get_index())
        tf = pt.BatchRetrieve(dataset.get_index(), wmodel="Tf")
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
            