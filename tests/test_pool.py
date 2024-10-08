from .base import BaseTestCase, parallel_test
import pandas as pd
import pickle
import pyterrier as pt
import datetime
class TestPool(BaseTestCase):

    @parallel_test
    def test_br_parallel(self):
        self.skip_windows()
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.terrier.Retriever(vaswani.get_index()) %2
        t = vaswani.get_topics()
        
        a = datetime.datetime.now()
        res = br(t)
        
        b = datetime.datetime.now()
        
        res2 = pt.parallel.PoolParallelTransformer(br, 3)(t)
        c = datetime.datetime.now()
        
        print("Sequential: %d" % ((b-a).total_seconds()) )
        print("Parallel: %d" % ((c-b).total_seconds()) )

        #indexes can differ, so we drop from both
        res = res.sort_values(["qid", "docno"]).reset_index(drop=True)
        res2 = res2.sort_values(["qid", "docno"]).reset_index(drop=True)
        self.assertEqual(len(res), len(res2))
        pd.testing.assert_frame_equal(res, res2)

    @parallel_test
    def test_br_multiprocess(self):
        return
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.terrier.Retriever(vaswani.get_index(), wmodel="BM25", controls={"c" : 0.75}, num_results=15)
        t = vaswani.get_topics().head()
        res1 = br(t)
        # Fortunately, there is a fork of the multiprocessing module called multiprocess that works just fine .... 
        # multiprocess uses dill instead of pickle to serialize Python objects. https://jstaf.github.io/hpc-python/parallel/
        from multiprocess import Pool

        with Pool(None, pt.java.parallel_init, pt.java.parallel_init_args(), 1) as pool:
            for res in pool.map(lambda topics : br(topics), [t, t, t]):
                pd.testing.assert_frame_equal(res1, res)

    @parallel_test
    def test_br_ray(self):
        self.skipTest("disabling ray")
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.terrier.Retriever(vaswani.get_index(), wmodel="BM25", controls={"c" : 0.75}, num_results=15)
        t = vaswani.get_topics().head()
        res1 = br(t).sort_values(["qid", "docno"])
        from ray.util.multiprocessing import Pool
        with Pool(4, pt.java.parallel_init, pt.java.parallel_init_args()) as pool:
            for res in pool.map(lambda topics : br(topics), [t, t, t]):
                res = res.sort_values(["qid", "docno"])
                pd.testing.assert_frame_equal(res1, res)

    @parallel_test
    def test_br_joblib(self):
        self.skip_windows()
        from pyterrier.parallel import _joblib_with_initializer

        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.terrier.Retriever(vaswani.get_index(), wmodel="BM25", controls={"c" : 0.75}, num_results=15)
        t = vaswani.get_topics().head()
        res1 = br(t).sort_values(["qid", "docno"])

        from joblib import Parallel, delayed
        with Parallel(n_jobs=2) as parallel:
            results = _joblib_with_initializer(parallel, pt.java.parallel_init, pt.java.parallel_init_args())(delayed(br)(topics) for topics in [t,t,t])
            self.assertTrue(3, len(results))
            for res in results:
                res = res.sort_values(["qid", "docno"])
                pd.testing.assert_frame_equal(res1, res)
