import pandas as pd
import pyterrier as pt
import unittest
import os
from .base import BaseTestCase
import shutil
import tempfile

class TestUtils(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_trec(self):
        res = pd.DataFrame([["1", "d1", 5.3, 1]], columns=['qid', 'docno', 'score', 'rank'])
        res_dict = res.set_index(['qid', 'docno']).to_dict()
        for filename in ["rtr.res", "rtr.res.gz", "rtr.res.bz2"]:
            filepath = os.path.join(self.test_dir, filename)
            pt.io.write_results(res, filepath, format="trec")
            res2 = pt.io.read_results(filepath)
            res2_dict = res2.set_index(['qid', 'docno']).to_dict()
            del res2_dict["name"]
            self.assertEqual(res_dict, res2_dict)

    def test_save_trec_generator(self):
        br = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="TF_IDF")
        filepath = os.path.join(self.test_dir, "test.res")
        pt.io.write_results(br.transform_gen(pt.get_dataset("vaswani").get_topics().head()), filepath, format="trec")

    def test_save_letor(self):
        import numpy as np
        res = pd.DataFrame([["1", "d1", 5.3, 1, np.array([1, 2])], ["1", "d2", 5.3, 1, np.array([2, 1])]], columns=['qid', 'docno', 'score', 'rank', 'features'])
        res_dict = res.set_index(['qid', 'docno']).to_dict()
        del res_dict["score"]
        del res_dict["rank"]
        for filename in ["rtr.letor", "rtr.letor.gz", "rtr.letor.bz2"]:
            filepath = os.path.join(self.test_dir, filename)
            pt.io.write_results(res, filepath, format="letor")
            res2 = pt.io.read_results(filepath, format="letor")

            for ((i1, row1), (i2, row2)) in zip(res.iterrows(), res2.iterrows()):
                self.assertEqual(row1["qid"], row2["qid"])
                self.assertEqual(row1["docno"], row2["docno"])
                self.assertEqual(row1["docno"], row2["docno"])
                self.assertEqual(row1["qid"], row2["qid"])
                self.assertTrue(np.array_equal(row1["features"], row2["features"]))        

    

if __name__ == "__main__":
    unittest.main()
