import pandas as pd
import pyterrier as pt
import unittest
import os
from .base import TempDirTestCase
import shutil
import tempfile

class TestUtils(TempDirTestCase):


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

    def test_finalized_open(self):
        self.assertFalse(os.path.exists('file.txt'))
        self.assertFalse(os.path.exists('file.tmp.txt'))

        try:
            with pt.io.finalized_open('file.txt', 't') as f:
                f.write('OK')
                self.assertFalse(os.path.exists('file.txt'))
                self.assertTrue(os.path.exists('file.tmp.txt'))
                raise Exception("test")
        except Exception as e:
            if e.args[0] != 'test':
                raise # raise any error but our test error
        # File *doesn't* exist
        self.assertFalse(os.path.exists('file.txt'))
        self.assertFalse(os.path.exists('file.tmp.txt'))

        with pt.io.finalized_open('file.txt', 't') as f:
            f.write('Guess who\'s back')
            self.assertFalse(os.path.exists('file.txt'))
            self.assertTrue(os.path.exists('file.tmp.txt'))
        # File *does* exist
        self.assertTrue(os.path.exists('file.txt'))
        self.assertFalse(os.path.exists('file.tmp.txt'))
        with open('file.txt', 'rt') as f:
            self.assertEqual(f.read(), 'Guess who\'s back')

        with pt.io.finalized_open('file.txt', 't') as f:
            f.write('Shady\'s back')
            self.assertTrue(os.path.exists('file.txt'))
            self.assertTrue(os.path.exists('file.tmp.txt'))
            with open('file.txt', 'rt') as f:
                self.assertEqual(f.read(), 'Guess who\'s back')
        # contents *are* updated
        self.assertTrue(os.path.exists('file.txt'))
        self.assertFalse(os.path.exists('file.tmp.txt'))
        with open('file.txt', 'rt') as f:
            self.assertEqual(f.read(), 'Shady\'s back')

        try:
            with pt.io.finalized_open('file.txt', 't') as f:
                f.write('Back again')
                self.assertTrue(os.path.exists('file.txt'))
                self.assertTrue(os.path.exists('file.tmp.txt'))
                with open('file.txt', 'rt') as f:
                    self.assertEqual(f.read(), 'Shady\'s back')
                raise Exception("test")
        except Exception as e:
            if e.args[0] != 'test':
                raise # raise any error but our test error
        # contents *aren't* updated
        self.assertTrue(os.path.exists('file.txt'))
        self.assertFalse(os.path.exists('file.tmp.txt'))
        with open('file.txt', 'rt') as f:
            self.assertEqual(f.read(), 'Shady\'s back')

    def test_finalized_autoopen(self):
        self.assertFalse(os.path.exists('file.gz'))
        self.assertFalse(os.path.exists('file.tmp.gz'))

        try:
            with pt.io.finalized_autoopen('file.gz', 't') as f:
                f.write('OK')
                self.assertFalse(os.path.exists('file.gz'))
                self.assertTrue(os.path.exists('file.tmp.gz'))
                raise Exception("test")
        except Exception as e:
            if e.args[0] != 'test':
                raise # raise any error but our test error
        # File *doesn't* exist
        self.assertFalse(os.path.exists('file.gz'))
        self.assertFalse(os.path.exists('file.tmp.gz'))

        with pt.io.finalized_autoopen('file.gz', 't') as f:
            f.write('Guess who\'s back')
            self.assertFalse(os.path.exists('file.gz'))
            self.assertTrue(os.path.exists('file.tmp.gz'))
        # File *does* exist
        self.assertTrue(os.path.exists('file.gz'))
        self.assertFalse(os.path.exists('file.tmp.gz'))
        with pt.io.autoopen('file.gz', 'rt') as f:
            self.assertEqual(f.read(), 'Guess who\'s back')

        with pt.io.finalized_autoopen('file.gz', 't') as f:
            f.write('Shady\'s back')
            self.assertTrue(os.path.exists('file.gz'))
            self.assertTrue(os.path.exists('file.tmp.gz'))
            with pt.io.autoopen('file.gz', 'rt') as f:
                self.assertEqual(f.read(), 'Guess who\'s back')
        # contents *are* updated
        self.assertTrue(os.path.exists('file.gz'))
        self.assertFalse(os.path.exists('file.tmp.gz'))
        with pt.io.autoopen('file.gz', 'rt') as f:
            self.assertEqual(f.read(), 'Shady\'s back')

        try:
            with pt.io.finalized_autoopen('file.gz', 't') as f:
                f.write('Back again')
                self.assertTrue(os.path.exists('file.gz'))
                self.assertTrue(os.path.exists('file.tmp.gz'))
                with pt.io.autoopen('file.gz', 'rt') as f:
                    self.assertEqual(f.read(), 'Shady\'s back')
                raise Exception("test")
        except Exception as e:
            if e.args[0] != 'test':
                raise # raise any error but our test error
        # contents *aren't* updated
        self.assertTrue(os.path.exists('file.gz'))
        self.assertFalse(os.path.exists('file.tmp.gz'))
        with pt.io.autoopen('file.gz', 'rt') as f:
            self.assertEqual(f.read(), 'Shady\'s back')

    def tearDown(self):
        for file in ['file.txt', 'file.tmp.txt', 'file.gz', 'file.tmp.gz']:
            if os.path.exists(file):
                os.remove(file)

    

if __name__ == "__main__":
    unittest.main()
