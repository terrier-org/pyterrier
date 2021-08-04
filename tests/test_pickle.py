from .base import TempDirTestCase
import pandas as pd
import pickle
import pyterrier as pt
import tempfile, shutil
from os import path
import os

class TestPickle(TempDirTestCase):

    def _fix_joblib(self):
        import joblib
        def dumps(obj):
            filename = path.join(self.test_dir, 'obj')
            f = open(filename, 'wb')
            joblib.dump(obj, f)
            f.close()
            f = open(filename, mode='rb')
            all_data = f.read()
            f.close()
            os.remove(filename)
            return all_data
        def loads(data):
            filename = path.join(self.test_dir, 'obj')
            f = open(filename, 'wb')
            f.write(data)
            f.close()
            obj = joblib.load(filename)
            os.remove(filename)
            return obj
        
        joblib.__dict__['dumps'] = dumps
        joblib.__dict__['loads'] = loads

    def _indexref(self, pickler):
        vaswani = pt.datasets.get_dataset("vaswani")
        ref = vaswani.get_index()
        b_ref = pickler.dumps(ref)
        ref2 = pickler.loads(b_ref)
        index = pt.IndexFactory.of(ref2)
        self.assertTrue(index.getCollectionStatistics().getNumberOfDocuments() > 0)
        index.close()

    def test_indexref_pickle(self):
        self._indexref(pickle)

    def test_indexref_joblib(self):
        import joblib
        self._fix_joblib()
        self._indexref(joblib)

    def _sourcetransformer(self, pickler):
        df = pd.DataFrame([["q1", "doc1", 5]], columns=["qid", "docno", "score"])
        t = pt.transformer.SourceTransformer(df)
        t_2 = pickler.loads(pickler.dumps(t))
        q  = pd.DataFrame([["q1", "query"]], columns=["qid", "query"])
        res = t_2(q)
        self.assertEqual(1, len(res))
        self.assertEqual("q1", res.iloc[0]["qid"])

    def test_sourcetransformer_pickle(self):
        self._sourcetransformer(pickle)

    def test_sourcetransformer_joblib(self):
        import joblib
        self._fix_joblib()
        self._sourcetransformer(joblib)

    def test_br_pickle(self):
        self._br(pickle)

    def test_br_dill_callback(self):
        import dill
        self._br(dill, wmodel=lambda keyFreq, posting, entryStats, collStats: posting.getFrequency())

    def test_br_pickle_callback(self):
        import pickle
        self._br(pickle, wmodel=lambda keyFreq, posting, entryStats, collStats: posting.getFrequency())

    def test_br_joblib_callback(self):
        import joblib
        self._fix_joblib()
        self._br(joblib, wmodel=lambda keyFreq, posting, entryStats, collStats: posting.getFrequency())

    def test_br_pickle_straightwmodel(self):
        self._br(pickle, wmodel=pt.autoclass("org.terrier.matching.models.BM25")())

    def test_br_joblib(self):
        import joblib
        self._fix_joblib()
        self._br(joblib)

    def test_fbr_pickle(self):
        self._fbr(pickle)

    def test_fbr_joblib(self):
        import joblib
        self._fix_joblib()
        self._fbr(joblib)

    def _br(self, pickler, wmodel='BM25'):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.BatchRetrieve(vaswani.get_index(), wmodel=wmodel, controls={"c" : 0.75}, num_results=15)
        q  = pd.DataFrame([["q1", "chemical"]], columns=["qid", "query"])
        res1 = br(q)
        br2 = pickler.loads(pickler.dumps(br))

        if isinstance(wmodel, str):
            self.assertEqual(wmodel, br2.controls["wmodel"])
        self.assertEqual(br.controls, br2.controls)
        self.assertEqual(br.properties, br2.properties)
        self.assertEqual(br.metadata, br2.metadata)

        self.assertEqual(repr(br), repr(br2))
        res2 = br2(q)
        
        pd.testing.assert_frame_equal(res1, res2)

    def _fbr(self, pickler):
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.FeaturesBatchRetrieve(vaswani.get_index(), wmodel="BM25", features=["WMODEL:DPH"], controls={"c" : 0.75}, num_results=15)
        q  = pd.DataFrame([["q1", "chemical"]], columns=["qid", "query"])
        res1 = br(q)
        br2 = pickler.loads(pickler.dumps(br))

        self.assertEqual("BM25", br2.controls["wmodel"])
        self.assertEqual(br.controls, br2.controls)
        self.assertEqual(br.properties, br2.properties)
        self.assertEqual(br.metadata, br2.metadata)
        self.assertEqual(br.features, br2.features)

        self.assertEqual(repr(br), repr(br2))
        res2 = br2(q)
        
        pd.testing.assert_frame_equal(res1, res2)
        