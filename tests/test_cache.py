import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import tempfile
import shutil
import os

def compare(df1, df2):
    df1 = df1.sort_values(["qid", "rank"])
    df2 = df2.sort_values(["qid", "rank"])
    import numpy as np
    for i, (rowA, rowB) in enumerate( zip(df1.itertuples(), df2.itertuples())):
        for col in ["qid", "query", "docno", "score", "rank"]:
            assert getattr(rowA, col) ==  getattr(rowB, col), (i,col, rowA, rowB)
        if hasattr(rowA, "features") or hasattr(rowB, "features"):
            assert np.array_equal(getattr(rowA, "features"),  getattr(rowB, "features"), (i,"features", rowA, rowB))
    return True


class TestCache(BaseTestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_complex(self):
        pt.cache.CACHE_DIR = self.test_dir
        dataset = pt.get_dataset("vaswani")
        index = dataset.get_index()
        firstpassUB = pt.BatchRetrieve(index, wmodel="PL2")
        features = [
            "SAMPLE", #ie PL2
            "WMODEL:BM25",
        ]
        stdfeatures = pt.FeaturesBatchRetrieve(index, features)
        stage12 = firstpassUB >> stdfeatures
        CfirstpassUB = ~firstpassUB
        Cstdfeatures = ~stdfeatures
        Cstdfeatures.on=['qid', 'docno']
        Cstage12 = CfirstpassUB >> Cstdfeatures
        COstage12 = ~stage12

        num_topics = 5
        test_topics = dataset.get_topics().head(num_topics)

        #res0 is the ground truth
        res0 = stage12(test_topics)
        Cstage12(test_topics)
        res1 = Cstage12(test_topics).reset_index(drop=True)
        self.assertEqual(num_topics, Cstage12[0].hits)
        COstage12(test_topics)
        res2 = COstage12(test_topics)
        self.assertEqual(num_topics, COstage12.hits)

        self.assertTrue(compare(res1, res0))
        self.assertTrue(compare(res2, res0))

    def test_cache_reranker(self):
        pt.cache.CACHE_DIR = self.test_dir
        class MyT(pt.transformer.TransformerBase):
            def transform(self, docs):
                docs = docs.copy()
                docs["score"] = docs.apply(lambda doc_row: len(doc_row["text"]), axis=1)
                return pt.model.add_ranks(docs)
            def __repr__(self):
                return "MyT"
        p = MyT()
        testDF = pd.DataFrame([["q1", "hello", "d1", "aa"]], columns=["qid", "query", "docno", "text"])
        rtr = p(testDF)
        
        cached = ~p
        cached.on = ["qid", "text"]
        #cached.debug = True
        rtr2 = cached(testDF)
        self.assertTrue(rtr.equals(rtr2))
        rtr3 = cached(testDF)
        #print(rtr)
        #print(rtr3)
        self.assertTrue(rtr.equals(rtr3))
        self.assertEqual(cached.requests, 2)
        self.assertEqual(cached.hits, 1)

    def test_cache_br(self):
        pt.cache.CACHE_DIR = self.test_dir
        import pandas as pd
        queries = pd.DataFrame([["q1", "chemical"]], columns=["qid", "query"])
        br = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index())
        cache = ~br
        self.assertEqual(0, len(cache.chest.keys()))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())
        cache.close()

        #lets see if another cache of the same object would see the same cache entries.
        cache2 = ~br
        cache2(queries)
        self.assertEqual(1, cache2.stats())

        pt.cache.CACHE_DIR = None

    def test_cache_compose(self):
        pt.cache.CACHE_DIR = self.test_dir
        import pandas as pd
        queries = pd.DataFrame([["q1", "chemical"]], columns=["qid", "query"])
        br1 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="TF_IDF")
        br2 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
        cache = ~ (br1 >> br2)
        self.assertEqual(0, len(cache.chest.keys()))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())
        del(cache)

        #lets see if another cache of the same object would see the same cache entries.
        cache2 = ~(br1 >> br2)
        cache2(queries)
        self.assertEqual(1, cache2.stats())

        pt.cache.CACHE_DIR = None

    def test_cache_compose_cache(self):
        pt.cache.CACHE_DIR = self.test_dir
        import pandas as pd
        queries = pd.DataFrame([["q1", "chemical"]], columns=["qid", "query"])
        br1 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="TF_IDF")
        br2 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
        cache = ~ (~br1 >> br2)
        self.assertEqual(0, len(cache.chest.keys()))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())

        #this is required for shelve
        cache.close()

        #lets see if another cache of the same object would see the same cache entries.
        cache2 = ~(~br1 >> br2)
        cache2.debug = True
        #print("found keys in cache 2")
        #print(cache2.chest.keys())
        cache2(queries)
        #print(cache2.hits)
        self.assertEqual(1, cache2.stats())
        
        # check that the cache report works
        all_report = pt.cache.list_cache()
        self.assertTrue(len(all_report) > 0)
        report = list(all_report.values())[0]
        self.assertTrue("transformer" in report)
        self.assertTrue("size" in report)
        self.assertTrue("lastmodified" in report)
        
        pt.cache.CACHE_DIR = None

