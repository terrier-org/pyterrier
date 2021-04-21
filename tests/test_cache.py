import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import tempfile
import shutil
import os

class TestCache(BaseTestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

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
        self.assertEqual(0, len(cache.chest._keys))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())

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
        self.assertEqual(0, len(cache.chest._keys))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())

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
        self.assertEqual(0, len(cache.chest._keys))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())

        #lets see if another cache of the same object would see the same cache entries.
        cache2 = ~(~br1 >> br2)
        cache2(queries)
        self.assertEqual(1, cache2.stats())
        
        # check that the cache report works
        all_report = pt.cache.list_cache()
        self.assertTrue(len(all_report) > 0)
        report = list(all_report.values())[0]
        self.assertEqual(1, report["queries"])
        self.assertTrue("transformer" in report)
        self.assertTrue("size" in report)
        self.assertTrue("lastmodified" in report)
        
        pt.cache.CACHE_DIR = None

