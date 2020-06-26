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

    def test_cache(self):
        pt.cache.CACHE_DIR = self.test_dir
        import pandas as pd
        queries = pd.DataFrame([["q1", "chemical"]], columns=["qid", "query"])
        br = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index())
        cache = ~br
        self.assertEqual(0, len(cache.chest._keys))
        cache(queries)
        cache(queries)
        self.assertEqual(0.5, cache.stats())
        pt.cache.CACHE_DIR = None
