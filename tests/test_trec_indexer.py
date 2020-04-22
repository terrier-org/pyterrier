import pyterrier as pt

import unittest
import os
import shutil
import tempfile
from .base import BaseTestCase

class TestTRECIndexer(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_TREC_indexing(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir)
        indexRef = indexer.index(pt.Utils.get_files_in_dir(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
