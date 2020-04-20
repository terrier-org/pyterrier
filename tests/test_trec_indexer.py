import pyterrier as pt

import unittest, math, os
import shutil, tempfile
from os import path

class TestTRECIndexer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTRECIndexer, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init(logging="DEBUG")
        # else:
        #     pt.setup_logging("DEBUG")
        self.here=os.path.dirname(os.path.realpath(__file__))

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