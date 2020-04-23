import pyterrier as pt

import unittest
import os
import shutil
import tempfile


class TestTRECIndexer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTRECIndexer, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init(logging="DEBUG")
        # else:
        #     pt.setup_logging("DEBUG")
        self.here = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_TREC_indexing(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir, single_pass=False)
        indexRef = indexer.index(pt.Utils.get_files_in_dir(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing_singlepass(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir, single_pass=True)
        indexRef = indexer.index(pt.Utils.get_files_in_dir(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))
