import pyterrier as pt

import unittest
import os
import shutil
import tempfile


class TestTRECIndexer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTRECIndexer, self).__init__(*args, **kwargs)
        if not pt.started():
            #pt.init(logging="DEBUG")
            pt.init()
        # else:
        #     pt.setup_logging("DEBUG")
        self.here = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_TREC_indexing_pbar(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir, verbose=True)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing_singlepass(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir, type=pt.IndexingType.SINGLEPASS)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing_memory(self):
        indexer = pt.TRECCollectionIndexer(self.test_dir, type=pt.IndexingType.MEMORY)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())