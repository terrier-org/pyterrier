import pyterrier as pt

import unittest
import os
import shutil
import tempfile
from .base import TempDirTestCase

class TestTRECIndexer(TempDirTestCase):

    def test_TREC_indexing_pbar(self):
        print("Writing index to " + self.test_dir)
        indexer = pt.TRECCollectionIndexer(self.test_dir, verbose=True)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing(self):
        print("Writing index to " + self.test_dir)
        indexer = pt.TRECCollectionIndexer(self.test_dir)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing_text(self):
        print("Writing index to " + self.test_dir)
        indexer = pt.TRECCollectionIndexer(
            self.test_dir,
            meta= {'docno' : 26, 'body' : 2048},
            # The tags from which to save the text. ELSE is special tag name, which means anything not consumed by other tags.
            meta_tags = {'body' : 'ELSE'}
            )
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertTrue("body" in index.getMetaIndex().getKeys())
        self.assertTrue("compact memories have flexible capacities" in index.getMetaIndex().getItem("body", 0) )
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing_singlepass(self):
        print("Writing index to " + self.test_dir)
        indexer = pt.TRECCollectionIndexer(self.test_dir, type=pt.IndexingType.SINGLEPASS)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_TREC_indexing_memory(self):
        print("Writing index to " + self.test_dir)
        indexer = pt.TRECCollectionIndexer(self.test_dir, type=pt.IndexingType.MEMORY)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())