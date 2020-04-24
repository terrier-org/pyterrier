import pyterrier as pt

import unittest
import tempfile
import shutil
import os

from .base import BaseTestCase

class TestDFIndexer(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _create_index(self, df, type):
        pd_indexer = pt.DFIndexer(self.test_dir, type=type)
        indexref = pd_indexer.index(df["text"], df["docno"])
        self.assertIsNotNone(indexref)
        return indexref

    def _make_check_index(self, n, single_pass):
        import pandas as pd
        df = pd.DataFrame({
            'docno': ['1', '2', '3'],
            'url':
                ['url1', 'url2', 'url3'],
            'text':
                ['He ran out of money, so he had to stop playing',
                 'The waves were crashing on the shore; it was a',
                 'The body may perhaps compensates for the loss']
        })
        df = df.head(n)
        if single_pass:
            indexref = self._create_index(df, pt.IndexingType.SINGLEPASS)
        else:
            indexref = self._create_index(df, pt.IndexingType.CLASSIC)
        index = pt.IndexFactory.of(indexref)
        self.assertIsNotNone(index)
        self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue("docno" in index.getMetaIndex().getKeys())
        # self.assertTrue("url" in index.getMetaIndex().getKeys())
        if single_pass:
            self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))
        else:
            self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_checkjavaDocIteratr(self):
        import pandas as pd
        df = pd.DataFrame({
            'docno':
                ['1', '2', '3'],
            'url':
                ['url1', 'url2', 'url3'],
            'text':
                ['He ran out of money, so he had to stop playing',
                 'The waves were crashing on the shore; it was a',
                 'The body may perhaps compensates for the loss']
        })

        from pyterrier import DFIndexUtils

        d1 = df.head(1)
        jIter1 = DFIndexUtils.create_javaDocIterator(d1["text"], d1["docno"])
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

        d2 = df.head(2)
        jIter1 = DFIndexUtils.create_javaDocIterator(d2["text"], d2["docno"])
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

    def test_createindex1(self):
        self._make_check_index(1, single_pass=False)

    def test_createindex2(self):
        self._make_check_index(2, single_pass=False)

    def test_createindex3(self):
        self._make_check_index(3, single_pass=False)

    def test_createindex1_single_pass(self):
        self._make_check_index(1, single_pass=True)

    def test_createindex2_single_pass(self):
        self._make_check_index(2, single_pass=True)

    def test_createindex3_single_pass(self):
        self._make_check_index(3, single_pass=True)

if __name__ == "__main__":
    unittest.main()
