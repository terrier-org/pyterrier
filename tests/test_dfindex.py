
import pyterrier as pt

import unittest, math, os
import shutil, tempfile
from os import path

class TestDFIndexer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDFIndexer, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init(logging="DEBUG")
        # else:
        #     pt.setup_logging("DEBUG")

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _create_index(self, df):
        pd_indexer = pt.DFIndexer(self.test_dir)
        indexref=pd_indexer.index(df["text"], df["docno"])
        self.assertIsNotNone(indexref)
        return indexref

    def _make_check_index(self, n):
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
        df = df.head(n)
        indexref = self._create_index(df)
        index = pt.autoclass("org.terrier.structures.IndexFactory").of(indexref)
        self.assertIsNotNone(index)
        self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())

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
        d1 = df.head(1)
        import pyterrier as pt
        from pyterrier import DFIndexUtils
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
        self._make_check_index(1)

    def test_createindex2(self):
        self._make_check_index(2)

    def test_createindex3(self):
        self._make_check_index(3)
        


if __name__ == "__main__":
    unittest.main()
