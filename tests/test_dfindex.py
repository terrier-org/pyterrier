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
        print("Created " + self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        print("Deleting " + self.test_dir)

    def _create_index(self, type, dfText, dfMeta):
        print("Writing index type "+str(type)+" to " + self.test_dir)
        pd_indexer = pt.DFIndexer(self.test_dir, type=type)
        indexref = pd_indexer.index(dfText, dfMeta)
        self.assertIsNotNone(indexref)
        return indexref

    def _make_check_index(self, n, index_type, include_urls=False):
        from pyterrier.index import IndexingType
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
        metadata = df[["docno", "url"]] if include_urls else df["docno"]
        indexref = self._create_index(index_type, df["text"], metadata)
        index = pt.IndexFactory.of(indexref)
        self.assertIsNotNone(index)
        self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue("docno" in index.getMetaIndex().getKeys())
        if include_urls:
            self.assertTrue("url" in index.getMetaIndex().getKeys())
        if index_type is IndexingType.CLASSIC:
            self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))
            self.assertTrue(index.hasIndexStructure("direct"))
        else:
            self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))

    def test_checkjavaDocIteratr_None(self):
        import pandas as pd
        df = pd.DataFrame({
            'docno':
                ['1', '2', '3'],
            'url':
                ['url1', 'url2', 'url3'],
            'text':
                [None,
                 'The waves were crashing on the shore; it was a',
                 'Some other text']
        })

        from pyterrier import DFIndexUtils

        d1 = df.head(1)
        jIter1, metalens = DFIndexUtils.create_javaDocIterator(d1["text"], d1["docno"])
        self.assertEqual(1, len(metalens))
        self.assertEqual(1, metalens["docno"])
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

        d2 = df.head(2)
        jIter1, metalens = DFIndexUtils.create_javaDocIterator(d2["text"], d2[["docno", "url"]])
        self.assertEqual(2, len(metalens))
        self.assertEqual(1, metalens["docno"])
        self.assertEqual(4, metalens["url"])
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

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
        jIter1, metalens = DFIndexUtils.create_javaDocIterator(d1["text"], d1["docno"])
        self.assertEqual(1, len(metalens))
        self.assertEqual(1, metalens["docno"])
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

        d2 = df.head(2)
        jIter1, metalens = DFIndexUtils.create_javaDocIterator(d2["text"], d2[["docno", "url"]])
        self.assertEqual(2, len(metalens))
        self.assertEqual(1, metalens["docno"])
        self.assertEqual(4, metalens["url"])
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

    def test_createindex1_two_metadata(self):
        from pyterrier.index import IndexingType
        self._make_check_index(1, IndexingType.CLASSIC, include_urls=True)

    def test_createindex1(self):
        from pyterrier.index import IndexingType
        self._make_check_index(1, IndexingType.CLASSIC)

    def test_createindex2(self):
        from pyterrier.index import IndexingType
        self._make_check_index(2, IndexingType.CLASSIC)

    def test_createindex3(self):
        from pyterrier.index import IndexingType
        self._make_check_index(3, IndexingType.CLASSIC)

    def test_createindex1_single_pass(self):
        from pyterrier.index import IndexingType
        self._make_check_index(1, IndexingType.SINGLEPASS)

    def test_createindex2_single_pass(self):
        from pyterrier.index import IndexingType
        self._make_check_index(2, IndexingType.SINGLEPASS)

    def test_createindex3_single_pass(self):
        from pyterrier.index import IndexingType
        self._make_check_index(3, IndexingType.SINGLEPASS)

    def test_createindex1_memory(self):
        from pyterrier.index import IndexingType
        self._make_check_index(1, IndexingType.MEMORY)

    def test_createindex2_memory(self):
        from pyterrier.index import IndexingType
        self._make_check_index(2, IndexingType.MEMORY)

    def test_createindex3_memory(self):
        from pyterrier.index import IndexingType
        self._make_check_index(3, IndexingType.MEMORY)

if __name__ == "__main__":
    unittest.main()
