import unittest
import pyterrier as pt
import tempfile
from .base import TempDirTestCase

class TestDunder(TempDirTestCase):


    def test_dunders(self):
        import pandas as pd
        df = pd.DataFrame({
            'docno':
                ['1', '2', '3'],
            'text':
                ['He ran out of money, so he had to stop playing',
                 'The waves were crashing on the shore; it was a',
                 'The body may perhaps compensates for the loss']
        })
        import pyterrier as pt
        pd_indexer = pt.DFIndexer(self.test_dir)
        pd_indexer.properties["termpipelines"] = ""
        indexref = pd_indexer.index(df["text"], df["docno"])
        index = pt.IndexFactory.of(indexref)
        self.assertIsNotNone(index)
        self.assertIsNotNone(index.getLexicon())
        self.assertTrue("__getitem__" in dir(index.getLexicon()))
        crashingSeen = False

        # test out __len__ mapping
        self.assertEqual(len(index.getLexicon()), index.getLexicon().numberOfEntries())
        # lexicon is Iterable, test the Iterable mapping of jnius
        for t in index.getLexicon():
            if t.getKey() == "crashing":
                crashingSeen = True
                break
        self.assertTrue(crashingSeen)
        # test our own __getitem__ mapping
        self.assertTrue("crashing" in index.getLexicon())
        self.assertEqual(1, index.getLexicon()["crashing"].getFrequency())
        self.assertFalse("dentish" in index.getLexicon())

        # now test that IterablePosting has had its dunder methods added
        postings = index.getInvertedIndex().getPostings(index.getLexicon()["crashing"])
        count = 0
        for p in postings:
            count += 1
            self.assertEqual(1, p.getId())
            self.assertEqual(1, p.getFrequency())
        self.assertEqual(1, count)

if __name__ == "__main__":
    unittest.main()
