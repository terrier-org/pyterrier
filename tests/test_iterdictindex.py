import pyterrier as pt

import itertools
import unittest
import tempfile
import shutil
import os

from .base import BaseTestCase

class TestIterDictIndexer(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _create_index(self, it, fields, meta, type):
        pd_indexer = pt.IterDictIndexer(self.test_dir, type=type)
        indexref = pd_indexer.index(it, fields, meta)
        self.assertIsNotNone(indexref)
        return indexref

    def _make_check_index(self, n, single_pass, fields=('text',), meta=('docno', 'url', 'title')):
        it = (
            {'docno': '1', 'url': 'url1', 'text': 'He ran out of money, so he had to stop playing', 'title': 'Woes of playing poker'},
            {'docno': '2', 'url': 'url2', 'text': 'The waves were crashing on the shore; it was a', 'title': 'Lovely sight'},
            {'docno': '3', 'url': 'url3', 'text': 'The body may perhaps compensates for the loss', 'title': 'Best of Viktor Prowoll'},
        )
        it = itertools.islice(it, n)
        if single_pass:
            indexref = self._create_index(it, fields, meta, pt.IndexingType.SINGLEPASS)
        else:
            indexref = self._create_index(it, fields, meta, pt.IndexingType.CLASSIC)
        index = pt.IndexFactory.of(indexref)
        self.assertIsNotNone(index)
        self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertEqual(len(set(meta)), len(index.getMetaIndex().getKeys()))
        for m in meta:
            self.assertTrue(m in index.getMetaIndex().getKeys())
        self.assertEqual(len(fields), index.getCollectionStatistics().numberOfFields)
        for f in fields:
            self.assertTrue(f in index.getCollectionStatistics().getFieldNames())
        if single_pass:
            self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))
        else:
            self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))
        inv = index.getInvertedIndex()
        lex = index.getLexicon()
        post = inv.getPostings(lex.getLexiconEntry('plai'))
        post.next()
        from jnius import cast
        post = cast("org.terrier.structures.postings.FieldPosting", post)
        if 'title' in fields:
            self.assertEqual(2, post.frequency)
            self.assertEqual(1, post.fieldFrequencies[0])
            self.assertEqual(1, post.fieldFrequencies[1])
        else:
            self.assertEqual(1, post.frequency)
            self.assertEqual(1, post.fieldFrequencies[0])


    def test_checkjavaDocIterator(self):
        it = [
            {'docno': '1', 'url': 'url1', 'text': 'He ran out of money, so he had to stop playing', 'title': 'Woes of playing poker'},
            {'docno': '2', 'url': 'url2', 'text': 'The waves were crashing on the shore; it was a', 'title': 'Lovely sight'},
            {'docno': '3', 'url': 'url3', 'text': 'The body may perhaps compensates for the loss', 'title': 'Best of Viktor Prowoll'},
        ]

        from pyterrier import FlatJSONDocumentIterator

        it1 = itertools.islice(it, 1)
        jIter1 = FlatJSONDocumentIterator(it1)
        self.assertTrue(jIter1.hasNext())
        self.assertIsNotNone(jIter1.next())
        self.assertFalse(jIter1.hasNext())

        it2 = itertools.islice(it, 2)
        jIter1 = FlatJSONDocumentIterator(it2)
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

    def test_createindex1_2fields(self):
        self._make_check_index(1, single_pass=False, fields=['text', 'title'])

    def test_createindex2_2fields(self):
        self._make_check_index(2, single_pass=False, fields=['text', 'title'])

    def test_createindex3_2fields(self):
        self._make_check_index(3, single_pass=False, fields=['text', 'title'])

    def test_createindex1_single_pass_2fields(self):
        self._make_check_index(1, single_pass=True, fields=['text', 'title'])

    def test_createindex2_single_pass_2fields(self):
        self._make_check_index(2, single_pass=True, fields=['text', 'title'])

    def test_createindex3_single_pass_2fields(self):
        self._make_check_index(3, single_pass=True, fields=['text', 'title'])

if __name__ == "__main__":
    unittest.main()
