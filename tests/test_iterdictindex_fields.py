import pyterrier as pt

import itertools
import unittest
import tempfile
import shutil
import os

from .base import TempDirTestCase, BaseTestCase

class TestIterDictIndexer(TempDirTestCase):
        

    def _create_index(self, it, indexer):
        print("Writing index to " + self.test_dir)
        indexref = indexer.index(it)
        self.assertIsNotNone(indexref)
        return indexref

    def _make_check_index(self, n, index_type, attrs=('text',), meta=('docno', 'url', 'title')):
        from pyterrier.terrier.index import IndexingType
        # Test both versions: _fifo (for UNIX) and _nofifo (for Windows)
        indexers = [
            pt.index._IterDictIndexer_fifo(self.test_dir, text_attrs=attrs, type=index_type, meta=meta, fields=True),
            pt.index._IterDictIndexer_fifo(self.test_dir, text_attrs=attrs, type=index_type, threads=4, meta=meta, fields=True),
            pt.index._IterDictIndexer_nofifo(self.test_dir, text_attrs=attrs, type=index_type, meta=meta, fields=True),
        ]
        if pt.utils.is_windows():
           indexers = [indexers[-1]] 
        for indexer in indexers:
            with self.subTest(indexer=indexer):
                it = (
                    {'docno': '1', 'url': 'url1', 'text': 'He ran out of money, so he had to stop playing', 'title': 'Woes of playing poker', 'raw_source': b'<some>xml</content>'},
                    {'docno': '2', 'url': 'url2', 'text': 'The waves were crashing on the shore; it was a', 'title': 'Lovely sight', 'raw_source': b'<some>xml</content>'},
                    {'docno': '3', 'url': 'url3', 'text': 'The body may perhaps compensates for the loss', 'title': 'Best of Viktor Prowoll', 'raw_source': b'<some>xml</content>'},
                )
                it = itertools.islice(it, n)
                indexref = self._create_index(it, indexer)
                index = pt.IndexFactory.of(indexref)
                self.assertIsNotNone(index)
                self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())
                self.assertEqual(len(set(meta)), len(index.getMetaIndex().getKeys()))
                for m in meta:
                    self.assertTrue(m in index.getMetaIndex().getKeys())
                self.assertEqual(len(attrs), index.getCollectionStatistics().numberOfFields)
                for f in attrs:
                    self.assertTrue(f in index.getCollectionStatistics().getFieldNames())
                if index_type is IndexingType.CLASSIC:
                    self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))
                    self.assertTrue(index.hasIndexStructure("direct"))
                else:
                    self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))
                inv = index.getInvertedIndex()
                lex = index.getLexicon()
                post = inv.getPostings(lex.getLexiconEntry('plai'))
                post.next()
                from jnius import cast
                post = cast("org.terrier.structures.postings.FieldPosting", post)
                if 'title' in attrs:
                    self.assertEqual(2, post.frequency)
                    self.assertEqual(1, post.fieldFrequencies[0])
                    self.assertEqual(1, post.fieldFrequencies[1])
                else:
                    self.assertEqual(1, post.frequency)
                    self.assertEqual(1, post.fieldFrequencies[0])
                index.close()
            # reset index directory for next run
            shutil.rmtree(self.test_dir)
            os.mkdir(self.test_dir)

    def test_createindex1_2fields(self):
        from pyterrier.terrier.index import IndexingType
        self._make_check_index(1, IndexingType.CLASSIC, attrs=['text', 'title'])

    def test_createindex2_2fields(self):
        from pyterrier.terrier.index import IndexingType
        self._make_check_index(2, IndexingType.CLASSIC, attrs=['text', 'title'])

    def test_createindex3_2fields(self):
        from pyterrier.terrier.index import IndexingType
        self._make_check_index(3, IndexingType.CLASSIC, attrs=['text', 'title'])

    def test_createindex1_single_pass_2fields(self):
        from pyterrier.terrier.index import IndexingType
        self._make_check_index(1, IndexingType.SINGLEPASS, attrs=['text', 'title'])

    def test_createindex2_single_pass_2fields(self):
        from pyterrier.terrier.index import IndexingType
        self._make_check_index(2, IndexingType.SINGLEPASS, attrs=['text', 'title'])

    def test_createindex3_single_pass_2fields(self):
        from pyterrier.terrier.index import IndexingType
        self._make_check_index(3, IndexingType.SINGLEPASS, attrs=['text', 'title'])


if __name__ == "__main__":
    unittest.main()
