import pyterrier as pt

import itertools
import unittest
import tempfile
import shutil
import os

from .base import TempDirTestCase, BaseTestCase

class TestIterDictIndexer(TempDirTestCase):
        

    def _create_index(self, it, fields, type, indexer):
        print("Writing index to " + self.test_dir)
        indexref = indexer.index(it, fields)
        self.assertIsNotNone(indexref)
        return indexref

    def _make_check_index(self, n, index_type, fields=('text',), meta=('docno', 'url', 'title')):
        from pyterrier.index import IndexingType
        # Test both versions: _fifo (for UNIX) and _nofifo (for Windows)
        indexers = [
            pt.index._IterDictIndexer_fifo(self.test_dir, type=index_type, meta=meta),
            pt.index._IterDictIndexer_fifo(self.test_dir, type=index_type, threads=4, meta=meta),
            pt.index._IterDictIndexer_nofifo(self.test_dir, type=index_type, meta=meta),
        ]
        if BaseTestCase.is_windows():
           indexers = [indexers[-1]] 
        for indexer in indexers:
            with self.subTest(indexer=indexer):
                it = (
                    {'docno': '1', 'url': 'url1', 'text': 'He ran out of money, so he had to stop playing', 'title': 'Woes of playing poker', 'raw_source': b'<some>xml</content>'},
                    {'docno': '2', 'url': 'url2', 'text': 'The waves were crashing on the shore; it was a', 'title': 'Lovely sight', 'raw_source': b'<some>xml</content>'},
                    {'docno': '3', 'url': 'url3', 'text': 'The body may perhaps compensates for the loss', 'title': 'Best of Viktor Prowoll', 'raw_source': b'<some>xml</content>'},
                )
                it = itertools.islice(it, n)
                indexref = self._create_index(it, fields, index_type, indexer)
                index = pt.IndexFactory.of(indexref)
                self.assertIsNotNone(index)
                self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())
                self.assertEqual(len(set(meta)), len(index.getMetaIndex().getKeys()))
                for m in meta:
                    self.assertTrue(m in index.getMetaIndex().getKeys())
                self.assertEqual(len(fields), index.getCollectionStatistics().numberOfFields)
                for f in fields:
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
                if 'title' in fields:
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

    # Issue #43

    # def test_createindex3_memory(self):
    #     from pyterrier.index import IndexingType
    #     self._make_check_index(3, IndexingType.MEMORY)

    # def test_createindex1_2fields_memory(self):
    #     from pyterrier.index import IndexingType
    #     self._make_check_index(1, IndexingType.MEMORY, fields=['text', 'title'])

    def test_createindex1_2fields(self):
        from pyterrier.index import IndexingType
        self._make_check_index(1, IndexingType.CLASSIC, fields=['text', 'title'])

    def test_createindex2_2fields(self):
        from pyterrier.index import IndexingType
        self._make_check_index(2, IndexingType.CLASSIC, fields=['text', 'title'])

    def test_createindex3_2fields(self):
        from pyterrier.index import IndexingType
        self._make_check_index(3, IndexingType.CLASSIC, fields=['text', 'title'])

    def test_createindex1_single_pass_2fields(self):
        from pyterrier.index import IndexingType
        self._make_check_index(1, IndexingType.SINGLEPASS, fields=['text', 'title'])

    def test_createindex2_single_pass_2fields(self):
        from pyterrier.index import IndexingType
        self._make_check_index(2, IndexingType.SINGLEPASS, fields=['text', 'title'])

    def test_createindex3_single_pass_2fields(self):
        from pyterrier.index import IndexingType
        self._make_check_index(3, IndexingType.SINGLEPASS, fields=['text', 'title'])

    def test_meta_init(self):
        it = [
            {'docno': '1', 'url': 'url1', 'text': 'He ran out of money, so he had to stop playing', 'title': 'Woes of playing poker'},
            {'docno': '2', 'url': 'url2', 'text': 'The waves were crashing on the shore; it was a', 'title': 'Lovely sight'},
            {'docno': '3', 'url': 'url3', 'text': 'The body may perhaps compensates for the loss', 'title': 'Best of Viktor Prowoll'},
        ]
        props={}
        props["termpipelines"] = ""
        
        indexer = pt.IterDictIndexer(self.test_dir, meta={'docno' : 10, 'url' : 10, 'text' : 100, 'title' : 100}, meta_reverse=['docno', 'url'])
        indexref = indexer.index(it)
        index = pt.IndexFactory.of(indexref)
        self.assertIn("docno", index.getMetaIndex().getKeys())
        self.assertIn("text", index.getMetaIndex().getKeys())
        self.assertIn("docno", index.getMetaIndex().getKeys())
        self.assertIn("url", index.getMetaIndex().getReverseKeys())


    def test_check_stemmer(self):
        it = [
            {'docno': '1', 'url': 'url1', 'text': 'He ran out of money, so he had to stop playing', 'title': 'Woes of playing poker'},
            {'docno': '2', 'url': 'url2', 'text': 'The waves were crashing on the shore; it was a', 'title': 'Lovely sight'},
            {'docno': '3', 'url': 'url3', 'text': 'The body may perhaps compensates for the loss', 'title': 'Best of Viktor Prowoll'},
        ]
        props={}
        props["termpipelines"] = ""
        
        indexer = pt.IterDictIndexer(self.test_dir)
        for k,v in props.items():
            indexer.setProperty(k, v)
        indexref = indexer.index(it)
        index = pt.IndexFactory.of(indexref)
        index = pt.cast("org.terrier.structures.IndexOnDisk", index)
        #restore setting after test
        pt.ApplicationSetup.setProperty("termpipelines", "Stopwords,PorterStemmer")
        self.assertEqual("", index.getIndexProperty("termpipelines", "bla"))

if __name__ == "__main__":
    unittest.main()
