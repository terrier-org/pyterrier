import pyterrier as pt

import itertools
import unittest
import tempfile
import shutil
import os
from unittest import SkipTest

from .base import TempDirTestCase, BaseTestCase

class TestIterDictIndexerPreTok(TempDirTestCase):

    def test_dpl(self):
        it1 = [
            {'docno': 'd1', 'url': 'url1', "toks" : {"a" : 1, "b" : 2}}
        ]
        it2 = [
            {'docno': 'd1', 'url': 'url1', "toks" : {"a" : 1, "b" : 2}},
            {'docno': 'd2', 'url': 'url1', "toks" : {"a" : 1, "b" : 1}}
        ]
        from pyterrier.terrier.index import DocListIterator
        
        iterator =  DocListIterator(iter(it1))
        self.assertTrue(iterator.hasNext())
        obj = iterator.next()
        self.assertIsNotNone(obj)
        self.assertEqual("d1", obj.getKey().get("docno"))
        self.assertEqual(2, obj.getValue().getFrequency("b"))
        self.assertEqual(3, obj.getValue().getDocumentLength())
        
        if iterator.hasNext():
            obj = iterator.next()
            self.assertIsNone(obj)
        self.assertFalse(iterator.hasNext())

        iterator =  DocListIterator(iter(it2))
        self.assertTrue(iterator.hasNext())
        obj = iterator.next()
        self.assertIsNotNone(obj)
        self.assertEqual("d1", obj.getKey().get("docno"))
        self.assertEqual(2, obj.getValue().getFrequency("b"))

        self.assertTrue(iterator.hasNext())
        obj = iterator.next()
        self.assertIsNotNone(obj)
        self.assertEqual(1, obj.getValue().getFrequency("b"))
        self.assertEqual("d2", obj.getKey().get("docno"))

        if iterator.hasNext():
            obj = iterator.next()
            self.assertIsNone(obj)
        self.assertFalse(iterator.hasNext())


    
    def test_json_pretok_iterator(self):
        if not pt.terrier.check_version("5.7") or not pt.terrier.check_helper_version("0.0.7"):
            raise SkipTest("Requires Terrier 5.7 and helper 0.0.7")

        it = [
            {'docno': 'd1', 'url': 'url1', "toks" : {"a" : 1, "b" : 2}}
        ]

        import json
        testfile = os.path.join(self.test_dir, "test.json")
        with open(testfile, 'wt') as file:
            for doc in it:
                file.write(json.dumps(doc))
        jparserCls = pt.java.autoclass("org.terrier.python.JsonlPretokenisedIterator")
        jparser = jparserCls(testfile)
        self.assertTrue(jparser.hasNext())
        nextRow = jparser.next()
        self.assertIsNotNone(nextRow)
        props = nextRow.getKey()
        self.assertEqual("d1", props.get("docno"))
        self.assertEqual("url1", props.get("url"))
        pl = nextRow.getValue()
        self.assertIn("a", pl.termSet())
        self.assertEqual(1, pl.getFrequency("a"))

    def _make_pretok_index(self, n, index_type, meta=('docno', 'url')):
        if not pt.terrier.check_version("5.7") or not pt.terrier.check_helper_version("0.0.7"):
            self.skipTest("Requires Terrier 5.7 and helper 0.0.7")
        
        from pyterrier.terrier.index import IndexingType
        # Test both versions: _fifo (for UNIX) and _nofifo (for Windows)
        indexers = [
            pt.index._IterDictIndexer_fifo(self.test_dir, type=index_type, meta=meta, pretokenised=True),
            pt.index._IterDictIndexer_fifo(self.test_dir, type=index_type, threads=4, meta=meta, pretokenised=True),
            pt.index._IterDictIndexer_nofifo(self.test_dir, type=index_type, meta=meta, pretokenised=True),
        ]
        if pt.utils.is_windows():
           indexers = [indexers[-1]] 
        for indexer in indexers:
            with self.subTest(indexer=indexer):
                it = [
                    {'docno': 'd1', 'url': 'url1', "toks" : {"a" : 1, "b" : 2.9123}},
                    {'docno': 'd2', 'url': 'url2', "toks" : {"a" : 1.5, "b" : 2.}}
                ]
                it = itertools.islice(it, n)
                print("Writing index to " + self.test_dir)
                indexref = indexer.index(it)
                self.assertIsNotNone(indexref)
                index = pt.IndexFactory.of(indexref)
                self.assertIsNotNone(index)
                self.assertEqual(n, index.getCollectionStatistics().getNumberOfDocuments())
                self.assertEqual(len(set(meta)), len(index.getMetaIndex().getKeys()))
                for m in meta:
                    self.assertTrue(m in index.getMetaIndex().getKeys())
                if index_type is IndexingType.CLASSIC:
                    self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))
                    self.assertTrue(index.hasIndexStructure("direct"))
                else:
                    self.assertFalse(os.path.isfile(self.test_dir + '/data.direct.bf'))
                inv = index.getInvertedIndex()
                lex = index.getLexicon()
                post = inv.getPostings(lex.getLexiconEntry('a'))
                post.next()
                self.assertEqual(1, post.frequency)

                pindex = pt.java.cast("org.terrier.structures.PropertiesIndex", index)
                self.assertEqual("", pindex.getIndexProperty("termpipelines", "BLA"))

                index.close()
            # reset index directory for next run
            shutil.rmtree(self.test_dir)
            os.mkdir(self.test_dir)

    def test_pretok_createindex1_basic(self):
        from pyterrier.terrier.index import IndexingType
        from jnius import JavaException
        try:
            self._make_pretok_index(1, IndexingType.CLASSIC)
        except JavaException as ja:
            print("\n\t".join(ja.stacktrace))
            raise ja

    def test_pretok_createindex2_basic(self):
        from pyterrier.terrier.index import IndexingType
        self._make_pretok_index(2, IndexingType.CLASSIC)

    def test_pretok_createindex1_single_pass(self):
        from pyterrier.terrier.index import IndexingType
        from jnius import JavaException
        try:
            self._make_pretok_index(1, IndexingType.SINGLEPASS)
        except JavaException as ja:
            print("\n\t".join(ja.stacktrace))
            raise ja

    def test_pretok_createindex2_single_pass(self):
        from pyterrier.terrier.index import IndexingType
        from jnius import JavaException
        try:
            self._make_pretok_index(2, IndexingType.SINGLEPASS)
        except JavaException as ja:
            print("\n\t".join(ja.stacktrace))
            raise ja

if __name__ == "__main__":
    unittest.main()
