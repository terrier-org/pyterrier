import pyterrier as pt

import unittest
import tempfile
import shutil
import os
import numpy as np
from .base import BaseTestCase

class TestBackground(BaseTestCase):

    @staticmethod
    def get_contents(docid, index):
        rtr = {}
        direct = index.getDirectIndex()
        lexicon = index.getLexicon()
        for p in direct.getPostings(index.getDocumentIndex().getDocumentEntry(docid)):
            rtr[lexicon.getLexiconEntry(p.getId()).getKey()] = p.getFrequency()
        return rtr
    
    def test2_manualM(self):
        self._test2_manual(pt.index.IndexingType.MEMORY)
    def test2_manualC(self):
        self._test2_manual(pt.index.IndexingType.CLASSIC)
    def test2_manualS(self):
        self._test2_manual(pt.index.IndexingType.SINGLEPASS)
                

    def _test2_manual(self, type):
        import pyterrier as pt
        #pt.logging("INFO")
        import pandas as pd
        df1 = pd.DataFrame({
            'docno': ['1048'],
            'body':
                ['h  f  noise radiators in ground flashes of tropical lightning  a '+
                'detailed analysis of h  f  noise sources in tropical ground flashes '+
                'v  l  f  phase characteristics deduced from atmospheric waveforms']
        })
        pd_indexer1 = pt.DFIndexer(tempfile.mkdtemp(), type=type)
        indexref1 = pd_indexer1.index(df1["body"], df1["docno"])
        index1 = pt.IndexFactory.of(indexref1)
        
        has_direct1 = index1.hasIndexStructure("direct")

        indexref_big = pt.get_dataset("vaswani").get_index()
        index_big = pt.IndexFactory.of(indexref_big)

        from pyterrier import autoclass
        stopwords = autoclass("org.terrier.terms.Stopwords")(None)
        stemmer = autoclass("org.terrier.terms.PorterStemmer")(None)

        q = "MATHEMATICAL ANALYSIS AND DESIGN DETAILS OF WAVEGUIDE FED MICROWAVE RADIATIONS"
        self.assertEqual("1048", index_big.getMetaIndex().getItem("docno", 1047))
        contents_big = TestBackground.get_contents(1047, index_big)

        def _check_index(index_small):
            if has_direct1:
                contents1 = TestBackground.get_contents(0, index_small)
                self.assertEqual(contents1, contents_big)

            inv1 = index_small.getInvertedIndex()
            print(inv1.getClass().getName())
            lex1 = index_small.getLexicon()
            for t in contents_big:
                pointer = lex1[t]
                print(pointer.toString())
                p = inv1.getPostings(pointer)
                print(p.getClass().getName())
                rtr = p.next()
                self.assertEqual(0, rtr)
                self.assertEqual(p.getDocumentLength(), index_big.getDocumentIndex().getDocumentLength(1047))
                self.assertEqual(contents_big[t], p.getFrequency())
                self.assertEqual(p.next(), p.EOL)

            from jnius import JavaException
            try:
                br1 = pt.BatchRetrieve(index_small, wmodel="Tf")
                brall = pt.BatchRetrieve(index_big, wmodel="Tf")
                with_doc = pd.DataFrame([["q1", q, "1048", 1047]], columns=["qid", "query", "docno", "docid"])
                rtr1 = br1.transform(q)
            except JavaException as ja:
                print(ja.stacktrace)
                raise ja
            rtrall = brall(with_doc)            
            self.assertTrue(np.array_equal(rtr1["score"].values, rtrall["score"].values))
        
        _check_index(index1)
        _check_index( pt.autoclass("org.terrier.python.IndexWithBackground")(index1, index_big))

    def test_itM(self):
        self._test_it(pt.index.IndexingType.MEMORY)
    def test_itC(self):
        self._test_it(pt.index.IndexingType.CLASSIC)
    def test_itS(self):
        self._test_it(pt.index.IndexingType.SINGLEPASS)
        

    def _test_it(self, type):
        import pyterrier as pt
        import pandas as pd
        df1 = pd.DataFrame({
            'docno': ['1', '2', '3'],
            'url':
                ['url1', 'url2', 'url3'],
            'text':
                ['He ran out of money, so he had to stop playing',
                 'The wave were crash on the shore; it was a',
                 'The body may perhaps compensates for the loss']
        })
        pd_indexer1 = pt.DFIndexer(tempfile.mkdtemp(), type=type)
        indexref1 = pd_indexer1.index(df1["text"], df1["docno"])

        df2 = pd.DataFrame({
            'docno': ['14'],
            'text': ['test wave']
        })

        from jnius import JavaException
        try:

            pd_indexer2 = pt.DFIndexer(tempfile.mkdtemp(), type=type)
            indexref2 = pd_indexer2.index(df2["text"], df2["docno"])
            
            index1 = pt.IndexFactory.of(indexref1)
            self.assertEqual(3, index1.getCollectionStatistics().getNumberOfDocuments())

            index2 = pt.IndexFactory.of(indexref2)
            self.assertEqual(1, index2.getCollectionStatistics().getNumberOfDocuments())

            index_combined = pt.autoclass("org.terrier.python.IndexWithBackground")(index2, index1)
            self.assertEqual(3, index_combined.getCollectionStatistics().getNumberOfDocuments())

            self.assertEqual(1, index_combined.getLexicon()["test"].getFrequency())

            # this is 1 as we used the background index for the background
            # WITHOUT adding the statistics of the local index
            self.assertEqual(1, index_combined.getLexicon()["wave"].getFrequency())
            

        except JavaException as ja:
            print(ja.stacktrace)
            raise ja
        
        

        