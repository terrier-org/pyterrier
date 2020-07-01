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

    def test2_manual(self):
        import pyterrier as pt
        import pandas as pd
        df1 = pd.DataFrame({
            'docno': ['1048'],
            'body':
                ['h  f  noise radiators in ground flashes of tropical lightning  a '+
                'detailed analysis of h  f  noise sources in tropical ground flashes '+
                'v  l  f  phase characteristics deduced from atmospheric waveforms']
        })
        pd_indexer1 = pt.DFIndexer(None, type=pt.index.IndexingType.MEMORY)
        indexref1 = pd_indexer1.index(df1["body"], df1["docno"])
        index1 = pt.IndexFactory.of(indexref1)

        indexref_big = pt.get_dataset("vaswani").get_index()
        index_big = pt.IndexFactory.of(indexref_big)

        from pyterrier import autoclass
        stopwords = autoclass("org.terrier.terms.Stopwords")(None)
        stemmer = autoclass("org.terrier.terms.PorterStemmer")(None)

        q = "MATHEMATICAL ANALYSIS AND DESIGN DETAILS OF WAVEGUIDE FED MICROWAVE RADIATIONS"
        self.assertEqual("1048", index_big.getMetaIndex().getItem("docno", 1047))
        contents_big = TestBackground.get_contents(1047, index_big)
        
        for index_small in [index1, pt.autoclass("org.terrier.python.IndexWithBackground")(index1, index_big)]:

            contents1 = TestBackground.get_contents(0, index_small)
            self.assertEqual(contents1, contents_big)

            inv1 = index_small.getInvertedIndex()
            lex1 = index_small.getLexicon()
            for t in contents_big:
                pointer = lex1[t]
                p = inv1.getPostings(pointer)
                rtr = p.next()
                self.assertEqual(0, rtr)
                self.assertEqual(contents_big[t], p.getFrequency())

            br1 = pt.BatchRetrieve(index_small, wmodel="Tf")
            brall = pt.BatchRetrieve(index_big, wmodel="Tf")
            with_doc = pd.DataFrame([["q1", q, "1048", 1047]], columns=["qid", "query", "docno", "docid"])
            rtr1 = br1.transform(q)
            rtrall = brall(with_doc)            
            self.assertTrue(np.array_equal(rtr1["score"].values, rtrall["score"].values))

    def test_it(self):
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
        pd_indexer1 = pt.DFIndexer(None, type=pt.index.IndexingType.MEMORY)
        indexref1 = pd_indexer1.index(df1["text"], df1["docno"])

        df2 = pd.DataFrame({
            'docno': ['14'],
            'text': ['test wave']
        })

        from jnius import JavaException
        try:

            pd_indexer2 = pt.DFIndexer(None, type=pt.index.IndexingType.MEMORY)
            indexref2 = pd_indexer1.index(df2["text"], df2["docno"])
            
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
        
        

        