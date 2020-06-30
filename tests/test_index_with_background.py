import pyterrier as pt

import unittest
import tempfile
import shutil
import os

from .base import BaseTestCase

class TestBackground(BaseTestCase):

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
            self.assertEqual(2, index_combined.getLexicon()["wave"].getFrequency())
            #we dont support terms that arent in the small index.
            #self.assertEqual(1, index_combined.getLexicon()["crash"].getFrequency())

        except JavaException as ja:
            print(ja.stacktrace)
            raise ja
        
        

        