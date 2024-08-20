import unittest
import pyterrier as pt
import tempfile
from .base import TempDirTestCase

class TestDunder(TempDirTestCase):

    def test_callable_wmodel_dunders(self):
        testPosting = pt.java.autoclass("org.terrier.structures.postings.BasicPostingImpl")(0,1)

        lambdafn = lambda keyFreq, posting, entryStats, collStats: posting.getFrequency()
        callback, wmodel = pt.terrier.retriever._function2wmodel(lambdafn)
        
        byterep = pt.java.bytebuffer_to_array(wmodel.scoringClass.serializeFn())
        import dill as pickle
        from dill import extend
        #see https://github.com/SeldonIO/alibi/issues/447#issuecomment-881552005
        extend(use_dill=False)    
        fn = pickle.loads(byterep)
        self.assertEqual(
            lambdafn(1, testPosting, None, None),
            fn(1, testPosting, None, None),
            )

        wmodel.__getstate__()
        rtr = wmodel.__reduce__()
        
        #check the byte array is picklable
        pickle.dumps(rtr[1][0])
        #check object is picklable
        pickle.dumps(wmodel)
        #check can be unpickled too
        wmodel2 = pickle.loads(pickle.dumps(wmodel))

        score1 = wmodel.score(testPosting)
        score2 = wmodel2.score(testPosting)
        self.assertEqual(score1, score2)

        #check newly unpickled can still be pickled
        pickle.dumps(wmodel2)
        wmodel3 = pickle.loads(pickle.dumps(wmodel2))
        score3 = wmodel3.score(testPosting)
        self.assertEqual(score1, score3)


    def test_wmodel_dunders(self):

        wmodel = pt.java.autoclass("org.terrier.matching.models.BM25")()
        wmodel.__reduce__()
        wmodel.__getstate__()
        rtr = wmodel.__reduce__()
        pt.java.cast("org.terrier.matching.models.BM25", rtr[0](*rtr[1]))
        import pickle
        #import dill as pickle
        #check the byte array is picklable
        print(rtr[1][0])
        pickle.dumps(rtr[1][0])
        pickle.dumps(wmodel)

    def test_index_dunders(self):
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        i1 = pt.IndexFactory.of(indexref)
        i2 = pt.IndexFactory.of(indexref)
        i12 = i1 + i2
        self.assertIsNotNone(i12)
        self.assertEqual(
            i12.getCollectionStatistics().getNumberOfDocuments(), 
            i1.getCollectionStatistics().getNumberOfDocuments()
            + i2.getCollectionStatistics().getNumberOfDocuments())
        self.assertEqual( len(i1), i1.getCollectionStatistics().getNumberOfDocuments() )
        self.assertEqual( len(i12), len(i1) + len(i2) )
            
        self.assertTrue(i12.hasIndexStructure("inverted"))
        self.assertTrue(i12.hasIndexStructure("lexicon"))
        self.assertTrue(i12.hasIndexStructure("document"))
        self.assertTrue(i12.hasIndexStructure("meta"))

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
        pd_indexer = pt.IterDictIndexer(self.test_dir, stopwords=pt.TerrierStopwords.none, stemmer=pt.TerrierStemmer.none)
        indexref = pd_indexer.index(df.to_dict(orient='records'))
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
        
        # test Map$Entry can be decoded like a tuple
        # both within an iterator, and separately
        x, y = index.getLexicon().getLexiconEntry(0)
        term_count = 0
        for term, entry in index.getLexicon():
            term_count += 1
        self.assertEqual(term_count, index.getLexicon().numberOfEntries())
        term_count = 0
        for lee in index.getLexicon():
            term_count += 1
            lee.getKey()
            lee.getValue()
        self.assertEqual(term_count, index.getLexicon().numberOfEntries())

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
