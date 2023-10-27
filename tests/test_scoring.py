import pandas as pd
import numpy as np
import unittest
import pyterrier as pt
from .base import TempDirTestCase
import tempfile
import shutil
import os

class TestScoring(TempDirTestCase):

    def test_scoring_text(self):
        pt.logging("DEBUG")
        dataset = pt.get_dataset("vaswani")
        indexer = pt.TRECCollectionIndexer(
            self.test_dir, 
            meta= {'docno' : 26, 'body' : 2048},
            # The tags from which to save the text. ELSE is special tag name, which means anything not consumed by other tags.
            meta_tags = {'body' : 'ELSE'}
        )
        indexref = indexer.index(dataset.get_corpus())
        index = pt.IndexFactory.of(indexref)
        meta = index.getMetaIndex()
        self.assertTrue( "body" in meta.getKeys() )
        self.assertTrue( "compact memories have" in meta.getItem("body", 0) )
        print( meta.getItem("body", 1047))

        self._test_scoring_text(dataset, index, "org.terrier.python.TestModel$Constant")
        self._test_scoring_text(dataset, index, "Tf")
        self._test_scoring_text(dataset, index, "org.terrier.python.TestModel$TFOverN")
        self._test_scoring_text(dataset, index, "org.terrier.python.TestModel$F")
        self._test_scoring_text(dataset, index, "org.terrier.python.TestModel$Nt")
        self._test_scoring_text(dataset, index, "DPH")

    def _test_scoring_text(self, dataset, index, wmodel):
        
        br1 = pt.BatchRetrieve(index, wmodel=wmodel, metadata=["docno", "body"], num_results=5)
        input = dataset.get_topics().head(10)
        output1 = br1(input)
        self.assertTrue( "body" in output1.columns )
        input2 = output1[["qid", "query", "docno", "body"]]
        br2 = pt.batchretrieve.TextScorer(background_index=index, wmodel=wmodel, verbose=True)
        output2 = br2(input2)
        self.assertTrue( "score" in output2.columns )

        joined = output1.merge(output2, on=["qid", "docno"])[["qid", "docno", "score_x", "score_y"]]
        print(joined)
        #TODO: there is a bug here. TextScorer should have the same score, but it doesnt; occasionally terms arent matched
        #self.assertTrue(np.array_equal(joined["score_x"].values, joined["score_y"].values))
        #self.assertEqual(pt.Evaluate(output1,dataset.get_qrels()), pt.Evaluate(output2,dataset.get_qrels()))
        

    def test_scoring_manual_empty(self):
        input = pd.DataFrame([["q1", "fox", "d1", ""]], columns=["qid", "query", "docno", "body"])
        from pyterrier.batchretrieve import TextScorer
        scorer = TextScorer(wmodel="Tf")
        rtr = scorer(input)
        self.assertEqual(1, len(rtr))
        self.assertTrue("score" in rtr.columns)
        self.assertTrue("body" in rtr.columns)
        self.assertEqual(0, rtr.iloc[0]["score"])

    def test_scoring_manual(self):
        input = pd.DataFrame([["q1", "fox", "d1", "all the fox were fox"]], columns=["qid", "query", "docno", "body"])
        from pyterrier.batchretrieve import TextScorer
        scorer = TextScorer(wmodel="Tf")
        rtr = scorer(input)
        self.assertEqual(1, len(rtr))
        self.assertTrue("score" in rtr.columns)
        self.assertTrue("body" in rtr.columns)
        self.assertEqual(2, rtr.iloc[0]["score"])

        scorer = TextScorer(wmodel="org.terrier.python.TestModel$TFOverN")
        rtr = scorer(input)
        self.assertEqual(1, len(rtr))
        self.assertTrue("score" in rtr.columns)
        self.assertEqual(2, rtr.iloc[0]["score"]) # tf / numdocs

    def test_scoring_manual_background(self):
        input = pd.DataFrame([["q1", "fox", "d1", "all the fox were fox"]], columns=["qid", "query", "docno", "body"])
        from pyterrier.batchretrieve import TextScorer
        scorer = TextScorer(wmodel="Tf", background_index=pt.get_dataset("vaswani").get_index())
        rtr = scorer(input)
        self.assertEqual(1, len(rtr))
        self.assertTrue("score" in rtr.columns)
        self.assertEqual(2, rtr.iloc[0]["score"])

        index_background = pt.IndexFactory.of( pt.get_dataset("vaswani").get_index() )
        scorer = TextScorer(wmodel="org.terrier.python.TestModel$TFOverN", background_index=index_background)
        rtr = scorer(input)
        self.assertEqual(1, len(rtr))
        self.assertTrue("score" in rtr.columns)
        self.assertEqual(2/index_background.getCollectionStatistics().getNumberOfDocuments(), rtr.iloc[0]["score"])
        
    def test_scoring_qe(self):
        input = pd.DataFrame([["q1", "fox", "d1", "all the fox were fox"]], columns=["qid", "query", "docno", "body"])
        from pyterrier.batchretrieve import TextScorer
        scorer = pt.batchretrieve.TextIndexProcessor(pt.rewrite.Bo1QueryExpansion, takes="docs", returns="queries")
        rtr = scorer(input)
        self.assertTrue("qid" in rtr.columns)
        self.assertTrue("query" in rtr.columns)
        self.assertTrue("docno" not in rtr.columns)
        self.assertTrue("^" in rtr.iloc[0]["query"])
