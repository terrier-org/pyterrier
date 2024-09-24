import pandas as pd # an error occurring here ususally means failed pip installation not detected by GHA
import pyterrier as pt
import os
import unittest
import warnings
from .base import TempDirTestCase
import tempfile
import shutil

class TestFlash(TempDirTestCase):

    def test_one_row_round(self):
        import pyterrier as pt
        vaswani = pt.datasets.get_dataset("vaswani")
        br = pt.terrier.Retriever(vaswani.get_index())
        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q", pt.measures.NDCG@5], round=2)
        self.assertEqual(str(rtr.iloc[0]["map"]), "0.31")
        self.assertEqual(str(rtr.iloc[0]["nDCG@5"]), "0.46")

        rtr = pt.Experiment([br], vaswani.get_topics().head(10), vaswani.get_qrels(), ["map", "ndcg", "num_q", pt.measures.NDCG@5], round={"nDCG@5" : 1})
        self.assertEqual(str(rtr.iloc[0]["nDCG@5"]), "0.5")


    def test_TREC_indexing(self):
        print("Writing index to " + self.test_dir)
        indexer = pt.TRECCollectionIndexer(self.test_dir)
        indexRef = indexer.index(pt.io.find_files(self.here + "/fixtures/vaswani_npl/corpus/"))
        self.assertIsNotNone(indexRef)
        index = pt.IndexFactory.of(indexRef)
        self.assertEqual(11429, index.getCollectionStatistics().getNumberOfDocuments())
        self.assertTrue(os.path.isfile(self.test_dir + '/data.direct.bf'))
