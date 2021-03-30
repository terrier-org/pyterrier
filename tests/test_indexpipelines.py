import pyterrier as pt
import tempfile
import unittest
from .base import BaseTestCase
import os
import pandas as pd
import shutil

class TestIndexPipelines(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        print("Created " + self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        print("Deleting " + self.test_dir)

    def test_add_dup(self):
        def _first(df):
            df2 = df.copy()
            df2["docno"] = df2["docno"] + "bis" 
            return pd.concat([df, df2])

        slider = pt.apply.generic(_first)
        indexer = pt.IterDictIndexer(self.test_dir)
        pipeline = slider >> indexer
        dataset = pt.get_dataset("irds:vaswani")
        #print(next(dataset.get_corpus_iter().gen))
        indexref = pipeline.index(dataset.get_corpus_iter())
        self.assertIsNotNone(indexref)
        index = pt.IndexFactory.of(indexref)
        self.assertEqual( index.getCollectionStatistics().getNumberOfDocuments(),  2 * len(dataset.get_corpus_iter()))

    def test_sliding(self):
        slider = pt.text.sliding("text", 10, 10, prepend_attr=None)
        indexer = pt.IterDictIndexer(self.test_dir)
        pipeline = slider >> indexer
        dataset = pt.get_dataset("irds:vaswani")
        indexref = pipeline.index(dataset.get_corpus_iter())
        self.assertIsNotNone(indexref)
        index = pt.IndexFactory.of(indexref)
        self.assertTrue( index.getCollectionStatistics().getNumberOfDocuments() > len(dataset.get_corpus_iter()))
