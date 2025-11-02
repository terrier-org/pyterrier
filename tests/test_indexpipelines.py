import pyterrier as pt
import tempfile
import unittest
from .base import TempDirTestCase
import os
import pandas as pd
import shutil

class TestIndexPipelines(TempDirTestCase):

    def test_add_dup(self):
        def _first(df):
            df2 = df.copy()
            df2["docno"] = df2["docno"] + "bis" 
            return pd.concat([df, df2])

        slider = pt.apply.generic(_first)
        indexer = pt.IterDictIndexer(self.test_dir)
        pipeline = slider >> indexer
        dataset = pt.get_dataset("irds:vaswani")
        count = 200
        indexref = pipeline.index(dataset.get_corpus_iter(count=count))
        self.assertIsNotNone(indexref)
        index = pt.IndexFactory.of(indexref)
        self.assertEqual( index.getCollectionStatistics().getNumberOfDocuments(),  2 * count)

    def test_sliding_title_two(self):
        corpus = [{"docno" : "d1", "text" : "A B C", "title" : "this is a title"}]
        slider = pt.text.sliding("text", 2, 1, prepend_attr="title")
        indexer = pt.IterDictIndexer(self.test_dir)
        pipeline = slider >> indexer
        indexref = pipeline.index(corpus)
        self.assertIsNotNone(indexref)
        index = pt.IndexFactory.of(indexref)
        # we should get 2 passages in the resulting index
        self.assertEqual(2, index.getCollectionStatistics().getNumberOfDocuments())

    def test_sliding_title_one(self):
        corpus = [{"docno" : "d1", "text" : "A B", "title" : "this is a title"}]
        slider = pt.text.sliding("text", 2, 1, prepend_attr="title")
        indexer = pt.IterDictIndexer(self.test_dir)
        pipeline = slider >> indexer
        indexref = pipeline.index(corpus)
        self.assertIsNotNone(indexref)
        index = pt.IndexFactory.of(indexref)
        # we should get 1 passages in the resulting index
        self.assertEqual(1, index.getCollectionStatistics().getNumberOfDocuments())


    def test_sliding(self):
        slider = pt.text.sliding("text", 10, 10, prepend_attr=None)
        indexer = pt.IterDictIndexer(self.test_dir)
        pipeline = slider >> indexer
        dataset = pt.get_dataset("irds:vaswani")
        count = 200
        indexref = pipeline.index(dataset.get_corpus_iter(count=count))
        self.assertIsNotNone(indexref)
        index = pt.IndexFactory.of(indexref)
        self.assertTrue( index.getCollectionStatistics().getNumberOfDocuments() > count)
