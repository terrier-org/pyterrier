import pandas as pd
import unittest, math, os, ast, statistics
import pyterrier as pt




class TestDatasets(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDatasets, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init()

    def test_vaswani(self):
        import pyterrier as pt
        dataset = pt.datasets.get_dataset("vaswani")
        self.assertIsNotNone(dataset)

        topics = dataset.get_topics()
        self.assertIsNotNone(topics)
        self.assertEqual(len(topics), 93)

        qrels = dataset.get_qrels()
        self.assertIsNotNone(qrels)
        self.assertEqual(len(qrels), 2083)

        indexref = dataset.get_index()
        self.assertIsNotNone(indexref)
        with pt.IndexFactory.of(indexref) as index:
            self.assertIsNotNone(index)
            self.assertEqual(index.getCollectionStatistics().getNumberOfDocuments(), 11429)
        
        #do it once again, to ensure it works locally
        dataset = pt.datasets.get_dataset("vaswani")
        topics = dataset.get_topics()
        self.assertIsNotNone(topics)
        self.assertEqual(len(topics), 93)
