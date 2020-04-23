import unittest
import pyterrier as pt
from .base import BaseTestCase

class TestDatasets(BaseTestCase):

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

        # do it once again, to ensure it works locally
        dataset = pt.datasets.get_dataset("vaswani")
        topics = dataset.get_topics()
        self.assertIsNotNone(topics)
        self.assertEqual(len(topics), 93)

if __name__ == "__main__":
    unittest.main()