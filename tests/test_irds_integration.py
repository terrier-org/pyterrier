import pyterrier as pt
import tempfile
import unittest
from .base import BaseTestCase
import os
import pandas as pd

class TestIrDatasetsIntegration(BaseTestCase):

    def test_vaswani(self):
        dataset = pt.datasets.get_dataset('irds:vaswani')
        with self.subTest('topics'):
            topics = dataset.get_topics()
            self.assertEqual(len(topics), 93)
        with self.subTest('qrels'):
            qrels = dataset.get_qrels()
            self.assertEqual(len(qrels), 2083)
        with self.subTest('corpus'):
            corpus = dataset.get_corpus_iter()
            corpus = list(corpus)
            self.assertEqual(len(corpus), 11429)
            with tempfile.TemporaryDirectory() as d:
                indexer = pt.index.IterDictIndexer(d)
                indexref = indexer.index(dataset.get_corpus_iter(), fields=('text',))
                index = pt.IndexFactory.of(indexref)
                self.assertEqual(index.lexicon['bit'].frequency, 33)

    def test_cord19(self):
        if "PYTERRIER_TEST_IRDS_CORD" in os.environ:
            dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
            with self.subTest('topics'):
                topics = dataset.get_topics()
                self.assertEqual(len(topics), 50)
            with self.subTest('qrels'):
                qrels = dataset.get_qrels()
                self.assertEqual(len(qrels), 69318)
            with self.subTest('corpus'):
                corpus = dataset.get_corpus_iter()
                corpus = list(corpus)
                self.assertEqual(len(corpus), 192509)
            with self.subTest('indexing'):
                with tempfile.TemporaryDirectory() as d:
                    indexer = pt.index.IterDictIndexer(d)
                    indexref = indexer.index(dataset.get_corpus_iter(), fields=('title', 'abstract'))
                    index = pt.IndexFactory.of(indexref)
                    self.assertEqual(index.lexicon['covid'].frequency, 200582)

if __name__ == '__main__':
    unittest.main()
