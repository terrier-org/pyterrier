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
                index.close()

    def test_antique(self):
        dataset = pt.datasets.get_dataset('irds:antique/test')
        with self.subTest('topics'):
            topics = dataset.get_topics()
            self.assertEqual(len(topics), 200)
            self.assertEqual(topics['query'][0], 'how can we get concentration onsomething') # removes "?"
        with self.subTest('topics - no tokenisation'):
            topics = dataset.get_topics(tokenise_query=False)
            self.assertEqual(len(topics), 200)
            self.assertEqual(topics['query'][0], 'how can we get concentration onsomething?')

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

    def test_results(self):
        if "PYTERRIER_TEST_IRDS_RESULTS" in os.environ:
            dataset = pt.datasets.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
            with self.subTest('results'):
                results = dataset.get_results()
                self.assertEqual(41042, len(results))
                self.assertEqual(['qid', 'docno', 'score', 'query'], list(results.columns))
                self.assertEqual('1037798', results.iloc[0].qid)
                self.assertEqual('1031599', results.iloc[0].docno)
                self.assertEqual(0., results.iloc[0].score)
                self.assertEqual('who is robert gray', results.iloc[0].query)
                # ensure it's terrier-tokenised (orig text is "tracheids are part of _____.")
                self.assertEqual('tracheids are part of', results[results.qid=='1124210'].iloc[0].query)

if __name__ == '__main__':
    unittest.main()
