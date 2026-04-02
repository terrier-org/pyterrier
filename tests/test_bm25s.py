import os
import shutil
import tempfile
import unittest

import pandas as pd
import pyterrier as pt


CORPUS = [
    {'docno': 'd1', 'text': 'the cat sat on the mat'},
    {'docno': 'd2', 'text': 'the dog barked at the cat'},
    {'docno': 'd3', 'text': 'the bird flew away from the tree'},
    {'docno': 'd4', 'text': 'a fish swims in the pond every day'},
]


class TestBM25SIndex(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.test_dir, 'test_index')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def test_not_built_initially(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        self.assertFalse(index.built())

    def test_index_via_index_method(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        index.index(CORPUS)
        self.assertTrue(index.built())

    def test_index_via_indexer(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        indexer = index.indexer()
        self.assertIsInstance(indexer, pt.bm25s.BM25SIndexer)
        indexer.index(CORPUS)
        self.assertTrue(index.built())

    def test_index_raises_if_already_built(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        index.index(CORPUS)
        with self.assertRaises(AssertionError):
            index.index(CORPUS)

    def test_index_requires_docno_field(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        with self.assertRaises(AssertionError):
            index.index([{'text': 'hello world'}])

    def test_index_requires_text_field(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        with self.assertRaises(AssertionError):
            index.index([{'docno': 'd1', 'num': 42}])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_index(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        index.index(CORPUS)
        return index

    def test_bm25_returns_retriever(self):
        index = self._build_index()
        retriever = index.bm25()
        self.assertIsInstance(retriever, pt.bm25s.BM25SRetriever)

    def test_retriever_returns_dataframe(self):
        index = self._build_index()
        retriever = index.bm25()
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['cat']})
        results = retriever.transform(queries)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('qid', results.columns)
        self.assertIn('docno', results.columns)
        self.assertIn('score', results.columns)
        self.assertIn('rank', results.columns)

    def test_retrieval_returns_correct_docnos(self):
        index = self._build_index()
        retriever = index.bm25()
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['cat']})
        results = retriever.transform(queries)
        # cat appears in d1 and d2; both should be in top results
        returned_docnos = set(results['docno'].tolist())
        self.assertIn('d1', returned_docnos)
        self.assertIn('d2', returned_docnos)

    def test_retrieval_query_column_preserved(self):
        index = self._build_index()
        retriever = index.bm25()
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['cat']})
        results = retriever.transform(queries)
        self.assertIn('query', results.columns)
        self.assertTrue((results['query'] == 'cat').all())

    def test_retrieval_num_results(self):
        index = self._build_index()
        retriever = index.retriever(num_results=2)
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['the']})
        results = retriever.transform(queries)
        self.assertLessEqual(len(results[results['qid'] == 'q1']), 2)

    def test_retrieval_multiple_queries(self):
        index = self._build_index()
        retriever = index.bm25()
        queries = pd.DataFrame({
            'qid': ['q1', 'q2'],
            'query': ['cat', 'bird'],
        })
        results = retriever.transform(queries)
        self.assertEqual(set(results['qid'].unique()), {'q1', 'q2'})

    def test_retrieval_ranks_start_from_zero(self):
        index = self._build_index()
        retriever = index.bm25()
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['cat']})
        results = retriever.transform(queries)
        q1_results = results[results['qid'] == 'q1']
        self.assertEqual(q1_results['rank'].min(), 0)

    def test_retrieval_empty_query(self):
        index = self._build_index()
        retriever = index.bm25()
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['']})
        results = retriever.transform(queries)
        # Empty query should return no results
        self.assertEqual(len(results), 0)

    def test_search_shorthand(self):
        index = self._build_index()
        retriever = index.bm25()
        results = retriever.search('cat')
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)

    # ------------------------------------------------------------------
    # Pipeline composition
    # ------------------------------------------------------------------

    def test_pipeline_compose(self):
        index = self._build_index()
        pipeline = index.bm25() >> pt.apply.generic(lambda df: df.head(1))
        queries = pd.DataFrame({'qid': ['q1'], 'query': ['cat']})
        results = pipeline.transform(queries)
        self.assertEqual(len(results), 1)

    # ------------------------------------------------------------------
    # Artifact discovery
    # ------------------------------------------------------------------

    def test_artifact_load(self):
        index = self._build_index()
        loaded = pt.Artifact.load(self.index_path)
        self.assertIsInstance(loaded, pt.bm25s.BM25SIndex)

    def test_repr(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        self.assertIn('BM25SIndex', repr(index))

    # ------------------------------------------------------------------
    # Custom BM25 parameters
    # ------------------------------------------------------------------

    def test_custom_bm25_parameters(self):
        index = pt.bm25s.BM25SIndex(self.index_path)
        index.indexer(k1=1.2, b=0.8).index(CORPUS)
        retriever = index.bm25()
        results = retriever.search('cat')
        self.assertGreater(len(results), 0)

    def test_multiple_text_fields(self):
        corpus = [
            {'docno': 'd1', 'title': 'cats', 'body': 'the cat sat on the mat'},
            {'docno': 'd2', 'title': 'dogs', 'body': 'the dog barked at the cat'},
        ]
        index = pt.bm25s.BM25SIndex(self.index_path)
        index.indexer(text_attrs=['title', 'body']).index(corpus)
        retriever = index.bm25()
        results = retriever.search('cat')
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
