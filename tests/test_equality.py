import unittest
import pyterrier as pt
from pyterrier._ops import RankCutoff, ScalarProduct, Compose, FeatureUnion
from .base import TempDirTestCase


class TestOpsEquality(unittest.TestCase):
    """Tests for __eq__ and __hash__ on operator transformers (no Java required)."""

    # ── RankCutoff ──────────────────────────────────────────────────────────

    def test_rankcutoff_eq_same_k(self):
        self.assertEqual(RankCutoff(10), RankCutoff(10))

    def test_rankcutoff_eq_different_k(self):
        self.assertNotEqual(RankCutoff(10), RankCutoff(20))

    def test_rankcutoff_eq_default_k(self):
        self.assertEqual(RankCutoff(), RankCutoff(1000))

    def test_rankcutoff_eq_wrong_type(self):
        self.assertNotEqual(RankCutoff(10), "not a transformer")

    def test_rankcutoff_hash_same_k(self):
        self.assertEqual(hash(RankCutoff(10)), hash(RankCutoff(10)))

    def test_rankcutoff_hash_different_k(self):
        self.assertNotEqual(hash(RankCutoff(10)), hash(RankCutoff(20)))

    def test_rankcutoff_in_set(self):
        s = {RankCutoff(10), RankCutoff(10), RankCutoff(20)}
        self.assertEqual(len(s), 2)

    def test_rankcutoff_as_dict_key(self):
        d = {RankCutoff(10): "ten", RankCutoff(20): "twenty"}
        self.assertEqual(d[RankCutoff(10)], "ten")

    # ── ScalarProduct ────────────────────────────────────────────────────────

    def test_scalarproduct_eq_same(self):
        self.assertEqual(ScalarProduct(2.0), ScalarProduct(2.0))

    def test_scalarproduct_eq_different(self):
        self.assertNotEqual(ScalarProduct(2.0), ScalarProduct(3.0))

    def test_scalarproduct_eq_wrong_type(self):
        self.assertNotEqual(ScalarProduct(2.0), 2.0)

    def test_scalarproduct_hash_same(self):
        self.assertEqual(hash(ScalarProduct(2.0)), hash(ScalarProduct(2.0)))

    def test_scalarproduct_hash_different(self):
        self.assertNotEqual(hash(ScalarProduct(2.0)), hash(ScalarProduct(3.0)))

    def test_scalarproduct_in_set(self):
        s = {ScalarProduct(2.0), ScalarProduct(2.0), ScalarProduct(3.0)}
        self.assertEqual(len(s), 2)

    # ── Compose ──────────────────────────────────────────────────────────────

    def test_compose_eq_same(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        b = RankCutoff(10) >> ScalarProduct(2.0)
        self.assertEqual(a, b)

    def test_compose_eq_different_order(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        b = ScalarProduct(2.0) >> RankCutoff(10)
        self.assertNotEqual(a, b)

    def test_compose_eq_different_k(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        b = RankCutoff(99) >> ScalarProduct(2.0)
        self.assertNotEqual(a, b)

    def test_compose_eq_wrong_type(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        self.assertNotEqual(a, "not a pipeline")

    def test_compose_hash_same(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        b = RankCutoff(10) >> ScalarProduct(2.0)
        self.assertEqual(hash(a), hash(b))

    def test_compose_hash_different(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        b = RankCutoff(99) >> ScalarProduct(2.0)
        self.assertNotEqual(hash(a), hash(b))

    def test_compose_in_set(self):
        a = RankCutoff(10) >> ScalarProduct(2.0)
        b = RankCutoff(10) >> ScalarProduct(2.0)
        c = RankCutoff(99) >> ScalarProduct(2.0)
        s = {a, b, c}
        self.assertEqual(len(s), 2)

    def test_compose_three_stages_eq(self):
        a = RankCutoff(10) >> ScalarProduct(2.0) >> RankCutoff(5)
        b = RankCutoff(10) >> ScalarProduct(2.0) >> RankCutoff(5)
        self.assertEqual(a, b)

    # ── FeatureUnion ─────────────────────────────────────────────────────────

    def test_featureunion_eq_same(self):
        a = ScalarProduct(1.0) ** ScalarProduct(2.0)
        b = ScalarProduct(1.0) ** ScalarProduct(2.0)
        self.assertEqual(a, b)

    def test_featureunion_eq_different(self):
        a = ScalarProduct(1.0) ** ScalarProduct(2.0)
        b = ScalarProduct(1.0) ** ScalarProduct(9.0)
        self.assertNotEqual(a, b)

    def test_featureunion_hash_same(self):
        a = ScalarProduct(1.0) ** ScalarProduct(2.0)
        b = ScalarProduct(1.0) ** ScalarProduct(2.0)
        self.assertEqual(hash(a), hash(b))

    def test_featureunion_in_set(self):
        a = ScalarProduct(1.0) ** ScalarProduct(2.0)
        b = ScalarProduct(1.0) ** ScalarProduct(2.0)
        c = ScalarProduct(1.0) ** ScalarProduct(9.0)
        s = {a, b, c}
        self.assertEqual(len(s), 2)


class TestRetrieverEquality(TempDirTestCase):
    """Tests for Retriever.__eq__ and __hash__ (requires Java + index)."""

    def _make_index(self):
        import pandas as pd
        docs = pd.DataFrame({
            'docno': ['d1', 'd2', 'd3'],
            'text': ['cat sat mat', 'dog ran far', 'bird flew high']
        })
        indexer = pt.IterDictIndexer(
            self.test_dir,
            stopwords=pt.TerrierStopwords.none,
            stemmer=pt.TerrierStemmer.none,
        )
        return indexer.index(docs.to_dict(orient='records'))

    def test_retriever_eq_same_config(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="BM25")
        self.assertEqual(r1, r2)

    def test_retriever_eq_different_wmodel(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="TF_IDF")
        self.assertNotEqual(r1, r2)

    def test_retriever_eq_different_num_results(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25", num_results=10)
        r2 = pt.terrier.Retriever(indexref, wmodel="BM25", num_results=100)
        self.assertNotEqual(r1, r2)

    def test_retriever_eq_wrong_type(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        self.assertNotEqual(r1, "not a retriever")

    def test_retriever_hash_same_config(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="BM25")
        self.assertEqual(hash(r1), hash(r2))

    def test_retriever_hash_different_wmodel(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="TF_IDF")
        self.assertNotEqual(hash(r1), hash(r2))

    def test_retriever_in_set(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r3 = pt.terrier.Retriever(indexref, wmodel="TF_IDF")
        s = {r1, r2, r3}
        self.assertEqual(len(s), 2)

    def test_retriever_as_dict_key(self):
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="BM25")
        d = {r1: "bm25"}
        self.assertEqual(d[r2], "bm25")

    def test_compose_retriever_eq(self):
        """Two identical pipelines starting with the same Retriever should be equal."""
        indexref = self._make_index()
        r1 = pt.terrier.Retriever(indexref, wmodel="BM25")
        r2 = pt.terrier.Retriever(indexref, wmodel="BM25")
        cutoff1 = r1 % 10
        cutoff2 = r2 % 10
        self.assertEqual(cutoff1, cutoff2)


if __name__ == "__main__":
    unittest.main()