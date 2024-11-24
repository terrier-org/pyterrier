import unittest
import pandas as pd
import pyterrier as pt
from .base import BaseTestCase

class TestIndexer(BaseTestCase):
    def test_transform_impl_none(self):
        class TestIndexer(pt.Indexer):
            pass
        indexer = TestIndexer()
        with self.assertRaises(NotImplementedError):
            indexer.transform(pd.DataFrame())
        with self.assertRaises(NotImplementedError):
            indexer.transform_iter([])
        with self.assertRaises(NotImplementedError):
            indexer([])

    def test_transform_impl_transform(self):
        class TestIndexer(pt.Indexer):
            def transform(self, inp):
                return inp
        indexer = TestIndexer()
        inp = pd.DataFrame()
        pd.testing.assert_frame_equal(inp, indexer.transform(inp))
        inp = []
        self.assertEqual([], list(indexer.transform_iter(inp)))
        self.assertEqual([], indexer(inp))

    def test_transform_impl_transform_iter(self):
        class TestIndexer(pt.Indexer):
            def transform_iter(self, inp):
                return inp
        indexer = TestIndexer()
        inp = pd.DataFrame()
        pd.testing.assert_frame_equal(inp, indexer.transform(inp))
        inp = []
        self.assertEqual(inp, indexer.transform_iter(inp))
        self.assertEqual([], indexer(inp))

    def test_transform_impl_both(self):
        class TestIndexer(pt.Indexer):
            def transform_iter(self, inp):
                return inp
            def transform(self, inp):
                return inp
        indexer = TestIndexer()
        inp = pd.DataFrame()
        pd.testing.assert_frame_equal(inp, indexer.transform(inp))
        inp = []
        self.assertEqual(inp, indexer.transform_iter(inp))
        self.assertEqual([], indexer(inp))


if __name__ == "__main__":
    unittest.main()
