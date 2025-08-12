import unittest
import warnings
import pandas as pd
import pyterrier as pt


class TestValidate(unittest.TestCase):
    def test_columns_valid(self):
        with self.subTest('df'):
            pt.validate.columns(
                pd.DataFrame({'qid': [1, 2], 'text': ['hello', 'world']}),
                includes=['qid'],
                excludes=['docno'])
        with self.subTest('cols'):
            pt.validate.columns(
                ['qid', 'text'],
                includes=['qid'],
                excludes=['docno'])
        with self.subTest('iterdict'):
            pt.validate.columns_iter(
                pt.utils.peekable([{'qid': '1', 'text': '0'}]),
                includes=['qid'],
                excludes=['docno'])

    def test_columns_missing_column(self):
        with self.subTest('df'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.columns(
                    pd.DataFrame({'text': ['hello', 'world']}),
                    includes=['qid'])
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
        with self.subTest('cols'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.columns(
                    ['text', 'other'],
                    includes=['qid'])
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
        with self.subTest('iterdict'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.columns_iter(
                    pt.utils.peekable([{'text': 'hello', 'other': 'world'}]),
                    includes=['qid'])
            self.assertIn('qid', cm.exception.modes[0].missing_columns)

    def test_columns_extra_column(self):
        with self.subTest('df'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.columns(
                    pd.DataFrame({'qid': [1, 2], 'docno': [3, 4]}),
                    includes=['qid'],
                    excludes=['docno'])
            self.assertIn('docno', cm.exception.modes[0].extra_columns)
        with self.subTest('cols'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.columns(
                    ['qid', 'docno'],
                    includes=['qid'],
                    excludes=['docno'])
            self.assertIn('docno', cm.exception.modes[0].extra_columns)
        with self.subTest('iterdict'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.columns_iter(
                    pt.utils.peekable([{'qid': '1', 'docno': '2'}]),
                    includes=['qid'],
                    excludes=['docno'])
            self.assertIn('docno', cm.exception.modes[0].extra_columns)

    def test_columns_warn_mode(self):
        with self.subTest('df'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pt.validate.columns(pd.DataFrame({'text': ['hello', 'world']}), includes=['qid'], warn=True)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, pt.validate.InputValidationWarning))
                self.assertIn("DataFrame does not match required columns", str(w[0].message))
        with self.subTest('cols'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pt.validate.columns(['text', 'other'], includes=['qid'], warn=True)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, pt.validate.InputValidationWarning))
                self.assertIn("DataFrame does not match required columns", str(w[0].message))
        with self.subTest('iterdict'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pt.validate.columns_iter(
                    pt.utils.peekable([{'text': 'hello', 'other': 'world'}]),
                    includes=['qid'],
                    warn=True)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, pt.validate.InputValidationWarning))
                self.assertIn("Input does not match required columns", str(w[0].message))

    def test_query_frame_valid(self):
        with self.subTest('df'):
            pt.validate.query_frame(pd.DataFrame({'qid': [1, 2], 'query': ['cat', 'dog'], 'a': [1, 2]}), extra_columns=['a'])
        with self.subTest('cols'):
            pt.validate.query_frame(['qid', 'query', 'a'], extra_columns=['a'])
        with self.subTest('iterdict'):
            pt.validate.query_iter(pt.utils.peekable([{'qid': '1', 'query': 'cat', 'a': 1}]), extra_columns=['a'])

    def test_query_frame_invalid(self):
        with self.subTest('df'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.query_frame(pd.DataFrame({'query': ['cat', 'dog']}))
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
        with self.subTest('cols'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.query_frame(['query'])
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
        with self.subTest('iterdict'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.query_iter(pt.utils.peekable([{'query': 'cat'}]))
            self.assertIn('qid', cm.exception.modes[0].missing_columns)

    def test_document_frame_valid(self):
        with self.subTest('df'):
            pt.validate.document_frame(pd.DataFrame({'docno': [101, 102], 'text': ['hello', 'world']}))
        with self.subTest('cols'):
            pt.validate.document_frame(['docno', 'text'])
        with self.subTest('iterdict'):
            pt.validate.document_iter(pt.utils.peekable([{'docno': '101', 'text': 'hello'}]))

    def test_document_frame_invalid(self):
        with self.subTest('df'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.document_frame(pd.DataFrame({'text': ['hello', 'world']}))
            self.assertIn('docno', cm.exception.modes[0].missing_columns)
        with self.subTest('cols'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.document_frame(['text'])
            self.assertIn('docno', cm.exception.modes[0].missing_columns)
        with self.subTest('iterdict'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.document_iter(pt.utils.peekable([{'text': 'hello'}]))
            self.assertIn('docno', cm.exception.modes[0].missing_columns)

    def test_result_frame_valid(self):
        with self.subTest('df'):
            pt.validate.result_frame(pd.DataFrame({'qid': [1, 2], 'docno': [101, 102], 'score': [0.8, 0.6]}))
        with self.subTest('cols'):
            pt.validate.result_frame(['qid', 'docno', 'score'])
        with self.subTest('iterdict'):
            pt.validate.result_iter(pt.utils.peekable([{'qid': '1', 'docno': '101', 'score': 0.8}]))

    def test_result_frame_invalid(self):
        with self.subTest('df'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.result_frame(pd.DataFrame({'score': [0.8, 0.6]}))
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
            self.assertIn('docno', cm.exception.modes[0].missing_columns)
        with self.subTest('cols'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.result_frame(['score'])
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
            self.assertIn('docno', cm.exception.modes[0].missing_columns)
        with self.subTest('iterdict'):
            with self.assertRaises(pt.validate.InputValidationError) as cm:
                pt.validate.result_iter(pt.utils.peekable([{'score': 0.8}]))
            self.assertIn('qid', cm.exception.modes[0].missing_columns)
            self.assertIn('docno', cm.exception.modes[0].missing_columns)


if __name__ == '__main__':
    unittest.main()
