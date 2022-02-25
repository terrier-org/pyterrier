import unittest
import pyterrier as pt, pandas as pd
from .base import BaseTestCase

class TestDebug(BaseTestCase):

    def test_num_rows(self):
        df = pt.new.queries(["a"])
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            pt.debug.print_num_rows().transform(df)
        self.assertTrue("num_rows 1: 1" in f.getvalue())