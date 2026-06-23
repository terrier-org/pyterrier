# this has test cases that do not involve calling pt.Experiment directly, but test underlying functionality that is used by pt.Experiment. These tests are not specific to any execution plan, but are included here for completeness.

import pandas as pd
import pyterrier as pt
import os
from .base import BaseTestCase
import ir_measures


class TestExperimentOther(BaseTestCase):

    def test_mrt_is_average(self):
        from pyterrier._evaluation._rendering import RenderFromPerQuery
        # 2 queries, mrt passed in as 500ms (already averaged)
        r = RenderFromPerQuery(['bm25'])
        r.add_metrics(0, {'q1': {'AP': 1.0}, 'q2': {'AP': 0.8}}, 500.0)
        df = r.averages(mrt_needed=True)
        self.assertEqual(500.0, df['mrt'].iloc[0])

    def test_experiment_render(self):
        from pyterrier._evaluation._rendering import RenderFromPerQuery
        r = RenderFromPerQuery(['bm25'])
        r.add_metrics(0, {'q1': {'AP' : 1.0}, 'q2': {'AP' : 0.8}}, 1000)
        df = r.perquery()
        self.assertEqual(2, len(df))
        self.assertEqual(1.8, df['value'].sum())
        df = r.averages()
        self.assertEqual(1, len(df))
        self.assertEqual(0.9, df['AP'].iloc[0])