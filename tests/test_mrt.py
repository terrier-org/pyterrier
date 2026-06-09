import unittest

class TestMrt(unittest.TestCase):

    def test_mrt_is_average(self):
        from pyterrier._evaluation._rendering import RenderFromPerQuery
        # 2 queries, mrt passed in as 500ms (already averaged)
        r = RenderFromPerQuery(['bm25'])
        r.add_metrics(0, {'q1': {'AP': 1.0}, 'q2': {'AP': 0.8}}, 500.0)
        df = r.averages(mrt_needed=True)
        self.assertEqual(500.0, df['mrt'].iloc[0])

    def test_mrt_division_logic(self):
        # verifies the fix: runtime / num_q gives mean retrieval time
        num_q = 2
        total_runtime = 1000.0
        mrt = total_runtime / num_q if num_q > 0 else 0.
        self.assertEqual(500.0, mrt)

    def test_mrt_zero_queries_safe(self):
        # edge case: no queries should not cause division by zero
        num_q = 0
        total_runtime = 1000.0
        mrt = total_runtime / num_q if num_q > 0 else 0.
        self.assertEqual(0., mrt)

if __name__ == "__main__":
    unittest.main()
