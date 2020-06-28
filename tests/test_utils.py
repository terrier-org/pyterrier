import pandas as pd

import pyterrier as pt
import unittest
import os
from .base import BaseTestCase
import shutil
import tempfile

class TestUtils(BaseTestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_trec(self):
        res = pd.DataFrame([["1", "d1", 5.3, 1]], columns=['qid', 'docno', 'score', 'rank'])
        res_dict = res.set_index(['qid', 'docno']).to_dict()
        for filename in ["rtr.res", "rtr.res.gz", "rtr.res.bz2"]:
            filepath = os.path.join(self.test_dir, filename)
            pt.Utils.write_results_trec(res, filepath)
            res2 = pt.Utils.parse_results_file(filepath)
            res2_dict = res2.set_index(['qid', 'docno']).to_dict()
            del res2_dict["name"]
            self.assertEqual(res_dict, res2_dict)

    def test_save_letor(self):
        import numpy as np
        res = pd.DataFrame([["1", "d1", 5.3, 1, np.array([1, 2])]], columns=['qid', 'docno', 'score', 'rank', 'features'])
        res_dict = res.set_index(['qid', 'docno']).to_dict()
        del res_dict["score"]
        del res_dict["rank"]
        for filename in ["rtr.letor", "rtr.letor.gz", "rtr.letor.bz2"]:
            filepath = os.path.join(self.test_dir, filename)
            pt.Utils.write_results_letor(res, filepath)
            res2 = pt.Utils.parse_letor_results_file(filepath)

            for ((i1, row1), (i2, row2)) in zip(res.iterrows(), res2.iterrows()):
                self.assertEqual(row1["qid"], row2["qid"])
                self.assertEqual(row1["docno"], row2["docno"])
                self.assertEqual(row1["docno"], row2["docno"])
                self.assertEqual(row1["qid"], row2["qid"])
                self.assertTrue(np.array_equal(row1["features"], row2["features"]))        

    def test_parse_trec_topics_file(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.trec"
        exp_result = pd.DataFrame([["1", "light"], ["2", "radiowave"], ["3", "sound"]], columns=['qid', 'query'])
        result = pt.Utils.parse_trec_topics_file(input)
        self.assertTrue(exp_result.equals(result))

    def test_convert_df_to_pytrec_eval_float(self):
        input = pd.DataFrame([["1", "1", 12.5], ["1", "7", 4.3], ["2", "12", 8.5]], columns=["qid", "docno", "score"])
        exp_result = {"1": {"1": 12.5, "7": 4.3}, "2": {"12": 8.5}}
        result = pt.Utils.convert_res_to_dict(input)
        self.assertEqual(exp_result, result)

    def test_convert_df_to_pytrec_eval_int(self):
        input = pd.DataFrame([["1", "1", 1], ["1", "7", 0], ["2", "12", 1]], columns=["qid", "docno", "score"])
        exp_result = {"1": {"1": 1, "7": 0}, "2": {"12": 1}}
        result = pt.Utils.convert_res_to_dict(input)
        self.assertEqual(exp_result, result)

    def test_parse_qrels(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/qrels"
        exp_result = pd.DataFrame([["1", "13", 1], ["1", "15", 1], ["2", "8", 1], ["2", "4", 1], ["2", "17", 1], ["3", "2", 1]], columns=['qid', 'docno', 'label'])
        result = pt.Utils.parse_qrels(input)
        #print(exp_result)
        #print(result)
        pd.testing.assert_frame_equal(exp_result, result)

    def test_evaluate(self):
        input_qrels = pd.DataFrame([["1", "12", 1], ["1", "26", 1], ["1", "5", 1], ["1", "6", 1], ["2", "12", 1], ["2", "13", 1], ["2", "7", 1], ["2", "17", 1]], columns=["qid", "docno", "label"])
        input_res = pd.DataFrame([["1", "12", 3.917300970672472], ["1", "17", 3.912008156607317], ["1", "5", 3.895776784815295], ["1", "6", 1.6976053561565434], ["1", "11394", 1.419217511596875],
                                  ["2", "12", 3.352655284198764], ["2", "13", 3.3410694508732677], ["2", "7", 3.32843147860022], ["2", "15", 3.210614190096991], ["2", "17", 1.3688610792424558], ["2", "25", 1.2673250497019404]],
                                 columns=['qid', 'docno', 'score'])
        exp_result = [0.6042, 0.9500]
        result = pt.Utils.evaluate(input_res, input_qrels, perquery=True)
        self.assertAlmostEqual(sum(exp_result) / len(exp_result), 0.7771, places=4)
        for i, item in enumerate(exp_result):
            self.assertAlmostEqual(result[str(i + 1)]["map"], item, places=4)
    
    def test_evaluate_ndcg_cut(self):
        input_qrels = pd.DataFrame([["1", "12", 1], ["1", "26", 1], ["1", "5", 1], ["1", "6", 1], ["2", "12", 1], ["2", "13", 1], ["2", "7", 1], ["2", "17", 1]], columns=["qid", "docno", "label"])
        input_res = pd.DataFrame([["1", "12", 3.917300970672472], ["1", "17", 3.912008156607317], ["1", "5", 3.895776784815295], ["1", "6", 1.6976053561565434], ["1", "11394", 1.419217511596875],
                                  ["2", "12", 3.352655284198764], ["2", "13", 3.3410694508732677], ["2", "7", 3.32843147860022], ["2", "15", 3.210614190096991], ["2", "17", 1.3688610792424558], ["2", "25", 1.2673250497019404]],
                                 columns=['qid', 'docno', 'score'])
        exp_result = [0.6042, 0.9500]
        result = pt.Utils.evaluate(input_res, input_qrels, metrics=["ndcg_cut_5"])
        self.assertTrue("ndcg_cut_5" in result)
        self.assertFalse("ndcg_cut_10" in result)

        result = pt.Utils.evaluate(input_res, input_qrels, perquery=True, metrics=["ndcg_cut_5"])
        self.assertTrue("ndcg_cut_5" in result["1"])
        self.assertFalse("ndcg_cut_10" in result["1"])


if __name__ == "__main__":
    unittest.main()
