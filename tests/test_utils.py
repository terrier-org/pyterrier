
import pandas as pd

import pyterrier as pt
import unittest, math, os, ast, statistics

   

class TestUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUtils, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init()

    def test_parse_trec_topics_file(self):
        input=os.path.dirname(os.path.realpath(__file__))+"/fixtures/topics.trec"
        exp_result=pd.DataFrame([["1","light"],["2","radiowave"],["3","sound"]],columns=['qid','query'])
        result=pt.Utils.parse_trec_topics_file(input)
        self.assertTrue(exp_result.equals(result))

    def test_convert_df_to_pytrec_eval_float(self):
        input=pd.DataFrame([["1","1",12.5],["1","7",4.3],["2","12",8.5]],columns=["qid","docno","score"])
        exp_result={"1":{"1":12.5,"7":4.3},"2":{"12":8.5}}
        result=pt.Utils.convert_res_to_dict(input)
        self.assertEqual(exp_result,result)

    def test_convert_df_to_pytrec_eval_int(self):
        input=pd.DataFrame([["1","1",1],["1","7",0],["2","12",1]],columns=["qid","docno","score"])
        exp_result={"1":{"1":1,"7":0},"2":{"12":1}}
        result=pt.Utils.convert_res_to_dict(input)
        self.assertEqual(exp_result,result)

    def test_parse_qrels(self):
        input=os.path.dirname(os.path.realpath(__file__))+"/fixtures/qrels"
        exp_result=pd.DataFrame(
            [["1","13",1],["1","15",1],["2","8",1],["2","4",1],["2","17",1],["3","2",1]]
            ,columns=['qid','docno','label'])
        result=pt.Utils.parse_qrels(input)
        print(exp_result)
        print(result)
        pd.testing.assert_frame_equal(exp_result,result)

    def test_evaluate(self):
        input_qrels=pd.DataFrame([  ["1","12",1],["1","26",1],["1","5",1],["1","6",1],
                                    ["2","12",1],["2","13",1],["2","7",1],["2","17",1]],
                                    columns=["qid","docno","label"])
        input_res = pd.DataFrame([  ["1","12",3.917300970672472],["1","17",3.912008156607317],["1","5",3.895776784815295],["1","6",1.6976053561565434],["1","11394",1.419217511596875],
                                    ["2","12",3.352655284198764],["2","13",3.3410694508732677],["2","7",3.32843147860022],["2","15",3.210614190096991],["2","17",1.3688610792424558],["2","25",1.2673250497019404]],
                                    columns=['qid','docno','score'])
        exp_result=[0.6042,0.9500]
        result = pt.Utils.evaluate(input_res,input_qrels, perquery=True)
        #mapValue=result["map"]
        #result = ast.literal_eval(result)
        self.assertAlmostEqual(sum(exp_result)/len(exp_result),0.7771, places=4)
        for i,item in enumerate(exp_result):
            self.assertAlmostEqual(result[str(i+1)]["map"], item,places=4)



if __name__ == "__main__":
    unittest.main()
