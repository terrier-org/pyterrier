import unittest, jnius_config,math, os, ast, statistics
from main import BatchRetrieve, Utils
import pandas as pd
# jnius_config.set_classpath("terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass

def parse_query_result(filename):
    results=[]
    with open(filename, 'r') as file:
        for line in file:
            split_line=line.strip("\n").split(" ")
            results.append([split_line[1],float(split_line[2])])
    return results

def parse_res_file(filename):
    results=[]
    with open(filename, 'r') as file:
        for line in file:
            split_line=line.strip("\n").split(" ")
            results.append([split_line[0],split_line[2],float(split_line[4])])
    return results

class TestUtils(unittest.TestCase):
    def test_parse_trec_topics_file(self):
        input=os.path.dirname(os.path.realpath(__file__))+"/fixtures/topics.trec"
        exp_result=pd.DataFrame([["1","light"],["2","radiowave"],["3","sound"]],columns=['qid','query'])
        result=Utils.parse_trec_topics_file(input)
        self.assertTrue(exp_result.equals(result))

    def test_convert_df_to_pytrec_eval_float(self):
        input=pd.DataFrame([["1","1",12.5],["1","7",4.3],["2","12",8.5]],columns=["qid","docno","score"])
        exp_result={"1":{"1":12.5,"7":4.3},"2":{"12":8.5}}
        result=Utils.convert_df_to_pytrec_eval(input, score_int=False)
        self.assertEquals(exp_result,result)

    def test_convert_df_to_pytrec_eval_int(self):
        input=pd.DataFrame([["1","1",1],["1","7",0],["2","12",1]],columns=["qid","docno","score"])
        exp_result={"1":{"1":1,"7":0},"2":{"12":1}}
        result=Utils.convert_df_to_pytrec_eval(input, score_int=True)
        self.assertEquals(exp_result,result)

    def test_parse_qrels(self):
        input=os.path.dirname(os.path.realpath(__file__))+"/fixtures/qrels"
        exp_result=pd.DataFrame([["1","13","1"],["1","15","1"],["2","8","1"],["2","4","1"],["2","17","1"],["3","2","1"]],columns=['qid','docno','score'])
        result=Utils.parse_qrels(input)
        self.assertTrue(exp_result.equals(result))

    def test_evaluate(self):
        input_qrels=pd.DataFrame([  ["1","12",1],["1","26",1],["1","5",1],["1","6",1],
                                    ["2","12",1],["2","13",1],["2","7",1],["2","17",1]],
                                    columns=["qid","docno","score"])
        input_res = pd.DataFrame([  ["1","12",3.917300970672472],["1","17",3.912008156607317],["1","5",3.895776784815295],["1","6",1.6976053561565434],["1","11394",1.419217511596875],
                                    ["2","12",3.352655284198764],["2","13",3.3410694508732677],["2","7",3.32843147860022],["2","15",3.210614190096991],["2","17",1.3688610792424558],["2","25",1.2673250497019404]],
                                    columns=['qid','docno','score'])
        exp_result=[0.6042,0.9500]
        result = Utils.evaluate(input_res,input_qrels)
        result = ast.literal_eval(result)
        self.assertAlmostEquals(sum(exp_result)/len(exp_result),0.7771, places=4)
        for i,item in enumerate(exp_result):
            self.assertAlmostEquals(result[str(i+1)]["map"], item,places=4)

class TestBatchRetrieve(unittest.TestCase):
    def test_form_dataframe_with_string(self):
        input="light"
        exp_result = pd.DataFrame([["1", "light"]],columns=['qid','query'])
        result=Utils.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_form_dataframe_with_list(self):
        input=["light","mathematical","electronic"]
        exp_result = pd.DataFrame([["1", "light"],["2", "mathematical"],["3", "electronic"]],columns=['qid','query'])
        result=Utils.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_form_dataframe_throws_assertion_error(self):
        input=("light","mathematical",25)
        self.assertRaises(AssertionError,Utils.form_dataframe,input)

    def test_form_dataframe_with_tuple(self):
        input=("light","mathematical","electronic")
        exp_result = pd.DataFrame([["1", "light"],["2", "mathematical"],["3", "electronic"]],columns=['qid','query'])
        result=Utils.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_one_term_query_correct_docid_and_score(self):
        JIR = autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of("./index/data.properties")
        retr = BatchRetrieve(indexref)
        result = retr.transform("light")
        exp_result = parse_query_result(os.path.dirname(os.path.realpath(__file__))+"/fixtures/light_results")
        for index,row in result.iterrows():
            self.assertEquals(row['docno'], exp_result[index][0])
            self.assertAlmostEquals(row['score'], exp_result[index][1])

    def test_two_term_query_correct_qid_docid_score(self):
        JIR = autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of("./index/data.properties")
        retr = BatchRetrieve(indexref)
        input=pd.DataFrame([["1", "Stability"],["2", "Generator"]],columns=['qid','query'])
        result = retr.transform(input)
        exp_result=parse_res_file(os.path.dirname(os.path.realpath(__file__))+"/fixtures/two_queries_result")
        for index,row in result.iterrows():
            self.assertEquals(row['qid'], exp_result[index][0])
            self.assertEquals(row['docno'], exp_result[index][1])
            self.assertAlmostEquals(row['score'], exp_result[index][2])




if __name__ == "__main__":
    unittest.main()
