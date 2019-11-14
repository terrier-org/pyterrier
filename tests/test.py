import unittest, jnius_config,math, os
from main import BatchRetrieve
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

class TestBatchRetrieve(unittest.TestCase):
    def test_form_dataframe_with_string(self):
        input="light"
        exp_result = pd.DataFrame([["1", "light"]],columns=['qid','query'])
        result=BatchRetrieve.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_form_dataframe_with_list(self):
        input=["light","mathematical","electronic"]
        exp_result = pd.DataFrame([["1", "light"],["2", "mathematical"],["3", "electronic"]],columns=['qid','query'])
        result=BatchRetrieve.form_dataframe(input)
        self.assertTrue(exp_result.equals(result))

    def test_form_dataframe_with_tuple(self):
        input=("light","mathematical","electronic")
        exp_result = pd.DataFrame([["1", "light"],["2", "mathematical"],["3", "electronic"]],columns=['qid','query'])
        result=BatchRetrieve.form_dataframe(input)
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
