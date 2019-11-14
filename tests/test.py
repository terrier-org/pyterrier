import unittest, jnius_config,math, os
from main import BatchRetrieve
import pandas as pd
# jnius_config.set_classpath("terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass

def parse_res_file(filename):
    results=[]
    with open(filename, 'r') as file:
        for line in file:
            split_line=line.strip("\n").split(" ")
            results.append([split_line[1],float(split_line[2])])
    return results

class TestBatchRetrieve(unittest.TestCase):
    def test_one_term_query_correct_docid_and_score(self):
        JIR = autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of("./index/data.properties")
        retr = BatchRetrieve(indexref)
        result = retr.transform("light")
        exp_result = parse_res_file(os.path.dirname(os.path.realpath(__file__))+"/fixtures/light_results")
        for index,row in result.iterrows():
            self.assertEquals(row['docno'], exp_result[index][0])
            self.assertAlmostEquals(row['score'], exp_result[index][1])

    def test_two_term_query_correct_docid_and_score(self):
        JIR = autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of("./index/data.properties")
        retr = BatchRetrieve(indexref)
        input=pd.DataFrame([["1", "stability"],["2", "generator"]],columns=['qid','query'])






if __name__ == "__main__":
    unittest.main()
