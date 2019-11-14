import unittest
from main import BatchRetrieve
import jnius_config,math
# jnius_config.set_classpath("terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass



class TestBatchRetrieve(unittest.TestCase):
    def test_one_term_query_correct_docid_and_score(self):
        JIR = autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of("./index/data.properties")
        retr = BatchRetrieve(indexref)
        result = retr.transform("light")
        exp_result =[
        ["0", "11067", 5.177300241106782],
        ["1", "6999", 4.968141993817469],
        ["2", "5996", 4.897958615783841],
        ["3", "11232", 4.8791159927346115],
        ["4", "211", 4.526725003783039],
        ["5", "403", 4.524341784111971]]
        for index,row in result[:5].iterrows():
            self.assertEquals(row['docno'], exp_result[index][1])
            self.assertAlmostEquals(row['score'], exp_result[index][2])



if __name__ == "__main__":
    unittest.main()
