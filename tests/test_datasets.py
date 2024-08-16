import unittest
import pyterrier as pt
from .base import BaseTestCase

class TestDatasets(BaseTestCase):

    def test_list_datasets(self):
        df = pt.list_datasets()
        self.assertIsNotNone(df)
        self.assertTrue(len(df) > 2)

    def test_webtrack_gov(self):
        import pyterrier as pt
        import requests, urllib
        try:
            for k in ["trec-wt-2002", "trec-wt-2003", "trec-wt-2004"]:
                ds = pt.get_dataset(k)
                for t in ["np", "td", "hp"]:
                    if k != "trec-wt-2004":
                        #HP finding only for the 2004 task?
                        continue
                    topics = ds.get_topics(t)
                    qrels = ds.get_qrels(t)
                    
                    #check that the qrels qid match the topics.
                    join = topics.merge(qrels, on=["qid"])
                    self.assertTrue(len(join) > 0)
        except requests.exceptions.ConnectionError:
            self.skipTest("NIST not reachable")
        except urllib.error.URLError:
            self.skipTest("NIST not reachable")

    def test_webtrack_cw09(self):
        import pyterrier as pt
        import requests, urllib
        try:
            for k in ["trec-wt-2009", "trec-wt-2010", "trec-wt-2011", "trec-wt-2012"]:
                ds = pt.get_dataset(k)
                topics = ds.get_topics()
                qrels = ds.get_qrels("adhoc")
                
                #check that the qrels match the topics.
                join = topics.merge(qrels, on=["qid"])
                self.assertTrue(len(join) > 0)
        except requests.exceptions.ConnectionError:
            self.skipTest("NIST not reachable")
        except urllib.error.URLError:
            self.skipTest("NIST not reachable")
    
    def test_vaswani_corpus_iter(self):
        import pyterrier as pt
        dataset = pt.datasets.get_dataset("vaswani")
        self.assertIsNotNone(dataset)
        iter = dataset.get_corpus_iter()
        self.assertIsNotNone(iter)
        doc = next(iter)
        self.assertEqual(doc["docno"], "1")
        self.assertTrue(doc["text"].startswith("compact memories have flexible capacities"))

    def test_vaswani_from_dataset(self):
        import pyterrier as pt
        dataset = pt.datasets.get_dataset("vaswani")
        br = pt.terrier.Retriever.from_dataset(dataset)
        br.search("chemical reactions")
        
    def test_vaswani(self):
        import pyterrier as pt
        dataset = pt.datasets.get_dataset("vaswani")
        self.assertIsNotNone(dataset)

        topics = dataset.get_topics()
        self.assertIsNotNone(topics)
        self.assertEqual(len(topics), 93)

        qrels = dataset.get_qrels()
        self.assertIsNotNone(qrels)
        self.assertEqual(len(qrels), 2083)

        indexref = dataset.get_index()
        self.assertIsNotNone(indexref)
        with pt.IndexFactory.of(indexref) as index:
            self.assertIsNotNone(index)
            self.assertEqual(index.getCollectionStatistics().getNumberOfDocuments(), 11429)

        # do it once again, to ensure it works locally
        dataset = pt.datasets.get_dataset("vaswani")
        topics = dataset.get_topics()
        self.assertIsNotNone(topics)
        self.assertEqual(len(topics), 93)

        # test the newer get_topicsqrels
        pt.Experiment([pt.terrier.Retriever(dataset.get_index())], *dataset.get_topicsqrels(), ["map"])

if __name__ == "__main__":
    unittest.main()