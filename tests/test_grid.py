import pandas as pd
import pyterrier as pt
from .base import BaseTestCase


class TestGrid(BaseTestCase):

    def test_gridscan_2pipe_2params(self):
        dataset = pt.get_dataset("vaswani")
        index = pt.IndexFactory.of(dataset.get_index())
        PL2 = pt.BatchRetrieve(index, wmodel="PL2", controls={'c' : 1})
        bo1 = pt.rewrite.Bo1QueryExpansion(index)
        pipe = PL2 >> bo1 >> PL2                

        rtr = pt.GridScan(
            pipe, 
            {
                PL2 : {'c' : [0.1, 1]},
                bo1 : {'fb_terms' : [2,4]}}, 
            dataset.get_topics().head(2),
            dataset.get_qrels(),
            dataframe=False
        )
        self.assertEqual(4, len(rtr))
        print(rtr)

    def test_gridscan_2params(self):
        dataset = pt.get_dataset("vaswani")
        pipe = pt.BatchRetrieve(dataset.get_index(), wmodel="PL2", controls={'c' : 1})
        self.assertEqual(1, pipe.get_parameter('c'))
        self.assertEqual("PL2", pipe.get_parameter('wmodel'))
        rtr = pt.GridScan(
            pipe, 
            {pipe : {'c' : [0.1, 1], 'wmodel' : ["PL2", "BM25"]}}, 
            dataset.get_topics().head(2),
            dataset.get_qrels(),
            dataframe=False
        )
        self.assertEqual(4, len(rtr))
        print(rtr)

    def test_gridscan_1param(self):
        dataset = pt.get_dataset("vaswani")
        pipe = pt.BatchRetrieve(dataset.get_index(), wmodel="PL2", controls={'c' : 1})
        self.assertEqual(1, pipe.get_parameter('c'))
        rtr = pt.GridScan(
            pipe, 
            {pipe : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            dataset.get_topics().head(5),
            dataset.get_qrels(),
            dataframe=False
        )
        self.assertEqual(6, len(rtr))
        print(rtr)
    
    def test_gridscan_1param_df(self):
        dataset = pt.get_dataset("vaswani")
        pipe = pt.BatchRetrieve(dataset.get_index(), wmodel="PL2", controls={'c' : 1})
        self.assertEqual(1, pipe.get_parameter('c'))
        rtr = pt.GridScan(
            pipe, 
            {pipe : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            dataset.get_topics().head(5),
            dataset.get_qrels(),
            dataframe=True,
        )
        self.assertEqual(6, len(rtr))
        self.assertTrue(isinstance(rtr, pd.DataFrame))
        print(rtr)

    def test_gridscan_joblib2(self):
        dataset = pt.get_dataset("vaswani")
        pipe = pt.BatchRetrieve(dataset.get_index(), wmodel="PL2", controls={'c' : 1})
        self.assertEqual(1, pipe.get_parameter('c'))
        rtr = pt.GridScan(
            pipe, 
            {pipe : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            dataset.get_topics().head(5),
            dataset.get_qrels(),
            jobs=2,
            dataframe=False
        )
        self.assertEqual(6, len(rtr))
        print(rtr)

    def test_gridsearch(self):
        dataset = pt.get_dataset("vaswani")
        pipe = pt.BatchRetrieve(dataset.get_index(), wmodel="PL2", controls={'c' : 1})
        rtr = pt.pipelines.GridSearch(
            pipe, 
            {pipe : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            dataset.get_topics().head(5),
            dataset.get_qrels()
        )
        self.assertEqual(100, rtr.get_parameter("c"))
