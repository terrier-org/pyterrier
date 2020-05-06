from .utils import Utils
from tqdm import tqdm
from .batchretrieve import BatchRetrieveBase

import pandas as pd
import numpy as np
from tqdm import tqdm

anserini_monkey=False
def init_anserini():
    global anserini_monkey
    if anserini_monkey:
        return
    import jnius_config
    anserini_found = False
    for j in jnius_config.get_classpath():
        if "anserini" in j:
            anserini_found = True
            break
    assert anserini_found, 'Anserini jar not found: you should start Pyterrier with '\
        + 'pt.init(boot_packages=["io.anserini:anserini:0.9.0:fatjar"])'
    jnius_config.set_classpath = lambda x: x
    anserini_monkey = True

class AnseriniBatchRetrieve(BatchRetrieveBase):
    def __init__(self, index_location, **kwargs):
        super().__init__(kwargs)
        self.index_location = index_location
        init_anserini()
        from pyserini.search import pysearch
        self.searcher = pysearch.SimpleSearcher(index_location)
    
    def transform(self, queries):
        """
        Performs the retrieval

        Args:
            queries: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']

        Returns:
            pandas.Dataframe with columns=['qid', 'docno', 'rank', 'score']
        """
        results=[]
        if not isinstance(queries, pd.DataFrame):
            queries=Utils.form_dataframe(queries)
        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "scores" in queries.columns
        if docno_provided or docid_provided:
            assert False

        for index,row in tqdm(queries.iterrows(), total=queries.shape[0], unit="q") if self.verbose else queries.iterrows():
            rank = 0
            qid = str(row['qid'])
            
            hits = self.searcher.search(row['query'], k=1000)
            for i in range(0, min(len(hits), 1000)):
                res = [str(row['qid']), hits[i].docid,rank, hits[i].score]
                rank += 1
                results.append(res)   
                
        res_dt = pd.DataFrame(results, columns=['qid', ] + ["docno"] + ['rank', 'score'])
        return res_dt