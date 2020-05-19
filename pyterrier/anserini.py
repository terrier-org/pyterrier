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

    # jnius monkypatching
    import jnius_config
    anserini_found = False
    for j in jnius_config.get_classpath():
        if "anserini" in j:
            anserini_found = True
            break
    assert anserini_found, 'Anserini jar not found: you should start Pyterrier with '\
        + 'pt.init(boot_packages=["io.anserini:anserini:0.9.2:fatjar"])'
    jnius_config.set_classpath = lambda x: x
    anserini_monkey = True

    #this is the Anserini early rank cutoff rule
    from matchpy import Wildcard, ReplacementRule, Pattern
    from .transformer import RankCutoffTransformer, rewrite_rules
    x = Wildcard.dot('x')
    _brAnserini = Wildcard.symbol('_brAnserini', AnseriniBatchRetrieve)

    def set_k(_brAnserini, x):
        _brAnserini.k = int(x.value)
        return _brAnserini

    rewrite_rules.append(ReplacementRule(
            Pattern(RankCutoffTransformer(_brAnserini, x) ),
            set_k
    ))



class AnseriniBatchRetrieve(BatchRetrieveBase):
    def __init__(self, index_location, k=1000, **kwargs):
        super().__init__(kwargs)
        self.index_location = index_location
        self.k = k
        init_anserini()
        from pyserini.search import pysearch
        self.searcher = pysearch.SimpleSearcher(index_location)

    def __str__(self):
        return "AnseriniBatchRetrieve()"
    
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
        if docid_provided and not docno_provided:
            raise KeyError("Anserini doesnt expose Lucene's internal docids, you need the docnos")
        if docno_provided: #we are re-ranking
            from . import autoclass
            indexreaderutils = autoclass("io.anserini.index.IndexReaderUtils")
            indexreader = self.searcher.object.reader
            rank = 0
            last_qid = None
            for index,row in tqdm(queries.iterrows(), total=queries.shape[0], unit="d") if self.verbose else queries.iterrows():
                qid = row["qid"]
                query = row["query"]
                docno = row["docno"]
                if last_qid is None or last_qid != qid:
                    rank = 0
                rank += 1
                score = indexreaderutils.computeQueryDocumentScore(indexreader, docno, query)
                results.append([qid, query, docno, rank, score])

        else: #we are searching, no candidate set provided
            for index,row in tqdm(queries.iterrows(), total=queries.shape[0], unit="q") if self.verbose else queries.iterrows():
                rank = 0
                qid = str(row['qid'])
                query = row['query']
                
                hits = self.searcher.search(query, k=self.k)
                for i in range(0, min(len(hits), self.k)):
                    res = [qid, query,hits[i].docid,rank, hits[i].score]
                    rank += 1
                    results.append(res)   
                
        res_dt = pd.DataFrame(results, columns=['qid', 'query'] + ["docno"] + ['rank', 'score'])
        return res_dt