from warnings import warn
import pandas as pd
import pyterrier as pt
from pyterrier.anserini.java import required


@required
class AnseriniBatchRetrieve(pt.Transformer):
    """
        Allows retrieval from an Anserini index. To use this class, you must first enable anserini using `pt.anserini.enable()`.
    """
    def __init__(self, index_location, k=1000, wmodel="BM25", verbose=False):
        """
            Construct an AnseriniBatchRetrieve retrieve from pyserini.search.lucene.LuceneSearcher. 

            Args:

                index_location(str): The location of the Anserini index.
                wmodel(str): Weighting models supported by Anserini. There are three options: 
                
                 * `"BM25"` - the BM25 weighting model
                 * `"QLD"`  - Dirichlet language modelling
                 *  `"TFIDF"` - Lucene's `ClassicSimilarity <https://lucene.apache.org/core/8_1_0/core/org/apache/lucene/search/similarities/ClassicSimilarity.html>`_.
                k(int): number of results to return. Default is 1000.
        """
        self.index_location = index_location
        self.k = k
        self.wmodel = wmodel
        self.verbose = verbose
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_location)
        self._setsimilarty(wmodel)

    def __reduce__(self):
        return (
            self.__class__,
            (self.index_location, self.k, self.wmodel, self.verbose),
            self.__getstate__()
        )

    def __getstate__(self): 
        return  {}

    def __setstate__(self, d): 
        pass

    def _setsimilarty(self, wmodel):
        if wmodel == "BM25":
            self.searcher.set_bm25(k1=0.9, b=0.4)
        elif wmodel == "QLD":
            self.searcher.object.set_qld(1000.0)
        elif wmodel == "TFIDF":
            self.searcher.object.similarty = pt.anserini.J.ClassicSimilarity()
        else:
            raise ValueError("wmodel %s not support in AnseriniBatchRetrieve" % wmodel) 

    def _getsimilarty(self, wmodel):
        if wmodel == "BM25":
            return pt.anserini.J.BM25Similarity(0.9, 0.4)#(self.searcher.object.bm25_k1, self.searcher.object.bm25_b)
        elif wmodel == "QLD":
            return pt.anserini.J.LMDirichletSimilarity(1000.0)# (self.searcher.object.ql_mu)
        elif wmodel == "TFIDF":
            return pt.anserini.J.ClassicSimilarity()
        else:
            raise ValueError("wmodel %s not support in AnseriniBatchRetrieve" % wmodel) 

    def __str__(self):
        return "AnseriniBatchRetrieve()"

    def __repr__(self):
        return "AnseriniBatchRetrieve("+self.wmodel + ","+self.k+")"
    
    def transform(self, queries):
        """
        Performs the retrieval

        Args:
            queries: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']

        Returns:
            pandas.DataFrame with columns=['qid', 'docno', 'rank', 'score']
        """
        results=[]
        if not isinstance(queries, pd.DataFrame):
            warn(".transform() should be passed a dataframe. Use .search() to execute a single query.", DeprecationWarning, 2)
            queries=pt.model.coerce_queries_dataframe(queries)
        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        if docid_provided and not docno_provided:
            raise KeyError("Anserini doesnt expose Lucene's internal docids, you need the docnos")
        if docno_provided: #we are re-ranking
            indexreaderutils = pt.anserini.J.IndexReaderUtils
            indexreader = self.searcher.object.reader
            rank = 0
            last_qid = None
            sim = self._getsimilarty(self.wmodel)
            for row in pt.tqdm(queries.itertuples(), desc=self.name, total=queries.shape[0], unit="d") if self.verbose else queries.itertuples():
                qid = str(row.qid)
                query = row.query
                docno = row.docno
                if last_qid is None or last_qid != qid:
                    rank = 0
                rank += 1
                score = indexreaderutils.computeQueryDocumentScoreWithSimilarity(indexreader, docno, query, sim)
                results.append([qid, query, docno, rank, score])

        else: #we are searching, no candidate set provided
            for row in pt.tqdm(queries.itertuples(), desc=self.name, total=queries.shape[0], unit="q") if self.verbose else queries.itertuples():
                rank = 0
                qid = str(row.qid)
                query = row.query
                
                hits = self.searcher.search(query, k=self.k)
                for i in range(0, min(len(hits), self.k)):
                    res = [qid, query,hits[i].docid,rank, hits[i].score]
                    rank += 1
                    results.append(res)   
                
        res_dt = pd.DataFrame(results, columns=['qid', 'query'] + ["docno"] + ['rank', 'score'])
        return res_dt
