import pyterrier as pt
from jnius import cast
import pandas as pd
from .batchretrieve import _parse_index_like
from .transformer import TransformerBase, Symbol
from . import tqdm
from warnings import warn
from typing import List

TerrierQLParser = pt.autoclass("org.terrier.querying.TerrierQLParser")()
TerrierQLToMatchingQueryTerms = pt.autoclass("org.terrier.querying.TerrierQLToMatchingQueryTerms")()
QueryResultSet = pt.autoclass("org.terrier.matching.QueryResultSet")
DependenceModelPreProcess = pt.autoclass("org.terrier.querying.DependenceModelPreProcess")



_terrier_prf_package_loaded = False
_terrier_prf_message = 'terrier-prf jar not found: you should start PyTerrier with '\
    + 'pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])'

def _check_terrier_prf():
    import jnius_config
    global _terrier_prf_package_loaded
    if _terrier_prf_package_loaded:
        return
    
    for j in jnius_config.get_classpath():
        if "terrier-prf" in j:
            _terrier_prf_package_loaded = True
            break
    assert _terrier_prf_package_loaded, _terrier_prf_message

def reset() -> TransformerBase:
    """
        Undoes a previous query rewriting operation. This results in the query formulation stored in the `"query_0"`
        attribute being moved to the `"query"` attribute, and, if present, the `"query_1"` being moved to
        `"query_0"` and so on. This transformation is useful if you have rewritten the query for the purposes
        of one retrieval stage, but wish a subquent transformer to be applies on the original formulation.

        Internally, this function applies `pt.model.pop_queries()`.

        Example::

            firststage = pt.rewrite.SDM() >> pt.BatchRetrieve(index, wmodel="DPH")
            secondstage = pyterrier_bert.cedr.CEDRPipeline()
            fullranker = firststage >> pt.rewrite.reset() >> secondstage

    """
    from .model import pop_queries
    return pt.apply.generic(lambda topics: pop_queries(topics))

class SDM(TransformerBase):
    '''
        Implements the sequential dependence model, which Terrier supports using its
        Indri/Galagoo compatible matchop query language. The rewritten query is derived using
        the Terrier class DependenceModelPreProcess. 

        This transformer changes the query. It must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.
    '''

    def __init__(self, verbose = 0, remove_stopwords = True, prox_model = None, **kwargs):
        super().__init__(**kwargs)
        self.verbose = 0
        self.prox_model = prox_model
        self.remove_stopwords = remove_stopwords
        from . import check_version
        assert check_version("5.3")
        self.ApplyTermPipeline_stopsonly = pt.autoclass("org.terrier.querying.ApplyTermPipeline")("Stopwords")

    def transform(self, topics_and_res):
        results = []
        from .model import ranked_documents_to_queries, push_queries
        queries = ranked_documents_to_queries(topics_and_res)

        # instantiate the DependenceModelPreProcess, specifying a proximity model if specified
        sdm = DependenceModelPreProcess() if self.prox_model is None else DependenceModelPreProcess(self.prox_model)
        
        for row in tqdm(queries.itertuples(), desc=self.name, total=queries.shape[0], unit="q") if self.verbose else queries.itertuples():
            qid = row.qid
            query = row.query
            # parse the querying into a MQT
            rq = pt.autoclass("org.terrier.querying.Request")()
            rq.setQueryID(qid)
            rq.setOriginalQuery(query)
            TerrierQLParser.process(None, rq)
            TerrierQLToMatchingQueryTerms.process(None, rq)
            if self.remove_stopwords:
                self.ApplyTermPipeline_stopsonly.process(None, rq)

            # rewrite the query
            sdm.expandQuery(rq.getMatchingQueryTerms(), rq)
            new_query = ""

            # put the query back into a matchopql form that Terrier can parse later 
            for me in rq.getMatchingQueryTerms():
                term = me.getKey().toString()
                w = me.getValue().getWeight()
                prefix = ""
                if w != 1.0 or me.getValue().termModels.size() > 0:
                    prefix="#combine"
                    if w != 1:
                        prefix += ":0=" + str(w)
                    if me.getValue().termModels.size() == 1:
                        prefix += ":wmodel=" + me.getValue().termModels[0].getClass().getName()
                    term = prefix + "(" + term + ")"
                new_query += term + " "
            new_query = new_query[:-1]
            results.append([qid, new_query])
        new_queries = pd.DataFrame(results, columns=["qid", "query"])
        # restore any other columns, e.g. put back docs if we are re-ranking
        return new_queries.merge(push_queries(topics_and_res, inplace=True) , on="qid")

class SequentialDependence(SDM):
    '''
        Implements the sequential dependence model, which Terrier supports using its
        Indri/Galagoo compatible matchop query language. The rewritten query is derived using
        the Terrier class DependenceModelPreProcess. 

        This transformer changes the query. It must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.
    '''
    pass
    
class QueryExpansion(TransformerBase):
    '''
        A base class for applying different types of query expansion using Terrier's classes.
        This transformer changes the query. It must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.

        Instance Attributes:
         - fb_terms(int): number of feedback terms. Defaults to 10
         - fb_docs(int): number of feedback documents. Defaults to 3
         
    '''

    def __init__(self, index_like, fb_terms=10, fb_docs=3, qeclass="org.terrier.querying.QueryExpansion", verbose=0, properties={}, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        if isinstance(qeclass, str):
            self.qe = pt.autoclass(qeclass)()
        else:
            self.qe = qeclass
        self.indexref = _parse_index_like(index_like)
        for k,v in properties.items():
            pt.ApplicationSetup.setProperty(k, str(v))
        self.applytp = pt.autoclass("org.terrier.querying.ApplyTermPipeline")()
        self.fb_terms = fb_terms
        self.fb_docs = fb_docs
        self.manager = pt.autoclass("org.terrier.querying.ManagerFactory")._from_(self.indexref)

    def _populate_resultset(self, topics_and_res, qid, index):
        
        docids=None
        scores=None
        occurrences=None
        if "docid" in topics_and_res.columns:
            # we need .tolist() as jnius cannot convert numpy arrays
            docids = topics_and_res[topics_and_res["qid"] == qid]["docid"].values.tolist()
            scores = [0.0] * len(docids)
            occurrences = [0] * len(docids)

        elif "docno" in topics_and_res.columns:
            docnos = topics_and_res[topics_and_res["qid"] == qid]["docno"].values
            docids = []
            scores = []
            occurrences = []
            metaindex = index.getMetaIndex()
            skipped = 0
            for docno in docnos:
                docid = metaindex.getDocument("docno", docno)
                if docid == -1:
                    skipped +=1 
                assert docid != -1, "could not match docno" + docno + " to a docid for query " + qid    
                docids.append(docid)
                scores.append(0.0)
                occurrences.append(0)
            if skipped > 0:
                if skipped == len(docnos):
                    warn("*ALL* %d feedback docnos for qid %s could not be found in the index" % (skipped, qid))
                else:
                    warn("%d feedback docnos for qid %s could not be found in the index" % (skipped, qid))
        else:
            raise ValueError("Input resultset has neither docid nor docno")
        return QueryResultSet(docids, scores, occurrences)

    def _configure_request(self, rq):
        rq.setControl("qe_fb_docs", str(self.fb_docs))
        rq.setControl("qe_fb_terms", str(self.fb_terms))

    def transform(self, topics_and_res):

        results = []

        from .model import push_queries, ranked_documents_to_queries
        queries = ranked_documents_to_queries(topics_and_res)
        #queries = topics_and_res[query_columns(topics_and_res, qid=True)].dropna(axis=0, subset=query_columns(topics_and_res, qid=False)).drop_duplicates()
                
        for row in tqdm(queries.itertuples(), desc=self.name, total=queries.shape[0], unit="q") if self.verbose else queries.itertuples():
            qid = row.qid
            query = row.query
            srq = self.manager.newSearchRequest(qid, query)
            rq = cast("org.terrier.querying.Request", srq)
            self.qe.configureIndex(rq.getIndex())
            self._configure_request(rq)

            # generate the result set from the input
            rq.setResultSet(self._populate_resultset(topics_and_res, qid, rq.getIndex()))

            
            TerrierQLParser.process(None, rq)
            TerrierQLToMatchingQueryTerms.process(None, rq)
            # how to make sure this happens/doesnt happen when appropriate.
            self.applytp.process(None, rq)
            # to ensure weights are identical to Terrier
            rq.getMatchingQueryTerms().normaliseTermWeights();
            self.qe.expandQuery(rq.getMatchingQueryTerms(), rq)

            # this control for Terrier stops it re-stemming the expanded terms
            new_query = "applypipeline:off "
            for me in rq.getMatchingQueryTerms():
                new_query += me.getKey().toString() + ( "^%.9f "  % me.getValue().getWeight() ) 
            # remove trailing space
            new_query = new_query[:-1]
            results.append([qid, new_query])
        new_queries = pd.DataFrame(results, columns=["qid", "query"])
        return push_queries(queries, inplace=True).merge(new_queries, on="qid")

class DFRQueryExpansion(QueryExpansion):

    def __init__(self, *args, qemodel="Bo1", **kwargs):
        super().__init__(*args, **kwargs)
        self.qemodel = qemodel

    def _configure_request(self, rq):
        super()._configure_request(rq)
        rq.setControl("qemodel", self.qemodel)

class Bo1QueryExpansion(DFRQueryExpansion):
    '''
        Applies the Bo1 query expansion model from the Divergence from Randomness Framework, as provided by Terrier.
        It must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.

        Instance Attributes:
         - fb_terms(int): number of feedback terms. Defaults to 10
         - fb_docs(int): number of feedback documents. Defaults to 3  
    '''

    def __init__(self, *args, **kwargs):
        """
        Args:
            index_like: the Terrier index to use.
            fb_terms(int): number of terms to add to the query. Terrier's default setting is 10 expansion terms.
            fb_docs(int): number of feedback documents to consider. Terrier's default setting is 3 feedback documents.
        """
        kwargs["qemodel"] = "Bo1"
        super().__init__(*args, **kwargs)

class KLQueryExpansion(DFRQueryExpansion):
    '''
        Applies the KL query expansion model from the Divergence from Randomness Framework, as provided by Terrier.
        This transformer must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.

        Instance Attributes:
         - fb_terms(int): number of feedback terms. Defaults to 10
         - fb_docs(int): number of feedback documents. Defaults to 3  
    '''
    def __init__(self, *args, **kwargs):
        """
        Args:
            index_like: the Terrier index to use
            fb_terms(int): number of terms to add to the query. Terrier's default setting is 10 expansion terms.
            fb_docs(int): number of feedback documents to consider. Terrier's default setting is 3 feedback documents.
        """
        kwargs["qemodel"] = "KL"
        super().__init__(*args, **kwargs)

class RM3(QueryExpansion):
    '''
        Performs query expansion using RM3 relevance models. RM3 relies on an external Terrier plugin, 
        `terrier-prf <https://github.com/terrierteam/terrier-prf/>`_. You should start PyTerrier with 
        `pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])`.

        This transformer must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.

        Instance Attributes:
         - fb_terms(int): number of feedback terms. Defaults to 10
         - fb_docs(int): number of feedback documents. Defaults to 3
         - fb_lambda(float): lambda in RM3, i.e. importance of relevance model viz feedback model. Defaults to 0.6.

        Example:: 

            bm25 = pt.BatchRetrieve(index, wmodel="BM25")
            rm3_pipe = bm25 >> pt.rewrite.RM3(index) >> bm25
            pt.Experiment([bm25, rm3_pipe],
                        dataset.get_topics(),
                        dataset.get_qrels(),
                        ["map"]
                        )
 
    '''
    def __init__(self, *args, fb_terms=10, fb_docs=3, fb_lambda=0.6, **kwargs):
        """
        Args:
            index_like: the Terrier index to use
            fb_terms(int): number of terms to add to the query. Terrier's default setting is 10 expansion terms.
            fb_docs(int): number of feedback documents to consider. Terrier's default setting is 3 feedback documents.
            fb_lambda(float): lambda in RM3, i.e. importance of relevance model viz feedback model. Defaults to 0.6.
        """
        _check_terrier_prf()
        rm = pt.autoclass("org.terrier.querying.RM3")()
        self.fb_lambda = fb_lambda
        kwargs["qeclass"] = rm
        super().__init__(*args, fb_terms=fb_terms, fb_docs=fb_docs, **kwargs)

    def _configure_request(self, rq):
        super()._configure_request(rq)
        rq.setControl("rm3.lambda", str(self.fb_lambda))
        
    def transform(self, queries_and_docs):
        self.qe.fbTerms = self.fb_terms
        self.qe.fbDocs = self.fb_docs
        return super().transform(queries_and_docs)

class AxiomaticQE(QueryExpansion):
    '''
        Performs query expansion using axiomatic query expansion. This class relies on an external Terrier plugin, 
        `terrier-prf <https://github.com/terrierteam/terrier-prf/>`_. You should start PyTerrier with 
        `pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])`.

        This transformer must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` column, which can be restored using `pt.rewrite.reset()`.

        Instance Attributes:
         - fb_terms(int): number of feedback terms. Defaults to 10
         - fb_docs(int): number of feedback documents. Defaults to 3
    '''
    def __init__(self, *args, fb_terms=10, fb_docs=3, **kwargs):
        """
        Args:
            index_like: the Terrier index to use
            fb_terms(int): number of terms to add to the query. Terrier's default setting is 10 expansion terms.
            fb_docs(int): number of feedback documents to consider. Terrier's default setting is 3 feedback documents.
        """
        _check_terrier_prf()
        rm = pt.autoclass("org.terrier.querying.AxiomaticQE")()
        self.fb_terms = fb_terms
        self.fb_docs = fb_docs
        kwargs["qeclass"] = rm
        super().__init__(*args, **kwargs)

    def transform(self, queries_and_docs):
        self.qe.fbTerms = self.fb_terms
        self.qe.fbDocs = self.fb_docs
        return super().transform(queries_and_docs)

def stash_results(clear=True) -> TransformerBase:
    """
    Stashes (saves) the current retrieved documents for each query into the column `"stashed_results_0"`.
    This means that they can be restored later by using `pt.rewrite.reset_results()`.
    thereby converting a retrieved documents dataframe into one of queries.

    Args: 
    clear(bool): whether to drop the document and retrieved document related columns. Defaults to True.

    """
    return _StashResults(clear)
    
def reset_results() -> TransformerBase:
    """
    Applies a transformer that undoes a `pt.rewrite.stash_results()` transformer, thereby restoring the
    ranked documents.
    """
    return _ResetResults()

class _StashResults(TransformerBase):

    def __init__(self, clear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear = clear

    def transform(self, topics_and_res: pd.DataFrame) -> pd.DataFrame:
        from .model import document_columns, query_columns
        if "stashed_results_0" in topics_and_res.columns:
            raise ValueError("Cannot apply pt.rewrite.stash_results() more than once")
        doc_cols = document_columns(topics_and_res)
        
        rtr =  []
        if self.clear:
            query_cols = query_columns(topics_and_res)            
            for qid, groupDf in topics_and_res.groupby("qid"):
                documentsDF = groupDf[doc_cols]
                queryDf = groupDf[query_cols].iloc[0]
                queryDict = queryDf.to_dict()
                queryDict["stashed_results_0"] = documentsDF.to_dict(orient='records')
                rtr.append(queryDict)
            return pd.DataFrame(rtr)
        else:
            for qid, groupDf in topics_and_res.groupby("qid"):
                groupDf = groupDf.reset_index().copy()
                documentsDF = groupDf[doc_cols]
                docsDict = documentsDF.to_dict(orient='records')
                groupDf["stashed_results_0"] = None
                for i in range(len(groupDf)):
                    groupDf.at[i, "stashed_results_0"] = docsDict
                rtr.append(groupDf)
            return pd.concat(rtr)        

class _ResetResults(TransformerBase):

    def transform(self, topics_with_saved_docs : pd.DataFrame) -> pd.DataFrame:
        if not "stashed_results_0" in topics_with_saved_docs.columns:
            raise ValueError("Cannot apply pt.rewrite.reset_results() without pt.rewrite.stash_results() - column stashed_results_0 not found")
        from .model import query_columns
        query_cols = query_columns(topics_with_saved_docs)
        rtr = []
        for row in topics_with_saved_docs.itertuples():
            docsdf = pd.DataFrame.from_records(row.stashed_results_0)
            docsdf["qid"] = row.qid
            querydf = pd.DataFrame(data=[row])
            querydf.drop("stashed_results_0", axis=1, inplace=True)
            finaldf = querydf.merge(docsdf, on="qid")
            rtr.append(finaldf)
        return pd.concat(rtr)

def linear(weightCurrent : float, weightPrevious : float, format="terrierql", **kwargs) -> TransformerBase:
    """
    Applied to make a linear combination of the current and previous query formulation. The implementation
    is tied to the underlying query language used by the retrieval/re-ranker transformers. Two of Terrier's
    query language formats are supported by the `format` kwarg, namely `"terrierql"` and `"matchoptql"`. 
    Their exact respective formats are `detailed in the Terrier documentation <https://github.com/terrier-org/terrier-core/blob/5.x/doc/querylanguage.md>`_.

    Args:
        weightCurrent(float): weight to apply to the current query formulation.
        weightPrevious(float): weight to apply to the previous query formulation.
        format(str): which query language to use to rewrite the queries, one of "terrierql" or "matchopql".

    Example::

        pipeTQL = pt.apply.query(lambda row: "az") >> pt.rewrite.linear(0.75, 0.25, format="terrierql")
        pipeMQL = pt.apply.query(lambda row: "az") >> pt.rewrite.linear(0.75, 0.25, format="matchopql")
        pipeT.search("a")
        pipeM.search("a")

    Example outputs of `pipeTQL` and `pipeMQL` corresponding to the query "a" above:

    - Terrier QL output: `"(az)^0.750000 (a)^0.250000"`
    - MatchOp QL output: `"#combine:0:0.750000:1:0.250000(#combine(az) #combine(a))"`

    """
    return _LinearRewriteMix([weightCurrent, weightPrevious], format, **kwargs)

class _LinearRewriteMix(TransformerBase):

    def __init__(self, weights : List[float], format : str = 'terrierql', **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self.format = format
        if format not in ["terrierql", "matchopql"]:
            raise ValueError("Format must be one of 'terrierql', 'matchopql'")

    def _terrierql(self, row):
        return "(%s)^%f (%s)^%f" % (
            row["query_0"],
            self.weights[0],
            row["query_1"],
            self.weights[1])
    
    def _matchopql(self, row):
        return "#combine:0:%f:1:%f(#combine(%s) #combine(%s))" % (
            self.weights[0],
            self.weights[1],
            row["query_0"],
            row["query_1"])

    def transform(self, topics_and_res):
        from .model import push_queries
        
        fn = None
        if self.format == "terrierql":
            fn = self._terrierql
        elif self.format == "matchopql":
            fn = self._matchopql

        newDF = push_queries(topics_and_res)
        newDF["query"] = newDF.apply(fn, axis=1)
        return newDF
