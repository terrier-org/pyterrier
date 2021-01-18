import pyterrier as pt
from jnius import cast
import pandas as pd
from .batchretrieve import _parse_index_like
from .transformer import TransformerBase, Symbol
from . import tqdm
from warnings import warn

TerrierQLParser = pt.autoclass("org.terrier.querying.TerrierQLParser")()
TerrierQLToMatchingQueryTerms = pt.autoclass("org.terrier.querying.TerrierQLToMatchingQueryTerms")()
QueryResultSet = pt.autoclass("org.terrier.matching.QueryResultSet")
DependenceModelPreProcess = pt.autoclass("org.terrier.querying.DependenceModelPreProcess")

def reset() -> TransformerBase:
    """
        Undoes a query rewriting operation. This results in the query formulation stored in the `"query_0"`
        attribute being moved to the `"query"` attribute, and, if present, the `"query_1"` being moved to
        `"query_0"` and so on. This transformation is useful if you have rewritten the query for the purposes
        of one retrieval stage, but wish a subquent transformer to be applies on the original formulation.

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
        The original query is saved in the `"query_0"` atribute, which can be restored using `pt.rewrite.reset()`.
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
        from .model import query_columns, push_queries
        queries = topics_and_res[query_columns(topics_and_res, qid=True)].dropna(axis=0, subset=query_columns(topics_and_res, qid=False)).drop_duplicates()

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
        return push_queries(queries, inplace=True).merge(new_queries, on="qid")

class SequentialDependence(SDM):
    ''' alias for SDM '''
    pass
    
class QueryExpansion(TransformerBase):
    '''
        A base class for applying different types of query expansion using Terrier's classes.
        This transformer changes the query. It must be followed by a Terrier Retrieve() transformer.
        The original query is saved in the `"query_0"` atribute, which can be restored using `pt.rewrite.reset()`.
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

        from .model import query_columns, push_queries
        queries = topics_and_res[query_columns(topics_and_res, qid=True)].dropna(axis=0, subset=query_columns(topics_and_res, qid=False)).drop_duplicates()
                
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
    def __init__(self, *args, **kwargs):
        """
        Args:
            index_like: the Terrier index to use.
            fb_terms(int): number of terms to add to the query
            fb_docs(int): number of feedback documents to consider
        """
        kwargs["qemodel"] = "Bo1"
        super().__init__(*args, **kwargs)

class KLQueryExpansion(DFRQueryExpansion):
    def __init__(self, *args, **kwargs):
        """
        Args:
            index_like: the Terrier index to use
            fb_terms(int): number of terms to add to the query
            fb_docs(int): number of feedback documents to consider
        """
        kwargs["qemodel"] = "KL"
        super().__init__(*args, **kwargs)

terrier_prf_package_loaded = False
class RM3(QueryExpansion):
    '''
        Performs query expansion using RM3 relevance models
    '''
    def __init__(self, *args, fb_terms=10, fb_docs=3, fb_lambda=0.6, **kwargs):
        """
        Args:
            index_like: the Terrier index to use
            fb_terms(int): number of terms to add to the query
            fb_docs(int): number of feedback documents to consider
        """
        global terrier_prf_package_loaded

        #if not terrier_prf_package_loaded:
        #    pt.extend_classpath("org.terrier:terrier-prf")
        #    terrier_prf_package_loaded = True
        #rm = pt.ApplicationSetup.getClass("org.terrier.querying.RM3").newInstance()
        import jnius_config
        prf_found = False
        for j in jnius_config.get_classpath():
            if "terrier-prf" in j:
                prf_found = True
                break
        assert prf_found, 'terrier-prf jar not found: you should start Pyterrier with '\
            + 'pt.init(boot_packages=["org.terrier:terrier-prf:0.0.1-SNAPSHOT"])'
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
        Performs query expansion using axiomatic query expansion
    '''
    def __init__(self, *args, fb_terms=10, fb_docs=3, **kwargs):
        """
        Args:
            index_like: the Terrier index to use
            fb_terms(int): number of terms to add to the query
            fb_docs(int): number of feedback documents to consider
        """
        global terrier_prf_package_loaded

        #if not terrier_prf_package_loaded:
        #    pt.extend_classpath("org.terrier:terrier-prf")
        #    terrier_prf_package_loaded = True
        #rm = pt.ApplicationSetup.getClass("org.terrier.querying.RM3").newInstance()
        import jnius_config
        prf_found = False
        for j in jnius_config.get_classpath():
            if "terrier-prf" in j:
                prf_found = True
                break
        assert prf_found, 'terrier-prf jar not found: you should start Pyterrier with '\
            + 'pt.init(boot_packages=["org.terrier:terrier-prf:0.0.1-SNAPSHOT"])'
        rm = pt.autoclass("org.terrier.querying.AxiomaticQE")()
        self.fb_terms = fb_terms
        self.fb_docs = fb_docs
        kwargs["qeclass"] = rm
        super().__init__(*args, **kwargs)

    def transform(self, queries_and_docs):
        self.qe.fbTerms = self.fb_terms
        self.qe.fbDocs = self.fb_docs
        return super().transform(queries_and_docs)

