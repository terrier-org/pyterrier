import pyterrier as pt
from jnius import cast
from tqdm import tqdm
import pandas as pd
from .batchretrieve import parse_index_like
from .transformer import TransformerBase, Symbol

TerrierQLParser = pt.autoclass("org.terrier.querying.TerrierQLParser")()
TerrierQLToMatchingQueryTerms = pt.autoclass("org.terrier.querying.TerrierQLToMatchingQueryTerms")()
ApplyTermPipeline_default = pt.autoclass("org.terrier.querying.ApplyTermPipeline")()
QueryResultSet = pt.autoclass("org.terrier.matching.QueryResultSet")
DependenceModelPreProcess = pt.autoclass("org.terrier.querying.DependenceModelPreProcess")

class SDM(TransformerBase, Symbol):

    def __init__(self, verbose = 0, remove_stopwords = True, prox_model = None, **kwargs):
        super().__init__(kwargs)
        self.verbose = 0
        self.prox_model = prox_model
        self.remove_stopwords = remove_stopwords
        from . import check_version
        assert check_version("5.3")
        self.ApplyTermPipeline_stopsonly = pt.autoclass("org.terrier.querying.ApplyTermPipeline")("Stopwords")

    def transform(self, topics_and_res):
        results = []
        queries = topics_and_res[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()

        # instantiate the DependenceModelPreProcess, specifying a proximity model if specified
        sdm = DependenceModelPreProcess() if self.prox_model is None else DependenceModelPreProcess(self.prox_model)
        
        for index,row in tqdm(queries.iterrows(), total=queries.shape[0], unit="q") if self.verbose else queries.iterrows():
            qid = row["qid"]
            query = row["query"]
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
        return pd.DataFrame(results, columns=["qid", "query"])
    


class QueryExpansion(TransformerBase, Symbol):

    def __init__(self, index_like, fb_terms=10, fb_docs=3, qeclass="org.terrier.querying.QueryExpansion", verbose=0, **kwargs):
        super().__init__(kwargs)
        self.verbose = verbose
        if isinstance(qeclass, str):
            self.qe = pt.autoclass(qeclass)()
        else:
            self.qe = qeclass
        self.indexref = parse_index_like(index_like)
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
            metaindex = index.getMetaIndex()
            for docno in docnos:
                docid = metaindex.getDocument("docno", docno)
                assert docid != -1, "could not match docno" + docno + " to a docid for query " + qid    
                docids.append(docid)
                scores.append(0.0)
                occurrences.append(0)
        return QueryResultSet(docids,scores, occurrences)

    def transform(self, topics_and_res):

        results = []

        queries = topics_and_res[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
                
        for index,row in tqdm(queries.iterrows(), total=queries.shape[0], unit="q") if self.verbose else queries.iterrows():
            qid = row["qid"]
            query = row["query"]
            srq = self.manager.newSearchRequest(qid, query)
            rq = cast("org.terrier.querying.Request", srq)
            self.qe.configureIndex(rq.getIndex())

            # generate the result set from the input
            rq.setResultSet(self._populate_resultset(topics_and_res, qid, rq.getIndex()))

            rq.setControl("qe_fb_docs", str(self.fb_docs))
            rq.setControl("qe_fb_terms", str(self.fb_terms))
            
            TerrierQLParser.process(None, rq)
            TerrierQLToMatchingQueryTerms.process(None, rq)
            # how to make sure this happens/doesnt happen when appropriate.
            ApplyTermPipeline_default.process(None, rq)
            # to ensure weights are identical to Terrier
            rq.getMatchingQueryTerms().normaliseTermWeights();
            self.qe.expandQuery(rq.getMatchingQueryTerms(), rq)

            # this control for Terrier stops it re-stemming the expanded terms
            new_query = "applypipeline:off "
            for me in rq.getMatchingQueryTerms():
                new_query += me.getKey().toString() + "^" + str(me.getValue().getWeight()) + " "
            # remove trailing space
            new_query = new_query[:-1]
            results.append([qid, new_query])
        return pd.DataFrame(results, columns=["qid", "query"])

terrier_prf_package_loaded = False
class RM3(QueryExpansion):
     def __init__(self, *args, fb_terms=10, fb_docs=3, **kwargs):
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
            + 'pt.init(boot_packages=["org.terrierorg:terrier-prf:0.0.1-SNAPSHOT"])'
        rm = pt.autoclass("org.terrier.querying.RM3")()
        rm.fbTerms = fb_terms
        rm.fbDocs = fb_docs
        kwargs["qeclass"] = rm
        super().__init__(*args, **kwargs)
