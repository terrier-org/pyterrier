from jnius import autoclass, cast
from typing import Union
import pandas as pd
import numpy as np
from . import tqdm, check_version
from warnings import warn
from .index import Indexer
from .datasets import Dataset
from .transformer import TransformerBase, Symbol, is_lambda
from .model import coerce_queries_dataframe, FIRST_RANK
import deprecation
import concurrent
from concurrent.futures import ThreadPoolExecutor

def importProps():
    from . import properties as props
    # Make import global
    globals()["props"] = props
props = None

_matchops = ["#combine", "#uw", "#1", "#tag", "#prefix", "#band", "#base64", "#syn"]
def _matchop(query):
    for m in _matchops:
        if m in query:
            return True
    return False

def _function2wmodel(function):
    from . import autoclass
    from jnius import PythonJavaClass, java_method

    class PythonWmodelFunction(PythonJavaClass):
        __javainterfaces__ = ['org/terrier/python/CallableWeightingModel$Callback']

        def __init__(self, fn):
            super(PythonWmodelFunction, self).__init__()
            self.fn = fn
            
        @java_method('(DLorg/terrier/structures/postings/Posting;Lorg/terrier/structures/EntryStatistics;Lorg/terrier/structures/CollectionStatistics;)D', name='score')
        def score(self, keyFreq, posting, entryStats, collStats):
            return self.fn(keyFreq, posting, entryStats, collStats)

        @java_method('()Ljava/nio/ByteBuffer;')
        def serializeFn(self):
            import dill as pickle
            #see https://github.com/SeldonIO/alibi/issues/447#issuecomment-881552005
            from dill import extend
            extend(use_dill=False)
            byterep = pickle.dumps(self.fn)
            byterep = autoclass("java.nio.ByteBuffer").wrap(byterep)
            return byterep

    callback = PythonWmodelFunction(function)
    wmodel = autoclass("org.terrier.python.CallableWeightingModel")( callback )
    return callback, wmodel

def _mergeDicts(defaults, settings):
    KV = defaults.copy()
    if settings is not None and len(settings) > 0:
        KV.update(settings)
    return KV

def _parse_index_like(index_location):
    JIR = autoclass('org.terrier.querying.IndexRef')
    JI = autoclass('org.terrier.structures.Index')

    if isinstance(index_location, JIR):
        return index_location
    if isinstance(index_location, JI):
        return cast('org.terrier.structures.Index', index_location).getIndexRef()
    if isinstance(index_location, str) or issubclass(type(index_location), Indexer):
        if issubclass(type(index_location), Indexer):
            return JIR.of(index_location.path)
        return JIR.of(index_location)

    raise ValueError(
        f'''index_location is current a {type(index_location)},
        while it needs to be an Index, an IndexRef, a string that can be
        resolved to an index location (e.g. path/to/index/data.properties),
        or an pyterrier.Indexer object'''
    )

class BatchRetrieveBase(TransformerBase, Symbol):
    """
    A base class for retrieval

    Attributes:
        verbose(bool): If True transform method will display progress
    """
    def __init__(self, verbose=0, **kwargs):
        super().__init__(kwargs)
        self.verbose = verbose

def _from_dataset(dataset : Union[str,Dataset], 
            clz,
            variant : str = None, 
            version='latest',            
            **kwargs):

    from . import get_dataset
    from .io import autoopen
    import os
    import json
    
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)
    if version != "latest":
        raise ValueError("index versioning not yet supported")
    indexref = dataset.get_index(variant)

    classname = clz.__name__
    # now look for, e.g., BatchRetrieve.args.json file, which will define the args for BatchRetrieve, e.g. stemming
    indexdir = indexref #os.path.dirname(indexref.toString())
    argsfile = os.path.join(indexdir, classname + ".args.json")
    if os.path.exists(argsfile):
        with autoopen(argsfile, "rt") as f:
            args = json.load(f)
            # anything specified in kwargs of this methods overrides the .args.json file
            args.update(kwargs)
            kwargs = args
    return clz(indexref, **kwargs)   
                
class BatchRetrieve(BatchRetrieveBase):
    """
    Use this class for retrieval by Terrier
    """

    @staticmethod
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):
        """
        Instantiates a BatchRetrieve object from a pre-built index access via a dataset.
        Pre-built indices are ofen provided via the `Terrier Data Repository <http://data.terrier.org/>`_.

        Examples::

            dataset = pt.get_dataset("vaswani")
            bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
            #or
            bm25 = pt.BatchRetrieve.from_dataset("vaswani", "terrier_stemmed", wmodel="BM25")

        **Index Variants**:

        There are a number of standard index names.
         - `terrier_stemmed` - a classical index, removing Terrier's standard stopwords, and applying Porter's English stemmer
         - `terrier_stemmed_positions` - as per `terrier_stemmed`, but also containing position information
         - `terrier_unstemmed` - a classical index, without applying stopword removal or stemming
         - `terrier_stemmed_text` - as per `terrier_stemmed`, but also containing the raw text of the documents
         - `terrier_unstemmed_text` - as per `terrier_stemmed`, but also containing the raw text of the documents

        """
        return _from_dataset(dataset, variant=variant, version=version, clz=BatchRetrieve, **kwargs)

    #: default_controls(dict): stores the default controls
    default_controls = {
        "terrierql": "on",
        "parsecontrols": "on",
        "parseql": "on",
        "applypipeline": "on",
        "localmatching": "on",
        "filters": "on",
        "decorate": "on",
        "wmodel": "DPH",
    }

    #: default_properties(dict): stores the default properties
    default_properties = {
        "querying.processes": "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,context_wmodel:org.terrier.python.WmodelFromContextProcess,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess",
        "querying.postfilters": "decorate:SimpleDecorate,site:SiteFilter,scope:Scope",
        "querying.default.controls": "wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on",
        "querying.allowed.controls": "scope,qe,qemodel,start,end,site,scope,applypipeline",
        "termpipelines": "Stopwords,PorterStemmer"
    }

    def __init__(self, index_location, controls=None, properties=None, metadata=["docno"],  num_results=None, wmodel=None, threads=1, **kwargs):
        """
            Init method

            Args:
                index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
                controls(dict): A dictionary with the control names and values
                properties(dict): A dictionary with the property keys and values
                verbose(bool): If True transform method will display progress
                num_results(int): Number of results to retrieve. 
                metadata(list): What metadata to retrieve
        """
        super().__init__(kwargs)
        from . import autoclass
        self.indexref = _parse_index_like(index_location)
        self.appSetup = autoclass('org.terrier.utility.ApplicationSetup')
        self.properties = _mergeDicts(BatchRetrieve.default_properties, properties)
        self.concurrentIL = autoclass("org.terrier.structures.ConcurrentIndexLoader")
        if check_version(5.5) and "SimpleDecorateProcess" not in self.properties["querying.processes"]:
            self.properties["querying.processes"] += ",decorate:SimpleDecorateProcess"
        self.metadata = metadata
        self.threads = threads
        self.RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")
        self.search_context = {}

        if props is None:
            importProps()
        for key, value in self.properties.items():
            self.appSetup.setProperty(key, str(value))
        
        self.controls = _mergeDicts(BatchRetrieve.default_controls, controls)
        if wmodel is not None:
            from .transformer import is_lambda, is_function
            if isinstance(wmodel, str):
                self.controls["wmodel"] = wmodel
            elif is_lambda(wmodel) or is_function(wmodel):
                callback, wmodelinstance = _function2wmodel(wmodel)
                #save the callback instance in this object to prevent being GCd by Python
                self._callback = callback
                self.search_context['context_wmodel'] = wmodelinstance
                self.controls['context_wmodel'] = 'on'
            elif isinstance(wmodel, autoclass("org.terrier.matching.models.WeightingModel")):
                self.search_context['context_wmodel'] = wmodel
                self.controls['context_wmodel'] = 'on'
            else:
                raise ValueError("Unknown parameter type passed for wmodel argument: %s" % str(wmodel))
                  
        if self.threads > 1:
            warn("Multi-threaded retrieval is experimental, YMMV.")
            assert check_version(5.5), "Terrier 5.5 is required for multi-threaded retrieval"

            # we need to see if our indexref is concurrent. if not, we upgrade it using ConcurrentIndexLoader
            # this will upgrade the underlying index too.
            if not self.concurrentIL.isConcurrent(self.indexref):
                warn("Upgrading indexref %s to be concurrent" % self.indexref.toString())
                self.indexref = self.concurrentIL.makeConcurrent(self.indexref)

        if num_results is not None:
            if num_results > 0:
                self.controls["end"] = str(num_results -1)
            elif num_results == 0:
                del self.controls["end"]
            else: 
                raise ValueError("num_results must be None, 0 or positive")


        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")
        self.manager = MF._from_(self.indexref)
    
    def get_parameter(self, name : str):
        if name in self.controls:
            return self.controls[name]
        elif name in self.properties:
            return self.properties[name]
        else:
            return super().get_parameter(name)

    def set_parameter(self, name : str, value):
        if name in self.controls:
            self.controls[name] = value
        elif name in self.properties:
            self.properties[name] = value
        else:
            super().set_parameter(name,value)

    def __reduce__(self):
        return (
            self.__class__,
            (self.indexref,),
            self.__getstate__()
        )

    def __getstate__(self): 
        return  {
                'context' : self.search_context,
                'controls' : self.controls, 
                'properties' : self.properties, 
                'metadata' : self.metadata,
                }

    def __setstate__(self, d): 
        self.controls = d["controls"]
        self.metadata = d["metadata"]
        self.search_context = d["context"]
        self.properties.update(d["properties"])
        for key,value in d["properties"].items():
            self.appSetup.setProperty(key, str(value))

    def _retrieve_one(self, row, input_results=None, docno_provided=False, docid_provided=False, scores_provided=False):
        rank = FIRST_RANK
        qid = str(row.qid)
        query = row.query
        if len(query) == 0:
            warn("Skipping empty query for qid %s" % qid)
            return []

        srq = self.manager.newSearchRequest(qid, query)
        
        for control, value in self.controls.items():
            srq.setControl(control, str(value))

        for key, value in self.search_context.items():
            srq.setContextObject(key, value)

        # this is needed until terrier-core issue #106 lands
        if "applypipeline:off" in query:
            srq.setControl("applypipeline", "off")
            srq.setOriginalQuery(query.replace("applypipeline:off", ""))

        # transparently detect matchop queries
        if _matchop(query):
            srq.setControl("terrierql", "off")
            srq.setControl("parsecontrols", "off")
            srq.setControl("parseql", "off")
            srq.setControl("matchopql", "on")

        #ask decorate only to grab what we need
        srq.setControl("decorate", ",".join(self.metadata))

        # this handles the case that a candidate set of documents has been set. 
        num_expected = None
        if docno_provided or docid_provided:
            # we use RequestContextMatching to make a ResultSet from the 
            # documents in the candidate set. 
            matching_config_factory = self.RequestContextMatching.of(srq)
            input_query_results = input_results[input_results["qid"] == qid]
            num_expected = len(input_query_results)
            if docid_provided:
                matching_config_factory.fromDocids(input_query_results["docid"].values.tolist())
            elif docno_provided:
                matching_config_factory.fromDocnos(input_query_results["docno"].values.tolist())
            # batch retrieve is a scoring process that always overwrites the score; no need to provide scores as input
            #if scores_provided:
            #    matching_config_factory.withScores(input_query_results["score"].values.tolist())
            matching_config_factory.build()
            srq.setControl("matching", "org.terrier.matching.ScoringMatching" + "," + srq.getControl("matching"))

        # now ask Terrier to run the request
        self.manager.runSearchRequest(srq)
        result = srq.getResults()

        # check we got all of the expected metadata (if the resultset has a size at all)
        if len(result) > 0 and len(set(self.metadata) & set(result.getMetaKeys())) != len(self.metadata):
            raise KeyError("Mismatch between requested and available metadata in %s. Requested metadata: %s, available metadata %s" % 
                (str(self.indexref), str(self.metadata), str(result.getMetaKeys()))) 

        if num_expected is not None:
            assert(num_expected == len(result))
        
        rtr_rows=[]
        # prepare the dataframe for the results of the query
        for item in result:
            metadata_list = []
            for meta_column in self.metadata:
                metadata_list.append(item.getMetadata(meta_column))
            res = [qid, item.getDocid()] + metadata_list + [rank, item.getScore()]
            rank += 1
            rtr_rows.append(res)

        return rtr_rows


    def transform(self, queries):
        """
        Performs the retrieval

        Args:
            queries: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']. For re-ranking,
                the DataFrame may also have a 'docid' and or 'docno' column.

        Returns:
            pandas.Dataframe with columns=['qid', 'docno', 'rank', 'score']
        """
        results=[]
        if not isinstance(queries, pd.DataFrame):
            warn(".transform() should be passed a dataframe. Use .search() to execute a single query.", FutureWarning, 2)
            queries = coerce_queries_dataframe(queries)
        
        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "score" in queries.columns
        input_results = None
        if docno_provided or docid_provided:
            assert check_version(5.3)
            input_results = queries

            # query is optional, and functionally dependent on qid.
            # Hence as long as one row has the query for each qid, 
            # the rest can be None
            queries = input_results[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
            
        # make sure queries are a String
        if queries["qid"].dtype == np.int64:
            queries['qid'] = queries['qid'].astype(str)

        if self.threads > 1:

            if not self.concurrentIL.isConcurrent(self.indexref):
                raise ValueError("Threads must be set >1 in constructor and/or concurrent indexref used")
            
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                
                # we must detatch jnius to prevent thread leaks through JNI
                from jnius import detach
                def _one_row(*args, **kwargs):
                    rtr = self._retrieve_one(*args, **kwargs)
                    detach()
                    return rtr
                
                # create a future for each query, and submit to Terrier
                future_results = {
                    executor.submit(_one_row, row, input_results, docno_provided=docno_provided, docid_provided=docid_provided, scores_provided=scores_provided) : row.qid 
                    for row in queries.itertuples()}                
                
                # as these futures complete, wait and add their results
                iter = concurrent.futures.as_completed(future_results)
                if self.verbose:
                    iter = tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
                
                for future in iter:
                    res = future.result()
                    results.extend(res)
        else:
            iter = queries.itertuples()
            if self.verbose:
                iter = tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
            for row in iter:
                res = self._retrieve_one(row, input_results, docno_provided=docno_provided, docid_provided=docid_provided, scores_provided=scores_provided)
                results.extend(res)

        res_dt = pd.DataFrame(results, columns=['qid', 'docid' ] + self.metadata + ['rank', 'score'])
        # ensure to return the query and any other input columns
        input_cols = queries.columns[ (queries.columns == "qid") | (~queries.columns.isin(res_dt.columns))]
        res_dt = res_dt.merge(queries[input_cols], on=["qid"])
        return res_dt
        
    def __repr__(self):
        return "BR(" + ",".join([
            self.indexref.toString(),
            str(self.controls),
            str(self.properties)
            ]) + ")"

    def __str__(self):
        return "BR(" + self.controls["wmodel"] + ")"

    def setControls(self, controls):
        for key, value in controls.items():
            self.controls[key] = value

    def setControl(self, control, value):
        self.controls[control] = value



class TextIndexProcessor(TransformerBase):
    '''
        Creates a new MemoryIndex based on the contents of documents passed to it.
        It then creates a new instance of the innerclass and passes the topics to that.

        This class is the base class for TextScorer, but can be used in other settings as well, 
        for instance query expansion based on text.
    '''

    def __init__(self, innerclass, takes="queries", returns="docs", body_attr="body", background_index=None, verbose=False, **kwargs):
        #super().__init__(**kwargs)
        self.innerclass = innerclass
        self.takes = takes
        self.returns = returns
        self.body_attr = body_attr
        if background_index is not None:
            self.background_indexref = _parse_index_like(background_index)
        else:
            self.background_indexref = None
        self.kwargs = kwargs
        self.verbose = verbose

    def transform(self, topics_and_res):
        from . import DFIndexer, autoclass, IndexFactory
        from .index import IndexingType
        documents = topics_and_res[["docno", self.body_attr]].drop_duplicates(subset="docno")
        indexref = DFIndexer(None, type=IndexingType.MEMORY, verbose=self.verbose).index(documents[self.body_attr], documents["docno"])
        docno2docid = { docno:id for id, docno in enumerate(documents["docno"]) }
        index_docs = IndexFactory.of(indexref)
        docno2docid = {}
        for i in range(0, index_docs.getCollectionStatistics().getNumberOfDocuments()):
            docno2docid[index_docs.getMetaIndex().getItem("docno", i)] = i
        assert len(docno2docid) == index_docs.getCollectionStatistics().getNumberOfDocuments(), "docno2docid size (%d) doesnt match index (%d)" % (len(docno2docid), index_docs.getCollectionStatistics().getNumberOfDocuments())
        
        # if a background index is set, we create an "IndexWithBackground" using both that and our new index
        if self.background_indexref is None:
            index = index_docs
        else:
            index_background = IndexFactory.of(self.background_indexref)
            index = autoclass("org.terrier.python.IndexWithBackground")(index_docs, index_background)          

        topics = topics_and_res[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
        
        if self.takes == "queries":
            # we have provided the documents, so we dont need a docno or docid column that will confuse 
            # BR and think it is re-ranking. In fact, we only need qid and query
            input = topics
        elif self.takes == "docs":
            # we have to pass the documents, but its desirable to have the docids mapped to the new index already
            # build a mapping, as the metaindex may not have reverse lookups enabled
            input = topics_and_res.copy()
            # add the docid to the dataframe
            input["docid"] = input.apply(lambda row: docno2docid[row["docno"]], axis=1)


        # and then just instantiate BR using the our new index 
        # we take all other arguments as arguments for BR
        inner = self.innerclass(index, **(self.kwargs))
        inner.verbose = self.verbose
        inner_res = inner.transform(input)

        if self.returns == "docs":
            # as this is a new index, docids are not meaningful externally, so lets drop them
            inner_res.drop(columns=['docid'], inplace=True)

            topics_columns = topics_and_res.columns[(topics_and_res.columns.isin(["qid", "docno"])) | (~topics_and_res.columns.isin(inner_res.columns))]
            if len(inner_res) < len(topics_and_res):
                inner_res = topics_and_res[topics_columns].merge(inner_res, on=["qid", "docno"], how="left")
                inner_res["score"] = inner_res["score"].fillna(value=0)
            else:
                inner_res = topics_and_res[ topics_columns ].merge(inner_res, on=["qid", "docno"])
        elif self.returns == "queries":
            if len(inner_res) < len(topics):
                inner_res = topics.merge(on=["qid"], how="left")
        else:
            raise ValueError("returns attribute should be docs of queries")
        return inner_res

class TextScorer(TextIndexProcessor):
    """
        A re-ranker class, which takes the queries and the contents of documents, indexes the contents of the documents using a MemoryIndex, and performs ranking of those documents with respect to the queries.
        Unknown kwargs are passed to BatchRetrieve.

        Arguments:
            takes(str): configuration - what is needed as input: `"queries"`, or `"docs"`.
            returns(str): configuration - what is needed as output: `"queries"`, or `"docs"`.
            body_attr(str): what dataframe input column contains the text of the document. Default is `"body"`.
            wmodel(str): example of configuration passed to BatchRetrieve.

        Example::

            df = pd.DataFrame(
                [
                    ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
                    ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
                ], columns=["qid", "query", "text"])
            textscorer = pt.TextScorer(takes="docs", body_attr="text", wmodel="TF_IDF")
            rtr = textscorer.transform(df)
            #rtr will score each document for the query "chemical reactions" based on the provided document contents
    """

    def __init__(self, **kwargs):
        super().__init__(BatchRetrieve, **kwargs)

class FeaturesBatchRetrieve(BatchRetrieve):
    """
    Use this class for retrieval with multiple features
    """

    #: FBR_default_controls(dict): stores the default properties for a FBR
    FBR_default_controls = BatchRetrieve.default_controls.copy()
    FBR_default_controls["matching"] = "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull"
    del FBR_default_controls["wmodel"]
    #: FBR_default_properties(dict): stores the default properties
    FBR_default_properties = BatchRetrieve.default_properties.copy()

    def __init__(self, index_location, features, controls=None, properties=None, threads=1, **kwargs):
        """
            Init method

            Args:
                index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
                features(list): List of features to use
                controls(dict): A dictionary with the control names and values
                properties(dict): A dictionary with the property keys and values
                verbose(bool): If True transform method will display progress
                num_results(int): Number of results to retrieve. 
        """
        controls = _mergeDicts(FeaturesBatchRetrieve.FBR_default_controls, controls)
        properties = _mergeDicts(FeaturesBatchRetrieve.FBR_default_properties, properties)
        self.features = features
        properties["fat.featured.scoring.matching.features"] = ";".join(features)

        # record the weighting model
        self.wmodel = None
        if "wmodel" in kwargs:
            assert isinstance(kwargs["wmodel"], str), "Non-string weighting models not yet supported by FBR"
            self.wmodel = kwargs["wmodel"]
        if "wmodel" in controls:
            self.wmodel = controls["wmodel"]
        if threads > 1:
            raise ValueError("Multi-threaded retrieval not yet supported by FeaturesBatchRetrieve")
        
        super().__init__(index_location, controls, properties, **kwargs)

    def __reduce__(self):
        return (
            self.__class__,
            (self.indexref, self.features),
            self.__getstate__()
        )

    def __getstate__(self): 
        return  {
                'controls' : self.controls, 
                'properties' : self.properties, 
                'metadata' : self.metadata,
                'features' : self.features,
                'wmodel' : self.wmodel
                #TODO consider the context state?
                }

    def __setstate__(self, d): 
        self.controls = d["controls"]
        self.metadata = d["metadata"]
        self.features = d["features"]
        self.wmodel = d["wmodel"]
        self.properties.update(d["properties"])
        for key,value in d["properties"].items():
            self.appSetup.setProperty(key, str(value))
        #TODO consider the context state?

    @staticmethod 
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):
        return _from_dataset(dataset, variant=variant, version=version, clz=FeaturesBatchRetrieve, **kwargs)

    @staticmethod 
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):
        return _from_dataset(dataset, variant=variant, version=version, clz=FeaturesBatchRetrieve, **kwargs)

    def transform(self, queries):
        """
        Performs the retrieval with multiple features

        Args:
            queries: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']. For re-ranking,
                the DataFrame may also have a 'docid' and or 'docno' column.

        Returns:
            pandas.DataFrame with columns=['qid', 'docno', 'score', 'features']
        """
        results = []
        if not isinstance(queries, pd.DataFrame):
            warn(".transform() should be passed a dataframe. Use .search() to execute a single query.", FutureWarning, 2)
            queries = coerce_queries_dataframe(queries)

        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "score" in queries.columns
        if docno_provided or docid_provided:
            #re-ranking mode
            from . import check_version
            assert check_version(5.3)
            input_results = queries

            # query is optional, and functionally dependent on qid.
            # Hence as long as one row has the query for each qid, 
            # the rest can be None
            queries = input_results[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
            RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")

            if not scores_provided and self.wmodel is None:
                raise ValueError("We're in re-ranking mode, but input does not have scores, and wmodel is None")
        else:
            assert not scores_provided

            if self.wmodel is None:
                raise ValueError("We're in retrieval mode (input columns were "+str(queries.columns)+"), but wmodel is None. FeaturesBatchRetrieve requires a wmodel be set for identifying the candidate set. "
                    +" Hint: wmodel argument for FeaturesBatchRetrieve, e.g. FeaturesBatchRetrieve(index, features, wmodel=\"DPH\")")

        if queries["qid"].dtype == np.int64:
            queries['qid'] = queries['qid'].astype(str)

        newscores=[]
        for row in tqdm(queries.itertuples(), desc=str(self), total=queries.shape[0], unit="q") if self.verbose else queries.itertuples():
            qid = str(row.qid)
            query = row.query
            if len(query) == 0:
                warn("Skipping empty query for qid %s" % qid)
                continue

            srq = self.manager.newSearchRequest(qid, query)

            for control, value in self.controls.items():
                srq.setControl(control, str(value))

            # this is needed until terrier-core issue #106 lands
            if "applypipeline:off" in query:
                srq.setControl("applypipeline", "off")
                srq.setOriginalQuery(query.replace("applypipeline:off", ""))

            # transparently detect matchop queries
            if _matchop(query):
                srq.setControl("terrierql", "off")
                srq.setControl("parsecontrols", "off")
                srq.setControl("parseql", "off")
                srq.setControl("matchopql", "on")

            # this handles the case that a candidate set of documents has been set. 
            if docno_provided or docid_provided:
                # we use RequestContextMatching to make a ResultSet from the 
                # documents in the candidate set. 
                matching_config_factory = RequestContextMatching.of(srq)
                input_query_results = input_results[input_results["qid"] == qid]
                if docid_provided:
                    matching_config_factory.fromDocids(input_query_results["docid"].values.tolist())
                elif docno_provided:
                    matching_config_factory.fromDocnos(input_query_results["docno"].values.tolist())
                if scores_provided:
                    if self.wmodel is None:
                        # we provide the scores, so dont use a weighting model, and pass the scores through Terrier
                        matching_config_factory.withScores(input_query_results["score"].values.tolist())
                        srq.setControl("wmodel", "Null")
                    else:
                        srq.setControl("wmodel", self.wmodel)
                matching_config_factory.build()
                srq.setControl("matching", ",".join(["FatFeaturedScoringMatching","ScoringMatchingWithFat", srq.getControl("matching")]))
            
            self.manager.runSearchRequest(srq)
            srq = cast('org.terrier.querying.Request', srq)
            fres = cast('org.terrier.learning.FeaturedResultSet', srq.getResultSet())
            feat_names = fres.getFeatureNames()

            docids=fres.getDocids()
            scores= fres.getScores()
            metadata_list = [fres.getMetaItems(meta_column) for meta_column in self.metadata]
            feats_values = [fres.getFeatureScores(feat) for feat in feat_names]
            rank = FIRST_RANK
            for i in range(fres.getResultSize()):
                doc_features = np.array([ feature[i] for feature in feats_values])
                meta=[ metadata_col[i] for metadata_col in metadata_list]
                results.append( [qid, query, docids[i], rank, doc_features ] + meta )
                newscores.append(scores[i])
                rank += 1

        res_dt = pd.DataFrame(results, columns=["qid", "query", "docid", "rank", "features"] + self.metadata)
        res_dt["score"] = newscores
        # ensure to return the query and any other input columns
        input_cols = queries.columns[ (queries.columns == "qid") | (~queries.columns.isin(res_dt.columns))]
        res_dt = res_dt.merge(queries[input_cols], on=["qid"])
        return res_dt

    def __repr__(self):
        return "FBR(" + ",".join([
            self.indexref.toString(),
            str(self.features),
            str(self.controls),
            str(self.properties)
        ]) + ")"

    def __str__(self):
        if self.wmodel is None:
            return "FBR(" + str(len(self.features)) + " features)"
        return "FBR(" + self.controls["wmodel"] + " and " + str(len(self.features)) + " features)"
