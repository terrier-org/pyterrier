from typing import Union, Optional, Callable, List, Any
import pandas as pd
import numpy as np
import re
from warnings import warn
from pyterrier.datasets import Dataset
from pyterrier.model import FIRST_RANK
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pyterrier as pt
from typing import Dict

from pyterrier.terrier.stemmer import TerrierStemmer
from pyterrier.terrier.stopwords import TerrierStopwords
from pyterrier.terrier.tokeniser import TerrierTokeniser

def _query_needs_tokenised(query: str) -> bool:
    # detects if we need to tokenise the query
    if _matchop(query):
        return False
    
    if "applypipeline" in query:
        return False
    
    # does it contains in terrier language 
    termweightsre = re.compile(r'\^\d+(\.\d+)?')
    if termweightsre.match(query):
        return False

    # we dont include : in this list, as it denotes a control key:value pair in TerrierQL
    bad_chars = [",", ".", ";", "!", "?", "(", ")", "[", "]", "{", "}", "<", ">", "\"", "^", 
                 "'", "`", "~", "@", "#", "$", "%", "&", "*", "-", "+", "=", "|", "\\", "/"]

    if any(c in query for c in bad_chars):
        return True
    
    return False

_matchops = ["#combine", "#uw", "#1", "#tag", "#prefix", "#band", "#base64", "#syn"]

def _matchop(query):
    for m in _matchops:
        if m in query:
            return True
    return False

def _querytoks2matchop(query_toks: Dict[str,float]) -> str:
    def _matchop_tok(t, w):
        import base64
        import string
        if not all(a in string.ascii_letters + string.digits for a in t):
            encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8") 
            t = f'#base64({encoded})'
        if w != 1:
            t = f'#combine:0={w:f}({t})'
        return t
    return ' '.join([ _matchop_tok(t, w) for (t,w) in query_toks.items() ])

@pt.java.required
def _function2wmodel(function):
    from jnius import PythonJavaClass, java_method

    @pt.java.required
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
            # keep both python and java representations around to prevent them being GCd in respective VMs 
            self.pbyterep = pickle.dumps(self.fn)
            self.jbyterep = pt.java.autoclass("java.nio.ByteBuffer").wrap(self.pbyterep)
            return self.jbyterep

    callback = PythonWmodelFunction(function)
    wmodel = pt.java.autoclass("org.terrier.python.CallableWeightingModel")( callback )
    return callback, wmodel

def _mergeDicts(defaults : Dict[str,str], settings : Optional[Dict[str,str]] = None) -> Dict[str,str]:
    KV = defaults.copy()
    if settings is not None and len(settings) > 0:
        KV.update(settings)
    return KV

@pt.java.required
def _parse_index_like(index_location):
    JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
    JI = pt.java.autoclass('org.terrier.structures.Index')
    from pyterrier.terrier import TerrierIndexer

    if isinstance(index_location, pt.terrier.TerrierIndex):
        return index_location.index_ref()
    if isinstance(index_location, JIR):
        return index_location
    if isinstance(index_location, JI):
        return pt.java.cast('org.terrier.structures.Index', index_location).getIndexRef()
    if isinstance(index_location, str) or issubclass(type(index_location), TerrierIndexer):
        if issubclass(type(index_location), TerrierIndexer):
            return JIR.of(index_location.path)
        return JIR.of(index_location)

    raise ValueError(
        f'''index_location is current a {type(index_location)},
        while it needs to be a TerrierIndex, an Index, an IndexRef, a string that can be
        resolved to an index location (e.g. path/to/index/data.properties),
        or a pyterrier.index.TerrierIndexer object'''
    )

  
@pt.java.required
class Retriever(pt.Transformer):
    """
    Use this class for retrieval by Terrier
    """

    @staticmethod
    def matchop(t, w=1):
        """
        Static method used for rewriting a query term to use a MatchOp operator if it contains
        anything except ASCII letters or digits.
        """
        import base64
        import string
        if not all(a in string.ascii_letters + string.digits for a in t):
            encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8") 
            t = f'#base64({encoded})'
        if w != 1:
            t = f'#combine:0={w}({t})'
        return t


    @staticmethod
    def from_dataset(dataset : Union[str,Dataset], 
            variant : Optional[str] = None, 
            version='latest',            
            **kwargs):
        """
        Static method that instantiates a Retriever object from a pre-built index access via a dataset.
        Pre-built indices are ofen provided via the `Terrier Data Repository <http://data.terrier.org/>`_.

        :param dataset: The name of the dataset, or a Dataset object that contains the index
        :param variant: The variant of the index to use. If None, the default index will be used.
        :param version: The version of the dataset to use. If None, the latest version will be used.

        Examples::

            dataset = pt.get_dataset("vaswani")
            bm25 = pt.terrier.Retriever.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
            #or
            bm25 = pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed", wmodel="BM25")

        **Index Variants**:

        There are a number of standard index names.
         - `terrier_stemmed` - a classical index, removing Terrier's standard stopwords, and applying Porter's English stemmer
         - `terrier_stemmed_positions` - as per `terrier_stemmed`, but also containing position information
         - `terrier_unstemmed` - a classical index, without applying stopword removal or stemming
         - `terrier_stemmed_text` - as per `terrier_stemmed`, but also containing the raw text of the documents
         - `terrier_unstemmed_text` - as per `terrier_stemmed`, but also containing the raw text of the documents

        """
        return pt.datasets.transformer_from_dataset(dataset, variant=variant, version=version, clz=Retriever, **kwargs)

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

    def __init__(self, 
                 index_location : Union[str,Any], 
                 controls : Optional[Dict[str,str]] = None, 
                 properties : Optional[Dict[str,str]]= None, 
                 metadata : List[str] = ["docno"], 
                 num_results : Optional[int] = None, 
                 wmodel : Optional[Union[str, Callable]] = None, 
                 tokeniser : Union[str,TerrierTokeniser] = TerrierTokeniser.english,
                 threads : int = 1, 
                 verbose : bool = False):
        """
            Init method

            :param index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
            :param controls: A dictionary with the control names and values
            :param properties: A dictionary with the property keys and values
            :param verbose: If True transform method will display progress
            :param num_results: Number of results to retrieve. 
            :param metadata: What metadata to retrieve. Default is ["docno"]. 
        """
        self.indexref = _parse_index_like(index_location)
        self.properties = _mergeDicts(Retriever.default_properties, properties)
        self.concurrentIL = pt.java.autoclass("org.terrier.structures.ConcurrentIndexLoader")
        if pt.terrier.check_version(5.5) and "SimpleDecorateProcess" not in self.properties["querying.processes"]:
            self.properties["querying.processes"] += ",decorate:SimpleDecorateProcess"
            controls = controls or {}
            controls["decorate_batch"] = "on"
        self.metadata = metadata
        self.threads = threads
        self.RequestContextMatching = pt.java.autoclass("org.terrier.python.RequestContextMatching")
        self.search_context = {}
        self.verbose = verbose
        self.tokeniser = TerrierTokeniser.java_tokeniser(TerrierTokeniser._to_obj(tokeniser))

        for key, value in self.properties.items():
            pt.terrier.J.ApplicationSetup.setProperty(str(key), str(value))
        
        self.controls = _mergeDicts(Retriever.default_controls, controls)
        if wmodel is not None:
            from pyterrier.transformer import is_lambda, is_function
            if isinstance(wmodel, str):
                self.controls["wmodel"] = wmodel
            elif is_lambda(wmodel) or is_function(wmodel):
                callback, wmodelinstance = _function2wmodel(wmodel)
                #save the callback instance in this object to prevent being GCd by Python
                self._callback = callback
                self.search_context['context_wmodel'] = wmodelinstance
                self.controls['context_wmodel'] = 'on'
            elif isinstance(wmodel, pt.java.autoclass("org.terrier.matching.models.WeightingModel")):
                self.search_context['context_wmodel'] = wmodel
                self.controls['context_wmodel'] = 'on'
            else:
                raise ValueError("Unknown parameter type passed for wmodel argument: %s" % str(wmodel))
                  
        if self.threads > 1:
            warn(
                "Multi-threaded retrieval is experimental, YMMV.")
            assert pt.terrier.check_version(5.5), "Terrier 5.5 is required for multi-threaded retrieval"

            # we need to see if our indexref is concurrent. if not, we upgrade it using ConcurrentIndexLoader
            # this will upgrade the underlying index too.
            if not self.concurrentIL.isConcurrent(self.indexref):
                warn(
                    "Upgrading indexref %s to be concurrent" % self.indexref.toString())
                self.indexref = self.concurrentIL.makeConcurrent(self.indexref)

        if num_results is not None:
            if num_results > 0:
                self.controls["end"] = str(num_results -1)
            elif num_results == 0:
                del self.controls["end"]
            else: 
                raise ValueError("num_results must be None, 0 or positive")


        MF = pt.java.autoclass('org.terrier.querying.ManagerFactory')
        self.RequestContextMatching = pt.java.autoclass("org.terrier.python.RequestContextMatching")
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
            pt.terrier.J.ApplicationSetup.setProperty(key, str(value))

    def _retrieve_one(self, row, input_results=None, docno_provided=False, docid_provided=False, scores_provided=False):
        rank = FIRST_RANK
        qid = str(row.qid)

        # row is a namedtuple, whose fields are exposed in _fields
        query_toks_present = 'query_toks' in row._fields
        if query_toks_present:
            query = '' # Clear the query so it doesn't match the "applypipeline:off" or "_matchop" condictions below... The query_toks query is converted below.
            srq = self.manager.newSearchRequest(qid)
        else:
            query = row.query
            if _query_needs_tokenised(query):
                query = ' '.join(self.tokeniser.getTokens(query))
            if len(query) == 0:
                warn(
                    "Skipping empty query for qid %s" % qid)
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

        if query_toks_present:
            if len(row.query_toks) == 0:
                warn(
                    "Skipping empty query_toks for qid %s" % qid)
                return []
            srq.setControl("terrierql", "off")
            srq.setControl("parsecontrols", "off")
            srq.setControl("parseql", "off")
            srq.setControl("matchopql", "on")
            query = _querytoks2matchop(row.query_toks)
            srq.setOriginalQuery(query)

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

        :param queries: a pandas.Dataframe with columns=['qid', 'query']. For re-ranking,
                the DataFrame may also have a 'docid' and or 'docno' column.

        :return: pandas.Dataframe with columns=['qid', 'docno', 'rank', 'score']
        """
        results=[]
        if not isinstance(queries, pd.DataFrame):
            raise ValueError(".transform() should be passed a DataFrame (found %s). Use .search() to execute a single query; Use .transform_iter() for iter-dicts" % str(type(queries)))
        
        # use pt.validate - this makes inspection of input columns better
        with pt.validate.any(queries) as v:
            v.columns(includes=['qid', 'query'], excludes=['docid', 'docno'], mode='retrieve') # query based frame without docno or docid
            v.columns(includes=['qid', 'query', 'docid'], mode='rerank') # docid-based results frame
            v.query_frame(extra_columns=['query_toks'], mode='retrieve_toks')
            v.result_frame(extra_columns=['query'], mode='rerank')
            
        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "score" in queries.columns
        input_results = None
        if v.mode == 'rerank':
            assert docno_provided or docid_provided, "For reranking, either docno or docid must be provided"
            assert pt.terrier.check_version(5.3)
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
                if self.verbose and len(queries):
                    iter = pt.tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
                
                for future in iter:
                    res = future.result()
                    results.extend(res)
        else:
            iter = queries.itertuples()
            if self.verbose and len(queries):
                iter = pt.tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
            for row in iter:
                res = self._retrieve_one(row, input_results, docno_provided=docno_provided, docid_provided=docid_provided, scores_provided=scores_provided)
                results.extend(res)

        res_dt = pd.DataFrame(results, columns=['qid', 'docid' ] + self.metadata + ['rank', 'score'])
        # ensure to return the query and any other input columns
        input_cols = queries.columns[ (queries.columns == "qid") | (~queries.columns.isin(res_dt.columns))]
        res_dt = res_dt.merge(queries[input_cols], on=["qid"])
        return res_dt
        
    def __repr__(self):
        return "TerrierRetr(" + ",".join([
            self.indexref.toString(),
            str(self.controls),
            str(self.properties)
            ]) + ")"

    def __str__(self):
        return "TerrierRetr(" + self.controls["wmodel"] + ")"

    def setControls(self, controls):
        for key, value in controls.items():
            self.controls[str(key)] = str(value)

    def setControl(self, control, value):
        self.controls[str(control)] = str(value)

    def schematic(self, *, input_columns = None): 
        return {'label': self.controls['wmodel']}

    def fuse_rank_cutoff(self, k: int) -> Optional[pt.Transformer]:
        """
        Support fusing with RankCutoffTransformer.
        """
        if float(self.controls.get('end', float('inf'))) < k:
            return self # the applied rank cutoff is greater than the one already applied
        if self.controls.get('context_wmodel') == 'on':
            return None # we don't store the original wmodel value so we can't reconstruct
        # apply the new k as num_results
        return pt.inspect.transformer_apply_attributes(self, num_results=k)

    def fuse_feature_union(self, other: pt.Transformer, is_left: bool) -> Optional[pt.Transformer]:
        if isinstance(other, Retriever) and \
           self.indexref == other.indexref and \
           self.controls.get('context_wmodel') != 'on' and \
           other.controls.get('context_wmodel') != 'on':
            features = ["WMODEL:" + self.controls['wmodel'], "WMODEL:" + other.controls['wmodel']] if is_left else ["WMODEL:" + other.controls['wmodel'], "WMODEL:" + self.controls['wmodel']]
            controls = dict(self.controls)
            del controls['wmodel']
            return FeaturesRetriever(self.indexref, features, controls=controls, properties=self.properties,
                metadata=self.metadata, threads=self.threads, verbose=self.verbose)
        return None

    def attributes(self):
        if 'wmodel' in self.controls:
            wmodel = self.controls['wmodel']
        elif self.controls.get('context_wmodel') == 'on':
            wmodel = self.search_context['context_wmodel']
        return [
            pt.inspect.TransformerAttribute('index_location', self.indexref),
            pt.inspect.TransformerAttribute('num_results', int(self.controls.get('end', '999')) + 1),
            pt.inspect.TransformerAttribute('metadata', self.metadata),
            pt.inspect.TransformerAttribute('wmodel', wmodel),
            pt.inspect.TransformerAttribute('threads', self.threads),
            pt.inspect.TransformerAttribute('verbose', self.verbose),
        ] + [
            pt.inspect.TransformerAttribute(key, value)
            for key, value in self.controls.items()
            if key not in ('end', 'wmodel')
        ] + [
            pt.inspect.TransformerAttribute(key, value)
            for key, value in self.properties.items()
        ]

    def apply_attributes(self, **kwargs):
        for attr in self.attributes():
            if attr.name not in kwargs:
                kwargs[attr.name] = attr.value
        kwargs['controls'] = {}
        kwargs['properties'] = {}
        for key, value in list(kwargs.items()):
            if key in ('index_location', 'controls', 'properties', 'metadata', 'num_results', 'wmodel', 'threads', 'verbose'):
                pass
            elif key in self.properties:
                kwargs['properties'][key] = value
                del kwargs[key]
            else:
                kwargs['controls'][key] = value
                del kwargs[key]
        return Retriever(**kwargs)


@pt.java.required
class TextIndexProcessor(pt.Transformer):
    '''
        Creates a new MemoryIndex based on the contents of documents passed to it.
        It then creates a new instance of the innerclass and passes the topics to that.

        This class is the base class for TextScorer, but can be used in other settings as well, 
        for instance query expansion based on text.
    '''

    def __init__(self, innerclass, 
                 takes="queries", 
                 returns="docs", 
                 body_attr="body", 
                 background_index=None, 
                 stemmer : Union[None, str, TerrierStemmer] = TerrierStemmer.porter,
                 stopwords : Union[None, TerrierStopwords, List[str]] = TerrierStopwords.terrier,
                 tokeniser : Union[str,TerrierTokeniser] = TerrierTokeniser.english,
                 verbose=False, 
                 **kwargs):
        #super().__init__(**kwargs)
        self.innerclass = innerclass
        self.takes = takes
        self.returns = returns
        assert isinstance(body_attr, str)
        self.body_attr = body_attr
        if background_index is not None:
            self.background_indexref = _parse_index_like(background_index)
        else:
            self.background_indexref = None
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.tokeniser = tokeniser
        self.kwargs = kwargs
        self.verbose = verbose

    def transform(self, topics_and_res):
        # we use _IterDictIndexer_nofifo, as _IterDictIndexer_fifo (which is default on unix) doesnt support IndexingType.MEMORY as a destination
        from pyterrier.terrier import IndexFactory
        from pyterrier.terrier.index import IndexingType, _IterDictIndexer_nofifo
        pt.validate.result_frame(topics_and_res, extra_columns=[self.body_attr, 'query'])
        documents = topics_and_res[["docno", self.body_attr]].drop_duplicates(subset="docno").rename(columns={self.body_attr:'text'})
        indexref = _IterDictIndexer_nofifo(
            None, 
            type=IndexingType.MEMORY, 
            verbose=self.verbose, 
            stemmer = self.stemmer, 
            stopwords = self.stopwords,
            tokeniser = self.tokeniser
            ).index(documents.to_dict(orient='records'))
        
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
            index = pt.java.autoclass("org.terrier.python.IndexWithBackground")(index_docs, index_background)          

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
            input["docid"] = input.apply(lambda row: docno2docid[row["docno"]], axis=1, result_type='reduce')

        if self.innerclass == Retriever:
            # build up a termpipeline based on the stemmer and stopwords settings
            tp = []
            stops, _ = TerrierStopwords._to_obj(self.stopwords or 'none')
            if stops == TerrierStopwords.terrier:
                tp.append("Stopwords")
            elif stops == TerrierStopwords.none:
                pass # noop
            else:
                # sadly PyTerrierCustomStopwordList only support instantiation from ApplicationSetup or Index properties, not controls
                raise KeyError("Only TerrierStopwords.terrier and TerrierStopwords.none are supported for stopwords in TextIndexProcessor, found %s" % str(stops))       
            stemmer = TerrierStemmer._to_obj(self.stemmer or 'none')
            if stemmer is not TerrierStemmer.none:
                tp.append(TerrierStemmer._to_class(stemmer))

            if 'controls' not in self.kwargs:
                self.kwargs['controls'] = {}
            if not len(tp):
                tp = ["NoOp"] # an empty termpipeline will be detected as missing by ApplyTermPipeline in Terrier
            self.kwargs['controls']['termpipelines'] = ",".join(tp)
        else:
            # if not Retriever, we dont know how to set the termpipeline
            pass

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
        Unknown kwargs are passed to Retriever.

        :param takes: configuration - what is needed as input: `"queries"`, or `"docs"`. Default is `"docs"` since v0.8.
        :param returns: configuration - what is needed as output: `"queries"`, or `"docs"`. Default is `"docs"`.
        :param body_attr: what dataframe input column contains the text of the document. Default is `"body"`.
        :param wmodel: name of the weighting model to use for scoring.
        :param background_index: An optional background index to use for term and collection statistics. If a weighting
            model such as BM25 or TF_IDF or PL2 is used without setting the background_index, the background statistics
            will be calculated from the dataframe, which is ususally not the desired behaviour.

        Example::

            df = pd.DataFrame(
                [
                    ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
                    ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
                ], columns=["qid", "query", "text"])
            textscorer = pt.TextScorer(takes="docs", body_attr="text", wmodel="Tf")
            rtr = textscorer.transform(df)
            #rtr will score each document by term frequency for the query "chemical reactions" based on the provided document contents

        Example::

            df = pd.DataFrame(
                [
                    ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
                    ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
                ], columns=["qid", "query", "text"])
            existing_index = pt.IndexFactory.of(...)
            textscorer = pt.TextScorer(takes="docs", body_attr="text", wmodel="TF_IDF", background_index=existing_index)
            rtr = textscorer.transform(df)
            #rtr will score each document by TF_IDF for the query "chemical reactions" based on the provided document contents
    """

    def __init__(self, takes="docs", **kwargs):
        super().__init__(Retriever, takes=takes, **kwargs)


@pt.java.required
class FeaturesRetriever(Retriever):
    """
    Use this class for retrieval with multiple features
    """

    #: FBR_default_controls(dict): stores the default properties for a FBR
    FBR_default_controls = Retriever.default_controls.copy()
    FBR_default_controls["matching"] = "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull"
    del FBR_default_controls["wmodel"]
    #: FBR_default_properties(dict): stores the default properties
    FBR_default_properties = Retriever.default_properties.copy()

    def __init__(self, 
                 index_location : Union[str,Any], 
                 features : List[str], 
                 controls : Optional[Dict[str,str]] = None, 
                 properties : Optional[Dict[str,str]] = None, 
                 threads : int = 1, 
                 **kwargs):
        """
            Init method

            :param index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
            :param features: List of features to use
            :param controls: A dictionary with the control names and values
            :param properties: A dictionary with the property keys and values
            :param verbose: If True transform method will display progress
            :param num_results: Number of results to retrieve. 
        """
        controls = _mergeDicts(FeaturesRetriever.FBR_default_controls, controls)
        properties = _mergeDicts(FeaturesRetriever.FBR_default_properties, properties)
        self.features = features
        properties["fat.featured.scoring.matching.features"] = ";".join(features)

        # record the weighting model
        self.wmodel = None
        if "wmodel" in kwargs and kwargs['wmodel'] is not None:
            assert isinstance(kwargs["wmodel"], str), "Non-string weighting models not yet supported by FBR"
            self.wmodel = kwargs["wmodel"]
        if "wmodel" in controls:
            self.wmodel = controls["wmodel"]
        
        # check for terrier-core#246 bug usiung FatFull
        if self.wmodel is not None:    
            assert pt.terrier.check_version(5.9), "Terrier 5.9 is required for this functionality, see https://github.com/terrier-org/terrier-core/pull/246"
            
        if threads > 1:
            raise ValueError("Multi-threaded retrieval not yet supported by FeaturesRetriever")
        
        super().__init__(index_location, controls, properties, **kwargs)
        if self.wmodel is None and 'wmodel' in self.controls:
            del self.controls['wmodel'] # Retriever sets a default controls['wmodel'], we only want this

    def __reduce__(self):
        return (
            self.__class__,
            (self.indexref, self.features),
            self.__getstate__()
        )
    
    def schematic(self, *, input_columns = None): 
        if self.wmodel is None:
            return {'label': "FeaturesRetriever: %df" % len(self.features)}
        return {'label': "FeaturesRetriever: %s + %df" % (self.wmodel, len(self.features))}

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
            pt.terrier.J.ApplicationSetup.setProperty(key, str(value))
        #TODO consider the context state?

    @staticmethod 
    def from_dataset(dataset : Union[str,Dataset], 
            variant : Optional[str] = None, 
            version='latest',            
            **kwargs):
        return pt.datasets.transformer_from_dataset(dataset, variant=variant, version=version, clz=FeaturesRetriever, **kwargs)

    def transform(self, queries):
        """
        Performs the retrieval with multiple features

        :param queries: A pandas.Dataframe with columns=['qid', 'query']. For re-ranking,
                the DataFrame may also have a 'docid' and or 'docno' column.

        :return: a pandas.DataFrame with columns=['qid', 'docno', 'score', 'rank, 'features']
        """
        results = []
        if not isinstance(queries, pd.DataFrame):
            raise ValueError(".transform() should be passed a dataframe. Use .search() to execute a single query; Use .transform_iter() for iter-dicts")

        # use pt.validate - this makes inspection of input columns better
        with pt.validate.any(queries) as v:
            v.columns(includes=['qid', 'query', 'docid'], mode='rerank') # docid-based results frame
            v.query_frame(extra_columns=['query_toks'], mode='retrieve_toks')
            v.query_frame(extra_columns=['query'], mode='retrieve')
            v.result_frame(extra_columns=['query'], mode='rerank')

        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "score" in queries.columns
        if v.mode == 'rerank':
            assert docno_provided or docid_provided, "For reranking, either docno or docid must be provided"
            #re-ranking mode
            assert pt.terrier.check_version(5.3)
            input_results = queries

            # query is optional, and functionally dependent on qid.
            # Hence as long as one row has the query for each qid, 
            # the rest can be None
            queries = input_results[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
            RequestContextMatching = pt.java.autoclass("org.terrier.python.RequestContextMatching")

            if not scores_provided and self.wmodel is None:
                raise ValueError("We're in re-ranking mode, but input does not have scores, and wmodel is None")
        else:
            assert v.mode == 'retrieve' or v.mode == 'retrieve_toks'
            assert not scores_provided

            if self.wmodel is None:
                raise ValueError("We're in retrieval mode (input columns were "+str(queries.columns)+"), but wmodel is None. FeaturesRetriever requires a wmodel be set for identifying the candidate set. "
                    +" Hint: wmodel argument for FeaturesRetriever, e.g. FeaturesRetriever(index, features, wmodel=\"DPH\")")

        if queries["qid"].dtype == np.int64:
            queries['qid'] = queries['qid'].astype(str)

        newscores=[]
        iter = queries.itertuples()
        if self.verbose and len(queries):
            iter = pt.tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
        for row in iter:
            qid = str(row.qid)
            query_toks_present = 'query_toks' in row._fields
            if query_toks_present:
                # Even though it might look like we should parse the query toks here, we don't want the resulting query to be caught by the conditions
                # that come before the "if query_toks_present" check. So we set it to an empty string and handle the parsing below.
                query = ''
                srq = self.manager.newSearchRequest(qid)
            else:
                query = row.query
                if _query_needs_tokenised(query):
                    query = ' '.join(self.tokeniser.getTokens(query))
                if len(query) == 0:
                    warn(
                        "Skipping empty query for qid %s" % qid)
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

            if query_toks_present:
                if len(row.query_toks) == 0:
                    warn(
                        "Skipping empty query_toks for qid %s" % qid)
                    return []
                srq.setControl("terrierql", "off")
                srq.setControl("parsecontrols", "off")
                srq.setControl("parseql", "off")
                srq.setControl("matchopql", "on")
                query = _querytoks2matchop(row.query_toks)
                srq.setOriginalQuery(query)

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
            srq = pt.java.cast('org.terrier.querying.Request', srq)
            fres = pt.java.cast('org.terrier.learning.FeaturedResultSet', srq.getResultSet())
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
        return "TerrierFeatRetr(" + ",".join([
            self.indexref.toString(),
            str(self.features),
            str(self.controls),
            str(self.properties)
        ]) + ")"

    def __str__(self):
        if self.wmodel is None:
            return "TerrierFeatRetr(" + str(len(self.features)) + " features)"
        return "TerrierFeatRetr(" + self.controls["wmodel"] + " and " + str(len(self.features)) + " features)"

    def fuse_left(self, left: pt.Transformer) -> Optional[pt.Transformer]:
        # Can merge Retriever >> FeaturesRetriever into a single FeaturesRetriever that also retrieves
        # if the indexref matches and the current FeaturesRetriever isn't already reranking.
        if isinstance(left, Retriever) and \
           self.indexref == left.indexref and \
           left.controls.get('context_wmodel') != 'on' and \
           self.wmodel is None:
            return FeaturesRetriever(
                self.indexref,
                self.features,
                controls=self.controls,
                properties=self.properties,
                threads=self.threads,
                wmodel=left.controls['wmodel'],
            )
        return None

    def fuse_rank_cutoff(self, k: int) -> Optional[pt.Transformer]:
        """
        Support fusing with RankCutoffTransformer.
        """
        if float(self.controls.get('end', float('inf'))) < k:
            return self # the applied rank cutoff is greater than the one already applied
        if self.wmodel is None:
            return None # not a retriever
        # apply the new k as num_results
        return FeaturesRetriever(self.indexref, self.features, controls=self.controls, properties=self.properties,
            threads=self.threads, wmodel=self.wmodel, verbose=self.verbose, num_results=k)

    def fuse_feature_union(self, other: pt.Transformer, is_left: bool) -> Optional[pt.Transformer]:
        if isinstance(other, FeaturesRetriever) and \
           self.indexref == other.indexref and \
           self.wmodel is None and \
           other.wmodel is None:
            features = self.features + other.features if is_left else other.features + self.features
            return FeaturesRetriever(self.indexref, features, controls=self.controls, properties=self.properties,
                threads=self.threads, wmodel=self.wmodel, verbose=self.verbose)

        if isinstance(other, Retriever) and \
           self.indexref == other.indexref and \
           self.wmodel is None  and \
           other.controls.get('context_wmodel') != 'on':
            features = self.features + ["WMODEL:" + other.controls['wmodel']] if is_left else ["WMODEL:" + other.controls['wmodel']] + self.features
            return FeaturesRetriever(self.indexref, features, controls=self.controls, properties=self.properties,
                threads=self.threads, wmodel=self.wmodel, verbose=self.verbose)
        
        return None
