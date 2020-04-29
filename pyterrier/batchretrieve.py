from jnius import autoclass, cast
import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import Utils
from .index import Indexer
from .transformer import TransformerBase, Symbol
from tqdm import tqdm

# import time

def importProps():
    from . import properties as props
    # Make import global
    globals()["props"] = props
props = None


def parse_index_like(index_location):
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

class BatchRetrieve(BatchRetrieveBase):
    """
    Use this class for retrieval by Terrier

    Attributes:
        default_controls(dict): stores the default controls
        default_properties(dict): stores the default properties
        IndexRef: stores the index reference object
        appSetup: stores the Terrier ApplicationSetup object
        verbose(bool): If True transform method will display progress
        properties(dict): Current properties
        controls(dict): Current controls
    """
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

    default_properties = {
        "querying.processes": "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess",
        "querying.postfilters": "decorate:SimpleDecorate,site:SiteFilter,scope:Scope",
        "querying.default.controls": "wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on",
        "querying.allowed.controls": "scope,qe,qemodel,start,end,site,scope",
        "termpipelines": "Stopwords,PorterStemmer"
    }

    """
        Init method

        Args:
            index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
            controls(dict): A dictionary with with the control names and values
            properties(dict): A dictionary with with the property keys and values
            verbose(bool): If True transform method will display progress
    """
    def __init__(self, index_location, controls=None, properties=None, **kwargs):
        super().__init__(kwargs)
        
        self.indexref = parse_index_like(index_location)
        self.appSetup = autoclass('org.terrier.utility.ApplicationSetup')
        self.properties = _mergeDicts(BatchRetrieve.default_properties, properties)

        if props is None:
            importProps()
        for key, value in self.properties.items():
            self.appSetup.setProperty(key, value)

        self.controls = _mergeDicts(BatchRetrieve.default_controls, controls)

        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.manager = MF._from_(self.indexref)

    def transform(self, queries, metadata=["docno"]):
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
            from . import check_version
            assert check_version(5.3)
            input_results = queries

            # query is optional, and functionally dependent on qid.
            # Hence as long as one row has the query for each qid, 
            # the rest can be None
            queries = input_results[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
            RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")

        for index,row in tqdm(queries.iterrows(), total=queries.shape[0], unit="q") if self.verbose else queries.iterrows():
            rank = 0
            qid = str(row['qid'])
            srq = self.manager.newSearchRequest(qid, row['query'])
            for control, value in self.controls.items():
                srq.setControl(control, value)

            # this handles the case that a candidate set of documents has been set. 
            if docno_provided or docid_provided:
                # we use RequestContextMatching to make a ResultSet from the 
                # documents in the candidate set. 
                matching_config_factory = RequestContextMatching.of(srq)
                input_query_results = input_results[input_results["qid"] == qid]
                if docno_provided:
                    matching_config_factory.fromDocnos(input_query_results["docno"].values.tolist())
                elif docid_provided:
                    matching_config_factory.fromDocids(input_query_results["docid"].values.tolist())
                if scores_provided:
                    matching_config_factory.withScores(input_query_results["scores"].values.tolist())
                matching_config_factory.build()
                srq.setControl("matching", "org.terrier.matching.ScoringMatching" + "," + srq.getControl("matching"))
            
            # now ask Terrier to run the request
            self.manager.runSearchRequest(srq)
            result = srq.getResults()
            for item in result:
                metadata_list = []
                for meta_column in metadata:
                    metadata_list.append(item.getMetadata(meta_column))
                res = [str(row['qid'])] + metadata_list + [rank, item.getScore()]
                rank += 1
                results.append(res)
        res_dt = pd.DataFrame(results, columns=['qid', ] + metadata + ['rank', 'score'])
        return res_dt

    def __str__(self):
        return "BR(" + self.controls["wmodel"] + ")"

    def saveResult(self, result, path, run_name=None):
        if run_name is None:
            run_name = self.controls["wmodel"]
        res_copy = result.copy()[["qid", "docno", "rank", "score"]]
        res_copy.insert(1, "Q0", "Q0")
        res_copy.insert(5, "run_name", run_name)
        res_copy.to_csv(path, sep=" ", header=False, index=False)

    def setControls(self, controls):
        for key, value in controls.items():
            self.controls[key] = value

    def setControl(self, control, value):
        self.controls[control] = value

def _mergeDicts(defaults, settings):
    KV = defaults.copy()
    if settings is not None and len(settings) > 0:
        KV.update(settings)
    return KV

class FeaturesBatchRetrieve(BatchRetrieve):
    """
    Use this class for retrieval with multiple features

    Attributes:
        default_controls(dict): stores the default controls
        default_properties(dict): stores the default properties
        IndexRef: stores the index reference object
        appSetup: stores the Terrier ApplicationSetup object
        verbose(bool): If True transform method will display progress
        properties(dict): Current properties
        controls(dict): Current controls
    """
    FBR_default_controls = BatchRetrieve.default_controls.copy()
    FBR_default_controls["matching"] = "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull"
    FBR_default_properties = BatchRetrieve.default_properties.copy()

    def __init__(self, index_location, features, controls={}, properties={}, verbose=0):
        """
            Init method

            Args:
                index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
                features(list): List of features to use
                controls(dict): A dictionary with with the control names and values
                properties(dict): A dictionary with with the control names and values
                verbose(bool): If True transform method will display progress
        """
        # if props==None:
        #     importProps()
        controls = _mergeDicts(FeaturesBatchRetrieve.FBR_default_controls, controls)
        properties = _mergeDicts(FeaturesBatchRetrieve.FBR_default_properties, properties)
        self.features = features
        properties["fat.featured.scoring.matching.features"] = ";".join(features)
        super().__init__(index_location, controls=controls, properties=properties, verbose=verbose)

    def transform(self, topics):
        """
        Performs the retrieval with multiple features

        Args:
            topics: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']

        Returns:
            pandas.Dataframe with columns=['qid', 'docno', 'score', 'features']
        """
        results = []
        queries = Utils.form_dataframe(topics)
        for index, row in tqdm(queries.iterrows(), total=queries.shape[0], unit="q") if self.verbose else queries.iterrows():
            srq = self.manager.newSearchRequest(row['qid'], row['query'])
            for control, value in self.controls.items():
                srq.setControl(control, value)
            self.manager.runSearchRequest(srq)
            srq = cast('org.terrier.querying.Request', srq)
            fres = cast('org.terrier.learning.FeaturedResultSet', srq.getResultSet())
            feat_names = fres.getFeatureNames()
            feats_values = []
            for feat in feat_names:
                feats_values.append(fres.getFeatureScores(feat))
            for i in range(fres.getResultSize()):
                elem = []
                # start_time = time.time()
                elem.append(row["qid"])
                elem.append(fres.getMetaItems("docno")[i])
                elem.append(fres.getScores()[i])
                feats_array = []
                for j in range(len(feats_values)):
                    feats_array.append(feats_values[j][i])
                feats_array = np.array(feats_array)
                # start_time = time.time()
                elem.append(feats_array)
                results.append(elem)

        res_dt = pd.DataFrame(results, columns=["qid", "docno", "score", "features"])
        return res_dt

    def __str__(self):
        return "FBR(" + self.controls["wmodel"] + " and " + str(len(self.features)) + " features)"
