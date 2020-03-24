# import jnius_config
# jnius_config.add_classpath("../terrier-project-5.1-jar-with-dependencies.jar")
# from jnius import autoclass, cast

# import pytrec_eval
import os, json, wget
import numpy as np
import pandas as pd
from xml.dom import minidom
import urllib.request


def setup_terrier(file_path, version):
    """
    Download the Terrier.jar file for the given version at the given file_path
    Called inside init()

    Args:
        file_path(str): Where to download
        version(str): Which version
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(file_path,"terrier-assemblies-"+version+"-jar-with-dependencies.jar")):
        print("Terrier "+ version +" not found, downloading...")
        url = "https://repo.maven.apache.org/maven2/org/terrier/terrier-assemblies/"+version+"/terrier-assemblies-"+version+"-jar-with-dependencies.jar"
        wget.download(url, file_path)
        print("Done")

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
properties = None

def init(version=None, mem="4096", packages=[]):
    """
    Function necessary to be called before Terrier classes and methods can be used.
    Loads the Terrier.jar file and imports classes. Also finds the correct version of Terrier to download if no version is specified.

    Args:
        version(str): Which version of Terrier to download. Default=None.
            If None, find the newest Terrier version in maven and download it.
        mem: Maximum memory allocated for java heap in MB. Default=4096.
        packages: Extra .jar files to load.
    """
    global ApplicationSetup
    global properties
    # If version is not specified, find newest and download it
    if version is None:
        url_str = "https://repo1.maven.org/maven2/org/terrier/terrier-assemblies/maven-metadata.xml"
        with urllib.request.urlopen(url_str) as url:
            xml_str = url.read()
        xmldoc = minidom.parseString(xml_str)
        obs_values = xmldoc.getElementsByTagName("latest")
        version = obs_values[0].firstChild.nodeValue
    else: version = str(version)
    url = "https://repo.maven.apache.org/maven2/org/terrier/terrier-assemblies/"+version+"/terrier-assemblies-"+version+"-jar-with-dependencies.jar"
    setup_terrier(file_path, version)

    # Import pyjnius and other classes
    import jnius_config
    jnius_config.set_classpath(os.path.join(file_path,"terrier-assemblies-"+version+"-jar-with-dependencies.jar"))
    jnius_config.add_options('-Xmx'+str(mem)+'m')
    from jnius import autoclass, cast
    # Properties = autoclass('java.util.Properties')
    properties = autoclass('java.util.Properties')()
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
    from utils import Utils
    from batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer

    # Make imports global
    globals()["Utils"]=Utils
    globals()["autoclass"] = autoclass
    globals()["cast"] = cast
    globals()["BatchRetrieve"] = BatchRetrieve
    globals()["Indexer"] = Indexer
    globals()["FeaturesBatchRetrieve"] = FeaturesBatchRetrieve
    globals()["TRECCollectionIndexer"] = TRECCollectionIndexer
    globals()["FilesIndexer"] = FilesIndexer
    globals()["DFIndexer"] = DFIndexer
    globals()["ApplicationSetup"] = ApplicationSetup

    # Import other java packages
    if packages != []:
        pkgs_string = ",".join(packages)
        properties.put("terrier.mvn.coords",pkgs_string)
    ApplicationSetup.bootstrapInitialisation(properties)

def set_property(property):
    ApplicationSetup.bootstrapInitialisation(properties)
def set_properties(properties):
    for control,value in kwargs.items():
        self.properties.put(control,value)
    ApplicationSetup.bootstrapInitialisation(self.properties)

def Experiment(topics,retr_systems,eval_metrics,qrels, names=None, perquery=False, dataframe=True):
    """
    Cornac style experiment. Combines retrieval and evaluation.
    Allows easy comparison of multiple retrieval systems with different properties and controls.

    Args:
        topics: Either a path to a topics file or a pandas.Dataframe with columns=['qid', 'query']
        retr_systems(list): A list of BatchRetrieve objects to compare
        eval_metrics(list): Which evaluation metrics to use. E.g. ['map']
        qrels: Either a path to a qrels file or a pandas.Dataframe with columns=['qid','docno', 'relevancy']
        names(list)=List of names for each retrieval system when presenting the results.
            Defaul=None. If None: Use names of weighting models for each retrieval system.
        perquery(bool): If true return each metric for each query, else return mean metrics. Default=False.
        dataframe(bool): If True return results as a dataframe. Else as a dictionary of dictionaries. Default=True.

    Returns:
        A Dataframe with each retrieval system with each metric evaluated.
    """
    if type(topics)==type(""):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if type(qrels)==type(""):
        if os.path.isfile(qrels):
            qrels = Utils.parse_qrels(qrels)

    results = []
    neednames = names is None
    if neednames:
        names = []
    for system in retr_systems:
        results.append(system.transform(topics))
        if neednames:
            names.append(system.controls["wmodel"])
    evals={}

    for weight,res in zip(names,results):
        evals[weight]=Utils.evaluate(res,qrels, metrics=eval_metrics, perquery=perquery)
    if dataframe:
        evals = pd.DataFrame.from_dict(evals, orient='index')
    return evals

class LTR_pipeline():
    """
    This class simplifies the use of Scikit-learn's techniques for learning-to-rank.
    """
    def __init__(self, index, model, features, qrels, LTR):
        """
        Init method

        Args:
            index: The index which to query.
                Can be an Indexer object(Can be parent Indexer or any of its child classes)
                or a string with the path to the index_dir/data.properties
            model(str): The weighting model to use. E.g. "PL2"
            features(list): A list of the feature names to use
            qrels(DataFrame): Dataframe with columns=['qid','docno', 'relevancy']
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
        """
        self.feat_retrieve = FeaturesBatchRetrieve(index, features)
        self.feat_retrieve.setControl('wmodel', model)
        self.qrels = qrels
        self.LTR = LTR
        self.controls = self.feat_retrieve.controls

    def fit(self, topicsTrain):
        """
        Trains the model with the given topics.

        Args:
            topicsTrain(DataFrame): A dataframe with the topics to train the model
        """
        if len(topicsTrain) == 0:
            raise ValueError("No topics to fit to")
        train_DF = self.feat_retrieve.transform(topicsTrain)
        if not 'features' in train_DF.columns:
            raise ValueError("No features column retrieved")
        train_DF = train_DF.merge(self.qrels, on=['qid','docno'], how='left').fillna(0)
        self.LTR.fit(list(train_DF["features"]), train_DF["relevancy"].values)

    def transform(self, topicsTest):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = self.feat_retrieve.transform(topicsTest)
        test_DF["score"] = self.LTR.predict(list(test_DF["features"]))
        return test_DF

class XGBoostLTR_pipeline(LTR_pipeline):
    """
    This class simplifies the use of XGBoost's techniques for learning-to-rank.
    """

    def __init__(self, index, model, features, qrels, LTR, validqrels):
        """
        Init method

        Args:
            index: The index which to query.
                Can be an Indexer object(Can be parent Indexer or any of its child classes)
                or a string with the path to the index_dir/data.properties
            model(str): The weighting model to use. E.g. "PL2"
            features(list): A list of the feature names to use
            qrels(DataFrame): Dataframe with columns=['qid','docno', 'relevancy']
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
            validqrels(DataFrame): The qrels which to use for the validation.
        """
        super().__init__(index,model,features, qrels, LTR)
        self.validqrels = validqrels

    def transform(self, topicsTest):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = self.feat_retrieve.transform(topicsTest)
        #xgb is more sensitive about the type of the values.
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

    def fit(self, topicsTrain, topicsValid):
        """
        Trains the model with the given training and validation topics.

        Args:
            topicsTrain(DataFrame): A dataframe with the topics to train the model
            topicsValid(DataFrame): A dataframe with the topics for validation
        """
        if len(topicsTrain) == 0:
            raise ValueError("No training topics to fit to")
        if len(topicsValid) == 0:
            raise ValueError("No training topics to fit to")

        tr_res = self.feat_retrieve.transform(topicsTrain)
        va_res = self.feat_retrieve.transform(topicsValid)
        if not 'features' in tr_res.columns:
            raise ValueError("No features column retrieved")
        if not 'features' in va_res.columns:
            raise ValueError("No features column retrieved")

        tr_res = tr_res.merge(self.qrels, on=['qid','docno'], how='left').fillna(0)
        va_res = va_res.merge(self.validqrels, on=['qid','docno'], how='left').fillna(0)

        self.LTR.fit(
            np.stack(tr_res["features"].values),
            tr_res["relevancy"].values, tr_res.groupby(["qid"]).count()["docno"].values,
            eval_set=[(np.stack(va_res["features"].values),
            va_res["relevancy"].values)],
            eval_group=[va_res.groupby(["qid"]).count()["docno"].values]
        )
