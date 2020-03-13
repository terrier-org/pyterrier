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
    file_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(file_path,"terrier-assemblies-"+version+"-jar-with-dependencies.jar")):
        print("Terrier "+ version +" not found, downloading")
        url = "https://repo.maven.apache.org/maven2/org/terrier/terrier-assemblies/"+version+"/terrier-assemblies-"+version+"-jar-with-dependencies.jar"
        wget.download(url, file_path)

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
properties = None

def init(version=None, mem="4096", packages=[]):
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
    from index import FilesIndexer, TRECCollectionIndexer, DFIndexer

    # Make imports global
    globals()["Utils"]=Utils
    globals()["autoclass"] = autoclass
    globals()["cast"] = cast
    globals()["BatchRetrieve"] = BatchRetrieve
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
    # properties = Properties()
    ApplicationSetup.bootstrapInitialisation(properties)
def set_properties(properties):
    # properties = Properties()
    for control,value in kwargs.items():
        self.properties.put(control,value)
    ApplicationSetup.bootstrapInitialisation(self.properties)

def Experiment(topics,retr_systems,eval_metrics,qrels, perquery=False, dataframe=True):
    if type(topics)==type(""):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if type(qrels)==type(""):
        if os.path.isfile(qrels):
            qrels = Utils.parse_qrels(qrels)

    results = []
    weightings = []
    for system in retr_systems:
        results.append(system.transform(topics))
        weightings.append(system.controls["wmodel"])
    evals={}

    for weight,res in zip(weightings,results):
        evals[weight]=Utils.evaluate(res,qrels, metrics=eval_metrics, perquery=perquery)
    if dataframe:
        evals = pd.DataFrame(evals)
    return evals

class LTR_pipeline():
    def __init__(self, index, topics, model, features, qrels, LTR):
        self.feat_retrieve = FeaturesBatchRetrieve(index, features)
        self.feat_retrieve.setControl('wmodel', model)
        self.qrels = qrels
        self.LTR = LTR

    def fit(self, topicsTrain):
        if len(topicsTrain) == 0:
            raise ValueError("No topics to fit to")
        train_DF = feat_retrieve.transform(topicsTrain)
        if not 'features' in train_DF.columns:
            raise ValueError("No features column retrieved")
        train_DF = train_DF.merge(qrels, on=['qid','docno'], how='left')
        self.LTR.fit(list(train_DF["features"]),train_DF["relevancy"].values)

    def transform(self, topicsTest):
        test_DF = feat_retrieve.transform(topicsTest)
        test_DF["predicted"] = self.LTR.predict(list(test_DF["features"]))
        return test_DF
