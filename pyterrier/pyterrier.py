import os, json, wget
import numpy as np
import pandas as pd

TERRIER_PKG = "org.terrier"

import mavenresolver

def setup_terrier(file_path, terrier_version=None, helper_version=None):
    """
    Download the Terrier.jar file for the given version at the given file_path
    Called inside init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier
        helper_version(str): Which version of the helper
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    # If version is not specified, find newest and download it
    if terrier_version is None:
        terrier_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else: 
        terrier_version = str(terrier_version) #just in case its a float
    #obtain the fat jar from Maven
    trJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-assemblies", terrier_version, file_path, "jar-with-dependencies")
    
    #now the helper classes
    if helper_version is None:
        helper_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else: 
        helper_version = str(helper_version) #just in case its a float
    helperJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-python-helper", helper_version, file_path, "jar")
    return [trJar, helperJar]

   
file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
properties = None


def init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=False, logging='WARN'):
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
    global firstInit
    global file_path
    
    classpathTrJars = setup_terrier(file_path, version)

    # Import pyjnius and other classes
    import jnius_config
    for jar in classpathTrJars:
        jnius_config.add_classpath(jar)
    if jvm_opts is not None :
        for opt in jvm_opts:
            jnius_config.add_options(opt)
    if mem is not None:
        jnius_config.add_options('-Xmx'+str(mem)+'m')
    from jnius import autoclass, cast
    properties = autoclass('java.util.Properties')()
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
    from utils import Utils
    from batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils

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
    globals()["DFIndexUtils"] = DFIndexUtils
    globals()["ApplicationSetup"] = ApplicationSetup

    #append the python helpers
    if packages is None:
        packages = []

    # Import other java packages
    if packages != []:
        pkgs_string = ",".join(packages)
        properties.put("terrier.mvn.coords", pkgs_string)
    ApplicationSetup.bootstrapInitialisation(properties)

    if redirect_io:
        raise ValueError("Sorry, this doesnt work here. Call pt.redirect_stdouterr() yourself later")
    #if redirect_io:
        #this ensures that the python stdout/stderr and the Java are matched
    #    redirect_stdouterr()
    setup_logging(logging)
    firstInit = True

def started():
    return(firstInit)

def redirect_stdouterr():
    from jnius import autoclass
    from utils import MyOut
    import sys
    jls = autoclass("java.lang.System")
    jls.setOut(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')( MyOut(sys.stdout)), 
            signature="(Ljava/io/OutputStream;)V"))
    jls.setErr(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')( MyOut(sys.stderr)), 
            signature="(Ljava/io/OutputStream;)V"))

def setup_logging(level):
    from jnius import autoclass
    autoclass("org.terrier.python.PTUtils").setLogLevel(level, None)


def set_property(k,v):
    # properties = Properties()
    properties[k] = v
    ApplicationSetup.bootstrapInitialisation(properties)

def set_properties(kwargs):
    # properties = Properties()
    for control,value in kwargs.items():
        properties.put(control,value)
    ApplicationSetup.bootstrapInitialisation(properties)

def run(cmd, args=[]):
    from jnius import autoclass
    autoclass("org.terrier.applications.CLITool").main([cmd] + args)

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
