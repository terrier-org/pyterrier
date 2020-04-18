import os
import pandas as pd
from .bootstrap import logging, setup_logging, setup_terrier, setup_jnius
from . import mavenresolver
from . utils import Utils
from . import datasets

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
properties = None



def init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN'):
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

    from .batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from .utils import Utils
    from .index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils
    from .pipelines import LTR_pipeline, XGBoostLTR_pipeline

    # Make imports global
    globals()["autoclass"] = autoclass
    globals()["cast"] = cast
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
       # this ensures that the python stdout/stderr and the Java are matched
       redirect_stdouterr()
    logging(logging)
    setup_jnius()

    globals()["BatchRetrieve"] = BatchRetrieve
    globals()["Indexer"] = Indexer
    globals()["FeaturesBatchRetrieve"] = FeaturesBatchRetrieve
    globals()["TRECCollectionIndexer"] = TRECCollectionIndexer
    globals()["FilesIndexer"] = FilesIndexer
    globals()["DFIndexer"] = DFIndexer
    globals()["DFIndexUtils"] = DFIndexUtils
    globals()["Utils"] = Utils
    globals()["LTR_pipeline"] = LTR_pipeline
    globals()["XGBoostLTR_pipeline"] = XGBoostLTR_pipeline

    firstInit = True

def started():
    return(firstInit)

def redirect_stdouterr():
    from . import bootstrap
    bootstrap.redirect_stdouterr()

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
