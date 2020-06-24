__version__ = "0.3.0.dev"

import os
from .bootstrap import _logging, setup_terrier, setup_jnius
from . import datasets

import importlib

#sub modules
rewrite = None
index = None
pipelines = None
anserini = None
transformer = None

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
IndexFactory = None
IndexRef = None
properties = None

HOME_DIR = None

def init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN', home_dir=None, boot_packages=[]):
    """
    Function necessary to be called before Terrier classes and methods can be used.
    Loads the Terrier.jar file and imports classes. Also finds the correct version of Terrier to download if no version is specified.

    Args:
        version(str): Which version of Terrier to download. Default=None.
            If None, find the newest Terrier version in maven and download it.
        mem(str): Maximum memory allocated for java heap in MB. Default is 1/4 of physical memory.
        packages(list(str)): Extra maven package coordinates files to load. Default=[]. More information at https://github.com/terrier-org/terrier-core/blob/5.x/doc/terrier_develop.md
        jvm_opts(list(str)): Extra options to pass to the JVM. Default=[].
        redirect_io(boolean): If True, the Java System.out and System.err will be redirected to Pythons sys.out and sys.err. Default=True.
        logging(str): the logging level to use.
                      Can be one of 'INFO', 'DEBUG', 'TRACE', 'WARN', 'ERROR'. The latter is the quietest.
                      Default='WARN'.
        home_dir(str): the home directory to use. Default to PYTERRIER_HOME environment variable.
    """
    global ApplicationSetup
    global properties
    global firstInit
    global file_path
    global HOME_DIR

    # we keep a local directory
    if home_dir is not None:
        HOME_DIR = home_dir
    if "PYTERRIER_HOME" in os.environ:
        HOME_DIR = os.environ["PYTERRIER_HOME"]
    else:
        from os.path import expanduser
        userhome = expanduser("~")
        HOME_DIR = os.path.join(userhome, ".pyterrier")
        if not os.path.exists(HOME_DIR):
            os.mkdir(HOME_DIR)

    # get the initial classpath for the JVM
    classpathTrJars = setup_terrier(HOME_DIR, version, boot_packages=boot_packages)

    # Import pyjnius and other classes
    import jnius_config
    for jar in classpathTrJars:
        jnius_config.add_classpath(jar)
    if jvm_opts is not None:
        for opt in jvm_opts:
            jnius_config.add_options(opt)
    if mem is not None:
        jnius_config.add_options('-Xmx' + str(mem) + 'm')
    from jnius import autoclass, cast
    properties = autoclass('java.util.Properties')()
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')

    from .batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from .utils import Utils
    from .datasets import get_dataset, list_datasets
    from .index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, FlatJSONDocumentIterator, IndexingType
    from .pipelines import LTR_pipeline, XGBoostLTR_pipeline, Experiment

    # Make imports global
    globals()["autoclass"] = autoclass
    globals()["cast"] = cast
    globals()["ApplicationSetup"] = ApplicationSetup

    global rewrite
    global anserini
    global pipelines
    global index
    global transformer

    rewrite = importlib.import_module('.rewrite', package='pyterrier') 
    anserini = importlib.import_module('.anserini', package='pyterrier') 
    pipelines = importlib.import_module('.pipelines', package='pyterrier') 
    index = importlib.import_module('.index', package='pyterrier') 
    transformer = importlib.import_module('.transformer', package='pyterrier') 


    # append the python helpers
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
    _logging(logging)
    setup_jnius()

    globals()["get_dataset"] = get_dataset
    globals()["list_datasets"] = list_datasets
    globals()["Experiment"] = Experiment
    globals()["BatchRetrieve"] = BatchRetrieve
    globals()["Indexer"] = Indexer
    globals()["FeaturesBatchRetrieve"] = FeaturesBatchRetrieve
    globals()["TRECCollectionIndexer"] = TRECCollectionIndexer
    globals()["FilesIndexer"] = FilesIndexer
    globals()["DFIndexer"] = DFIndexer
    globals()["DFIndexUtils"] = DFIndexUtils
    globals()["IterDictIndexer"] = IterDictIndexer
    globals()["FlatJSONDocumentIterator"] = FlatJSONDocumentIterator
    globals()["Utils"] = Utils
    globals()["LTR_pipeline"] = LTR_pipeline
    globals()["XGBoostLTR_pipeline"] = XGBoostLTR_pipeline
    globals()["IndexFactory"] = autoclass("org.terrier.structures.IndexFactory")
    globals()["IndexRef"] = autoclass("org.terrier.querying.IndexRef")
    globals()["IndexingType"] = IndexingType

    firstInit = True

def started():
    return(firstInit)

def check_version(min):
    from jnius import autoclass
    from packaging.version import Version
    min = Version(str(min))
    currentVer = Version(autoclass("org.terrier.Version").VERSION.replace("-SNAPSHOT", ""))
    return currentVer >= min

def redirect_stdouterr():
    from . import bootstrap
    bootstrap.redirect_stdouterr()

def logging(level):
    from . import bootstrap
    bootstrap.logging(level)

def set_property(k, v):
    # properties = Properties()
    properties[k] = v
    ApplicationSetup.bootstrapInitialisation(properties)

def set_properties(kwargs):
    # properties = Properties()
    for control, value in kwargs.items():
        properties.put(control, value)
    ApplicationSetup.bootstrapInitialisation(properties)

def run(cmd, args=[]):
    from jnius import autoclass
    autoclass("org.terrier.applications.CLITool").main([cmd] + args)

def extend_classpath(mvnpackages):
    assert check_version(5.3), "Terrier 5.3 required for this functionality"
    if isinstance(mvnpackages, str):
        mvnpackages = [mvnpackages]
    from jnius import autoclass, cast
    thelist = autoclass("java.util.ArrayList")()
    for pkg in mvnpackages:
        thelist.add(pkg)
    mvnr = ApplicationSetup.getPlugin("MavenResolver")
    assert mvnr is not None
    mvnr = cast("org.terrier.utility.MavenResolver", mvnr)
    mvnr.addDependencies(thelist)
