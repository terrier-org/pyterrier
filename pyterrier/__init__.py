__version__ = "0.3.0.dev"

import os
from .bootstrap import _logging, setup_terrier, setup_jnius

import importlib

#sub modules
anserini = None
cache = None
index = None
io = None
model = None
pipelines = None
rewrite = None
transformer = None
apply = None

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
IndexFactory = None
IndexRef = None
properties = None
tqdm = None
HOME_DIR = None

def init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN', home_dir=None, boot_packages=[], tqdm=None):
    """
    Function necessary to be called before Terrier classes and methods can be used.
    Loads the Terrier .jar file and imports classes. Also finds the correct version of Terrier to download if no version is specified.

    Args:
        version(str): Which version of Terrier to download. Default is `None`.

         * If None, find the newest Terrier released version in MavenCentral and download it.
         * If `"snapshot"`, will download the latest build from `Jitpack <https://jitpack.io/>`_.

        mem(str): Maximum memory allocated for the Java virtual machine heap in MB. Corresponds to java `-Xmx` commandline argument. Default is 1/4 of physical memory.
        boot_packages(list(str)): Extra maven package coordinates files to load before starting Java. Default=`[]`. There is more information about loading packages in the `Terrier documentation <https://github.com/terrier-org/terrier-core/blob/5.x/doc/terrier_develop.md>`_
        packages(list(str)): Extra maven package coordinates files to load, using the Terrier classloader. Default=`[]`. See also `boot_packages` above.
        jvm_opts(list(str)): Extra options to pass to the JVM. Default=`[]`. For instance, you may enable Java assertions by setting `jvm_opts=['-ea']`
        redirect_io(boolean): If True, the Java `System.out` and `System.err` will be redirected to Pythons sys.out and sys.err. Default=True.
        logging(str): the logging level to use:

         * Can be one of `'INFO'`, `'DEBUG'`, `'TRACE'`, `'WARN'`, `'ERROR'`. The latter is the quietest.
         * Default is `'WARN'`.

        home_dir(str): the home directory to use. Default to PYTERRIER_HOME environment variable.
        tqdm: The `tqdm <https://tqdm.github.io/>`_ instance to use for progress bars within PyTerrier. Defaults to tqdm.tqdm. Available options are `'tqdm'`, `'auto'` or `'notebook'`.

   
    **Locating the Terrier .jar file:** PyTerrier is not tied to a specific version of Terrier and will automatically locate and download a recent Terrier .jar file. However, inevitably, some functionalities will require more recent Terrier versions. 
    
     * If set, PyTerrier uses the `version` init kwarg to determine the .jar file to look for.
     * If the `version` init kwarg is not set, Terrier will query MavenCentral to determine the latest Terrier release.
     * If `version` is set to `"snapshot"`, the latest .jar file build derived from the `Terrier Github repository <https://github.com/terrier-org/terrier-core/>`_ will be downloaded from `Jitpack <https://jitpack.io/>`_.
     * Otherwise the local (`~/.mvn`) and MavenCentral repositories are searched for the jar file at the given version.
    In this way, the default setting is to download the latest release of Terrier from MavenCentral. The user is also able to use a locally installed copy in their private Maven repository, or track the latest build of Terrier from Jitpack.
    
    """
    set_tqdm(tqdm)

    global ApplicationSetup
    global properties
    global firstInit
    global file_path
    global HOME_DIR

    # we keep a local directory
    if home_dir is not None:
        HOME_DIR = home_dir
    elif "PYTERRIER_HOME" in os.environ:
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

    # we only accept Java version 11 and newer; so anything starting 1. or 9. is too old
    java_version = autoclass("java.lang.System").getProperty("java.version")
    if java_version.startswith("1.") or java_version.startswith("9."):
        raise RuntimeError("Pyterrier requires Java 11 or newer, we only found Java version %s;"
            +" install a more recent Java, or change os.environ['JAVA_HOME'] to point to the proper Java installation",
            java_version)
    
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

    
    global anserini
    global cache
    global index
    global io
    global apply
    global model
    global pipelines
    global rewrite
    global transformer
    anserini = importlib.import_module('.anserini', package='pyterrier') 
    cache = importlib.import_module('.cache', package='pyterrier')
    index = importlib.import_module('.index', package='pyterrier') 
    io = importlib.import_module('.io', package='pyterrier')
    apply = importlib.import_module('.apply', package='pyterrier')
    model = importlib.import_module('.model', package='pyterrier')
    pipelines = importlib.import_module('.pipelines', package='pyterrier') 
    rewrite = importlib.import_module('.rewrite', package='pyterrier')
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

def set_tqdm(type):
    """
        Set the tqdm type that Pyterrier will use internally.
    """
    global tqdm
    
    if type is None or type == 'tqdm':
        from tqdm import tqdm as bartype
        tqdm = bartype
    elif type == 'notebook':
        from tqdm.notebook import tqdm as bartype
        tqdm = bartype
    elif type == 'auto':
        from tqdm.auto import tqdm as bartype
        tqdm = bartype
    else:
        raise ValueError("Unknown tqdm type %s" % str(type))
    tqdm.pandas()
    

def started():
    """
        Returns `True` if `init()` has already been called, false otherwise. Typical usage::

            import pyterrier as pt
            if not pt.started():
                pt.init()
    """
    return(firstInit)

def version():
    """
        Returns the version string from the underlying Terrier platform.
    """
    from jnius import autoclass
    return autoclass("org.terrier.Version").VERSION

def check_version(min):
    """
        Returns True iff the underlying Terrier version is no older than the requested version.
    """
    from jnius import autoclass
    from packaging.version import Version
    min = Version(str(min))
    currentVer = Version(version().replace("-SNAPSHOT", ""))
    return currentVer >= min

def redirect_stdouterr():
    """
        Ensure that stdout and stderr have been redirected. Equivalent to setting the redirect_io parameter to init() as `True`.
    """
    from . import bootstrap
    bootstrap.redirect_stdouterr()

def logging(level):
    """
        Set the logging level. Equivalent to setting the logging= parameter to init().
    """
    from . import bootstrap
    bootstrap.logging(level)

def set_property(k, v):
    """
        Allows to set a property in Terrier's global properties configuration. Example::

            pt.set_property("termpipelines", "")
    """
    properties[k] = v
    ApplicationSetup.bootstrapInitialisation(properties)

def set_properties(kwargs):
    """
        Allows to set many properties in Terrier's global properties configuration
    """
    for control, value in kwargs.items():
        properties.put(control, value)
    ApplicationSetup.bootstrapInitialisation(properties)

def run(cmd, args=[]):
    """
        Allows to run a Terrier executable class, i.e. one that can be access from the `bin/terrier` commandline programme.
    """
    from jnius import autoclass
    autoclass("org.terrier.applications.CLITool").main([cmd] + args)

def extend_classpath(mvnpackages):
    """
        Allows to add packages to Terrier's classpath after the JVM has started.
    """
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
