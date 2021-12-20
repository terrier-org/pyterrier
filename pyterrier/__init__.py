__version__ = "0.7.2"

import os
from .bootstrap import _logging, setup_terrier, setup_jnius, is_windows

import importlib

#sub modules
anserini = None
apply = None
cache = None
debug = None
index = None
io = None
ltr = None
measures = None
model = None
new = None
parallel = None
pipelines = None
rewrite = None
text = None
transformer = None

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
IndexFactory = None
IndexRef = None
properties = None
tqdm = None
HOME_DIR = None
init_args ={}

def init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN', home_dir=None, boot_packages=[], tqdm=None, no_download=False,helper_version = None):
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
        helper_version(str): Which version of the helper.

   
    **Locating the Terrier .jar file:** PyTerrier is not tied to a specific version of Terrier and will automatically locate and download a recent Terrier .jar file. However, inevitably, some functionalities will require more recent Terrier versions. 
    
     * If set, PyTerrier uses the `version` init kwarg to determine the .jar file to look for.
     * If the `version` init kwarg is not set, Terrier will query MavenCentral to determine the latest Terrier release.
     * If `version` is set to `"snapshot"`, the latest .jar file build derived from the `Terrier Github repository <https://github.com/terrier-org/terrier-core/>`_ will be downloaded from `Jitpack <https://jitpack.io/>`_.
     * Otherwise the local (`~/.mvn`) and MavenCentral repositories are searched for the jar file at the given version.
    In this way, the default setting is to download the latest release of Terrier from MavenCentral. The user is also able to use a locally installed copy in their private Maven repository, or track the latest build of Terrier from Jitpack.
    
    If you wish to run PyTerrier in an offline enviroment, you should ensure that the "terrier-assemblies-{your version}-jar-with-dependencies.jar" and "terrier-python-helper-{your helper version}.jar"
    are in the  "~/.pyterrier" (if they are not present, they will be downloaded the first time). Then you should set their versions when calling ``init()`` function. For example:
    ``pt.init(version = 5.5, helper_version = "0.0.6")``.
    """
    global firstInit
    if firstInit:
        raise RuntimeError("pt.init() has already been called. Check pt.started() before calling pt.init()")

    # check python version
    import platform
    from packaging.version import Version
    if Version(platform.python_version()) < Version('3.7.0'):
        from warnings import warn
        warn("From PyTerrier 0.8, Python 3.7 will be required, you currently have %s" % platform.python_version())

    set_tqdm(tqdm)

    global ApplicationSetup
    global properties
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
    classpathTrJars = setup_terrier(HOME_DIR, version, helper_version = helper_version, boot_packages=boot_packages, force_download=not no_download)
    
    if is_windows():
        if "JAVA_HOME" in os.environ:
            java_home =  os.environ["JAVA_HOME"]
            fix = '%s\\jre\\bin\\server\\;%s\\jre\\bin\\client\\;%s\\bin\\server\\' % (java_home, java_home, java_home)
            os.environ["PATH"] = os.environ["PATH"] + ";" + fix

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
    
    tr_version = autoclass('org.terrier.Version')
    version_string = tr_version.VERSION
    if "BUILD_DATE" in dir(tr_version):
        version_string += " (built by %s on %s)" % (tr_version.BUILD_USER, tr_version.BUILD_DATE)
    print("PyTerrier %s has loaded Terrier %s" % (__version__, version_string))
    properties = autoclass('java.util.Properties')()
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')

    from .batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from .utils import Utils
    from .datasets import get_dataset, find_datasets, list_datasets
    from .index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, FlatJSONDocumentIterator, IndexingType
    from .pipelines import Experiment, GridScan, GridSearch, KFoldGridSearch

    # Make imports global
    globals()["autoclass"] = autoclass
    globals()["cast"] = cast
    globals()["ApplicationSetup"] = ApplicationSetup

    # apply is an object, not a module, as it also has __get_attr__() implemented
    from .apply import _apply
    globals()['apply'] = _apply()

    for sub_module_name in ['anserini', 'cache', 'debug', 'index', 'io', 'measures', 'model', 'new', 'ltr', 'parallel', 'pipelines', 'rewrite', 'text', 'transformer']:
        globals()[sub_module_name] = importlib.import_module('.' + sub_module_name, package='pyterrier') 

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
    init_args["logging"] = logging
    _logging(logging)
    setup_jnius()

    globals()["get_dataset"] = get_dataset
    globals()["list_datasets"] = list_datasets
    globals()["find_datasets"] = find_datasets
    globals()["Experiment"] = Experiment
    globals()["BatchRetrieve"] = BatchRetrieve
    globals()["TerrierRetrieve"] = BatchRetrieve # TerrierRetrieve is an alias to BatchRetrieve
    globals()["Indexer"] = Indexer
    globals()["FeaturesBatchRetrieve"] = FeaturesBatchRetrieve
    globals()["TRECCollectionIndexer"] = TRECCollectionIndexer
    globals()["FilesIndexer"] = FilesIndexer
    globals()["DFIndexer"] = DFIndexer
    globals()["DFIndexUtils"] = DFIndexUtils
    globals()["IterDictIndexer"] = IterDictIndexer
    globals()["FlatJSONDocumentIterator"] = FlatJSONDocumentIterator
    globals()["Utils"] = Utils
    globals()["IndexFactory"] = autoclass("org.terrier.structures.IndexFactory")
    globals()["IndexRef"] = autoclass("org.terrier.querying.IndexRef")
    globals()["IndexingType"] = IndexingType
    globals()["GridScan"] = GridScan
    globals()["GridSearch"] = GridSearch
    globals()["KFoldGridSearch"] = KFoldGridSearch
    
    
    # we save the pt.init() arguments so that other processes,
    # started by joblib or ray can booted with same options
    init_args["version"] = version
    init_args["mem"] = mem
    init_args["packages"] = packages
    init_args["jvm_opts"] = jvm_opts
    init_args["redirect_io"] = redirect_io
    init_args["home_dir"] = home_dir
    init_args["boot_packages"] = boot_packages
    init_args["tqdm"] = tqdm
    firstInit = True

def set_tqdm(type):
    """
        Set the tqdm progress bar type that Pyterrier will use internally.
        Many PyTerrier transformations can be expensive to apply in some settings - users can
        view progress by using the verbose=True kwarg to many classes, such as BatchRetrieve.

        The `tqdm <https://tqdm.github.io/>`_ progress bar can be made prettier when using appropriately configured Jupyter notebook setups.
        We use this automatically when Google Colab is detected.

        Allowable options for type are:

         - `'tqdm'`: corresponds to the standard text progresss bar, ala `from tqdm import tqdm`.
         - `'notebook'`: corresponds to a notebook progress bar, ala `from tqdm.notebook import tqdm`
         - `'auto'`: allows tqdm to decide on the progress bar type, ala `from tqdm.auto import tqdm`. Note that this works fine on Google Colab, but not on Jupyter unless the `ipywidgets have been installed <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`_.
    """
    global tqdm

    import sys
    if type is None:
        if 'google.colab' in sys.modules:
            type = 'notebook'
        else:
            type = 'tqdm'
    
    if type == 'tqdm':
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
        The following string values are allowed, corresponding to Java logging levels:
        
         - `'ERROR'`: only show error messages
         - `'WARN'`: only show warnings and error messages (default)
         - `'INFO'`: show information, warnings and error messages
         - `'DEBUG'`: show debugging, information, warnings and error messages

    """
    from . import bootstrap
    bootstrap.logging(level)

def set_property(k, v):
    """
        Allows to set a property in Terrier's global properties configuration. Example::

            pt.set_property("termpipelines", "")

        While Terrier has a variety of properties -- as discussed in its 
        `indexing <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_indexing.md>`_ 
        and `retrieval <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_retrieval.md>`_ 
        configuration guides -- in PyTerrier, we aim to expose Terrier configuration through appropriate 
        methods or arguments. So this method should be seen as a safety-valve - a way to override the 
        Terrier configuration not explicitly supported by PyTerrier.
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
