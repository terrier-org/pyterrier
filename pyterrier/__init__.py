__version__ = "0.10.1"

from deprecated import deprecated

from pyterrier import utils
from pyterrier.transformer import Transformer, Estimator, Indexer

from pyterrier import java
from pyterrier.java import started, redirect_stdouterr # for backward compat, maybe remove/deprecate some day?

from pyterrier import terrier
from pyterrier.terrier import BatchRetrieve, TerrierRetrieve, FeaturesBatchRetrieve, IndexFactory, set_property, set_properties, run, rewrite, index, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, TerrierStemmer, TerrierStopwords, TerrierTokeniser

from pyterrier import anserini
from pyterrier import cache
from pyterrier import debug
from pyterrier import io
from pyterrier import measures
from pyterrier import model
from pyterrier import new
from pyterrier import ltr
from pyterrier import parallel
from pyterrier import pipelines
from pyterrier import text
from pyterrier import transformer
from pyterrier import datasets
from pyterrier.datasets import get_dataset, find_datasets, list_datasets
from pyterrier.pipelines import Experiment, GridScan, GridSearch, KFoldGridSearch, Evaluate

# old name
Utils = utils

# will be set in java._post_init once java is loaded
HOME_DIR = None

# will be set in terrier.java._post_init once java is loaded
IndexRef = None
ApplicationSetup = None
properties = None


@java.before_init
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

    # Set the corresponding options
    java.set_memory_limit(mem)
    java.set_redirect_io(redirect_io)
    java.set_log_level(logging)
    for package in boot_packages:
        java.add_package(*package.split(':')) # format: org:package:version:filetype (where version and filetype are optional)
    for opt in jvm_opts:
        java.add_option(opt)
    terrier.set_terrier_version(version)
    terrier.set_helper_version(helper_version)
    utils.set_tqdm(tqdm)

    # TODO: missing no_download. Is this something like pt.java.set_offline(True)?

    java.init()

    # Import other java packages
    if packages:
        pkgs_string = ",".join(packages)
        set_property("terrier.mvn.coords", pkgs_string)


# deprecated functions explored to the main namespace, which will be removed in a future version
logging = deprecated(version='0.11.0', reason="use pt.java.set_log_level(...) instead")(java.set_log_level)
version = deprecated(version='0.11.0', reason="use pt.terrier.version() instead")(terrier.version)
check_version = deprecated(version='0.11.0', reason="use pt.terrier.check_version(...) instead")(terrier.check_version)
extend_classpath = deprecated(version='0.11.0', reason="use pt.terrier.extend_classpath(...) instead")(terrier.extend_classpath)
set_tqdm = deprecated(version='0.11.0', reason="use pt.utils.set_tqdm(...) instead")(utils.set_tqdm)


# Additional setup performed in a function to avoid polluting the namespace with other imports like platform
def _():
    # check python version
    import platform
    from packaging.version import Version
    if Version(platform.python_version()) < Version('3.7.0'):
        raise RuntimeError("From PyTerrier 0.8, Python 3.7 minimum is required, you currently have %s" % platform.python_version())

    # apply is an object, not a module, as it also has __get_attr__() implemented
    from pyterrier.apply import _apply
    globals()['apply'] = _apply()

    utils.set_tqdm()
_()

__all__ = [
    'java', 'terrier', 'anserini', 'cache', 'debug', 'io', 'measures', 'model', 'new', 'ltr', 'parallel', 'pipelines',
    'text', 'transformer', 'datasets', 'get_dataset', 'find_datasets', 'list_datasets', 'Experiment', 'GridScan',
    'GridSearch', 'KFoldGridSearch', 'Evaluate',
    'utils', 'Utils', 'Transformer', 'Estimator', 'Indexer', 'started', 'redirect_stdouterr',
    'BatchRetrieve', 'TerrierRetrieve', 'FeaturesBatchRetrieve', 'IndexFactory', 'set_property', 'set_properties',
    'run', 'rewrite', 'index', 'FilesIndexer', 'TRECCollectionIndexer', 'DFIndexer', 'DFIndexUtils', 'IterDictIndexer',
    'IndexingType', 'TerrierStemmer', 'TerrierStopwords', 'TerrierTokeniser',
    'HOME_DIR', 'IndexRef', 'ApplicationSetup', 'properties', 'init',

    # Deprecated:
    'logging', 'version', 'check_version', 'extend_classpath', 'set_tqdm',
]
