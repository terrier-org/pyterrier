__version__ = "0.10.1"

import os
from deprecated import deprecated

from pyterrier import utils
from pyterrier.utils import Utils

# definitive API used by others, now available before pt.init
from .transformer import Transformer, Estimator, Indexer

from pyterrier import java

from pyterrier import terrier
from pyterrier.terrier import BatchRetrieve, TerrierRetrieve, FeaturesBatchRetrieve, IndexFactory, set_property, set_properties, run

from tqdm.auto import tqdm



from . import anserini
from . import cache
from . import debug
from . import index
from . import io
from . import measures
from . import model
from . import new
from . import ltr
from . import parallel
from . import pipelines
from . import rewrite
from . import text
from . import transformer
from .datasets import get_dataset, find_datasets, list_datasets


from .index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, TerrierStemmer, TerrierStopwords, TerrierTokeniser
from .pipelines import Experiment, GridScan, GridSearch, KFoldGridSearch, Evaluate

import importlib


file_path = os.path.dirname(os.path.abspath(__file__))

# will be set in java._legacy_post_init once java is loaded
ApplicationSetup = None
properties = None

# will be set in terrier._java_post_init once java is loaded
IndexRef = None
HOME_DIR = None

# TODO
_helper_version = None


@java.before_init()
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
    set_tqdm(tqdm)

    # TODO: missing no_download. Is this something like pt.java.set_offline(True)?

    java.init()

    # Import other java packages
    if packages:
        pkgs_string = ",".join(packages)
        set_property("terrier.mvn.coords", pkgs_string)


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
    return java.started()

@java.required()
def version():
    """
        Returns the version string from the underlying Terrier platform.
    """
    return terrier.J.Version.VERSION

def check_version(min, helper=False):
    """
        Returns True iff the underlying Terrier version is no older than the requested version.
    """
    from packaging.version import Version
    from . import terrier
    currentVer = terrier.java._resolved_helper_version if helper else version()
    assert currentVer is not None, "Could not obtain Terrier version (helpher=%s)" % str(helper)
    currentVer = Version(currentVer.replace("-SNAPSHOT", ""))

    min = Version(str(min))
    return currentVer >= min

def redirect_stdouterr():
    """
        Ensure that stdout and stderr have been redirected. Equivalent to setting the redirect_io parameter to init() as `True`.
    """
    java.redirect_stdouterr()


@java.required()
@deprecated(version="0.11", reason="Use pt.java.set_log_level() instead")
def logging(level):
    """
        Set the logging level. Equivalent to setting the logging= parameter to init().
        The following string values are allowed, corresponding to Java logging levels:
        
         - `'ERROR'`: only show error messages
         - `'WARN'`: only show warnings and error messages (default)
         - `'INFO'`: show information, warnings and error messages
         - `'DEBUG'`: show debugging, information, warnings and error messages

    """
    java.set_log_level(level)


@java.required()
def extend_classpath(mvnpackages):
    """
        Allows to add packages to Terrier's classpath after the JVM has started.
    """
    if isinstance(mvnpackages, str):
        mvnpackages = [mvnpackages]
    for package in mvnpackages:
        terrier.extend_package(package)


def _():
    # apply is an object, not a module, as it also has __get_attr__() implemented
    from pyterrier.apply import _apply
    globals()['apply'] = _apply()

    # check python version
    import platform
    from packaging.version import Version
    if Version(platform.python_version()) < Version('3.7.0'):
        raise RuntimeError("From PyTerrier 0.8, Python 3.7 minimum is required, you currently have %s" % platform.python_version())
_()
