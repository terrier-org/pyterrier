__version__ = '1.0.3'
# NB: version number must be the first line and must use single quotes for the sed expression in .github/workflows/publish-to-pypi.yml

from typing import Any
from deprecated import deprecated

from pyterrier import model, utils, validate, testing
from pyterrier.transformer import Transformer, Estimator, Indexer
from pyterrier._ops import RankCutoff, Compose
from pyterrier._artifact import Artifact

from pyterrier import java

from pyterrier import terrier
from pyterrier.terrier import BatchRetrieve, TerrierRetrieve, FeaturesBatchRetrieve, IndexFactory, run, rewrite, index, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, TerrierStemmer, TerrierStopwords, TerrierTokeniser
from pyterrier._evaluation import Experiment, GridScan, GridSearch, Evaluate, KFoldGridSearch

from pyterrier import debug
from pyterrier import documentation
from pyterrier import io
from pyterrier import inspect
from pyterrier import measures
from pyterrier import new
from pyterrier import ltr
from pyterrier import pipelines
from pyterrier import text
from pyterrier import transformer
from pyterrier import schematic
from pyterrier import datasets
from pyterrier.datasets import get_dataset, find_datasets, list_datasets
from pyterrier import apply as _apply_base

# old name
Utils = utils

# will be set in terrier.terrier.java once java is loaded
IndexRef = None

# these will be set once _() runs, but we need to define them here to get type checking to work properly
tqdm: Any
apply: _apply_base._apply


# deprecated functions explored to the main namespace, which will be removed in a future version
init = java.legacy_init # java.legacy_init raises a deprecated warning internally
started = deprecated(version='0.11.0', reason="use pt.java.started() instead")(java.started)
logging = deprecated(version='0.11.0', reason="use pt.java.set_log_level(...) instead")(java.set_log_level)
version = deprecated(version='0.11.0', reason="use pt.terrier.version() instead")(terrier.version)
check_version = deprecated(version='0.11.0', reason="use pt.terrier.check_version(...) instead")(terrier.check_version)
extend_classpath = deprecated(version='0.11.0', reason="use pt.terrier.extend_classpath(...) instead")(terrier.extend_classpath)
set_tqdm = deprecated(version='0.11.0', reason="use pt.utils.set_tqdm(...) instead")(utils.set_tqdm)
set_property = deprecated(version='0.11.0', reason="use pt.terrier.set_property(...) instead")(terrier.set_property)
set_properties = deprecated(version='0.11.0', reason="use pt.terrier.set_properties(...) instead")(terrier.set_properties)
redirect_stdouterr = deprecated(version='0.11.0', reason="use pt.java.redirect_stdouterr(...) instead")(java.redirect_stdouterr)
autoclass = deprecated(version='0.11.0', reason="use pt.java.autoclass(...) instead")(java.autoclass)
cast = deprecated(version='0.11.0', reason="use pt.java.cast(...) instead")(java.cast)


__all__ = [
    'java', 'terrier', 'debug', 'documentation', 'io', 'inspect', 'measures', 'model', 'new', 'ltr', 'pipelines', 'schematic',
    'text', 'transformer', 'datasets', 'validate', 'testing', 'get_dataset', 'find_datasets', 'list_datasets', 'Experiment', 'GridScan',
    'GridSearch', 'KFoldGridSearch', 'Evaluate',
    'utils', 'Utils', 'Transformer', 'Estimator', 'Indexer', 'Artifact',
    'RankCutoff', 'Compose',
    'BatchRetrieve', 'TerrierRetrieve', 'FeaturesBatchRetrieve', 'IndexFactory',
    'run', 'rewrite', 'index', 'FilesIndexer', 'TRECCollectionIndexer', 'DFIndexer', 'DFIndexUtils', 'IterDictIndexer',
    'IndexingType', 'TerrierStemmer', 'TerrierStopwords', 'TerrierTokeniser',
    'IndexRef', 'ApplicationSetup', 'properties',

    # Deprecated:
    'init', 'started', 'logging', 'version', 'check_version', 'extend_classpath', 'set_tqdm', 'set_property', 'set_properties',
    'redirect_stdouterr', 'autoclass', 'cast',

    # Entry point modules (appended loaded below):
]


# Additional setup performed in a function to avoid polluting the namespace with other imports like platform
def _():
    from warnings import warn
    import platform
    from packaging.version import Version

    # check python version
    if Version(platform.python_version()) < Version('3.9.0'):
        raise RuntimeError("From PyTerrier 1.0, Python 3.9 minimum is required, you currently have %s" % platform.python_version())
    
    # check for both pyterrier and python-terrier installed
    import importlib.metadata
    try:
        old_pkg_version = importlib.metadata.distribution('python-terrier').version
        if Version(old_pkg_version) < Version('1.0'):
            raise RuntimeError(f"Both 'pyterrier' and 'python-terrier' packages are installed with mismatched versions ({__version__} and {old_pkg_version}). "
                                "This may lead to unexpected behaviour. Remove python-terrier, or upgrade to python-terrier>=1.0'")
    except importlib.metadata.PackageNotFoundError:
        pass
    
    globs = globals()

    # Load the _apply object as pt.apply so that the dynamic __getattr__ methods work
    from pyterrier.apply import _apply
    globs['apply'] = _apply()
    __all__.append('apply')

    # load modules defined as package entry points into the global pyterrier namespace
    for entry_point in utils.entry_points('pyterrier.modules'):
        if entry_point.name in globs:
            warn(f'skipping loading {entry_point} because a module with this name is already loaded.')
            continue
        module = entry_point.load()
        if callable(module): # if the entry point refers to an function/class, call it to get the module
            module = module()
        globs[entry_point.name] = module
        __all__.append(entry_point.name)

    # guess the environment and set an appropriate tqdm as pt.tqdm
    utils.set_tqdm()
_()
