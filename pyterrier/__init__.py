__version__ = "0.11.0"

from deprecated import deprecated

from pyterrier import model, utils
from pyterrier.transformer import Transformer, Estimator, Indexer
from pyterrier._ops import RankCutoff, Compose

from pyterrier import java

from pyterrier import terrier
from pyterrier.terrier import BatchRetrieve, TerrierRetrieve, FeaturesBatchRetrieve, IndexFactory, run, rewrite, index, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, TerrierStemmer, TerrierStopwords, TerrierTokeniser

from pyterrier import cache
from pyterrier import debug
from pyterrier import io
from pyterrier import measures
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

# will be set in terrier.terrier.java once java is loaded
IndexRef = None


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
    'java', 'terrier', 'cache', 'debug', 'io', 'measures', 'model', 'new', 'ltr', 'parallel', 'pipelines',
    'text', 'transformer', 'datasets', 'get_dataset', 'find_datasets', 'list_datasets', 'Experiment', 'GridScan',
    'GridSearch', 'KFoldGridSearch', 'Evaluate',
    'utils', 'Utils', 'Transformer', 'Estimator', 'Indexer',
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
    if Version(platform.python_version()) < Version('3.7.0'):
        raise RuntimeError("From PyTerrier 0.8, Python 3.7 minimum is required, you currently have %s" % platform.python_version())

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
