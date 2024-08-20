__version__ = "0.10.1"

from deprecated import deprecated

from pyterrier import utils
from pyterrier.transformer import Transformer, Estimator, Indexer

from pyterrier import java

from pyterrier import terrier
from pyterrier.terrier import BatchRetrieve, TerrierRetrieve, FeaturesBatchRetrieve, IndexFactory, run, rewrite, index, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, TerrierStemmer, TerrierStopwords, TerrierTokeniser

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
    'utils', 'Utils', 'Transformer', 'Estimator', 'Indexer',
    'BatchRetrieve', 'TerrierRetrieve', 'FeaturesBatchRetrieve', 'IndexFactory',
    'run', 'rewrite', 'index', 'FilesIndexer', 'TRECCollectionIndexer', 'DFIndexer', 'DFIndexUtils', 'IterDictIndexer',
    'IndexingType', 'TerrierStemmer', 'TerrierStopwords', 'TerrierTokeniser',
    'IndexRef', 'ApplicationSetup', 'properties', 'apply',

    # Deprecated:
    'init', 'started', 'logging', 'version', 'check_version', 'extend_classpath', 'set_tqdm', 'set_property', 'set_properties',
    'redirect_stdouterr', 'autoclass', 'cast',
]
