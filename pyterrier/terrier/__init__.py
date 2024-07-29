# java stuff
from pyterrier.terrier import java
from pyterrier.terrier.java import configure, set_terrier_version, set_helper_version, enable_prf, extend_classpath, J, set_property, set_properties, run, version, check_version, check_helper_version
from pyterrier.terrier.retriever import BatchRetrieve, FeaturesBatchRetrieve
from pyterrier.terrier.index_factory import IndexFactory
from pyterrier.terrier import index
from pyterrier.terrier.index import TerrierIndexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, TerrierStemmer, TerrierStopwords, TerrierTokeniser
from pyterrier.terrier import rewrite

TerrierRetrieve = BatchRetrieve # BatchRetrieve is an alias to TerrierRetrieve


__all__ = [
    # java stuff
    'java', 'configure', 'set_terrier_version', 'set_helper_version', 'enable_prf', 'extend_classpath', 'J', 'version', 'check_version', 'check_helper_version',

    # retrieval
    'BatchRetrieve', 'TerrierRetrieve', 'FeaturesBatchRetrieve',

    # indexing
    'index', 'TerrierIndexer', 'FilesIndexer', 'TRECCollectionIndexer', 'DFIndexer', 'DFIndexUtils', 'IterDictIndexer', 'IndexingType', 'TerrierStemmer', 'TerrierStopwords', 'TerrierTokeniser',

    # rewriting
    'rewrite',

    # misc
    'IndexFactory', 'set_property', 'set_properties', 'run', 
]
