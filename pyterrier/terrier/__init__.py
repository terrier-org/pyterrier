# java stuff
from pyterrier.terrier import java
from pyterrier.terrier.java import configure, set_version, set_helper_version, enable_prf, extend_classpath, J, set_property, set_properties, run, version, check_version, check_helper_version
from pyterrier.terrier.retriever import Retriever, FeaturesRetriever, TextScorer
from pyterrier.terrier.index_factory import IndexFactory
from pyterrier.terrier.stemmer import TerrierStemmer
from pyterrier.terrier.tokeniser import TerrierTokeniser
from pyterrier.terrier.stopwords import TerrierStopwords
from pyterrier.terrier import index
from pyterrier.terrier.index import TerrierIndexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, treccollection2textgen
from pyterrier.terrier import rewrite
from deprecated import deprecated


TerrierRetrieve = deprecated(version='0.11.0', reason="use pt.terrier.Retriever() instead")(Retriever)
BatchRetrieve = deprecated(version='0.11.0', reason="use pt.terrier.Retriever() instead")(Retriever)
FeaturesBatchRetrieve = deprecated(version='0.11.0', reason="use pt.terrier.FeaturesRetriever() instead")(FeaturesRetriever)

__all__ = [
    # java stuff
    'java', 'configure', 'set_version', 'set_helper_version', 'set_prf_version', 'extend_classpath', 'J', 'version', 'check_version', 'check_helper_version',

    # retrieval
    'Retriever', 'BatchRetrieve', 'TerrierRetrieve', 'FeaturesRetriever', 'FeaturesBatchRetrieve', 'TerrierRetrieve', 'TextScorer',

    # indexing
    'index', 'TerrierIndexer', 'FilesIndexer', 'TRECCollectionIndexer', 'DFIndexer', 'DFIndexUtils', 'IterDictIndexer', 'IndexingType', 'treccollection2textgen',

    # rewriting
    'rewrite',

    # misc
    'TerrierStemmer', 'TerrierStopwords', 'TerrierTokeniser',
    'IndexFactory', 'set_property', 'set_properties', 'run', 
]
