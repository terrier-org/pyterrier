# java stuff
from pyterrier.terrier import java
from pyterrier.terrier._index import TerrierIndex, TerrierModel
from pyterrier.terrier._text_loader import TerrierTextLoader, terrier_text_loader
from pyterrier.terrier.java import configure, set_version, set_helper_version, extend_classpath, J, set_property, set_properties, run, version, check_version, check_helper_version
from pyterrier.terrier.retriever import Retriever, FeaturesRetriever, TextScorer
from pyterrier.terrier.index_factory import IndexFactory
from pyterrier.terrier.stemmer import TerrierStemmer
from pyterrier.terrier.tokeniser import TerrierTokeniser
from pyterrier.terrier.stopwords import TerrierStopwords
from pyterrier.terrier import index
from pyterrier.terrier.index import TerrierIndexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, IndexingType, treccollection2textgen
from pyterrier.terrier import rewrite
from deprecated import deprecated
import pyterrier as pt


@deprecated(version='0.11.0', reason="use pt.terrier.Retriever() instead")
class TerrierRetrieve(Retriever):
    
    @staticmethod
    @deprecated(version='0.11.0', reason="use pt.terrier.Retriever.from_dataset() instead")
    def from_dataset(*args, **kwargs):
        return Retriever.from_dataset(*args, **kwargs)


@deprecated(version='0.11.0', reason="use pt.terrier.Retriever() instead")
class BatchRetrieve(Retriever):

    @staticmethod
    @deprecated(version='0.11.0', reason="use pt.terrier.Retriever.from_dataset() instead")
    def from_dataset(*args, **kwargs):
        return Retriever.from_dataset(*args, **kwargs)


@deprecated(version='0.11.0', reason="use pt.terrier.FeaturesRetriever() instead")
class FeaturesBatchRetrieve(FeaturesRetriever):
    
    @staticmethod
    @deprecated(version='0.11.0', reason="use pt.terrier.FeaturesRetriever.from_dataset() instead")
    def from_dataset(*args, **kwargs):
        return FeaturesRetriever.from_dataset(*args, **kwargs)


@deprecated(version='0.12.0', reason="This class provides no functionality; inherit from pt.Transformer and set a verbose flag in your constructor instead")
class RetrieverBase(pt.Transformer):
    def __init__(self, verbose=0, **kwargs):
        super().__init__(kwargs)
        self.verbose = verbose


@deprecated(version='0.12.0', reason="This class provides no functionality; inherit from pt.Transformer and set a verbose flag in your constructor instead")
class BatchRetrieveBase(pt.Transformer):
    def __init__(self, verbose=0, **kwargs):
        super().__init__(kwargs)
        self.verbose = verbose


__all__ = [
    # java stuff
    'java', 'configure', 'set_version', 'set_helper_version', 'extend_classpath', 'J', 'version', 'check_version', 'check_helper_version',

    # retrieval
    'BatchRetrieveBase', 'Retriever', 'RetrieverBase', 'BatchRetrieve', 'TerrierRetrieve', 'FeaturesRetriever', 'FeaturesBatchRetrieve', 'TerrierRetrieve', 'TextScorer',

    # indexing
    'index', 'TerrierIndexer', 'FilesIndexer', 'TRECCollectionIndexer', 'DFIndexer', 'DFIndexUtils', 'IterDictIndexer', 'IndexingType', 'treccollection2textgen',

    # rewriting
    'rewrite',

    # misc
    'TerrierStemmer', 'TerrierStopwords', 'TerrierTokeniser',
    'IndexFactory', 'set_property', 'set_properties', 'run', 
    'TerrierTextLoader', 'terrier_text_loader',

    # Beta High-level API
    'TerrierIndex', 'TerrierModel',
]
