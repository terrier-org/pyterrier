# java stuff
from pyterrier.terrier import java
from pyterrier.terrier.java import configure, set_terrier_version, set_helper_version, enable_prf, extend_package, J, set_property, set_properties, run
from pyterrier.terrier.retriever import BatchRetrieve, FeaturesBatchRetrieve
from pyterrier.terrier.index_factory import IndexFactory

TerrierRetrieve = BatchRetrieve # BatchRetrieve is an alias to TerrierRetrieve


__all__ = [
    # java stuff
    'java', 'configure', 'set_terrier_version', 'set_helper_version', 'enable_prf', 'extend_package', 'J',

    # retrieval
    'BatchRetrieve', 'TerrierRetrieve', 'FeaturesBatchRetrieve',

    # misc
    'IndexFactory', 'set_property', 'set_properties', 'run', 
]
