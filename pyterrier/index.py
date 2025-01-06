"""DEPRECATED MODULE

Warning: This module is deprecated and will be removed in a future version.

It isn't loaded by __init__ -- it's intended to handle cases where users:
import pyterrier.index
or
from pyterrier.index import X
"""

from deprecated import deprecated
from enum import Enum
import pyterrier as pt

@deprecated(version='0.11.0', reason="use pt.terrier.IterDictIndexer() instead")
class IterDictIndexer(pt.terrier.IterDictIndexer): # type: ignore[valid-type,misc] # IterDictIndexer can be either fifo or non-fifo version
    pass

@deprecated(version='0.11.0', reason="use pt.terrier.DFIndexer() instead")
class DFIndexer(pt.terrier.DFIndexer):
    pass

@deprecated(version='0.11.0', reason="use pt.terrier.TRECCollectionIndexer() instead")
class TRECCollectionIndexer(pt.terrier.DFIndexer):
    pass

@deprecated(version='0.11.0', reason="use pt.terrier.FilesIndexer() instead")
class FilesIndexer(pt.terrier.DFIndexer):
    pass

# Enumerations are tricky since we can't subclass them :(. Just duplicate while deprecated

@deprecated(version='0.11.0', reason="use pt.terrier.IndexingType() instead")
class IndexingType(Enum):
    CLASSIC = 1
    SINGLEPASS = 2
    MEMORY = 3
