"""DEPRECATED MODULE

Warning: This module is deprecated and will be removed in a future version.

It isn't loaded by __init__ -- it's intended to handle cases where users:
import pyterrier.batchretrieve
or
from pyterrier.batchretrieve import X
"""
from deprecated import deprecated
import pyterrier as pt

BatchRetrieve = pt.terrier.BatchRetrieve # already has @deprecated
FeaturesBatchRetrieve = pt.terrier.FeaturesBatchRetrieve # already has @deprecated
BatchRetrieveBase = pt.terrier.BatchRetrieveBase # already has @deprecated

@deprecated(version='0.11.0', reason="use pt.terrier.TextScorer() instead")
class TextScorer(pt.terrier.TextScorer):
    @staticmethod
    @deprecated(version='0.11.0', reason="use pt.terrier.TextScorer.from_dataset() instead")
    def from_dataset(*args, **kwargs):
        return pt.terrier.TextScorer.from_dataset(*args, **kwargs)


_from_dataset = deprecated(version='0.11.0', reason="use pt.datasets.transformer_from_dataset() instead")(pt.datasets.transformer_from_dataset)
