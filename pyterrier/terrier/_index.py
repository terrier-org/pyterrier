import os
from typing import List, Dict
import pyterrier as pt


class TerrierIndex(pt.Artifact):
    """A Terrier index."""

    ARTIFACT_TYPE = 'sparse_index'
    ARTIFACT_FORMAT = 'terrier'

    def __init__(self, path):
        """Initialises a TerrierIndex for the given path."""
        super().__init__(path)
        self._index_ref = None
        self._index_obj = None

    def retriever(
        self,
        *,
        controls: Dict = None,
        properties: Dict = None,
        metadata: List[str] = ["docno"],
        num_results: int = 1000,
        wmodel: str = 'DPH',
        threads: int = 1,
    ) -> pt.Transformer:
        """Creates a ``pt.terrier.Retriever`` object for this index.

        Args:
            controls: The controls to set for this retriever. Controls are specific settings for a given search request.
            properties: The properties to use for this retriever. Properties are settings that apply globally to the index.
            metadata: The metadata fields to return for each search result.
            num_results: The maximum number of results to return per query.
            wmodel: The weighting model to use for scoring.
            threads: The number of threads to use during retrieval.

        Returns:
            A retriever object for this index.
        """
        return pt.terrier.Retriever(self.index_obj(), controls, properties, metadata, num_results, wmodel)

    def bm25(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        num_results: int = 1000,
        threads: int = 1,
    ) -> pt.Transformer:
        """Creates a BM25 retriever for this index.

        Args:
            k1: BM25's ``k1`` parameter, which controls TF saturation.
            b: BM25's ``b`` parameter, which controls the length penalty.
            num_results: The maximum number of results to return per query.
            threads: The number of threads to use during retrieval.
        """
        return self.retriever(
            wmodel='BM25',
            controls={'bm25.k_1': k1, 'bm25.b': b},
            num_results=num_results,
            threads=threads
        )

    def __repr__(self):
        return f'TerrierIndex({self.path!r})'

    @pt.java.required
    def index_ref(self):
        """Returns the internal Java index reference object for this index."""
        if self._index_ref is None:
            self._index_ref = pt.terrier.J.IndexRef.of(os.path.realpath(self.path))
        return self._index_ref

    def index_obj(self):
        """Returns the internal Java index object for this index."""
        if self._index_obj is None:
            self._index_obj = pt.terrier.IndexFactory.of(self.index_ref())
        return self._index_obj
