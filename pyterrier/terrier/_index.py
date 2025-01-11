import os
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
import pyterrier as pt


class TerrierModel(Enum):
    bm25 = 'bm25'
    dph = 'dph'


######################################
# This is a work-in-progress Artifact-compatible wrapper for a Terrier Index. It doesn't support most
# features yet, but does allow for uploading/downloading Terrier indexes from HuggingFace, etc.
######################################
class TerrierIndex(pt.Artifact):
    """A Terrier index."""

    ARTIFACT_TYPE = 'sparse_index'
    ARTIFACT_FORMAT = 'terrier'
    ARTIFACT_PACKAGE_HINT = 'python-terrier'

    def __init__(self, path, *, _index_ref=None, _index_obj=None):
        """Initialises a TerrierIndex for the given path."""
        super().__init__(path)
        if _index_ref is not None:
            assert path is pt.Artifact.NO_PATH and _index_obj is None
        self._index_ref = _index_ref
        if _index_obj is not None:
            assert path is pt.Artifact.NO_PATH and _index_ref is None
        self._index_obj = _index_obj

    def retriever(
        self,
        model: Union[TerrierModel, str],
        model_args: Optional[Dict[str, Any]] = None,
        *,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a retriever transformer for this index.

        Args:
            model: The weighting model to use for scoring.
            model_args: The arguments to pass to the weighting model.
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval

        Returns:
            A retriever transformer for this index.
        """
        if include_fields is None:
            include_fields = ['docno']
        else:
            if 'docno' not in include_fields:
                include_fields = ['docno'] + include_fields
        return pt.terrier.Retriever(
            self.index_obj(),
            controls=_map_controls(model_args),
            properties=_map_properties(model_args),
            metadata=include_fields,
            num_results=num_results,
            wmodel=_map_wmodel(model),
            threads=threads,
            verbose=verbose,
        )

    def bm25(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a BM25 retriever for this index.

        Args:
            k1: BM25's ``k1`` parameter, which controls TF saturation.
            b: BM25's ``b`` parameter, which controls the length penalty.
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval
        """
        return self.retriever(
            TerrierModel.bm25,
            {'bm25.k1': k1, 'bm25.b': b},
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def dph(
        self,
        *,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a DPH retriever for this index.

        Args:
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval
        """
        return self.retriever(
            TerrierModel.dph,
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def text_loader(self,
        index,
        fields: Union[List[str], str, Literal['*']] = '*',
        *,
        verbose=False
    ):
        """Creates a text loader transformer for this index.

        Args:
            fields: The fields to load from the index. If '*', all fields will be loaded.
            verbose: Whether to print progress information.
        """
        return pt.terrier.TerrierTextLoader(self, fields, verbose=verbose)

    def __repr__(self):
        return f'TerrierIndex({str(self.path)!r})'

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

    @classmethod
    @pt.java.required
    def coerce(cls, index_like: Union[str, Path, 'TerrierIndex']) -> 'TerrierIndex':
        """Attempts to build a :class:`TerrierIndex` from the given object.
        
        ``index_like`` can be either: (bulleted list below):
        - ``str`` or ``Path``: loads the index at the provided path
        - ``pyterrier.terrier.J.IndexRef``: TODO: How to handle?
        - ``pyterrier.terrier.J.Index``: TODO: How to handle?
        - ``TerrierIndex``: returns itself
        """
        if isinstance(index_like, TerrierIndex):
            return index_like
        if isinstance(index_like, (str, Path)):
            return TerrierIndex(index_like)
        if isinstance(index_like, pt.terrier.J.IndexRef):
            return TerrierIndex(pt.Artifect.NO_PATH, _index_ref=index_like)
        if isinstance(index_like, pt.terrier.J.Index):
            return TerrierIndex(pt.Artifect.NO_PATH, _index_obj=index_like)
        raise RuntimeError(f'Could not coerce {index_like!r} into a TerrierIndex')


_WMODEL_MAP = {
    TerrierModel.bm25: 'BM25',
    TerrierModel.dph: 'DPH',
}
def _map_wmodel(model):
    return _WMODEL_MAP[model]


_CONTROL_MAP = {
    'bm25.k1': 'bm25.k_1',
    'bm25.b': 'bm25.b',
}
def _map_controls(model_args):
    return {
        _CONTROL_MAP[k]: v
        for k, v in model_args.items()
        if k in _CONTROL_MAP
    }


_PROPERTY_MAP = {
    'bm25.k1': 'bm25.k_1',
    'bm25.b': 'bm25.b',
}
def _map_properties(model_args):
    return {
        _PROPERTY_MAP[k]: v
        for k, v in model_args.items()
        if k in _PROPERTY_MAP
    }


######################################
# Below is a proposal for a different (more declarative and more flexible?) API for retriever.
# But I'm not yet super convinced by it, so leaving it out for now.
######################################
# class TerrierRetriever(pt.Transformer):
#     """A Terrier retriever.

#     This is a simplified (but less powerful) wrapper around ``pt.terrier.Retriever``.
#     """

#     def __init__(self,
#         index: TerrierIndex,
#         model: Union[Model, str],
#         model_args: Optional[Dict[str, Any]] = None,
#         *,
#         num_results: int = 1000,
#         include_fields: Optional[List[str]] = None,
#         threads: int = 1,
#         verbose: bool = False,
#     ):
#         """Initialises a TerrierRetriever."""
#         self._index = index
#         self.model = Model(model)
#         self.model_args = model_args
#         self.num_results = num_results
#         self.include_fields = include_fields
#         self.threads = threads
#         self.verbose = verbose

#     def transform(self, queries):
#         """Retrieves documents for the given queries."""
#         include_fields = self.include_fields
#         if include_fields is None:
#             include_fields = ['docno']
#         else:
#             if 'docno' not in include_fields:
#                 include_fields = ['docno'] + include_fields
#         retr = pt.terrier.Retriever(
#             self._index.index_obj(),
#             controls=_map_controls(self.model_args),
#             properties=_map_properties(self.model_args),
#             metadata=include_fields,
#             num_results=self.num_results,
#             wmodel=_map_wmodel(self.model),
#             threads=self.threads,
#             verbose=self.verbose,
#         )
#         return retr(queries)

#     def __repr__(self):
#         return (f'TerrierRetriever({self.index!r}, {self.model!r}, {self.model_args!r}, '
#                 f'num_results={self.num_results!r}, include_fields={self.include_fields!r}, threads={self.threads!r}, '
#                 f'verbose={self.verbose!r})')

#     def fuse_rank_cutoff(self, k: int) -> Optional[pt.Transformer]:
#         if self.num_results > k:
#             return TerrierRetriever(self.index, self.model, self.model_args, num_results=k, include_fields=self.include_fields, threads=self.threads, verbose=self.verbose)
