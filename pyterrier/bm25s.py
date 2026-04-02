"""PyTerrier integration for the bm25s library.

This module provides a :class:`BM25SIndex` artifact that wraps the ``bm25s`` library,
allowing fast pure-Python BM25 retrieval within PyTerrier pipelines.

Example usage::

    import pyterrier as pt

    # Build an index
    index = pt.bm25s.BM25SIndex('my_bm25s_index')
    index.index([
        {'docno': 'd1', 'text': 'the cat sat on the mat'},
        {'docno': 'd2', 'text': 'the dog barked at the cat'},
    ])

    # Retrieve
    pipeline = index.bm25()
    results = pipeline.search('cat')

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

import pyterrier as pt
from pyterrier.model import add_ranks

__all__ = ['BM25SIndex', 'BM25SRetriever', 'BM25SIndexer']


class BM25SRetriever(pt.Transformer):
    """A PyTerrier transformer that retrieves documents using a :class:`BM25SIndex`.

    Accepts a query frame (with ``qid`` and ``query`` columns) and returns a results
    frame with columns ``qid``, ``docno``, ``score``, and ``rank``.
    """

    def __init__(
        self,
        index: 'BM25SIndex',
        *,
        num_results: int = 1000,
        stopwords: Union[str, List[str]] = 'english',
        stemmer: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """
        Args:
            index: The :class:`BM25SIndex` to retrieve from.
            num_results: The maximum number of results to return per query.
            stopwords: Stopwords to remove from queries. Either ``'english'`` (default),
                ``None`` for no stopword removal, or a custom list of words.
            stemmer: An optional stemmer callable applied to each query token (must accept
                a list of strings and return a list of strings, e.g. ``PyStemmer``).
            verbose: Whether to display a progress bar during retrieval.
        """
        self.index = index
        self.num_results = num_results
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.verbose = verbose

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        """Retrieve documents for each query.

        Args:
            queries: A dataframe with columns ``qid`` and ``query``.

        Returns:
            A dataframe with columns ``qid``, ``docno``, ``score``, ``rank``, and any
            additional columns from the input.
        """
        import bm25s

        pt.validate.query_frame(queries)

        bm25_index, docnos = self.index._load_index()
        n_docs = len(docnos)
        k = min(self.num_results, n_docs)

        all_rows: List[Dict[str, Any]] = []

        iter_queries = queries.itertuples()
        if self.verbose:
            iter_queries = pt.tqdm(iter_queries, desc=str(self), total=len(queries), unit='q')

        for row in iter_queries:
            qid = str(row.qid)
            query_text = str(row.query)
            if not query_text.strip():
                continue

            query_tokens = bm25s.tokenize(
                query_text,
                stopwords=self.stopwords,
                stemmer=self.stemmer,
                show_progress=False,
                return_ids=True,
            )

            results, scores = bm25_index.retrieve(
                query_tokens,
                k=k,
                show_progress=False,
            )

            doc_ids = results[0]
            doc_scores = scores[0]

            for doc_id, score in zip(doc_ids, doc_scores):
                all_rows.append({
                    'qid': qid,
                    'docno': docnos[int(doc_id)],
                    'score': float(score),
                })

        if not all_rows:
            result_df = pd.DataFrame(columns=['qid', 'docno', 'score', 'rank'])
        else:
            result_df = pd.DataFrame(all_rows)
            result_df = add_ranks(result_df)

        # merge back any extra input columns (e.g. query text)
        input_cols = queries.columns[(queries.columns == 'qid') | (~queries.columns.isin(result_df.columns))]
        result_df = result_df.merge(queries[input_cols], on='qid')

        return result_df

    def __repr__(self) -> str:
        return f'BM25SRetriever({self.index!r}, num_results={self.num_results})'

    def __str__(self) -> str:
        return f'BM25SRetr(k={self.num_results})'


class BM25SIndexer(pt.Indexer):
    """Builds a :class:`BM25SIndex` from an iterable of documents.

    Documents should be dictionaries containing at minimum a ``docno`` field
    and one or more text fields (by default ``text``).
    """

    def __init__(
        self,
        index: 'BM25SIndex',
        *,
        text_attrs: List[str] = None,
        stopwords: Union[str, List[str], None] = 'english',
        stemmer: Optional[Callable] = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        method: str = 'lucene',
        verbose: bool = False,
    ):
        """
        Args:
            index: The :class:`BM25SIndex` that will store the built index.
            text_attrs: Document field names containing text to index.
            stopwords: Stopwords to remove during indexing. Either ``'english'``
                (default), ``None`` for no stopword removal, or a custom list.
            stemmer: An optional stemmer callable applied to each token during
                indexing (must accept a list of strings and return a list of
                strings, e.g. ``PyStemmer``).
            k1: BM25 ``k1`` parameter.
            b: BM25 ``b`` parameter.
            delta: BM25 ``delta`` parameter (used for ``bm25+`` / ``bm25l`` variants).
            method: BM25 variant to use. One of ``'robertson'``, ``'lucene'``
                (default), or ``'atire'``.
            verbose: Whether to display progress during indexing.
        """
        self._index = index
        self.text_attrs = list(text_attrs) if text_attrs is not None else ['text']
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.method = method
        self.verbose = verbose

    def index(self, iter_dict: Iterable[Dict[str, Any]]) -> 'BM25SIndex':
        """Build the index from an iterable of document dicts.

        Args:
            iter_dict: Iterable of dicts, each containing at least ``docno``
                and the fields listed in ``text_attrs``.

        Returns:
            The :class:`BM25SIndex` that was built.
        """
        import bm25s

        docnos: List[str] = []
        texts: List[str] = []

        iter_docs = iter_dict
        if self.verbose:
            iter_docs = pt.tqdm(iter_docs, desc='BM25SIndexer', unit='docs')

        for doc in iter_docs:
            docnos.append(str(doc['docno']))
            text_parts = [str(doc.get(attr, '')) for attr in self.text_attrs]
            texts.append(' '.join(text_parts))

        if not texts:
            raise ValueError('No documents were provided to the indexer')

        corpus_tokens = bm25s.tokenize(
            texts,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            show_progress=self.verbose,
        )

        retriever = bm25s.BM25(k1=self.k1, b=self.b, delta=self.delta, method=self.method)
        retriever.index(corpus_tokens, show_progress=self.verbose)

        # Persist the index
        path = str(self._index.path)
        os.makedirs(path, exist_ok=True)
        retriever.save(path)

        # Save the docno mapping and indexer parameters
        meta = {
            'docnos': docnos,
            'text_attrs': self.text_attrs,
            'stopwords': self.stopwords if self.stopwords is None or isinstance(self.stopwords, str) else list(self.stopwords),
            'k1': self.k1,
            'b': self.b,
            'delta': self.delta,
            'method': self.method,
        }
        with open(os.path.join(path, 'pt_bm25s_meta.json'), 'wt') as fout:
            json.dump(meta, fout)

        # Write pt_meta.json so the artifact discovery mechanism can find this type
        pt_meta = {
            'type': BM25SIndex.ARTIFACT_TYPE,
            'format': BM25SIndex.ARTIFACT_FORMAT,
            'package_hint': BM25SIndex.ARTIFACT_PACKAGE_HINT,
        }
        with open(os.path.join(path, 'pt_meta.json'), 'wt') as fout:
            json.dump(pt_meta, fout)

        # Invalidate cached index
        self._index._bm25_index = None
        self._index._docnos = None

        return self._index


class BM25SIndex(pt.Artifact, pt.Indexer):
    """A BM25S index artifact.

    A BM25S index is a pure-Python sparse inverted index that uses the ``bm25s``
    library for retrieval. It can be built from an iterable of document dicts and
    used to create retriever transformers for PyTerrier pipelines.

    Example::

        import pyterrier as pt

        # Build
        index = pt.bm25s.BM25SIndex('my_index')
        index.index([
            {'docno': 'd1', 'text': 'hello world'},
            {'docno': 'd2', 'text': 'goodbye cruel world'},
        ])

        # Retrieve
        results = index.bm25().search('hello world')
    """

    ARTIFACT_TYPE = 'sparse_index'
    ARTIFACT_FORMAT = 'bm25s'
    ARTIFACT_PACKAGE_HINT = 'pyterrier'

    def __init__(self, path: Union[str, Path]):
        """
        Args:
            path: The directory path where the index is (or will be) stored.
        """
        super().__init__(path)
        self._bm25_index = None
        self._docnos = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def indexer(
        self,
        *,
        text_attrs: List[str] = None,
        stopwords: Union[str, List[str], None] = 'english',
        stemmer: Optional[Callable] = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        method: str = 'lucene',
        verbose: bool = False,
    ) -> BM25SIndexer:
        """Returns a :class:`BM25SIndexer` that builds this index.

        Args:
            text_attrs: Document field names containing text to index.
            stopwords: Stopwords to remove. Either ``'english'`` (default),
                ``None``, or a custom list.
            stemmer: An optional stemmer callable.
            k1: BM25 ``k1`` parameter.
            b: BM25 ``b`` parameter.
            delta: BM25 ``delta`` parameter.
            method: BM25 variant (``'robertson'``, ``'lucene'``, ``'atire'``).
            verbose: Show progress bars.
        """
        return BM25SIndexer(
            self,
            text_attrs=text_attrs,
            stopwords=stopwords,
            stemmer=stemmer,
            k1=k1,
            b=b,
            delta=delta,
            method=method,
            verbose=verbose,
        )

    def index(self, iter_dict: Iterable[Dict[str, Any]], **kwargs: Any) -> 'BM25SIndex':
        """Build the index from an iterable of document dicts.

        The first document is inspected to infer text fields (all ``str``
        columns except ``docno``).

        Args:
            iter_dict: Iterable of dicts with at least ``docno`` and text fields.
            **kwargs: Forwarded to :meth:`indexer`.

        Returns:
            ``self`` after the index has been built.
        """
        assert not self.built(), 'an index is already built at this path'

        iter_docs = pt.utils.peekable(iter_dict)
        first_doc = iter_docs.peek()
        assert 'docno' in first_doc, "input documents must contain a 'docno' field"

        text_fields = kwargs.pop('text_attrs', None)
        if text_fields is None:
            text_fields = [k for k, v in sorted(first_doc.items()) if isinstance(v, str) and k != 'docno']
            assert text_fields, f"no str fields (besides 'docno') found in document: {list(first_doc.keys())}"

        indexer = self.indexer(text_attrs=text_fields, **kwargs)
        indexer.index(iter_docs)
        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retriever(
        self,
        *,
        num_results: int = 1000,
        stopwords: Union[str, List[str], None] = None,
        stemmer: Optional[Callable] = None,
        verbose: bool = False,
    ) -> BM25SRetriever:
        """Creates a :class:`BM25SRetriever` transformer for this index.

        When ``stopwords`` or ``stemmer`` are ``None`` (the default), the values
        recorded at index time are used automatically, ensuring that query and
        document tokenization are consistent.

        Args:
            num_results: The maximum number of results to return per query.
            stopwords: Override the stopwords used at query time.
            stemmer: Override the stemmer used at query time.
            verbose: Show a progress bar during retrieval.
        """
        # Fall back to the tokenizer settings recorded at index time
        if stopwords is None and stemmer is None:
            meta = self._load_meta()
            stopwords = meta.get('stopwords', 'english')

        return BM25SRetriever(
            self,
            num_results=num_results,
            stopwords=stopwords if stopwords is not None else 'english',
            stemmer=stemmer,
            verbose=verbose,
        )

    def bm25(
        self,
        *,
        num_results: int = 1000,
        verbose: bool = False,
    ) -> BM25SRetriever:
        """Creates a BM25 retriever for this index using the parameters set at index time.

        Args:
            num_results: The maximum number of results to return per query.
            verbose: Show a progress bar during retrieval.
        """
        return self.retriever(num_results=num_results, verbose=verbose)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> Dict[str, Any]:
        meta_path = os.path.join(str(self.path), 'pt_bm25s_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'rt') as fin:
                return json.load(fin)
        return {}

    def _load_index(self):
        """Load and cache the bm25s BM25 object and docno list."""
        import bm25s

        if self._bm25_index is None or self._docnos is None:
            assert self.built(), f'{self!r} has not been built yet'
            path = str(self.path)
            self._bm25_index = bm25s.BM25.load(path)
            meta = self._load_meta()
            self._docnos = meta.get('docnos', [])
        return self._bm25_index, self._docnos

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def built(self) -> bool:
        """Returns whether the index has been built."""
        return (
            isinstance(self.path, Path)
            and os.path.exists(os.path.join(str(self.path), 'pt_bm25s_meta.json'))
        )

    def __repr__(self) -> str:
        return f'BM25SIndex({str(self.path)!r})'
