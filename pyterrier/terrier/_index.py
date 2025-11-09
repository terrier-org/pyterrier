import os
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
import pyterrier as pt


class TerrierModel(Enum):
    """A built-in Terrier weighting (scoring) model.

    This enum is primarily used with :meth:`TerrierIndex.retriever` to specify the weighting model to use.
    """
    bm25 = 'bm25'
    dph = 'dph'
    pl2 = 'pl2'
    dirichlet_lm = 'dirichlet_lm' # dirichletlm.mu=2500
    hiemstra_lm = 'hiemstra_lm' # hiemstra_lm.lambda=0.15
    tf = 'tf'
    tf_idf = 'tf_idf'


class TerrierIndex(pt.Artifact, pt.Indexer):
    """Represents a Terrier index.

    A Terrier index is a sparse inverted index structure that supports a variety of operations. It can be used to
    create transformers that perform retrieval, re-ranking, pseudo-relevance feedback, and other operations.
    """

    ARTIFACT_TYPE = 'sparse_index'
    ARTIFACT_FORMAT = 'terrier'
    ARTIFACT_PACKAGE_HINT = 'python-terrier'

    def __init__(
        self,
        path: Union[str, Path],
        *,
        memory: bool = False,
        _index_ref: object = None, 
        _index_obj: object = None
    ):
        """
        Args:
            path: The path to the index on disk.
            memory: Whether to load the index fully into memory.
            _index_ref: **For internal use only.** The Java IndexRef object for this index.
            _index_obj: **For internal use only.** The Java Index object for this index.
        """
        super().__init__(path)
        if _index_ref is not None:
            assert path is pt.Artifact.NO_PATH and _index_obj is None
        self._index_ref = _index_ref
        if _index_obj is not None:
            assert path is pt.Artifact.NO_PATH and _index_ref is None
        self._index_obj = _index_obj
        self.memory = memory


    # ----------------------------------------------------
    # Retrieval
    # ----------------------------------------------------

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

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.retriever('BM25', num_results=10)

        Terrier retrievers can also perform re-ranking when they receive a result frame as input:

        .. schematic::
            :show_code:
            :input_columns: qid,query,docno

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            # As a re-ranker
            index.retriever('BM25', num_results=10)

        .. seealso::
            There are shorthand methods for creating common retrievers: :meth:`bm25`, :meth:`dph`.
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

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.bm25()

        .. cite.dblp:: conf/trec/RobertsonWB98
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

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.dph()

        .. cite.dblp:: conf/ecir/Amati06
        """
        return self.retriever(
            TerrierModel.dph,
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def pl2(
        self,
        *,
        c: float = 1.0,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a PL2 retriever for this index.

        Args:
            c: PL2's ``c`` parameter, which controls the length normalization.
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.pl2()

        .. cite.dblp:: journals/tois/AmatiR02
        """
        return self.retriever(
            TerrierModel.pl2,
            {'dfr.c': c},
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def dirichlet_lm(
        self,
        *,
        mu: float = 2500.0,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a Dirichlet Language Model retriever for this index.

        Args:
            mu: Dirichlet LM's ``mu`` parameter, which controls the strength of the prior.
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.dirichlet_lm()

        .. cite.dblp:: journals/tois/ZhaiL04
        """
        return self.retriever(
            TerrierModel.dirichlet_lm,
            {'dirichletlm.mu': mu},
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def hiemstra_lm(
        self,
        *,
        Lambda: float = 0.15,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a Hiemstra Language Model retriever for this index.

        Args:
            Lambda: Hiemstra LM's ``lambda`` parameter, which controls the interpolation weight.
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.hiemstra_lm()

        .. cite.dblp:: phd/basesearch/Hiemstra01
        """
        return self.retriever(
            TerrierModel.hiemstra_lm,
            {'hiemstra_lm.lambda': Lambda},
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def tf(
        self,
        *,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a raw Term Frequency (TF) retriever for this index.

        This is typically useful for retrieving from learned sparse models.

        Args:
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.tf()
        """
        return self.retriever(
            TerrierModel.tf,
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )

    def tf_idf(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        num_results: int = 1000,
        include_fields: Optional[List[str]] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Creates a TF-IDF retriever for this index.

        This retriever uses the Robertson formulation of TF and the Sparck Jones formulation of IDF.

        Args:
            k1: TF-IDF's ``k1`` parameter, which controls TF saturation.
            b: TF-IDF's ``b`` parameter, which controls the length penalty.
            num_results: The maximum number of results to return per query.
            include_fields: The metadata fields to return for each search result.
            threads: The number of threads to use during retrieval.
            verbose: Whether to progress information during retrieval

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.tf_idf()

        .. cite.dblp:: conf/trec/RobertsonWB98
        .. cite.dblp:: journals/jd/Jones04
        """
        return self.retriever(
            TerrierModel.tf_idf,
            {'tf_idf.k_1': k1, 'tf_idf.b': b},
            num_results=num_results,
            include_fields=include_fields,
            threads=threads,
            verbose=verbose,
        )


    # ----------------------------------------------------
    # Query Expansion
    # ----------------------------------------------------

    def rm3(
        self,
        *,
        fb_terms: int = 10,
        fb_docs: int = 3,
        fb_lambda: float = 0.6,
    ) -> pt.Transformer:
        """Creates an RM3 pseudo-relevance feedback transformer for this index.

        Args:
            fb_terms: The number of feedback terms to use.
            fb_docs: The number of feedback documents to use.
            fb_lambda: The interpolation weight between the original query and the feedback model.

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.bm25() >> index.rm3() >> index.bm25() >> pt.rewrite.reset()

        .. note::
            ``pt.rewrite.reset()`` is needed after the feedback step to reset the query to its original form.

        .. cite.dblp:: conf/trec/JaleelACDLLSW04
        """
        return pt.terrier.rewrite.RM3(
            self.index_obj(),
            fb_terms=fb_terms,
            fb_docs=fb_docs,
            fb_lambda=fb_lambda,
        )

    def bo1(
        self,
        *,
        fb_terms: int = 10,
        fb_docs: int = 3,
    ) -> pt.Transformer:
        """Creates a Bo1 pseudo-relevance feedback transformer for this index.

        Args:
            fb_terms: The number of feedback terms to use.
            fb_docs: The number of feedback documents to use.

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.bm25() >> index.bo1() >> index.bm25() >> pt.rewrite.reset()

        .. note::
            ``pt.rewrite.reset()`` is needed after the feedback step to reset the query to its original form.

        .. cite.dblp:: phd/ethos/Amati03
        """
        return pt.terrier.rewrite.Bo1QueryExpansion(
            self.index_obj(),
            fb_terms=fb_terms,
            fb_docs=fb_docs,
        )

    def kl(
        self,
        *,
        fb_terms: int = 10,
        fb_docs: int = 3,
    ) -> pt.Transformer:
        """Creates a KL-Divergence pseudo-relevance feedback transformer for this index.

        Args:
            fb_terms: The number of feedback terms to use.
            fb_docs: The number of feedback documents to use.

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.bm25() >> index.kl() >> index.bm25() >> pt.rewrite.reset()

        .. note::
            ``pt.rewrite.reset()`` is needed after the feedback step to reset the query to its original form.

        .. cite.dblp:: phd/ethos/Amati03
        """
        return pt.terrier.rewrite.KLQueryExpansion(
            self.index_obj(),
            fb_terms=fb_terms,
            fb_docs=fb_docs,
        )


    # ----------------------------------------------------
    # Loading
    # ----------------------------------------------------

    def text_loader(
        self,
        fields: Union[List[str], str, Literal['*']] = '*',
        *,
        verbose=False,
    ) -> pt.Transformer:
        """Creates a transformer that loads stored text content from this index.

        Args:
            fields: The metadata fields to load for each document. If ``"*"``, loads all available fields.

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex.example()
            # FOLD
            index.text_loader()

        """
        return pt.terrier.TerrierTextLoader(self, fields, verbose=verbose)


    # ----------------------------------------------------
    # Indexing
    # ----------------------------------------------------

    def indexer(
        self,
        *,
        meta: Dict = {'docno': 20},
        text_attrs: List[str] = ['text'],
        tokeniser: Union[str, 'pt.terrier.TerrierTokeniser'] = 'english',
        stemmer: Union[None, str, 'pt.terrier.TerrierStemmer'] = 'porter',
        stopwords: Union[None, 'pt.terrier.TerrierStopwords', str, List[str]] = 'terrier',
        store_separate_fields: bool = False,
        store_blocks: bool = False,
        threads: int = 1,
    ) -> pt.Indexer:
        """Returns an indexer that is used to build this index.

        Args:
            meta: The fields to store as metadata for each document. The keys are the metadata field names, and the values are the maximum lengths for each field.
            text_attrs: The text fields to index as text for each document.
            tokeniser: The tokeniser to use.
            stemmer: The stemmer to apply to each token.
            stopwords: The set of words to remove as stopwords.
            store_separate_fields: Whether to store each text attribute as a separate field in the index. This allows for fielded retrieval, but increases index size.
            store_blocks: Whether to store block information (i.e., positions) in the index. This allows for positional queries, but increases index size and retrieval time.
            threads: The number of threads to use during indexing.

        Example Pipeline:

        .. schematic::
            :show_code:

            index = pt.terrier.TerrierIndex('my_index.terrier')
            index.indexer()
        """
        return pt.terrier.IterDictIndexer(
            os.path.realpath(str(self.path)),
            meta=meta,
            text_attrs=text_attrs,
            tokeniser=tokeniser,
            stemmer=stemmer,
            stopwords=stopwords,
            fields=store_separate_fields,
            blocks=store_blocks,
            threads=threads,
        )

    def index(self, it: pt.model.IterDict, **kwargs: Any):
        """Indexes the given input data, creating the index if it does not yet exist, or raising an error if it does.

        This method is shorthand for ``self.indexer().index(iter)``.

        Args:
            it: The documents to index as an iterable of dicts.
        """
        assert len(kwargs) == 0, f"unknown keyword argument(s) given: {kwargs}"
        self.indexer().index(it)
        return self


    # ----------------------------------------------------
    # Index Data
    # ----------------------------------------------------


    @pt.java.required
    def index_ref(self):
        """The internal Java index reference object for this index.

        Returns:
            A Java `IndexRef <http://terrier.org/docs/current/javadoc/org/terrier/structures/IndexRef.html>`_ object for this index.
        """
        if self._index_ref is None:
            self._index_ref = pt.terrier.J.IndexRef.of(os.path.realpath(str(self.path)))
        return self._index_ref

    def index_obj(self):
        """The internal Java index object for this index.

        Returns:
            A Java `Index <http://terrier.org/docs/current/javadoc/org/terrier/structures/Index.html>`_ object for this index.
        """
        if self._index_obj is None:
            self._index_obj = pt.terrier.IndexFactory.of(self.index_ref(), memory=self.memory)
        return self._index_obj

    def collection_statistics(self):
        """Returns the collection statistics for this index.

        Example:

        .. code-block:: python
            :caption: Show collection statistics for a Terrier index.

            >>> stats = index.collection_statistics()
            >>> print(stats)
            Number of documents: 11429
            Number of terms: 7756
            Number of postings: 224573
            Number of fields: 0
            Number of tokens: 271581
            Field names: []
            Positions:   false

        In this example, the index has 11429 documents, which contained 271581 word occurrences. 7756 unique words were identified. The total number of postings in the inverted index is 224573.
        This index did not record fields during indexing (which can be useful for models such as BM25F). Similarly, positions, which are used for phrasal queries or proximity models were not recorded.

        Returns:
            A Java `CollectionStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/CollectionStatistics.html>`_ object for this index.
        """
        return self.index_obj().getCollectionStatistics()

    def lexicon(self):
        """The lexicon for this index.

        Note that the terms in the lexicon include all pre-processing, such as stemming. For example, the term 'chemical' would be
        stored as 'chemic' when using the default Porter stemmer.

        Returns:
            A Java `Lexicon <http://terrier.org/docs/current/javadoc/org/terrier/structures/Lexicon.html>`_ object for this index.
        """
        return self.index_obj().getLexicon()

    def inverted_index(self):
        """The inverted posting index for this index.

        Returns:
            A Java `PostingIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/PostingIndex.html>`_ object for this index's inverted index.
        """
        return self.index_obj().getInvertedIndex()

    def document_index(self):
        """The document index for this index.

        Returns:
            A Java `DocumentIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/DocumentIndex.html>`_ object for this index.
        """
        return self.index_obj().getDocumentIndex()

    def meta_index(self):
        """The meta index for this index.

        Example:

        .. code-block:: python
            :caption: Show metadata fields in a Terrier index.

            >>> print(index.meta_index().getKeys())
            ['docno', 'text']

        In this example, the index contains two metadata fields: ``docno``, which contains the document identifiers, and ``text``, which contains the raw text of each document.

        Returns:
            A Java `MetaIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/MetaIndex.html>`_ object for this index.
        """
        return self.index_obj().getMetaIndex()

    def direct_index(self):
        """The direct (forward) index for this index.

        Returns:
            A Java `PostingIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/PostingIndex.html>`_ object for this index's direct index.
        """
        return self.index_obj().getDirectIndex()

    # ----------------------------------------------------
    # Miscellaneous
    # ----------------------------------------------------

    def __repr__(self):
        return f'TerrierIndex({str(self.path)!r})'

    def built(self):
        """Returns whether the index has been built (or is a built in-memory index)."""
        return self.path == pt.Artifact.NO_PATH or os.path.exists(os.path.join(self.path, 'data.properties'))

    @classmethod
    @pt.java.required
    def coerce(cls, index_like: Union[str, Path, 'TerrierIndex']) -> 'TerrierIndex':
        """Attempts to build a :class:`TerrierIndex` from the given object.

        Args:
            index_like (object): The object to coerce into a TerrierIndex. If a ``str`` or ``Path``, it loads the index at the provided path. If
                a ``pt.terrier.J.IndexRef`` or ``pt.terrier.J.Index``, it creates a TerrierIndex from the Java object. If a ``pt.terrier.TerrierIndex``,
                it returns itself.
        """
        if isinstance(index_like, TerrierIndex):
            return index_like
        if isinstance(index_like, (str, Path)):
            return TerrierIndex(index_like)
        if isinstance(index_like, pt.terrier.J.IndexRef):
            return TerrierIndex(pt.Artifact.NO_PATH, _index_ref=index_like)
        if isinstance(index_like, pt.terrier.J.Index):
            return TerrierIndex(pt.Artifact.NO_PATH, _index_obj=index_like)
        raise RuntimeError(f'Could not coerce {index_like!r} into a TerrierIndex')

    @staticmethod
    def example():
        """Returns an example Terrier index."""
        return TerrierIndex.from_hf('pyterrier/sample.terrier')

    def get_corpus_iter(self, return_toks: bool = True) -> pt.model.IterDict:
        """Returns an iterable over the documents in this index's corpus.

        Args:
            return_toks: Whether to return tokenised text (list of strings) or raw text (string).

        A corpus iter from a Terrier index can be used for various purposes, including:
          - indexing the pre-tokenised Terrier index directly in another indexing pipeline
          - extracting document metadata for ingestion into another indexing pipeline
        """
        return self.index_obj().get_corpus_iter(return_toks=return_toks)


_WMODEL_MAP: Dict[TerrierModel, str] = {
    TerrierModel.bm25: 'BM25',
    TerrierModel.dph: 'DPH',
    TerrierModel.pl2: 'PL2',
    TerrierModel.dirichlet_lm: 'DirichletLM',
    TerrierModel.hiemstra_lm: 'Hiemstra_LM',
    TerrierModel.tf: 'Tf',
    TerrierModel.tf_idf: 'TF_IDF',
}

def _map_wmodel(model: Union[TerrierModel, str]) -> str:
    if isinstance(model, str):
        return model
    return _WMODEL_MAP[model]

_CONTROL_MAP: Dict[str, str] = {
    'bm25.k1': 'bm25.k_1',
    'bm25.b': 'bm25.b',
    'dfr.c': 'dfr.c',
    'tf_idf.k_1': 'tf_idf.k_1',
    'tf_idf.b': 'tf_idf.b',
    'dirichletlm.mu': 'dirichletlm.mu',
    'hiemstra_lm.lambda': 'hiemstra_lm.lambda',
}
def _map_controls(model_args):
    return {
        _CONTROL_MAP[k]: v
        for k, v in (model_args or {}).items()
        if k in _CONTROL_MAP
    }


_PROPERTY_MAP: Dict[str, str] = {}
def _map_properties(model_args):
    return {
        _PROPERTY_MAP[k]: v
        for k, v in (model_args or {}).items()
        if k in _PROPERTY_MAP
    }
