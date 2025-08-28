from functools import partial
from typing import Callable, Any, Dict, Union, Optional, Sequence, Literal, List
import numpy.typing as npt
import pandas as pd
import pyterrier as pt
from pyterrier.apply_base import ApplyDocumentScoringTransformer, ApplyQueryTransformer, ApplyDocFeatureTransformer, ApplyForEachQuery, ApplyIterForEachQuery, ApplyGenericTransformer, ApplyGenericIterTransformer, ApplyIndexer, DropColumnTransformer, ApplyByRowTransformer, RenameColumnsTransformer

def _bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method

def query(fn : Callable[[Union[pd.Series,pt.model.IterDictRecord]], str], *args, **kwargs) -> pt.Transformer:
    """
        Create a transformer that takes as input a query, and applies a supplied function to compute a new query formulation.

        The supplied function is called once for each query, and must return a string containing the new query formulation.
        Each time it is called, the function is supplied with a Panda Series or dict representing the attributes of the query.
        The particular type of the input is not controlled by the implementor, so the function should be written to support 
        both, e.g. using ``row["key"]`` notation and not the ``row.key`` that is supported by a Series.

        The previous query formulation is saved in the "query_0" column. If a later pipeline stage is intended to resort to
        be executed on the previous query formulation, a ``pt.rewrite.reset()`` transformer can be applied.  

        :param fn: the function to apply to each row. It must return a string containing the new query formulation.
        :param required_columns: If provided, should be a list of columns that must be present in the input dataframe.
        :param verbose: if set to True, a TQDM progress bar will be displayed

        Examples::

            # this will remove pre-defined stopwords from the query
            stops=set(["and", "the"])

            # a naieve function to remove stopwords - takes as input a Pandas Series, and returns a string
            def _remove_stops(q):
                terms = q["query"].split(" ")
                terms = [t for t in terms if not t in stops ]
                return " ".join(terms)

            # a query rewriting transformer that applies the _remove_stops to each row of an input dataframe
            p1 = pt.apply.query(_remove_stops) >> pt.terrier.Retriever(index, wmodel="DPH")

            # an equivalent query rewriting transformer using an anonymous lambda function
            p2 = pt.apply.query(
                    lambda q :  " ".join([t for t in q["query"].split(" ") if t not in stops ])
                ) >> pt.terrier.Retriever(index, wmodel="DPH")

        In both of the example pipelines above (`p1` and `p2`), the exact topics are not known until the pipeline is invoked, e.g.
        by using `p1.transform(topics)` on a topics dataframe, or within a ``pt.Experiment()``. When the pipeline 
        is invoked, the specified function (`_remove_stops` in the case of `p1`) is called for **each** row of the 
        input datatrame (becoming the `q` function argument).
            

    """
    return ApplyQueryTransformer(fn, *args, **kwargs)

def doc_score(fn : Union[Callable[[Union[pd.Series,pt.model.IterDictRecord]], float], Callable[[pd.DataFrame], Sequence[float]]], *args, batch_size=None, **kwargs) -> pt.Transformer:
    """
        Create a transformer that takes as input a ranked documents dataframe, and applies a supplied function to compute a new score.
        Ranks are automatically computed. doc_score() can operate row-wise, or batch-wise, depending on whether batch_size is set.

        The supplied function is called once for each document, and must return a float containing the new score for that document.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query and document.

        :param fn: the function to apply to each row
        :param batch_size: How many documents to operate on at once (batch-wise). If None, operates row-wise
        :param required_columns: If provided, should be a list of columns that must be present in the input dataframe.
        :param verbose: if set to True, a TQDM progress bar will be displayed

        Example (Row-wise)::

            # this transformer will subtract 5 from the score of each document
            p = pt.terrier.Retriever(index, wmodel="DPH") >> 
                pt.apply.doc_score(lambda doc : doc["score"] -5) # doc["score"] works for both a dict and Series

        Can be used in batch-wise manner, which is particularly useful for appling neural models. In this case,
        the scoring function receives a dataframe, rather than a single row::

            def _doclen(df):
                # returns series of lengths
                return df.text.str.len()
            
            pipe = pt.terrier.Retriever(index) >> pt.apply.doc_score(_doclen, batch_size=128)

        Can also be used to create individual features that are combined using the ``**`` feature-union operator::

            pipeline = bm25 >> ( some_features ** pt.apply.doc_score(_doclen) )

    """
    return ApplyDocumentScoringTransformer(fn, *args, batch_size=batch_size, **kwargs)

def doc_features(fn : Callable[[Union[pd.Series,pt.model.IterDictRecord]], npt.NDArray[Any]], *args, **kwargs) -> pt.Transformer:
    """
        Create a transformer that takes as input a ranked documents dataframe, and applies the supplied function to each document to compute feature scores. 

        The supplied function is called once for each document, must each time return a 1D numpy array.
        Each time it is called, the function is supplied with a Panda Series, or a dictionary, representing the attributes of the query and document. The
        particular type of the input is not controlled by the implementor, so the function should be written to support both, e.g. using ``row["key"]``
        notation and not the ``row.key`` that is supported by a Series.

        :param fn: the function to apply to each row. It must return a 1D numpy array
        :param required_columns: If provided, should be a list of columns that must be present in the input dataframe.
        :param verbose: if set to True, a TQDM progress bar will be displayed
        
        Example::

            # this transformer will compute the character and number of word in each document retrieved
            # using the contents of the document obtained from the MetaIndex

            def _features(row):
                docid = row["docid"]
                content = index.getMetaIndex().getItem("text", docid)
                f1 = len(content)
                f2 = len(content.split(" "))
                return np.array([f1, f2])

            p = pt.terrier.Retriever(index, wmodel="BM25") >> 
                pt.apply.doc_features(_features )

        NB: If you only want to calculate a single feature to add to existing features, it is better to use ``pt.apply.doc_score()`` 
        and the ``**`` feature union operator::

            pipeline = bm25 >> ( some_features ** pt.apply.doc_score(one_feature) )

    """
    return ApplyDocFeatureTransformer(fn, *args, **kwargs)

def indexer(fn : Callable[[pt.model.IterDict], Any], **kwargs) -> pt.Indexer:
    """
        Create an instance of pt.Indexer using a function that takes as input an interable dictionary.

        The supplied function is called once. It may optionally return something (typically a reference to the "index").

        :param fn: the function that consumes documents as IterDicts.

        Example::

            # make a pt.Indexer that returns the numnber of documents consumed
            def _counter(iter_dict):
                count = 0
                for d in iter_dict:
                    count += 1
                return count
            indexer = pt.apply.indexer(_counter)
            rtr = indexer.index([ {'docno' : 'd1'}, {'docno' : 'd2'}])
    """
    return ApplyIndexer(fn, **kwargs)

def rename(columns: Dict[str,str], *, errors: Literal['raise', 'ignore'] = 'raise') -> pt.Transformer:
    """
        Creates a transformer that renames columns in a dataframe. 

        :param columns: A dictionary mapping from old column name to new column name
        :param errors: Maps to df.rename() errors kwarg - default to 'raise', alternatively can be 'ignore'

        Example::
            
            pipe = pt.terrier.Retriever(index, metadata=["docno", "body"]) >> pt.apply.rename({'body':'text'})
    """
    return RenameColumnsTransformer(columns, errors=errors)

def generic(fn : Union[Callable[[pd.DataFrame], pd.DataFrame], Callable[[pt.model.IterDict], pt.model.IterDict]], *args, batch_size=None, iter=False, **kwargs) -> pt.Transformer:
    """
        Create a transformer that changes the input dataframe to another dataframe in an unspecified way.

        The supplied function is called once for an entire result set as a dataframe or iter-dict (which may contain one or 
        more queries and one or more documents). Each time it should return a new dataframe. The returned dataframe (or yielded row) 
        should abide by the general PyTerrier Data Model, for instance updating the rank column if the scores are amended.

        :param fn: the function to apply to each result set
        :param batch_size: whether to apply fn on batches of rows or all that are received
        :param required_columns: If provided, should be a list of columns that must be present in the input dataframe.
        :param verbose: Whether to display a progress bar over batches (only used if batch_size is set, and iter is not set).
        :param iter: Whether to use the iter-dict API - if-so, then ``fn`` receives an iterable, and returns an iterable. 

        Example (dataframe)::

            # this pipeline would remove all but the first two documents from a result set
            pipe = pt.terrier.Retriever(index) >> pt.apply.generic(lambda res : res[res["rank"] < 2])

        Example (iter-dict)::

            # this pipeline would simlarly remove all but the first two documents from a result set
            def _fn(iterdict):
                for result in iterdict:
                    if result["rank"] < 2:
                        yield result
            pipe1 = pt.terrier.Retriever(index) >> pt.apply.generic(_fn, iter=True)

            # transform_iter() can also return an iterable, so returning a list is also permissible
            pipe2 = pt.terrier.Retriever(index) >> pt.apply.generic(lambda res: [row for row in res if row["rank"] < 2], iter=True)

    """
    if iter:
        if kwargs.get("add_ranks", False):
            raise ValueError("add_ranks=True not supported with iter=True")
        return ApplyGenericIterTransformer(fn, *args, batch_size=batch_size, **kwargs)
    return ApplyGenericTransformer(fn, *args, batch_size=batch_size, **kwargs)

def by_query(fn : Union[Callable[[pd.DataFrame], pd.DataFrame], Callable[[pt.model.IterDict], pt.model.IterDict]], *args, batch_size=None, iter=False, verbose=False, **kwargs) -> pt.Transformer:
    """
        As `pt.apply.generic()` except that fn receives a dataframe (or iter-dict) for one query at at time, rather than all results at once.
        If batch_size is set, fn will receive no more than batch_size documents for any query. The verbose kwargs controls whether
        to display a progress bar over queries.  

        :param fn: the function to apply to each row. Should return a generator
        :param batch_size: whether to apply fn on batches of rows or all that are received.
        :param required_columns: If provided, should be a list of columns that must be present in the input dataframe.
        :param verbose: Whether to display a progress bar over batches (only used if batch_size is set, and iter is not set).
        :param iter: Whether to use the iter-dict API - if-so, then ``fn`` receives an iterable, and must return an iterable. 
    """
    if iter:
        if kwargs.get("add_ranks", False):
            raise ValueError("add_ranks=True not supported with iter=True")
        return ApplyIterForEachQuery(fn, *args, batch_size=batch_size, verbose=verbose, **kwargs)
    return ApplyForEachQuery(fn, *args, batch_size=batch_size, verbose=verbose, **kwargs)

class _apply:

    def __init__(self):
        _bind(self, lambda self, fn, *args, **kwargs : query(fn, *args, **kwargs), as_name='query')
        _bind(self, lambda self, fn, *args, **kwargs : doc_score(fn, *args, **kwargs), as_name='doc_score')
        _bind(self, lambda self, fn, *args, **kwargs : doc_features(fn, *args, **kwargs), as_name='doc_features')
        _bind(self, lambda self, fn, *args, **kwargs : indexer(fn, *args, **kwargs), as_name='indexer')
        _bind(self, lambda self, fn, *args, **kwargs : rename(fn, *args, **kwargs), as_name='rename')
        _bind(self, lambda self, fn, *args, **kwargs : by_query(fn, *args, **kwargs), as_name='by_query')
        _bind(self, lambda self, fn, *args, **kwargs : generic(fn, *args, **kwargs), as_name='generic')     
    
    def __getattr__(self, item: str) -> Callable[..., pt.Transformer]:
        return partial(generic_apply, item)

def generic_apply(
    name: str,
    fn=None,
    *,
    drop: bool = False,
    batch_size: Optional[int] = None,
    required_columns: Optional[List[str]] = None,
    verbose=False
) -> pt.Transformer:
    if drop:
        assert fn is None, "cannot provide both fn and drop=True"
        return DropColumnTransformer(name)

    return ApplyByRowTransformer(name, fn, batch_size=batch_size, required_columns=required_columns, verbose=verbose)
