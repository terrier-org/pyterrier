from functools import partial
from typing import Callable, Any, Dict, Union, Sequence
import numpy.typing as npt
import pandas as pd
import pyterrier as pt
from pyterrier.apply_base import ApplyDocumentScoringTransformer, ApplyQueryTransformer, ApplyDocFeatureTransformer, ApplyForEachQuery, ApplyGenericTransformer

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

def query(fn : Callable[[pd.Series], str], *args, **kwargs) -> pt.Transformer:
    """
        Create a transformer that takes as input a query, and applies a supplied function to compute a new query formulation.

        The supplied function is called once for each query, and must return a string containing the new query formulation.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query.

        The previous query formulation is saved in the "query_0" column. If a later pipeline stage is intended to resort to
        be executed on the previous query formulation, a `pt.rewrite.reset()` transformer can be applied.  

        Arguments:
            fn(Callable): the function to apply to each row. It must return a string containing the new query formulation.
            verbose(bool): if set to True, a TQDM progress bar will be displayed

        Examples::

            # this will remove pre-defined stopwords from the query
            stops=set(["and", "the"])

            # a naieve function to remove stopwords - takes as input a Pandas Series, and returns a string
            def _remove_stops(q):
                terms = q["query"].split(" ")
                terms = [t for t in terms if not t in stops ]
                return " ".join(terms)

            # a query rewriting transformer that applies the _remove_stops to each row of an input dataframe
            p1 = pt.apply.query(_remove_stops) >> pt.terrier.Retrieve(index, wmodel="DPH")

            # an equivalent query rewriting transformer using an anonymous lambda function
            p2 = pt.apply.query(
                    lambda q :  " ".join([t for t in q["query"].split(" ") if t not in stops ])
                ) >> pt.terrier.Retrieve(index, wmodel="DPH")

        In both of the example pipelines above (`p1` and `p2`), the exact topics are not known until the pipeline is invoked, e.g.
        by using `p1.transform(topics)` on a topics dataframe, or within a `pt.Experiment()`. When the pipeline 
        is invoked, the specified function (`_remove_stops` in the case of `p1`) is called for **each** row of the 
        input datatrame (becoming the `q` function argument).
            

    """
    return ApplyQueryTransformer(fn, *args, **kwargs)

def doc_score(fn : Union[Callable[[pd.Series], float], Callable[[pd.DataFrame], Sequence[float]]], *args, batch_size=None, **kwargs) -> pt.Transformer:
    """
        Create a transformer that takes as input a ranked documents dataframe, and applies a supplied function to compute a new score.
        Ranks are automatically computed. doc_score() can operate row-wise, or batch-wise, depending on whether batch_size is set.

        The supplied function is called once for each document, and must return a float containing the new score for that document.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query and document.

        Arguments:
            fn(Callable): the function to apply to each row
            batch_size (int or None). How many documents to operate on at once (batch-wise). If None, operates row-wise
            verbose(bool): if set to True, a TQDM progress bar will be displayed

        Example (Row-wise)::

            # this transformer will subtract 5 from the score of each document
            p = pt.terrier.Retrieve(index, wmodel="DPH") >> 
                pt.apply.doc_score(lambda doc : doc["score"] -5)

        Can be used in batch-wise manner, which is particularly useful for appling neural models. In this case,
        the scoring function receives a dataframe, rather than a single row::

            def _doclen(df):
                # returns series of lengths
                return df.text.str.len()
            
            pipe = pt.terrier.Retrieve(index) >> pt.apply.doc_score(_doclen, batch_size=128)

    """
    return ApplyDocumentScoringTransformer(fn, *args, batch_size=batch_size, **kwargs)

def doc_features(fn : Callable[[pd.Series], npt.NDArray[Any]], *args, **kwargs) -> pt.Transformer:
    """
        Create a transformer that takes as input a ranked documents dataframe, and applies the supplied function to each document to compute feature scores. 

        The supplied function is called once for each document, must each time return a 1D numpy array.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query and document.

        Arguments:
            fn(Callable): the function to apply to each row
            verbose(bool): if set to True, a TQDM progress bar will be displayed

        Example::

            # this transformer will compute the character and number of word in each document retrieved
            # using the contents of the document obtained from the MetaIndex

            def _features(row):
                docid = row["docid"]
                content = index.getMetaIndex().getItem("text", docid)
                f1 = len(content)
                f2 = len(content.split(" "))
                return np.array([f1, f2])

            p = pt.terrier.Retrieve(index, wmodel="BM25") >> 
                pt.apply.doc_features(_features )

    """
    return ApplyDocFeatureTransformer(fn, *args, **kwargs)

def rename(columns : Dict[str,str], *args, errors='raise', **kwargs) -> pt.Transformer:
    """
        Creates a transformer that renames columns in a dataframe. 

        Args:
            columns(dict): A dictionary mapping from old column name to new column name
            errors(str): Maps to df.rename() errors kwarg - default to 'raise', alternatively can be 'ignore'

        Example::
            
            pipe = pt.terrier.Retrieve(index, metadata=["docno", "body"]) >> pt.apply.rename({'body':'text'})
    """
    return ApplyGenericTransformer(lambda df: df.rename(columns=columns, errors=errors), *args, **kwargs)

def generic(fn : Callable[[pd.DataFrame], pd.DataFrame], *args, batch_size=None, **kwargs) -> pt.Transformer:
    """
        Create a transformer that changes the input dataframe to another dataframe in an unspecified way.

        The supplied function is called once for an entire result set as a dataframe (which may contain one of more queries).
        Each time it should return a new dataframe. The returned dataframe should abide by the general PyTerrier Data Model,
        for instance updating the rank column if the scores are amended.

        Arguments:
            fn(Callable): the function to apply to each row
            batch_size(int or None): whether to apply fn on batches of rows or all that are received
            verbose(bool): Whether to display a progress bar over batches (only used if batch_size is set).

        Example::

            # this transformer will remove all documents at rank greater than 2.

            # this pipeline would remove all but the first two documents from a result set
            pipe = pt.terrier.Retrieve(index) >> pt.apply.generic(lambda res : res[res["rank"] < 2])

    """
    return ApplyGenericTransformer(fn, *args, batch_size=batch_size, **kwargs)

def by_query(fn : Callable[[pd.DataFrame], pd.DataFrame], *args, batch_size=None, **kwargs) -> pt.Transformer:
    """
        As `pt.apply.generic()` except that fn receives a dataframe for one query at at time, rather than all results at once.
        If batch_size is set, fn will receive no more than batch_size documents for any query. The verbose kwargs controls whether
        to display a progress bar over queries.  
    """
    return ApplyForEachQuery(fn, *args, batch_size=batch_size, **kwargs)

class _apply:

    def __init__(self):
        _bind(self, lambda self, fn, *args, **kwargs : query(fn, *args, **kwargs), as_name='query')
        _bind(self, lambda self, fn, *args, **kwargs : doc_score(fn, *args, **kwargs), as_name='doc_score')
        _bind(self, lambda self, fn, *args, **kwargs : doc_features(fn, *args, **kwargs), as_name='doc_features')
        _bind(self, lambda self, fn, *args, **kwargs : rename(fn, *args, **kwargs), as_name='rename')
        _bind(self, lambda self, fn, *args, **kwargs : by_query(fn, *args, **kwargs), as_name='by_query')
        _bind(self, lambda self, fn, *args, **kwargs : generic(fn, *args, **kwargs), as_name='generic')
    
    def __getattr__(self, item):
        return partial(generic_apply, item)

def generic_apply(name, *args, drop=False, **kwargs) -> pt.Transformer:
    if drop:
        return ApplyGenericTransformer(lambda df : df.drop(name, axis=1), *args, **kwargs) 
    
    if len(args) == 0:
        raise ValueError("Must specify a fn, e.g. a lambda")

    fn = args[0]
    args=[]

    def _new_column(df):
        df[name] = df.apply(fn, axis=1, result_type='reduce')
        return df
    
    return ApplyGenericTransformer(_new_column, *args, **kwargs)
