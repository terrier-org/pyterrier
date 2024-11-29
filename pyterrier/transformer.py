import types
from matchpy import Wildcard, Symbol, Operation, Arity, ReplacementRule
from warnings import warn
import pandas as pd
from deprecated import deprecated
from typing import Iterator, List, Union, Tuple
import pyterrier as pt
from . import __version__

LAMBDA = lambda:0
def is_lambda(v):
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def is_function(v):
    return isinstance(v, types.FunctionType)

def is_transformer(v):
    if isinstance(v, Transformer):
        return True
    return False

def get_transformer(v, stacklevel=1):
    """ 
        Used to coerce functions, lambdas etc into transformers 
    """

    if isinstance(v, Wildcard):
        # get out of jail for matchpy
        return v
    if is_transformer(v):
        return v
    if is_lambda(v):
        warn('Coercion of a lambda into a transformer is deprecated; use a pt.apply instead', stacklevel=stacklevel, category=DeprecationWarning)
        from .apply_base import ApplyGenericTransformer
        return ApplyGenericTransformer(v)
    if is_function(v):
        from .apply_base import ApplyGenericTransformer
        warn('Coercion of a function (called "%s") into a transformer is deprecated; use a pt.apply instead' % v.__name__, stacklevel=stacklevel, category=DeprecationWarning)
        return ApplyGenericTransformer(v)
    if isinstance(v, pd.DataFrame):
        warn('Coercion of a dataframe into a transformer is deprecated; use a pt.Transformer.from_df() instead', stacklevel=stacklevel, category=DeprecationWarning)
        return SourceTransformer(v)
    raise ValueError("Passed parameter %s of type %s cannot be coerced into a transformer" % (str(v), type(v)))

rewrite_rules : List[ReplacementRule] = []


class Scalar(Symbol):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

class Transformer:
    """
        Base class for all transformers. Implements the various operators ``>>`` ``+`` ``*`` ``|`` ``&`` 
        as well as ``search()`` for executing a single query and ``compile()`` for rewriting complex pipelines into more simples ones.

        Its expected that either ``.transform()`` or ``.transform_iter()`` be implemented by any class extending this - this rule
        does not apply for indexers, which instead implement ``.index()``.
    """
    name = "Transformer"

    def __new__(cls, *args, **kwargs):
        if cls.transform == Transformer.transform and cls.transform_iter == Transformer.transform_iter:
            raise NotImplementedError("You need to implement either .transform() or .transform_iter() in %s" % str(cls))
        return super().__new__(cls)

    @staticmethod
    def identity() -> 'Transformer':
        """
        Instantiates a transformer that returns exactly its input. 
        
        This can be useful for adding the candidate ranking score
        as a feature in for learning-to-rank::

            bm25 = pt.terrier.Retriever(index, wmodel="BM25")
            two_feat_pipe = bm25 >> pt.Transformer.identify() ** pt.terrier.Retriever(index, wmodel="PL2")
        
        This will return a pipeline that produces a score column (BM25), but also has a features column containing
        BM25 and PL2 scores.
        
        """
        return IdentityTransformer()

    @staticmethod
    def from_df(input : pd.DataFrame, uniform=False) -> 'Transformer':
        """
        Instantiates a transformer from an input dataframe. Some rows from the input dataframe are returned
        in response to a query on the ``transform()`` method. Depending on the value `uniform`, the dataframe
        passed as an argument to ``transform()`` can affect this selection.

        If `uniform` is True, input will be returned in its entirety each time.
        If `uniform` is False, rows from input that match the qid values from the argument dataframe.
        
        """
        if uniform:
            return UniformTransformer(input)
        return SourceTransformer(input)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """
            Abstract method that runs the transformer over Pandas ``DataFrame`` objects.

            .. note::

                Either :meth:`transform` or :meth:`transform_iter` must be implemented for all transformers.
                If not, a runtime error will be raised when constructing the transformer.

                When :meth:`transform` is not implemented, the default implementation runs :meth:`transform_iter` and
                converts the output to a ``DataFrame``.

            Arguments:
                inp(``pd.DataFrame``): The input to the transformer (e.g., queries, documents, results, etc.)

            Returns:
                The output of the transformer (e.g., result of retrieval, re-writing, re-ranking, etc.)

            :rtype: ``pd.DataFrame``
        """
        # We should have no recursive transform <-> transform_iter problem, due to the __new__ check.
        return pd.DataFrame(list(self.transform_iter(inp.to_dict(orient='records'))))

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        """
            Abstract method that runs the transformer over iterable input (such as lists or generators),
            where each element is a dictionary record.

            This format can sometimes be easier to implement than :meth:`transform`. Furthermore, it avoids constructing
            expensive ``DataFrame`` objects. It is also used in the invocation of ``index()`` on a composed pipeline.

            .. note::

                Either :meth:`transform` or :meth:`transform_iter` must be implemented for all transformers.
                If not, a runtime error will be raised when constructing the transformer.

                When :meth:`transform_iter` is not implemented, the default implementation runs :meth:`transform` and
                converts the output to an iterable.

            Arguments:
                inp(``Iterable[Dict]``): The input to the transformer (e.g., queries, documents, results, etc.)

            Returns:
                The output of the transformer (e.g., result of retrieval, re-writing, re-ranking, etc.)

            :rtype: ``Iterable[Dict]``
        """
        # We should have no recursive transform <-> transform_iter problem, due to the __new__ check.
        return self.transform(pd.DataFrame(list(inp))).to_dict(orient='records')
    
    def __call__(self, inp: Union[pd.DataFrame, pt.model.IterDict, List[pt.model.IterDictRecord]]) -> Union[pd.DataFrame, pt.model.IterDict, List[pt.model.IterDictRecord]]:
        """
            Runs the transformer for the given input and returns its output as the same type as the input.

            - When ``inp`` is a DataFrame, invokes :meth:`transform` and returns a DataFrame
            - When ``inp`` is a list, invokes :meth:`transform_iter` and returns a list
            - Otherwise, invokes :meth:`transform_iter` and returns a generic iterable (returning whatever type is
              returned from :meth:`transform_iter()`.)

            Arguments:
                inp(``pd.DataFrame``, ``Iterable[Dict]``, ``List[Dict]``): The input to the transformer (e.g., queries,
                    documents, results, etc.)

            Returns:
                The output of the transformer (e.g., result of retrieval, re-writing, re-ranking, etc.) as the same
                type as the input.

            :rtype: ``pd.DataFrame``, ``Iterable[Dict]``, ``List[Dict]``
        """
        if isinstance(inp, pd.DataFrame):
            return self.transform(inp)
        out = self.transform_iter(inp)
        if isinstance(inp, list):
            return list(out)
        return out

    def transform_gen(self, input : pd.DataFrame, batch_size=1, output_topics=False) -> Union[Iterator[pd.DataFrame], Iterator[Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
            Method for executing a transformer pipeline on smaller batches of queries.
            The input dataframe is grouped into batches of batch_size queries, and a generator
            returned, such that transform() is only executed for a smaller batch at a time. 

            Arguments:
                input(DataFrame): a dataframe to process
                batch_size(int): how many input instances to execute in each batch. Defaults to 1.
            
        """
        docno_provided = "docno" in input.columns
        docid_provided = "docid" in input.columns
        
        if docno_provided or docid_provided:
            queries = input[["qid"]].drop_duplicates()
        else:
            queries = input
        batch : List[pd.DataFrame] = []      
        for query in queries.itertuples():
            if len(batch) == batch_size:
                batch_topics = pd.concat(batch)
                batch=[]
                res = self.transform(batch_topics)
                if output_topics:
                    yield res, batch_topics
                else:
                    yield res
            batch.append(input[input["qid"] == query.qid])
        if len(batch) > 0:
            batch_topics = pd.concat(batch)
            res = self.transform(batch_topics)
            if output_topics:
                yield res, batch_topics
            else:
                yield res

    def search(self, query : str, qid : str = "1", sort=True) -> pd.DataFrame:
        """
            Method for executing a transformer (pipeline) for a single query. 
            Returns a dataframe with the results for the specified query. This
            is a utility method, and most uses are expected to use the transform()
            method passing a dataframe.

            Arguments:
                query(str): String form of the query to run
                qid(str): the query id to associate to this request. defaults to 1.
                sort(bool): ensures the results are sorted by descending rank (defaults to True)

            Example::

                bm25 = pt.terrier.Retriever(index, wmodel="BM25")
                res = bm25.search("example query")

                # is equivalent to
                queryDf = pd.DataFrame([["1", "example query"]], columns=["qid", "query"])
                res = bm25.transform(queryDf)
            
            
        """
        import pandas as pd
        queryDf = pd.DataFrame([[qid, query]], columns=["qid", "query"])
        rtr = self.transform(queryDf)
        if "qid" in rtr.columns and "rank" in rtr.columns:
            rtr = rtr.sort_values(["qid", "rank"], ascending=[True,True])
        return rtr

    def compile(self) -> 'Transformer':
        """
        Rewrites this pipeline by applying of the Matchpy rules in rewrite_rules. Pipeline
        optimisation is discussed in the `ICTIR 2020 paper on PyTerrier <https://arxiv.org/abs/2007.14271>`_.
        """
        from matchpy import replace_all
        print("Applying %d rules" % len(rewrite_rules))
        return replace_all(self, rewrite_rules)

    def parallel(self, N : int, backend='joblib') -> 'Transformer':
        """
        Returns a parallelised version of this transformer. The underlying transformer must be "picklable".

        Args:
            N(int): how many processes/machines to parallelise this transformer over. 
            backend(str): which multiprocessing backend to use. Only two backends are supported, 'joblib' and 'ray'. Defaults to 'joblib'.
        """
        from .parallel import PoolParallelTransformer
        return PoolParallelTransformer(self, N, backend)

    # Get and set specific parameter value by parameter's name
    def get_parameter(self, name : str):
        """
            Gets the current value of a particular key of the transformer's configuration state.
            By default, this examines the attributes of the transformer object, using hasattr() and setattr().
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(("Invalid parameter name %s for transformer %s. " + 
                      "Check the list of available parameters") %(str(name), str(self)))

    def set_parameter(self, name : str, value):
        """
            Adjusts this transformer's configuration state, by setting the value for specific parameter.
            By default, this examines the attributes of the transformer object, using hasattr() and setattr().
        """
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError(('Invalid parameter name %s for transformer %s. '+
                    'Check the list of available parameters') %(name, str(self)))

    def __rshift__(self, right) -> 'Transformer':
        from .ops import ComposedPipeline
        return ComposedPipeline(self, right)

    def __rrshift__(self, left) -> 'Transformer':
        from .ops import ComposedPipeline
        return ComposedPipeline(left, self)

    def __add__(self, right : 'Transformer') -> 'Transformer':
        from .ops import CombSumTransformer
        return CombSumTransformer(self, right)

    def __pow__(self, right : 'Transformer') -> 'Transformer':
        from .ops import FeatureUnionPipeline
        return FeatureUnionPipeline(self, right)

    def __mul__(self, rhs : Union[float,int]) -> 'Transformer':
        assert isinstance(rhs, int) or isinstance(rhs, float)
        from .ops import ScalarProductTransformer
        return ScalarProductTransformer(self, rhs)

    def __rmul__(self, lhs : Union[float,int]) -> 'Transformer':
        assert isinstance(lhs, int) or isinstance(lhs, float)
        from .ops import ScalarProductTransformer
        return ScalarProductTransformer(self, lhs)

    def __or__(self, right : 'Transformer') -> 'Transformer':
        from .ops import SetUnionTransformer
        return SetUnionTransformer(self, right)

    def __and__(self, right : 'Transformer') -> 'Transformer':
        from .ops import SetIntersectionTransformer
        return SetIntersectionTransformer(self, right)

    def __mod__(self, right : int) -> 'Transformer':
        assert isinstance(right, int)
        from .ops import RankCutoffTransformer
        return RankCutoffTransformer(self, right)

    def __xor__(self, right : 'Transformer') -> 'Transformer':
        from .ops import ConcatenateTransformer
        return ConcatenateTransformer(self, right)

    @deprecated(version="0.11.1", reason="Use pyterrier-caching for more fine-grained caching, e.g. RetrieverCache or ScorerCache")
    def __invert__(self : 'Transformer') -> 'Transformer':
        from .cache import ChestCacheTransformer
        return ChestCacheTransformer(self)

    def __hash__(self):
        return hash(repr(self))

class TransformerBase(Transformer):
    @deprecated(version="0.9", reason="Use pt.Transformer instead of TransformerBase")
    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)

class Indexer(Transformer):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        # We have some patching do to in the (somewhat rare) case where the user implements transform/transform_iter
        # on an indexer. Normally these raise errors when called on an indexer, but in this case the user wants the
        # indexer to act as a transformer. So patch the complementary method to call the implemented one.
        if cls.transform != Indexer.transform and cls.transform_iter == Indexer.transform_iter:
            # User implemented transform on this indexer but not transform_iter. Replace transform_iter with the default
            # one, which invokes transform automatically.
            instance.transform_iter = types.MethodType(Transformer.transform_iter, instance)
        elif cls.transform == Indexer.transform and cls.transform_iter != Indexer.transform_iter:
            # User implemented transform_iter on this indexer but not transform. Replace transform with the default
            # one, which invokes transform_iter automatically.
            instance.transform = types.MethodType(Transformer.transform, instance)
        return instance

    def index(self, iter : pt.model.IterDict, **kwargs):
        """
            Takes an iterable of dictionaries ("iterdict"), and consumes them. The index method may return
            an instance of the index or retriever. This method is typically used to implement indexers that
            consume a corpus (or to consume the output of previous pipeline components that have
            transformer the documents being consumed).
        """
        pass

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('You called `transform()` on an indexer. Did you mean to call `index()`?')

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        raise NotImplementedError('You called `transform_iter()` on an indexer. Did you mean to call `index()`?')

class IterDictIndexerBase(Indexer):
    @deprecated(version="0.9", reason="Use pt.Indexer instead of IterDictIndexerBase")
    def __init__(self, *args, **kwargs):
        super(Indexer, self).__init__(*args, **kwargs)

class Estimator(Transformer):
    """
        This is a base class for things that can be fitted.
    """
    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va):
        """
            Method for training the transformer.

            Arguments:
                topics_or_res_tr(DataFrame): training topics (usually with documents)
                qrels_tr(DataFrame): training qrels
                topics_or_res_va(DataFrame): validation topics (usually with documents)
                qrels_va(DataFrame): validation qrels
        """
        pass

class EstimatorBase(Estimator):
    @deprecated(version="0.9", reason="Use pt.Estimator instead of EstimatorBase")
    def __init__(self, *args, **kwargs):
        super(Estimator, self).__init__(*args, **kwargs)

class IdentityTransformer(Transformer, Operation):
    """
        A transformer that returns exactly the same as its input.
    """
    arity = Arity.nullary

    def __init__(self, *args, **kwargs):
        super(IdentityTransformer, self).__init__(*args, **kwargs)
    
    def transform(self, topics):
        return topics

class SourceTransformer(Transformer, Operation):
    """
    A Transformer that can be used when results have been saved in a dataframe.
    It will select results on qid.
    If a column is in the dataframe passed in the constructor, this will override any
    column in the topics dataframe passed to the transform() method.
    """
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        super().__init__(operands=[], **kwargs)
        self.operands=[]
        self.df = rtr[0]
        assert "qid" in self.df.columns
    
    def transform(self, topics):
        import numpy as np
        assert "qid" in topics.columns
        keeping = topics.columns

        common_columns = np.intersect1d(topics.columns, self.df.columns)

        # we drop columns in topics that exist in the self.df
        drop_columns = common_columns[common_columns != "qid"]
        if len(drop_columns) > 0:
            keeping = topics.columns[~ topics.columns.isin(drop_columns)]
        
        rtr = topics[keeping].merge(self.df, on="qid")
        return rtr

class UniformTransformer(Transformer, Operation):
    """
        A transformer that returns the same dataframe every time transform()
        is called. This class is useful for testing. 
    """
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        super().__init__(operands=[], **kwargs)
        self.operands=[]
        self.rtr = rtr[0]
    
    def transform(self, topics):
        rtr = self.rtr.copy()
        return rtr
