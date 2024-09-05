
from .transformer import Transformer, Indexer
from .model import add_ranks, split_df
import pandas as pd
import pyterrier as pt

class ApplyTransformerBase(Transformer):
    """
        A base class for Apply*Transformers
    """
    def __init__(self, fn, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.verbose = verbose

    def __repr__(self):
        return "pt.apply.??()"

class ApplyForEachQuery(ApplyTransformerBase):
    def __init__(self, fn,  *args, add_ranks=True, batch_size=None, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns the new float doument score 
        """
        super().__init__(fn, *args, **kwargs)
        self.add_ranks = add_ranks
        self.batch_size = batch_size
    
    def __repr__(self):
        return "pt.apply.by_query()"
    
    def transform(self, res):
        if len(res) == 0:
            return self.fn(res)

        import math, pandas as pd
        from pyterrier.model import split_df

        it = res.groupby("qid")
        lastqid = None
        if self.verbose:
            it = pt.tqdm(it, unit='query')
        try:
            if self.batch_size is None:
                query_dfs = []
                for qid, group in it:
                    lastqid = qid
                    query_dfs.append(self.fn(group))
            else:
                # fn cannot be applied to more than batch_size rows at once
                # so we must split and reconstruct the output FOR EACH QUERY
                query_dfs = []
                for qid, group in it:
                    
                    num_chunks = math.ceil( len(group) / self.batch_size )
                    iterator = split_df(group, num_chunks)
                    query_dfs.append( pd.concat([self.fn(chunk_df) for chunk_df in iterator]) )
        except Exception as a:
            raise Exception("Problem applying %s for qid %s" % (self.fn, lastqid)) from a

        if self.add_ranks:
            try:
                query_dfs = [add_ranks(df, single_query=True) for df in query_dfs]
            except KeyError as ke:
                suffix = 'Try setting add_ranks=False'
                if len(query_dfs) > 0 and 'score' not in query_dfs[0].columns:
                    suffix = 'score column not present. Set add_ranks=False'
                raise ValueError("Cannot apply add_ranks in pt.apply.by_query - " + suffix) from ke
        rtr = pd.concat(query_dfs)
        return rtr

class ApplyDocumentScoringTransformer(ApplyTransformerBase):
    """
        Implements a transformer that can apply a function to perform document scoring. The supplied function 
        should take as input one row, and return a float for the score of the document.
        
        Usually accessed using pt.apply.doc_score()::

            def _score_fn(row):
                return float(row["url".count("/")])
            
            pipe = pt.terrier.Retriever(index) >> pt.apply.doc_score(_score_fn)

        Can be used in batching manner, which is particularly useful for appling neural models. In this case,
        the scoring function receives a dataframe, rather than a single row::

            def _doclen(df):
                return df.text.str.len()
            
            pipe = pt.terrier.Retriever(index) >> pt.apply.doc_score(_doclen)

    """
    def __init__(self, fn,  *args, batch_size=None, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns the new float doument score. If batch_size is set,
             takes a dataframe, and returns a sequence of floats representing scores for those documents.
             - batch_size (int or None). How many documents to operate on at once. If None, operates row-wise
        """
        super().__init__(fn, *args, **kwargs)
        self.batch_size = batch_size

    def __repr__(self):
        return "pt.apply.doc_score()"

    def _transform_rowwise(self, outputRes):
        fn = self.fn
        if len(outputRes) == 0:
            outputRes["score"] = pd.Series(dtype='float64')
            return outputRes
        if self.verbose:
            pt.tqdm.pandas(desc="pt.apply.doc_score", unit="d")
            outputRes["score"] = outputRes.progress_apply(fn, axis=1).astype('float64')
        else:
            outputRes["score"] = outputRes.apply(fn, axis=1).astype('float64')
        outputRes = add_ranks(outputRes)
        return outputRes
    
    def _transform_batchwise(self, outputRes):
        fn = self.fn
        outputRes["score"] = fn(outputRes)
        outputRes["score"] = outputRes["score"].astype('float64')
        return outputRes
    
    def transform(self, inputRes):
        outputRes = inputRes.copy()
        if len(outputRes) == 0:
            outputRes["score"] = pd.Series(dtype='float64')
            return add_ranks(outputRes)
        if self.batch_size is None:
            return self._transform_rowwise(inputRes)

        import math
        from .model import split_df
        num_chunks = math.ceil( len(inputRes) / self.batch_size )
        iterator = split_df(inputRes, num_chunks)
        iterator = pt.tqdm(iterator, desc="pt.apply", unit='row')
        rtr = pd.concat([self._transform_batchwise(chunk_df) for chunk_df in iterator])
        rtr = add_ranks(rtr)
        return rtr

class ApplyDocFeatureTransformer(ApplyTransformerBase):
    """
        Implements a transformer that can apply a function to perform feature scoring. The supplied function 
        should take as input one row, and return a numpy array for the features of the document.
        
        Usually accessed using pt.apply.doc_features()::

            def _feature_fn(row):
                return numpy.array([len(row["url"], row["url".count("/")])
            
            pipe = pt.terrier.Retriever(index) >> pt.apply.doc_features(_feature_fn) >> pt.LTRpipeline(xgBoost())
    """
    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns a new numpy array representing the features of that document
        """
        super().__init__(fn, *args, **kwargs)

    def __repr__(self):
        return "pt.apply.doc_features()"
    
    def transform_iter(self, iterdict):
        fn = self.fn
        # we assume that the function can take a dictionary as well as a pandas.Series. As long as [""] notation is used
        # to access fields, both should work
        def gen():
            for row in pt.tqdm(iterdict, desc="pt.apply.doc_features") if self.verbose else iterdict:
                row["features"] = self.fn(row)
                yield row
        return list(gen())

    def transform(self, inputRes):
        fn = self.fn
        outputRes = inputRes.copy()
        if self.verbose:
            pt.tqdm.pandas(desc="pt.apply.doc_features", unit="d")
            outputRes["features"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["features"] = outputRes.apply(fn, axis=1)
        return outputRes

class ApplyQueryTransformer(ApplyTransformerBase):
    """
        Implements a query rewriting transformer by passing a function to perform the rewriting. The function should take
        as input one row, and return the string form of the new query.
        
        Usually accessed using pt.apply.query() passing it the function::

            def _rewriting_fn(row):
                return row["query"] + " extra words"
            
            pipe = pt.apply.query(_rewriting_fn) >> pt.terrier.Retriever(index)

        Similarly, a lambda function can also be used::

            pipe = pt.apply.query(lambda row: row["query"] + " extra words") >> pt.terrier.Retriever(index)

        In the resulting dataframe, the previous query for each row can be found in the query_0 column.

    """
    def __init__(self, fn, *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing a query, and returns the new string query 
             - verbose (bool): Display a tqdm progress bar for this transformer
        """
        super().__init__(fn, *args, **kwargs)

    def __repr__(self):
        return "pt.apply.query()"
    
    def transform_iter(self, iterdict):
        fn = self.fn
        # we assume that the function can take a dictionary as well as a pandas.Series. As long as [""] notation is used
        # to access fields, both should work
        def gen():
            for row in pt.tqdm(iterdict, desc="pt.apply.query") if self.verbose else iterdict:
                if "query" in row:
                    pass
                    # we only push if a query already exists
                    # TODO implement push_queries for iter-dict
                row["query"] = self.fn(row)
                yield row
        return list(gen())

    def transform(self, inputRes):
        from .model import push_queries
        fn = self.fn        
        if "query" in inputRes.columns:
            # we only push if a query already exists
            outputRes = push_queries(inputRes.copy(), inplace=True, keep_original=True)
        else:
            outputRes = inputRes.copy()
        try:
            if self.verbose:
                pt.tqdm.pandas(desc="pt.apply.query", unit="d")
                outputRes["query"] = outputRes.progress_apply(fn, axis=1)
            else:
                outputRes["query"] = outputRes.apply(fn, axis=1)
        except ValueError as ve:
            msg = str(ve)
            if "Columns must be same length as key" in msg:
                raise TypeError("Could not coerce return from pt.apply.query function into a list of strings. Check your function returns a string.") from ve
            else:
                raise ve
        return outputRes

class ApplyGenericTransformer(ApplyTransformerBase):
    """
    Allows arbitrary pipelines components to be written as functions. The function should take as input
    a dataframe, and return a new dataframe. The function should abide by the main contracual obligations,
    e.g. updating then "rank" column.

    This class is normally accessed through pt.apply.generic()

    If you are scoring, query rewriting or calculating features, it is advised to use one of the other
    variants.

    Example::
        
        # this pipeline would remove all but the first two documents from a result set
        lp = ApplyGenericTransformer(lambda res : res[res["rank"] < 2])

        pipe = pt.terrier.Retriever(index) >> lp

    """

    def __init__(self, fn,  *args, batch_size=None, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda DataFrame, and returns a new Pandas DataFrame 
        """
        super().__init__(fn, *args, **kwargs)
        self.batch_size = batch_size

    def __repr__(self):
        return "pt.apply.generic()"

    def transform(self, inputRes):
        # no batching
        if self.batch_size is None:
            return self.fn(inputRes)

        # batching
        import math, pandas as pd
        from pyterrier.model import split_df
        num_chunks = math.ceil( len(inputRes) / self.batch_size )
        iterator = split_df(inputRes, num_chunks)
        if self.verbose:
            iterator = pt.tqdm(iterator, desc="pt.apply", unit='row') 
        rtr = pd.concat([self.fn(chunk_df) for chunk_df in iterator])
        return rtr

class ApplyIndexer(Indexer):
    """
    Allows arbitrary indexer pipelines components to be written as functions.
    """
    
    def __init__(self, fn,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn

    def index(self, iter_dict):
        return self.fn(iter_dict)