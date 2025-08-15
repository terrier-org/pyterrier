from typing import Callable, Any, Union, Optional, Iterable
import itertools
import more_itertools
import numpy.typing as npt
import pandas as pd
import pyterrier as pt


class DropColumnTransformer(pt.Transformer):
    """
    This transformer drops the provided column from the input.
    """
    def __init__(self, col: str):
        """
        Instantiates a DropColumnTransformer.

        Arguments:
            col: The column to drop
        """
        self.col = col

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the column from the input DataFrame.

        Arguments:
            inp: The input DataFrame

        Returns:
            The input DataFrame with the column dropped.
        """
        return inp.drop(columns=[self.col])

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        """
        Drops the column from the input IterDict.

        Arguments:
            inp: The input IterDict

        Returns:
            The input with the column dropped.
        """
        for rec in inp:
            new_rec = rec.copy()
            new_rec.pop(self.col, None) # None ensures no error if key doesn't exist
            yield new_rec

    def __repr__(self):
        return f"pt.apply.{self.col}(drop=True)"


class ApplyByRowTransformer(pt.Transformer):
    """
    This transformer applies a function to each row in the input and assigns the result to a new column.
    """
    def __init__(self,
        col: str,
        fn: Callable[[Union[pt.model.IterDictRecord, pd.Series]], Any],
        *,
        batch_size: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Instantiates a ApplyByRowTransformer.

        Arguments:
            col: The column to assign the result of the function to
            fn: The function to apply to each row
            batch_size: The number of rows to process at once. If None, processes in one batch. This only applies
                when processing DataFrames.
            verbose: Whether to display a progress bar when processing in batch mode.
        """
        self.col = col
        self.fn = fn
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the function to each row in the input DataFrame and assigns the result to a new column.

        Arguments:
            inp: The input DataFrame

        Returns:
            The input DataFrame with the new values assign to ``col``
        """
        if self.batch_size is None:
            return self._apply_df(inp)

        # batching
        iterator = pt.model.split_df(inp, batch_size=self.batch_size)
        if self.verbose:
            iterator = pt.tqdm(iterator, desc="pt.apply", unit='row')
        return pd.concat([self._apply_df(chunk_df) for chunk_df in iterator])

    def _apply_df(self, inp: pd.DataFrame) -> pd.DataFrame:
        new_vals = inp.apply(self.fn, axis=1, result_type='reduce')
        return inp.assign(**{self.col: new_vals})

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        """
        Applies the function to each row in the input IterDict and assigns the result to a new column.

        Arguments:
            inp: The input IterDict

        Returns:
            The input IterDict with the new values assign to ``col``
        """
        for rec in inp:
            yield dict(rec, **{self.col: self.fn(rec)})

    def __repr__(self):
        return f"pt.apply.{self.col}()"


class ApplyForEachQuery(pt.Transformer):
    def __init__(self,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        add_ranks: bool = True,
        batch_size: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Instantiates a ApplyForEachQuery.

        Arguments:
            fn: Takes as input a DataFrame representing all results for a query and returns a transformed DataFrame
            add_ranks: Whether to calcualte and add ranks to the output for each query
            batch_size: The number of results per query to process at once. If None, processes in one batch per query.
            verbose: Whether to display a progress bar
        """
        self.fn = fn
        self.add_ranks = add_ranks
        self.batch_size = batch_size
        self.verbose = verbose

    def __repr__(self):
        return "pt.apply.by_query()"

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        if len(res) == 0:
            return self.fn(res)

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
                # so we must split and reconstruct the output FOR EACH QUERY
                query_dfs = []
                for qid, group in it:
                    iterator = pt.model.split_df(group, batch_size=self.batch_size)
                    query_dfs.append( pd.concat([self.fn(chunk_df) for chunk_df in iterator]) )
        except Exception as a:
            raise Exception("Problem applying %r for qid %s" % (self.fn, lastqid)) from a # %r because its a function with bytes representation (mypy)

        if self.add_ranks:
            try:
                query_dfs = [pt.model.add_ranks(df, single_query=True) for df in query_dfs]
            except KeyError as ke:
                suffix = 'Try setting add_ranks=False'
                if len(query_dfs) > 0 and 'score' not in query_dfs[0].columns:
                    suffix = 'score column not present. Set add_ranks=False'
                raise ValueError("Cannot apply add_ranks in pt.apply.by_query - " + suffix) from ke
        rtr = pd.concat(query_dfs)
        return rtr


class ApplyIterForEachQuery(pt.Transformer):
    def __init__(self,
        fn: Callable[[pt.model.IterDict], pt.model.IterDict],
        *,
        verbose=False,
        batch_size=None):
        """
        Instantiates a ApplyIterForEachQuery.

        Arguments:
            fn: Takes as input an IterDict of dictionaries representing all results for a query and returns a transformed IterDict
            batch_size: The number of results per query to process at once. If None, processes in one batch per query.
        """
        self.fn = fn
        self.verbose = verbose
        self.batch_size = batch_size

    def __repr__(self):
        return "pt.apply.by_query()"

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        if self.verbose:
            inp = pt.tqdm(inp, desc="pt.apply.by_query()")
        if self.batch_size is not None:
            for _, group in itertools.groupby(inp, key=lambda row: row['qid']):
                for batch in more_itertools.ichunked(group, self.batch_size):
                    yield from self.fn(batch)
        else:
            for _, group in itertools.groupby(inp, key=lambda row: row['qid']):
                yield from self.fn(group)


class ApplyDocumentScoringTransformer(pt.Transformer):
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
    def __init__(self,
        fn: Union[
            Callable[[pd.Series], float],
            Callable[[pd.DataFrame], Iterable[float]],
        ],
        *,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Arguments:
            fn: Either takes a panda Series for a row (representing each document in the result set), and returns the
                new float doument score. Or, if batch_size is set, takes a DataFrame, and returns a sequence of floats
                representing scores for those documents.
            batch_size: How many documents to operate on at once. If None, operates row-wise.
            verbose: Whether to display a progress bar
        """
        self.fn = fn
        self.batch_size = batch_size
        self.verbose = verbose

    def __repr__(self):
        return "pt.apply.doc_score()"

    def _transform_rowwise(self, outputRes):
        if self.verbose:
            pt.tqdm.pandas(desc="pt.apply.doc_score", unit="d")
            outputRes["score"] = outputRes.progress_apply(self.fn, axis=1).astype('float64') # type: ignore
        else:
            outputRes["score"] = outputRes.apply(self.fn, axis=1).astype('float64')
        outputRes = pt.model.add_ranks(outputRes)
        return outputRes

    def _transform_batchwise(self, outputRes):
        outputRes["score"] = self.fn(outputRes)
        outputRes["score"] = outputRes["score"].astype('float64')
        return outputRes

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        outputRes = inp.copy()
        if len(outputRes) == 0:
            outputRes["score"] = pd.Series(dtype='float64')
            return pt.model.add_ranks(outputRes)

        if self.batch_size is None:
            return self._transform_rowwise(outputRes)

        iterator = pt.model.split_df(outputRes, batch_size=self.batch_size)
        if self.verbose:
            iterator = pt.tqdm(iterator, desc="pt.apply", unit='row')
        rtr = pd.concat([self._transform_batchwise(chunk_df) for chunk_df in iterator])
        rtr = pt.model.add_ranks(rtr)
        return rtr


class ApplyDocFeatureTransformer(pt.Transformer):
    """
        Implements a transformer that can apply a function to perform feature scoring. The supplied function 
        should take as input one row, and return a numpy array for the features of the document.
        
        Usually accessed using pt.apply.doc_features()::

            def _feature_fn(row):
                return numpy.array([len(row["url"]), row["url"].count("/")])
            
            pipe = pt.terrier.Retriever(index) >> pt.apply.doc_features(_feature_fn) >> pt.LTRpipeline(xgBoost())
    """
    def __init__(self,
        fn: Callable[[Union[pd.Series, pt.model.IterDictRecord]], npt.NDArray],
        *,
        verbose: bool = False
    ):
        """
        Arguments:
            fn: Takes as input a panda Series for a row representing that document, and returns a new numpy array representing the features of that document
            verbose: Whether to display a progress bar
        """
        self.fn = fn
        self.verbose = verbose

    def __repr__(self):
        return "pt.apply.doc_features()"
    
    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        # we assume that the function can take a dictionary as well as a pandas.Series. As long as [""] notation is used
        # to access fields, both should work
        if self.verbose:
            inp = pt.tqdm(inp, desc="pt.apply.doc_features")
        for row in inp:
            row["features"] = self.fn(row)
            yield row

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        fn = self.fn
        outputRes = inp.copy()
        if self.verbose:
            pt.tqdm.pandas(desc="pt.apply.doc_features", unit="d")
            outputRes["features"] = outputRes.progress_apply(fn, axis=1) # type: ignore
        else:
            outputRes["features"] = outputRes.apply(fn, axis=1)
        return outputRes


class ApplyQueryTransformer(pt.Transformer):
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
    def __init__(self,
        fn: Callable[[Union[pd.Series, pt.model.IterDictRecord]], str],
        *,
        verbose: bool = False
    ):
        """
        Arguments:
            fn: Takes as input a panda Series for a row representing a query, and returns the new string query 
            verbose: Display a tqdm progress bar for this transformer
        """
        self.fn = fn
        self.verbose = verbose

    def __repr__(self):
        return "pt.apply.query()"

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        # we assume that the function can take a dictionary as well as a pandas.Series. As long as [""] notation is used
        # to access fields, both should work
        if self.verbose:
            inp = pt.tqdm(inp, desc="pt.apply.query")
        for row in inp:
            row = row.copy()
            if "query" in row:
                row = pt.model.push_queries_dict(row, keep_original=True)
            row["query"] = self.fn(row)
            yield row

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:    
        if "query" in inp.columns:
            # we only push if a query already exists
            outputRes = pt.model.push_queries(inp, keep_original=True)
        else:
            outputRes = inp.copy()
        try:
            if self.verbose:
                pt.tqdm.pandas(desc="pt.apply.query", unit="d")
                outputRes["query"] = outputRes.progress_apply(self.fn, axis=1) # type: ignore
            else:
                outputRes["query"] = outputRes.apply(self.fn, axis=1)
        except ValueError as ve:
            msg = str(ve)
            if "Columns must be same length as key" in msg:
                raise TypeError("Could not coerce return from pt.apply.query function into a list of strings. Check your function returns a string.") from ve
            else:
                raise ve
        return outputRes


class ApplyGenericTransformer(pt.Transformer):
    """
    Allows arbitrary pipelines components to be written as functions. The function should take as input
    a dataframe, and return a new dataframe. The function should abide by the main contracual obligations,
    e.g. updating then "rank" column.

    This class is normally accessed through pt.apply.generic()

    If you are scoring, query rewriting or calculating features, it is advised to use one of the other
    variants.

    Example::
        
        # this pipeline would remove all but the first two documents from all result sets
        lp = ApplyGenericTransformer(lambda res : res[res["rank"] < 2])

        pipe = pt.terrier.Retriever(index) >> lp

    """

    def __init__(self,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        batch_size: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Arguments:
            fn: Takes as input a panda DataFrame, and returns a new Pandas DataFrame
            batch_size: The number of rows to process at once. If None, processes in one batch.
            verbose: When in batch model, display a tqdm progress bar
        """
        self.fn = fn
        self.batch_size = batch_size
        self.verbose = verbose

    def __repr__(self):
        return "pt.apply.generic()"

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        # no batching
        if self.batch_size is None:
            return self.fn(inp)

        # batching
        iterator = pt.model.split_df(inp, batch_size=self.batch_size)
        if self.verbose:
            iterator = pt.tqdm(iterator, desc="pt.apply", unit='row')
        rtr = pd.concat([self.fn(chunk_df) for chunk_df in iterator])
        return rtr


class ApplyGenericIterTransformer(pt.Transformer): 
    """

    As per ApplyGenericTransformer, but implements transform_iter(), not transform(). The supplied function
    is assumed to take Iterable[dict] and return Iterator[dict]

    This class is normally accessed through pt.apply.generic()

    If you are scoring, query rewriting or calculating features, it is advised to use one of the other
    variants.

    """
    def __init__(self,
        fn: Callable[[pt.model.IterDict], pt.model.IterDict],
        *,
        batch_size: Optional[int] = None
    ):
        """
        Arguments:
            fn: Takes as input a panda DataFrame, and returns a new Pandas DataFrame
            batch_size: The number of rows to process at once. If None, processes in one batch.
        """
        self.fn = fn
        self.batch_size = batch_size

    def __repr__(self):
        return "pt.apply.generic()"

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        if self.batch_size is None:
            # no batching
            yield from self.fn(inp)
        else:
            for batch in more_itertools.ichunked(inp, self.batch_size):
                yield from self.fn(batch)


class ApplyIndexer(pt.Indexer):
    """
    Allows arbitrary indexer pipelines components to be written as functions.
    """
    
    def __init__(self, fn: Callable[[pt.model.IterDict], Any]):
        self.fn = fn

    def index(self, iter_dict):
        return self.fn(iter_dict)
