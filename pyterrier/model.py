import re
import math
import itertools
from itertools import chain
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Sequence, Set, Optional, Union, overload

IterDictRecord = Dict[str, Any]
IterDict = Iterable[IterDictRecord]

# This file has useful methods for using the Pyterrier Pandas datamodel

# the first rank SHOULD be 0, see the standard "Welcome to TREC email"
FIRST_RANK = 0

# set to True to ensure that the resulting dataframe is correctly /ordered/
# as well as having correct ranks assigned
STRICT_SORT = False

def add_ranks(df : pd.DataFrame, single_query=False) -> pd.DataFrame:
    """
        Canonical method for adding a rank column which is calculated based on the score attribute
        for each query. Note that the dataframe is NOT sorted by this operation (this is defined by STRICT_SORT).
        The dataframe is modified, i.e. inplace.

        Arguments
            df: dataframe to create rank attribute for
            single_query (bool): whether the dataframe contains only a single-query or not. This method will be quicker for single-query dataframes
    """
    if "score" not in df.columns:
        raise KeyError("Must have score column to add ranks to dataframe. Found columns were %s" % str(df.columns.values.tolist()))
    df.drop(columns=["rank"], errors="ignore", inplace=True)
    if len(df) == 0:
        df["rank"] = pd.Series(index=df.index, dtype='int64')
        return df

    # remove group when single_query is True
    if single_query:
        # -1 assures that first rank will be FIRST_RANK
        df["rank"] = df["score"].rank(ascending=False, method="first").astype(int) -1 + FIRST_RANK
        if STRICT_SORT:
            df.sort_values(["rank"], ascending=True, inplace=True)
        return df

    # -1 assures that first rank will be FIRST_RANK
    df["rank"] = df.groupby("qid", sort=False)["score"].rank(ascending=False, method="first").astype(int) -1 + FIRST_RANK
    if STRICT_SORT:
        df.sort_values(["qid", "rank"], ascending=[True,True], inplace=True)
    return df

def document_columns(df : pd.DataFrame) -> Sequence[str]:
    """
        Given a dataframe, returns the names of all columns that contain attributes that are 
        concerned with a document, or the relationship between a document and a query.

        It is defined as the complement of query_columns().
    """
    return list(df.columns.difference(query_columns(df, qid=False)))

def query_columns(df : pd.DataFrame, qid=True) -> Sequence[str]:
    """
        Given a dataframe, returns the names of all columns that contain the current query or
        previous generations of the query (as performed by `push_queries()`). 

        Any saved_docs_0 column is also included.

        Arguments:
            df: Dataframe of queries to consider
            qid: whether to include the "qid" column in the returned list
    """
    columns=set(df.columns)
    rtr = []
    if qid and "qid" in columns:
        rtr.append("qid")
    if "query" in columns:
        rtr.append("query")
    for c in columns:
        if c.startswith("q") and c not in rtr:
            if c == 'qid' and not qid:
                continue
            rtr.append(c)
    import re
    saved_docs_col_re = re.compile('^stashed_results_[\\d]+')
    for c in columns:
        if saved_docs_col_re.search(c):
            rtr.append(c)
    return rtr


def push_queries(df: pd.DataFrame, *, keep_original: bool = False, inplace: bool = False) -> pd.DataFrame:
    """
        Changes a dataframe such that the "query" column becomes "query_0", and any
        "query_0" columns becames "query_1" etc.

        Arguments:
            df: Dataframe with a "query" column
            keep_original: if True, the query column is also left unchanged. Useful for client code. 
                Defaults to False.
            inplace: if False, a copy of the dataframe is returned. If True, changes are made to the
                supplied dataframe. Defaults to False. 
    """
    cols = set(df.columns)
    if "query" not in cols:
        raise KeyError(f"Expected a query column, but found {list(cols)}")
    if not inplace:
        df = df.copy()
    prev_col = 'query'
    rename_cols = {}
    for query_idx in itertools.count():
        next_col = f'query_{query_idx}'
        if prev_col in cols:
            rename_cols[prev_col] = next_col # map e.g., query_0 to be renamed to query_1
            prev_col = next_col
        else:
             break
    df = df.rename(columns=rename_cols)
    if keep_original:
        df['query'] = df["query_0"]
    return df


def push_queries_dict(inp: IterDictRecord, *, keep_original: bool = False, inplace: bool = False) -> IterDictRecord:
    """
    Works like ``push_queries`` but over a dict instead of a dataframe.
    """
    if "query" not in inp:
        raise KeyError(f"Expected a query column, but found {list(inp.keys())}")
    if not inplace:
        inp = inp.copy()
    prev_col = 'query'
    for query_idx in itertools.count():
        next_col = f'query_{query_idx}'
        if prev_col in inp:
            inp[next_col] = inp.pop(prev_col) # assign e.g., query_1 to query_0
            prev_col = next_col
        else:
             break
    if keep_original:
        inp['query'] = inp['query_0']
    return inp


def pop_queries(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """
        Changes a dataframe such that the "query_0" column becomes "query_1", and any
        "query_1" columns becames "query_0" etc. In effect, does the opposite of push_queries().
        The current "query" column is dropped.

        Arguments:
            df: Dataframe with a "query" column
            inplace: if False, a copy of the dataframe is returned. If True, changes are made to the
                supplied dataframe. Defaults to False. 
    """
    cols = set(df.columns)
    if "query_0" not in cols:
        raise KeyError(f"Expected a query_0 column, but found {list(cols)}")
    if not inplace:
        df = df.copy()
    df.drop(columns=["query"], inplace=True)
    prev_col = 'query'
    rename_cols = {}
    for query_idx in itertools.count():
        next_col = f'query_{query_idx}'
        if next_col in cols:
            rename_cols[next_col] = prev_col # map e.g., query_1 to be renamed to query_0
            prev_col = next_col
        else:
             break
    df = df.rename(columns=rename_cols)
    return df


def ranked_documents_to_queries(topics_and_res : pd.DataFrame):
    return topics_and_res[query_columns(topics_and_res, qid=True)].groupby(["qid"]).first().reset_index()


def coerce_queries_dataframe(query):
    """
    Convert either a string or a list of strings to a dataframe for use as topics in retrieval.

    Args:
        query: Either a string or a list of strings

    Returns:
        dataframe with columns=['qid','query']
    """
    if isinstance(query, pd.DataFrame):
        return query
    elif isinstance(query, str):
        return pd.DataFrame([["1", query]], columns=['qid', 'query'])
    # if queries is a list or tuple
    elif isinstance(query, list) or isinstance(query, tuple):
        # if the list or tuple is made of strings
        if query != [] and isinstance(query[0], str):
            indexed_query = []
            for i, item in enumerate(query):
                # all elements must be of same type
                assert isinstance(item, str), f"{item} is not a string"
                indexed_query.append([str(i + 1), item])
            return pd.DataFrame(indexed_query, columns=['qid', 'query'])
    # catch-all when we dont recognise the type
    raise ValueError("Could not coerce %s (type %s) into a DataFrame of queries" % (str(query), str(type(query))))


def coerce_dataframe_types(dataframe):
    """
    Changes data types to match standard values. The dataframe need not have all the columns,
    but if they are present, will cast the values to the proper types.
     - ``qid`` -> ``str``
     - ``docno`` -> ``str``
     - ``score`` -> ``float``

    Args:
        dataframe: a Pandas dataframe

    Returns:
        dataframe with data types properly set
    """
    TYPE_MAP = { # python type -> acceptable numpy types
        str: (np.dtype('O'),),
        float: (np.dtype('float32'), np.dtype('float64')),
    }
    COLUMN_MAP = { # column name -> python type
        'qid': str,
        'docno': str,
        'score': float,
    }
    for column, dtype in COLUMN_MAP.items():
        if column in dataframe.columns and dataframe[column].dtype not in TYPE_MAP[dtype]:
            dataframe[column] = dataframe[column].astype(dtype)
    return dataframe


def split_df(df : pd.DataFrame, N: Optional[int] = None, *, batch_size: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Splits a dataframe into N different chunks. Splitting will be sensitive to the primary datatype
    of the dataframe (Q,R,D).

    Either ``N`` (the number of chunks) or ``batch_size`` (the size of each chunk) should be provided (but not both).
    """
    assert (N is None) != (batch_size is None), "Either N or batch_size should be provided (and not both)"

    if N is None:
        assert batch_size is not None
        N = math.ceil(len(df) / batch_size)

    type = None
    if "qid" in df.columns:
        if "docno" in df.columns:
            type = "R"
        else:
            type = "Q"
    elif "docno" in df.columns:
        type = "D"
    else:
        raise ValueError("Dataframe is not of type D,Q,R")

    def chunks(df, n):
        """Yield successive n-sized chunks from df."""
        for i in range(0, len(df), n):
            yield df.iloc[ i: min(len(df),i + n)]
    
    if type == "Q" or type == "D":         
        splits = list( chunks(df, math.ceil(len(df)/N)))
        return splits

    rtr = []
    grouper = df.groupby("qid")
    this_group = []
    chunk_size = math.ceil(len(grouper)/N)
    for qid, group in grouper:
        this_group.append(group)
        if len(this_group) == chunk_size:
            rtr.append(pd.concat(this_group))
            this_group = []
    if len(this_group) > 0:
        rtr.append(pd.concat(this_group))
    return rtr


_ir_measures_to_pyterrier = {
    'query_id': 'qid',
    'doc_id': 'docno',
    'relevance': 'label',
}

@overload
def from_ir_measures(inp: str) -> str: ...
@overload
def from_ir_measures(inp: Dict[str, Any]) -> Dict[str, Any]: ...
@overload
def from_ir_measures(inp: List[str]) -> List[str]: ...
@overload
def from_ir_measures(inp: Set[str]) -> List[str]: ...
@overload
def from_ir_measures(inp: pd.DataFrame) -> pd.DataFrame: ...
def from_ir_measures(
    inp: Union[str, pd.DataFrame, Dict[str, Any], List[str], Set[str]],
) -> Union[str, pd.DataFrame, Dict[str, Any], List[str]]:
    """This function maps ir-measues column names to PyTerrier column names.

    It's useful when converting between PyTerrier and ir-measures data formats.

    .. seealso::
        :py:func:`pyterrier.model.to_ir_measures` for the reverse operation.
    """
    if isinstance(inp, str):
        return _ir_measures_to_pyterrier.get(inp, inp) # rename values in mapping, keep others the same
    elif isinstance(inp, pd.DataFrame):
        return inp.rename(columns=_ir_measures_to_pyterrier)
    elif isinstance(inp, dict):
        return { _ir_measures_to_pyterrier.get(k, k): v for k, v in inp.items() }
    else:
        return [_ir_measures_to_pyterrier.get(x, x) for x in inp]


_pyterrier_to_ir_measures = {
    'qid': 'query_id',
    'docno': 'doc_id',
    'label': 'relevance',
}

@overload
def to_ir_measures(inp: str) -> str: ...
@overload
def to_ir_measures(inp: Dict[str, Any]) -> Dict[str, Any]: ...
@overload
def to_ir_measures(inp: List[str]) -> List[str]: ...
@overload
def to_ir_measures(inp: pd.DataFrame) -> pd.DataFrame: ...
def to_ir_measures(
    inp: Union[str, pd.DataFrame, Dict[str, Any], List[str]],
) -> Union[str, pd.DataFrame, Dict[str, Any], List[str]]:
    """This function maps PyTerrier column names to ir-measures column names.

    It's useful when converting between PyTerrier and ir-measures data formats.

    .. seealso::
        :py:func:`pyterrier.model.from_ir_measures` for the reverse operation.
    """
    if isinstance(inp, str):
        return _pyterrier_to_ir_measures.get(inp, inp) # rename values in mapping, keep others the same
    elif isinstance(inp, pd.DataFrame):
        return inp.rename(columns=_pyterrier_to_ir_measures)
    elif isinstance(inp, dict):
        return { _pyterrier_to_ir_measures.get(k, k): v for k, v in inp.items() }
    return [_pyterrier_to_ir_measures.get(x, x) for x in inp]


def frame_info(columns : List[str]) -> Optional[Dict[str, str]]:
    """Returns a dict containing a short label and a short description for given set of columns."""
    if 'qid' in columns and 'docno' in columns and 'features' in columns:
        return {
            "label": 'R_f',
            "title": 'Result Frame with Features',
        }
    elif 'qid' in columns and 'docno' in columns:
        return {
            "label": 'R',
            "title": 'Result Frame',
        }
    elif 'qid' in columns and 'docid' in columns:
        return {
            "label": 'R',
            "title": 'Result Frame',
        }
    elif 'qanswer' in columns:
        return {
            "label": 'A',
            "title": 'Query Answer Frame',
        }
    elif 'qid' in columns:
        return {
            "label": 'Q',
            "title": 'Query Frame',
        }
    elif 'docno' in columns:
        return {
            "label": 'D',
            "title": 'Document Frame',
        }
    return None

def column_info(column: str) -> Optional[dict]:
    """Returns a dictionary with information about the specified column name."""
    if column == 'qid':
        return {
            'title': 'qid',
            'phrase': 'Query ID',
            'short_desc': 'ID of query in frame',
            'type': str,
        }
    if column == 'docno':
        return {
            'title': 'docno',
            'phrase': 'External Document ID',
            'short_desc': 'String ID of document in collection',
            'type': str,
        }
    if column == 'docid':
        return {
            'title': 'docid',
            'phrase': 'Internal Document ID',
            'short_desc': 'Integer ID of document in a specific index',
            'type': int,
        }
    if column == 'score':
        return {
            'title': 'score',
            'short_desc': 'Ranking score of document to query (higher=better)',
            'type': float,
        }
    if column == 'rank':
        return {
            'title': 'rank',
            'short_desc': 'Ranking order of document to query (lower=better)',
            'type': int,
        }
    if column == 'query':
        return {
            'title': 'query',
            'short_desc': 'Query text',
            'type': str,
        }
    if re.match(r'^query_[0-9]+$', column):
        return {
            'title': str(column),
            'short_desc': 'Stashed query text',
            'type': str,
        }
    if column == 'text':
        return {
            'title': "text",
            'short_desc': 'Document text',
            'type': str,
        }
    if column == 'title':
        return {
            'title': "title",
            'short_desc': 'Document title',
            'type': str,
        }
    if column == 'qanswer':
        return {
            'title': "qanswer",
            'short_desc': 'Answer to the query',
            'type': str,
        }
    if column == 'qcontext':
        return {
            'title': "qcontext",
            'short_desc': 'Context to the query',
            'type': str,
        }
    if column == 'features':
        return {
            'title': "features",
            'short_desc': 'Feature array for learning-to-rank',
            'type': np.array,
        }
    if column == 'query_vec':
        return {
            'title': "query_vec",
            'short_desc': 'Dense query vector',
            'type': np.array,
        }
    if column == 'doc_vec':
        return {
            'title': "doc_vec",
            'short_desc': 'Dense document vector',
            'type': np.array,
        }
    if column == 'query_toks':
        return {
            'title': "query_toks",
            'short_desc': 'Sparse query vector',
            'type': dict,
        }
    if column == 'toks':
        return {
            'title': "toks",
            'short_desc': 'Sparse document vector',
            'type': dict,
        }
    return None


class DataFrameBuilder:
    """Utility to build a DataFrame from a sequence of dictionaries.

    The dictionaries must have the same keys, and the values must be either scalars, or lists of the same length.

    Example::

        builder = pt.model.DataFrameBuilder(['docno', 'score'])
        for qid, results in retrieve_results():
            builder.extend({
                '_index': qid_index,
                'docno': results['docno'],
                'score': results['score'],
            })
        df = builder.to_df(merge_on_index=queries_df)
    """
    def __init__(self, columns: List[str]):
        """Create a DataFrameBuilder with the given columns.

        Args:
            columns: the columns of the resulting DataFrame, required to be present in each
                call to :meth:`~pyterrier.model.DataFrameBuilder.extend`.
        """
        if '_index' not in columns:
            columns = ['_index'] + columns
        self._data: Dict[str, list] = {c: [] for c in columns}
        self._auto_index = 0

    def extend(self, values: Dict[str, Any]) -> None:
        """Add a dictionary of values to the DataFrameBuilder.

        Args:
            values: a dictionary of values to add to the DataFrameBuilder. The keys must be the same as the columns
                provided to the constructor, and the values must be either scalars, or lists (all of the same length).
        """
        if '_index' not in values.keys():
            values['_index'] = self._auto_index
            self._auto_index += 1
        assert all(c in values.keys() for c in self._data), f"all columns must be provided: {list(self._data)}"
        lens = {k: len(v) for k, v in values.items() if hasattr(v, '__len__') and not isinstance(v, str) and len(v) > 1}
        if any(lens):
            first_len = list(lens.values())[0]
        else:
            first_len = 1  # if nothing has a len, everything is given a length of 1
        assert all(i == first_len for i in lens.values()), f"all values must have the same length {lens}"
        for k, v in values.items():
            if k not in lens:
                if isinstance(v, (tuple, list)) and len(v) == 1:
                    self._data[k].append(v * first_len)
                else:
                    self._data[k].append([v] * first_len)
            elif isinstance(v, pd.Series):
                self._data[k].append(v.values)
            else:
                self._data[k].append(v)

    def to_df(self, merge_on_index: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Convert the DataFrameBuilder to a DataFrame.

        Args:
            merge_on_index: an optional DataFrame to merge the resulting DataFrame on.
                Columns from ``merge_on_index`` come first in the result.

        Returns:
            A DataFrame with the values added to the DataFrameBuilder.
        """
        result = pd.DataFrame({
            k: (np.concatenate(v)
                if len(v) > 0 and not isinstance(v[0][0], np.ndarray) else
                list(chain.from_iterable(v))
               )
            for k, v in self._data.items()
        })
        if merge_on_index is not None:
            merge_on_index = merge_on_index.reset_index(drop=True)
            result = result.assign(**{
                col: merge_on_index[col].iloc[result['_index']].values
                for col in merge_on_index.columns
                if col not in result.columns
            })
            merge_columns = set(merge_on_index.columns)
            column_order = list(merge_on_index.columns) + [c for c in result.columns if c not in merge_columns]
            result = result[column_order]
        result = result.drop(columns=['_index'])
        return result
