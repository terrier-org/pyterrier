import math
import itertools
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Sequence, Optional, Union, Tuple

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
    import re
    query_col_re = re.compile('^query_[\\d]+')
    for c in columns:
        if query_col_re.search(c):
            rtr.append(c)
    saved_docs_col_re = re.compile('^stashed_results_[\\d]+')
    for c in columns:
        if saved_docs_col_re.search(c):
            rtr.append(c)
    return rtr


def push_columns(
    df: pd.DataFrame,
    *,
    keep_original: bool = False,
    inplace: bool = False,
    base_column: str = "query",
) -> pd.DataFrame:
    """
    Changes a dataframe such that the selected column becomes "<column>_0", and any
    "<column>_0" columns becomes "<column>_1" etc.

    Arguments:
        df: Dataframe with a "<column>" column
        keep_original: if True, the <column> column is also left unchanged. Useful for client code.
            Defaults to False.
        inplace: if False, a copy of the dataframe is returned. If True, changes are made to the
            supplied dataframe. Defaults to False.
    """
    cols = set(df.columns)
    if base_column not in cols:
        raise KeyError(f"Expected a {base_column} column, but found {list(cols)}")
    if not inplace:
        df = df.copy()
    prev_col = base_column
    rename_cols = {}
    for query_idx in itertools.count():
        next_col = f"{base_column}_{query_idx}"
        if prev_col in cols:
            rename_cols[prev_col] = next_col
            prev_col = next_col
        else:
            break
    # apply renaming in-place or on the copy
    df.rename(columns=rename_cols, inplace=True)
    if keep_original:
        df[base_column] = df[f"{base_column}_0"]
    return df


def push_columns_dict(
    inp: Union[Iterable[dict], dict],
    keep_original: bool = False,
    base_column: str = "query",
) -> Union[Iterable[dict], dict]:
    def per_element(i: dict):
        cols = i.keys()
        if base_column not in cols:
            raise KeyError(f"Expected a {base_column} column, but found {list(cols)}")
        prev_col = base_column
        rename_cols = {}
        for query_idx in itertools.count():
            next_col = f"{base_column}_{query_idx}"
            if prev_col in cols:
                rename_cols[prev_col] = next_col
                prev_col = next_col
            else:
                break

        renamed = {}
        for k, v in i.items():
            if k in rename_cols:
                renamed[rename_cols[k]] = v
            else:
                renamed[k] = v

        if keep_original:
            renamed[base_column] = renamed[f"{base_column}_0"]

        return renamed

    if isinstance(inp, dict):
        return per_element(inp)
    return [*map(per_element, inp)]


def find_maximum_push(inp: pd.DataFrame, base_column: str = "query") -> Tuple[str, int]:
    columns = inp.columns
    maxcol = None
    maxval = -1
    for col in columns:
        if col.startswith(f"{base_column}_"):
            val = int(col.split("_")[1])
            if val > maxval:
                maxval = val
                maxcol = col
    return maxcol, maxval


def find_maximum_push_dict(inp: Union[Iterable[dict], dict], base_column: str = "query") -> Tuple[str, int]:
    def per_element(i: dict):
        cols = i.keys()
        maxcol = None
        maxval = -1
        for col in cols:
            if col.startswith(f"{base_column}_"):
                val = int(col.split("_")[1])
                if val > maxval:
                    maxval = val
                    maxcol = col
        return maxcol, maxval

    if isinstance(inp, dict):
        return per_element(inp)
    return map(per_element, inp)


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
    assert not inplace, "push_queries no longer supports inplace"
    return push_columns(df, keep_original=keep_original, inplace=inplace, base_column="query")


def push_queries_dict(inp: IterDictRecord, *, keep_original: bool = False, inplace: bool = False) -> IterDictRecord:
    """
    Works like ``push_queries`` but over a dict instead of a dataframe.
    """
    assert not inplace, "push_queries_dict does not support inplace"
    return push_columns_dict(inp, keep_original=keep_original, base_column="query")


def pop_columns(df: pd.DataFrame, *, base_column="query") -> pd.DataFrame:
    cols = set(df.columns)
    if base_column + "_0" not in cols:
        raise KeyError(f"Expected a {base_column}_0 column, but found {list(cols)}")
    df = df.copy()
    df.drop(columns=[base_column], inplace=True)
    prev_col = base_column
    rename_cols = {}
    for query_idx in itertools.count():
        next_col = f'{base_column}_{query_idx}'
        if next_col in cols:
            rename_cols[next_col] = prev_col # map e.g., query_1 to be renamed to query_0
            prev_col = next_col
        else:
             break
    df = df.rename(columns=rename_cols)
    return df

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
    assert not inplace
    return pop_columns(df, base_column="query")

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
