import re
import math
import itertools
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Sequence, Optional, Union, overload, Tuple

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
def from_ir_measures(inp: pd.DataFrame) -> pd.DataFrame: ...
def from_ir_measures(
    inp: Union[str, pd.DataFrame, Dict[str, Any], List[str]],
) -> Union[str, pd.DataFrame, Dict[str, Any], List[str]]:
    """This function maps ir-measues column names to PyTerrier column names.

    It's useful when converting between PyTerrier and ir-measures data formats.

    .. seealso::
        :py:func:`pyterrier.utils.to_ir_measures` for the reverse operation.
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
        :py:func:`pyterrier.utils.from_ir_measures` for the reverse operation.
    """
    if isinstance(inp, str):
        return _pyterrier_to_ir_measures.get(inp, inp) # rename values in mapping, keep others the same
    elif isinstance(inp, pd.DataFrame):
        return inp.rename(columns=_pyterrier_to_ir_measures)
    elif isinstance(inp, dict):
        return { _pyterrier_to_ir_measures.get(k, k): v for k, v in inp.items() }
    return [_pyterrier_to_ir_measures.get(x, x) for x in inp]

def frame_info(columns : List[str]) -> Tuple[str,str]:
    """Returns a tuple containing a short label and a name for given set of columns."""
    if len(columns) == 0:
        df_label = '?'
        df_label_long = 'Unknown Frame'
    elif 'qid' in columns and 'docno' in columns and 'features' in columns:
        df_label = 'R<sub>f</sub>'
        df_label_long = 'Result Frame with Features'
    elif 'qid' in columns and 'docno' in columns:
        df_label = 'R'
        df_label_long = 'Result Frame'
    elif 'qanswer' in columns:
        df_label = 'A'
        df_label_long = 'Query Answer Frame'
    elif 'qcontext' in columns:
        df_label = 'Q<sub>c</sub>'
        df_label_long = 'Query Context Frame'
    elif 'qid' in columns:
        df_label = 'Q'
        df_label_long = 'Query Frame'
    elif 'docno' in columns:
        df_label = 'D'
        df_label_long = 'Document Frame'
    return df_label, df_label_long

def column_info(column: str) -> Optional[dict]:
    """Returns a dictionary with information about the specified column name."""
    if column == 'qid':
        return {
            'phrase': 'Query ID',
            'short_desc': 'ID of query in frame',
            'type': str,
        }
    if column == 'docno':
        return {
            'phrase': 'External Document ID',
            'short_desc': 'String ID of document in collection',
            'type': str,
        }
    if column == 'docid':
        return {
            'phrase': 'Internal Document ID',
            'short_desc': 'Integer ID of document in a specific index',
            'type': int,
        }
    if column == 'score':
        return {
            'short_desc': 'Ranking score of document to query (higher=better)',
            'type': float,
        }
    if column == 'rank':
        return {
            'short_desc': 'Ranking order of document to query (lower=better)',
            'type': int,
        }
    if column == 'query':
        return {
            'short_desc': 'Query text',
            'type': str,
        }
    if re.match(r'^query_[0-9]+$', column):
        return {
            'short_desc': 'Stashed query text',
            'type': str,
        }
    if column == 'text':
        return {
            'short_desc': 'Document text',
            'type': str,
        }
    if column == 'title':
        return {
            'short_desc': 'Document title',
            'type': str,
        }
    if column == 'qanswer':
        return {
            'short_desc': 'Answer to the query',
            'type': str,
        }
    if column == 'qcontext':
        return {
            'short_desc': 'Context to the query',
            'type': str,
        }
    if column == 'features':
        return {
            'short_desc': 'Feature array for learning-to-rank',
            'type': np.array,
        }
    return