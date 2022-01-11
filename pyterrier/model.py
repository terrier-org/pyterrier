import numpy as np
import pandas as pd
from typing import List, Sequence

# This file has useful methods for using the Pyterrier Pandas datamodel

# the first rank SHOULD be 0, see the standard "Welcome to TREC email"
FIRST_RANK = 0

# set to True to ensure that the resulting dataframe is correctly /ordered/
# as well as having correct ranks assigned
STRICT_SORT = False

def add_ranks(rtr : pd.DataFrame) -> pd.DataFrame:
    """
        Canonical method for adding a rank column which is calculated based on the score attribute
        for each query. Note that the dataframe is NOT sorted by this operation.

        Arguments
            df: dataframe to create rank attribute for
    """
    rtr.drop(columns=["rank"], errors="ignore", inplace=True)
    if len(rtr) == 0:
        rtr["rank"] = pd.Series(index=rtr.index, dtype='int64')
        return rtr

    # -1 assures that first rank will be FIRST_RANK
    rtr["rank"] = rtr.groupby("qid", sort=False)["score"].rank(ascending=False, method="first").astype(int) -1 + FIRST_RANK
    if STRICT_SORT:
        rtr.sort_values(["qid", "rank"], ascending=[True,True], inplace=True)
    return rtr

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

def _last_query(df : pd.DataFrame) -> int:
    """
        Returns the index of the last query column.
        Given a dataframe, returns:

            -1 is there is only a query column
            0 query_0 exists
            1 query_1 exists
            etc
        
    """
    last = -1
    columns = set(df.columns)
    while True:
        if not "query_%d" % (last+1) in columns:
            break
        last+=1
        
    #print("input %s rtr %d" % (str(columns), last))
    return last

def push_queries(df: pd.DataFrame, keep_original:bool=False, inplace:bool=False) -> pd.DataFrame:
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
    if "query" not in df.columns:
        raise TypeError("Expected a query column, but found %s" % df.columns) 
    df = df if inplace else df.copy()
    last_col = _last_query(df)
    rename_cols={}
    if last_col >= 0: 
        rename_cols = { "query_%d" % col_index : "query_%d" % (col_index+1) for col_index in range(0, last_col+1) }
    rename_cols["query"] = "query_0"
    df = df.rename(columns=rename_cols)
    if keep_original:
        df['query'] = df["query_0"]
    return df

def pop_queries(df: pd.DataFrame, inplace:bool=False):
    """
        Changes a dataframe such that the "query_0" column becomes "query_1", and any
        "query_1" columns becames "query_0" etc. In effect, does the opposite of push_queries().
        The current "query" column is dropped.

        Arguments:
            df: Dataframe with a "query" column
            inplace: if False, a copy of the dataframe is returned. If True, changes are made to the
                supplied dataframe. Defaults to False. 
    """
    if "query_0" not in df.columns:
        raise TypeError("Expected a query_0 column, but found %s" % df.columns) 
    last_col = _last_query(df)
    df = df if inplace else df.copy()
    df.drop(columns=["query"], inplace=True)
    rename_cols = { "query_%d" % (col_index+1) : "query_%d" % (col_index) for col_index in range(0, last_col+1) }
    rename_cols["query_0"] = "query"
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


def split_df(df : pd.DataFrame, N) -> List[pd.DataFrame]:
    """
    splits a dataframe into N different chunks. Splitting will be sensitive to the primary datatype
    of the dataframe (Q,R,D).
    """
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
    
    from math import ceil

    def chunks(df, n):
        """Yield successive n-sized chunks from df."""
        for i in range(0, len(df), n):
            yield df.iloc[ i: min(len(df),i + n)]
    
    if type == "Q" or type == "D":         
        splits = list( chunks(df, ceil(len(df)/N)))
        return splits

    rtr = []
    grouper = df.groupby("qid")
    this_group = []
    chunk_size = ceil(len(grouper)/N)
    for qid, group in grouper:
        this_group.append(group)
        if len(this_group) == chunk_size:
            rtr.append(pd.concat(this_group))
            this_group = []
    if len(this_group) > 0:
        rtr.append(pd.concat(this_group))
    return rtr
    
