import pandas as pd

# This file has useful methods for using the Pyterrier Pandas datamodel

# the first rank SHOULD be 0, see the standard "Welcome to TREC" email
FIRST_RANK = 0


def push_query(rtr, newquery=None):
    """
        For the purposes of query rewriting, this method renames the current `query` attribute to `query_prev`;
        Any *previous* `query_prev` is renamed to `query_prev_prev` etc.

        If the replacement queries are known, the newquery attribute can be used to populate the new column.
    """
    renames = []
    col_set = set(rtr.columns)
    inputCol = "query"
    while True:
        outputCol = inputCol + "_prev"
        renames.append( (inputCol,outputCol) )
        if outputCol in col_set:
            inputCol = outputCol
        else:
            break
    for (src,dest) in reversed(renames):
        rtr = rtr.rename(columns={src:dest})
    if newquery is not None:
        rtr["query"] = newquery
    return rtr
    
def pop_query(rtr):
    """
        Does the opposite of push_query(), in that `"query_prev"` attributes becomes the `"query"` attribute.
        Similarly `"query_prev_prev"` will become  `"query_prev"` etc. The current query is discarded. 
    """
    renames = []
    col_set = set(rtr.columns)
    outputCol = "query"
    while True:
        inputCol = outputCol + "_prev"
        renames.append( (inputCol,outputCol) )
        if inputCol in col_set:
            outputCol = inputCol
        else:
            break
    for (src,dest) in reversed(renames):
        rtr = rtr.rename(columns={src:dest})
    return rtr

def add_ranks(rtr):
    rtr.drop(columns=["rank"], errors="ignore", inplace=True)
    if len(rtr) == 0:
        rtr["rank"] = pd.Series(index=rtr.index, dtype='int64')
        return rtr
    # -1 assures that first rank for each query will be 0
    rtr["rank"] = rtr.groupby("qid").rank(ascending=False, method="first")["score"].astype(int) -1
    return rtr

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
