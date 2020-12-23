
from typing import Sequence, Union
import pyterrier as pt
import pandas as pd

def query(queries : Union[str, Sequence[str]], qids : Union[str, Sequence[str]] = None, **others) -> pd.DataFrame:
    """
        Creates a new queries dataframe. Will return a dataframe with the columns `["qid", "query"]`. 
        Any further lists in others will also be added.

        Arguments:
            queries: The search queries. Either a string, for a single query, or a sequence (e.g. list of strings)
            qids: Corresponding query ids. Either a string, for a single query, or a sequence (e.g. list of strings). Must have same length as queries.
            others: A dictionary of other attributes to add to the query dataframe           

        Examples::

            # create a dataframe with one query, qid "1"
            one_query = pt.new.queryies("what the noise was was the question")

            # create a dataframe with one query, qid "5"
            one_query = pt.new.queryies("what the noise was was the question", 5)

            # create a dataframe with two queries
            one_query = pt.new.queryies(["query text A", "query text B"], ["1", "2"])

            # create a dataframe with two queries
            one_query = pt.new.queryies(["query text A", "query text B"], ["1", "2"], categories=["catA", "catB"])

            
    """
    if type(queries) == str:
        if qids is None:
            qids = "1"
        assert type(qids) == str
        return pd.DataFrame({"qid" : [qids], "query" : [queries], **others})
    if qids is None:
        qids = map(str, range(1, len(queries)+1))
    return pd.DataFrame({"qid" : qids, "query" : queries, **others})

