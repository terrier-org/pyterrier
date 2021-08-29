
from typing import Sequence, Union
import pyterrier as pt
import pandas as pd
from .model import add_ranks

def empty_Q() -> pd.DataFrame:
    """
        Returns an empty dataframe with columns `["qid", "query"]`.
    """
    return pd.DataFrame(columns=["qid", "query"])

def queries(queries : Union[str, Sequence[str]], qid : Union[str, Sequence[str]] = None, **others) -> pd.DataFrame:
    """
        Creates a new queries dataframe. Will return a dataframe with the columns `["qid", "query"]`. 
        Any further lists in others will also be added.

        Arguments:
            queries: The search queries. Either a string, for a single query, or a sequence (e.g. list of strings)
            qids: Corresponding query ids. Either a string, for a single query, or a sequence (e.g. list of strings). Must have same length as queries.
            others: A dictionary of other attributes to add to the query dataframe           

        Examples::

            # create a dataframe with one query, qid "1"
            one_query = pt.new.queries("what the noise was was the question")

            # create a dataframe with one query, qid "5"
            one_query = pt.new.queries("what the noise was was the question", 5)

            # create a dataframe with two queries
            one_query = pt.new.queries(["query text A", "query text B"], ["1", "2"])

            # create a dataframe with two queries
            one_query = pt.new.queries(["query text A", "query text B"], ["1", "2"], categories=["catA", "catB"])
  
    """
    if type(queries) == str:
        if qid is None:
            qid = "1"
        assert type(qid) == str
        return pd.DataFrame({"qid" : [qid], "query" : [queries], **others})
    if qid is None:
        qid = map(str, range(1, len(queries)+1))
    return pd.DataFrame({"qid" : qid, "query" : queries, **others})

Q = queries

def empty_R() -> pd.DataFrame:
    """
        Returns an empty dataframe with columns `["qid", "query", "docno", "rank", "score"]`.
    """
    return pd.DataFrame([[]], columns=["qid", "query", "docno", "rank", "score"])

def ranked_documents(
        scores : Sequence[Sequence[float]], 
        qid : Sequence[str] = None, 
        docno=None, 
        **others) -> pd.DataFrame:
    """
        Creates a new ranked documents dataframe. Will return a dataframe with the columns `["qid", "docno", "score", "rank"]`. 
        Any further lists in others will also be added.

        Arguments:
            scores: The scores of the retrieved documents. Must be a list of lists.
            qid: Corresponding query ids. Must have same length as the first dimension of scores.
                If omitted, documents, qids are computed as strings starting from "1"
            docno: Corresponding docnos.  Must have same length as the first dimension of scores 
                and each 2nd dimension must be the same as the number of documents retrieved.
                If omitted, docnos are computed as strings starting from "d1" for each query.
            others: A dictionary of other attributes to add to the query dataframe.         

        Examples::

            # one query, one document
            R1 = pt.new.ranked_documents([[1]])

            # one query, two documents
            R2 = pt.new.ranked_documents([[1, 2]])

            # two queries, one documents each
            R3 = pt.new.ranked_documents([[1], [2]])

            # one query, one document, qid specified
            R4 = pt.new.ranked_documents([[1]], qid=["q100"])

            # one query, one document, qid and docno specified
            R5 = pt.new.ranked_documents([[1]], qid=["q100"], docno=[["d20"]])

    """
    from itertools import chain
    import numpy as np
    if len(scores) == 0:
        return empty_R()
    rtr = None
    if type(scores[0]) == list:
        # multiple queries
        if qid is None:
            qid = list(map(str, range(1, len(scores)+1)))
        else:
            assert len(qid) == len(scores)
        qid = list(chain.from_iterable([ [q] * len(score_array) for q, score_array in zip(qid, scores) ]))
        
        if docno is None:
            docno = [ list(map(lambda i: "d%d" % i, range(1, len(score_array)+1) ) ) for score_array in scores ]
        else:
            assert len(docno) == len(scores)
        
        from itertools import chain
        rtr = pd.DataFrame(list(chain.from_iterable(scores)), columns=["score"]) 
        
        rtr["docno"] = list(chain.from_iterable(docno))
        rtr["qid"] = qid
        #construct = {"qid" : qid, "docno" : docno, "score" : scores}
        for k, v in others.items():
            rtr[k] = list(chain.from_iterable(v))
            #assert len(v) == len(scores), "kwarg %s had length %d but was expected to have length %d" % (k, len(v), len(scores))
            #construct[k] = np.array( v ).flatten()
        #rtr = pd.DataFrame(construct)        
    else:
        raise ValueError("We assume multiple documents, for now")
    return add_ranks(rtr)

R = ranked_documents