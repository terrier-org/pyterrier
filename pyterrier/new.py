
from typing import Sequence, Union, Optional, cast, Iterable, Any, Dict, List
from itertools import chain
import numpy as np
import pandas as pd
from .model import add_ranks

def empty_Q() -> pd.DataFrame:
    """
        Returns an empty dataframe with columns `["qid", "query"]`.
    """
    return pd.DataFrame(columns=["qid", "query"])

def queries(queries : Union[str, Sequence[str]], qid : Optional[Union[str, Iterable[str]]] = None, **others) -> pd.DataFrame:
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
    if isinstance(queries, str):
        if qid is None:
            qid = "1"
        assert isinstance(qid, str)
        return pd.DataFrame({"qid" : [qid], "query" : [queries], **others})
    if qid is None:
        qid = cast(Iterable[str], map(str, range(1, len(queries)+1))) # noqa: PT100 (this is typing.cast, not jinus.cast)
    return pd.DataFrame({"qid" : qid, "query" : queries, **others})

Q = queries

def empty_R() -> pd.DataFrame:
    """
        Returns an empty dataframe with columns `["qid", "query", "docno", "rank", "score"]`.
    """
    return pd.DataFrame([[]], columns=["qid", "query", "docno", "rank", "score"])

def ranked_documents(
        scores : Sequence[Sequence[float]], 
        qid : Optional[Sequence[str]] = None, 
        docno : Optional[Sequence[Sequence[str]]] = None, 
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
    if len(scores) == 0:
        return empty_R()
    rtr = None
    if isinstance(scores[0], list):
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


class DataFrameBuilder:
    """Utility to build a DataFrame from a sequence of dictionaries.

    The dictionaries must have the same keys, and the values must be either scalars, or lists of the same length.

    Example::

        builder = pt.new.DataFrameBuilder(['docno', 'score'])
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
                call to :meth:`~pyterrier.new.DataFrameBuilder.extend`.
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
