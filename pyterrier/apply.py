from typing import Callable, Any
from .transformer import ApplyDocumentScoringTransformer, ApplyQueryTransformer, ApplyDocFeatureTransformer, ApplyGenericTransformer, TransformerBase
from nptyping import NDArray
import numpy as np
import pandas as pd

def query(fn : Callable[..., str], *args, **kwargs) -> TransformerBase:
    """
        Create a transformer that takes as input a query, and applies a supplied function to compute a new query formulation.

        The supplied function is called once for each query, and must return a string containing the new query formulation.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query.

        Examples::

            # this will remove pre-defined stopwords from the query
            stops=set(["and", "the"])

            # a naieve function to remove stopwords
            def _remove_stops(q):
                terms = q["query"].split(" ")
                terms = [t for t in terms if not t in stops ]
                return " ".join(terms)

            # a query rewriting transformer applying _remove_stops
            p1 = pt.apply.query(_remove_stops) >> pt.BatchRetrieve(index, wmodel="DPH")

            # an equivalent query rewriting transformer using an anonymous lambda function
            p2 = pt.apply.query(
                    lambda q :  " ".join([t for t in q["query"].split(" ") if t not in stops ])
                ) >> pt.BatchRetrieve(index, wmodel="DPH")

    """
    return ApplyQueryTransformer(fn, *args, **kwargs)

def doc_score(fn : Callable[..., float], *args, **kwargs) -> TransformerBase:
    """
        Create a transformer that takes as input query document pairs, and applies a supplied function to compute a new score.
        Ranks are automatically computed.

        The supplied function is called once for each document, and must return a float containing the new score for that document.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query and document.

        Example::

            # this transformer will subtract 5 from the score of each document
            p = pt.BatchRetrieve(index, wmodel="DPH") >> 
                pt.apply.doc_score(lambda doc : doc["score"] -5)

    """
    return ApplyDocumentScoringTransformer(fn, *args, **kwargs)

def doc_features(fn : Callable[..., NDArray[Any]], *args, **kwargs) -> TransformerBase:
    """
        Create a transformer that takes as input a query, and applies a supplied function to compute feature scores. 

        The supplied function is called once for each document, must each time return a 1D numpy array.
        Each time it is called, the function is supplied with a Panda Series representing the attributes of the query and document.

        Example::

            # this transformer will compute the character and number of word in each document retrieved
            # using the contents of the document obtained from the MetaIndex

            def _features(row):
                docid = row["docid"]
                content = index.getMetaIndex.getItem("text", docid)
                f1 = len(content)
                f2 = len(content.split(" "))
                return np.array([f1, f2])

            p = bt.BatchRetrieve(index, wmodel="BM25") >> 
                pt.apply.doc_features(_features )

    """
    return ApplyDocFeatureTransformer(fn, *args, **kwargs)

def generic(fn : Callable[[pd.DataFrame], pd.DataFrame], *args, **kwargs) -> TransformerBase:
    """
        Create a transformer that changes the input dataframe to another dataframe in an unspecified way.

        The supplied function is called once for an entire result set as a dataframe (which may contain one of more queries).
        Each time it should return a new dataframe. The returned dataframe should abide by the general PyTerrier Data Model,
        for instance updating the rank column if the scores are amended.

        Example::

            # this transformer will remove all documents at rank greater than 2.

            # this pipeline would remove all but the first two documents from a result set
            pipe = pt.BatchRetrieve(index) >> pt.apply.generic(lambda res : res[res["rank"] < 2])

    """
    return ApplyDocFeatureTransformer(fn, *args, **kwargs)
