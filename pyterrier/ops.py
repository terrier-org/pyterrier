from .transformer import Transformer, Estimator, get_transformer
from .model import add_ranks
from collections import deque
from warnings import warn
from typing import Iterable, List, Optional, Protocol, runtime_checkable
from itertools import chain
import pandas as pd

class NAryTransformerBase(Transformer):
    """
        A base class for all operator transformers that can combine the input of 2 or more transformers. 
    """
    def __init__(self, *transformers: Transformer):
        assert len(transformers) > 0
        # Flatten out multiple layers of the same NAryTransformer into one
        transformers = chain.from_iterable(
            (t._transformers if isinstance(t, type(self)) else [t])
            for t in transformers
        )
        # Coerce datatypes
        self._transformers = tuple(get_transformer(x, stacklevel=6) for x in transformers)

    def __getitem__(self, number) -> Transformer:
        """
            Allows access to the ith transformer.
        """
        return self._transformers[number]

    def __len__(self) -> int:
        """
            Returns the number of transformers in the operator.
        """
        return len(self._transformers)

class SetUnionTransformer(Transformer):
    """      
        This operator makes a retrieval set that includes documents that occur in the union (either) of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [1, "text1", doc1, 0.42] and right = [1, "text1", doc2, 0.24]. 
        Then, left | right will be a dataframe with only the columns [qid, query, docno] and two rows = [[1, "text1", doc1], [1, "text1", doc2]].
                
        In case of duplicated both containing (qid, docno), only the first occurrence will be used.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        import pandas as pd
        assert isinstance(res1, pd.DataFrame)
        assert isinstance(res2, pd.DataFrame)
        rtr = pd.concat([res1, res2])
        
        on_cols = ["qid", "docno"]     
        rtr = rtr.drop_duplicates(subset=on_cols)
        rtr = rtr.sort_values(by=on_cols)
        rtr.drop(columns=["score", "rank"], inplace=True, errors='ignore')
        return rtr

class SetIntersectionTransformer(Transformer):
    """
        This operator makes a retrieval set that only includes documents that occur in the intersection of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [[1, "text1", doc1, 0.42]] (one row) and right = [[1, "text1", doc1, 0.24],[1, "text1", doc2, 0.24]] (two rows).
        Then, left & right will be a dataframe with only the columns [qid, query, docno] and one single row = [[1, "text1", doc1]].
                
        For columns other than (qid, docno), only the left value will be used.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)  
        
        on_cols = ["qid", "docno"]
        rtr = res1.merge(res2, on=on_cols, suffixes=('','_y'))
        rtr.drop(columns=["score", "rank", "score_y", "rank_y", "query_y"], inplace=True, errors='ignore')
        for col in rtr.columns:
            if not '_y' in col:
                continue
            new_name = col.replace('_y', '')
            if new_name in rtr.columns:
                # duplicated column, drop
                rtr.drop(columns=[col], inplace=True)
                continue
            # column only from RHS, keep, but rename by removing '_y' suffix
            rtr.rename(columns={col:new_name}, inplace=True)

        return rtr

class CombSumTransformer(Transformer):
    """
        Adds the scores of documents from two different retrieval transformers.
        Documents not present in one transformer are given a score of 0.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    def transform(self, topics_and_res):
        res1 = self.left.transform(topics_and_res)
        res2 = self.right.transform(topics_and_res)
        both_cols = set(res1.columns) & set(res2.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        merged = res1.merge(res2, on=["qid", "docno"], suffixes=[None, "_r"], how='outer')
        merged["score"] = merged["score"].fillna(0) + merged["score_r"].fillna(0)
        merged = merged.drop(columns=["%s_r" % col for col in both_cols])
        merged = add_ranks(merged)
        return merged

class ConcatenateTransformer(Transformer):
    epsilon = 0.0001

    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    def transform(self, topics_and_res):
        import pandas as pd
        # take the first set as the top of the ranking
        res1 = self.left.transform(topics_and_res)
        # identify the lowest score for each query
        last_scores = res1[['qid', 'score']].groupby('qid').min().rename(columns={"score" : "_lastscore"})

        # the right hand side will provide the rest of the ranking        
        res2 = self.right.transform(topics_and_res)

        
        intersection = pd.merge(res1[["qid", "docno"]], res2[["qid", "docno"]].reset_index())
        remainder = res2.drop(intersection["index"])

        # we will append documents from remainder to res1
        # but we need to offset the score from each remaining document based on the last score in res1
        # explanation: remainder["score"] - remainder["_firstscore"] - self.epsilon ensures that the
        # first document in remainder has a score of -epsilon; we then add the score of the last document
        # from res1
        first_scores = remainder[['qid', 'score']].groupby('qid').max().rename(columns={"score" : "_firstscore"})

        remainder = remainder.merge(last_scores, on=["qid"]).merge(first_scores, on=["qid"])
        remainder["score"] = remainder["score"] - remainder["_firstscore"] + remainder["_lastscore"] - self.epsilon
        remainder = remainder.drop(columns=["_lastscore",  "_firstscore"])

        # now bring together and re-sort
        # this sort should match trec_eval
        rtr = pd.concat([res1, remainder]).sort_values(by=["qid", "score", "docno"], ascending=[True, False, True]) 

        # recompute the ranks
        rtr = add_ranks(rtr)
        return rtr

class ScalarProductTransformer(Transformer):
    """
        Multiplies the retrieval score by a scalar
    """
    def __init__(self, scalar: float):
        self.scalar = scalar

    def transform(self, inp):
        out = inp.assign(score=inp["score"] * self.scalar)
        if self.scalar < 0:
            out = add_ranks(out)
        return out

    def __repr__(self):
        return f'ScalarProductTransformer({self.scalar!r})'

class RankCutoffTransformer(Transformer):
    """
        Filters the input by rank<k for each query in the input
    """
    def __init__(self, k: int = 1000):
        self.k = k

    def transform(self, inp):
        assert 'rank' in inp.columns, "require rank to be present in the result set"
        res = inp[inp["rank"] < self.k]
        res = res.reset_index(drop=True)
        return res

    def __repr__(self):
        return f'RankCutoffTransformer({self.k!r})'

    def fuse_left(self, left: Transformer) -> Optional[Transformer]:
        # If the preceding component supports a native rank cutoff (via fuse_rank_cutoff), apply it.
        if isinstance(left, SupportsFuseRankCutoff):
            return left.fuse_rank_cutoff(self.k)

class FeatureUnionPipeline(NAryTransformerBase):
    """
        Implements the feature union operator.

        Example::
            cands = pt.terrier.Retriever(index wmodel="BM25")
            pl2f = pt.terrier.Retriever(index wmodel="PL2F")
            bm25f = pt.terrier.Retriever(index wmodel="BM25F")
            pipe = cands >> (pl2f ** bm25f)
    """
    def transform(self, inputRes):
        if not "docno" in inputRes.columns and "docid" in inputRes.columns:
            raise ValueError("FeatureUnion operates as a re-ranker, but input did not have either "
                "docno or docid columns, found columns were %s" %  str(inputRes.columns))

        num_results = len(inputRes)
        import numpy as np

        # a parent could be a feature union, but it still passes the inputRes directly, so inputRes should never have a features column
        if "features" in inputRes.columns:
            raise ValueError("FeatureUnion operates as a re-ranker. They can be nested, but input "
                "should not contain a features column; found columns were %s" %  str(inputRes.columns))
        
        all_results = []

        for i, m in enumerate(self._transformers):
            #IMPORTANT this .copy() is important, in case an operand transformer changes inputRes
            results = m.transform(inputRes.copy())
            if len(results) == 0 and num_results != 0:
                raise ValueError("Got no results from %s, expected %d" % (repr(m), num_results) )
            assert not "features_x" in results.columns 
            assert not "features_y" in results.columns
            all_results.append( results )

    
        for i, (m, res) in enumerate(zip(self._transformers, all_results)):
            #IMPORTANT: dont do this BEFORE calling subsequent feature unions
            if not "features" in res.columns:
                if not "score" in res.columns:
                    raise ValueError("Results from %s did not include either score or features columns, found columns were %s" % (repr(m), str(res.columns)) )

                if len(res) != num_results:
                    warn("Got number of results different expected from %s, expected %d received %d, feature scores for any "
                        "missing documents be 0, extraneous documents will be removed" % (repr(m), num_results, len(res)))
                    all_results[i] = res = inputRes[["qid", "docno"]].merge(res, on=["qid", "docno"], how="left")
                    res["score"] = res["score"].fillna(value=0)

                if len(res) == 0:
                    res["features"] = pd.Series([], dtype='float64')
                else:
                    res["features"] = res.apply(lambda row : np.array([row["score"]]), axis=1)
                res.drop(columns=["score"], inplace=True)
            assert "features" in res.columns
            #print("%d got %d features from operand %d" % ( id(self) ,   len(results.iloc[0]["features"]), i))

        def _concat_features(row):
            assert isinstance(row["features_x"], np.ndarray)
            assert isinstance(row["features_y"], np.ndarray)
            
            left_features = row["features_x"]
            right_features = row["features_y"]
            return np.concatenate((left_features, right_features))
        
        def _reduce_fn(left, right):
            import pandas as pd
            both_cols = set(left.columns) & set(right.columns)
            both_cols.remove("qid")
            both_cols.remove("docno")
            both_cols.remove("features")
            rtr = pd.merge(left, right, on=["qid", "docno"])
            rtr["features"] = rtr.apply(_concat_features, axis=1, result_type='reduce')
            rtr.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
            rtr.drop(columns=["features_x", "features_y"] + ["%s_y" % col for col in both_cols], inplace=True)
            return rtr
        
        from functools import reduce
        final_DF = reduce(_reduce_fn, all_results)

        # final_DF should have the features column
        assert "features" in final_DF.columns

        # we used .copy() earlier, inputRes should still have no features column
        assert not "features" in inputRes.columns

        # final merge - this brings us the score attribute from any previous transformer
        both_cols = set(inputRes.columns) & set(final_DF.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        final_DF = inputRes.merge(final_DF, on=["qid", "docno"])
        final_DF.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
        final_DF.drop(columns=["%s_y" % col for col in both_cols], inplace=True)
        # remove the duplicated columns
        #final_DF = final_DF.loc[:,~final_DF.columns.duplicated()]
        assert not "features_x" in final_DF.columns 
        assert not "features_y" in final_DF.columns 
        return final_DF

    def compile(self) -> Transformer:
        """
            Returns a new transformer that fuses feature unions where possible.
        """
        out = deque()
        inp = deque([t.compile() for t in self._transformers])
        while inp:
            right = inp.popleft()
            if out and isinstance(out[-1], SupportsFuseFeatureUnion) and (fused := out[-1].fuse_feature_union(right, is_left=True)) is not None:
                out.pop()
                inp.appendleft(fused)
            elif out and isinstance(right, SupportsFuseFeatureUnion) and (fused := right.fuse_feature_union(out[-1], is_left=False)) is not None:
                out.pop()
                inp.appendleft(fused)
            else:
                out.append(right)
        if len(out) == 1:
            return out[0]
        return FeatureUnionPipeline(*out)

    def __repr__(self):
        return '(' + ' ** '.join([str(t) for t in self._transformers]) + ')'

class ComposedPipeline(NAryTransformerBase):
    """ 
        This class allows pipeline components to be chained together using the "then" operator.

        :Example:

        >>> comp = ComposedPipeline([ DPH_br, ApplyGenericTransformer(lambda res : res[res["rank"] < 2])])
        >>> # OR
        >>> # we can even use lambdas as transformers
        >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
        >>> #this is equivelent
        >>> #comp = DPH_br >> lambda res : res[res["rank"] < 2]]
    """
    def index(self, iter : Iterable[dict], batch_size=100):
        """
        This methods implements indexing pipelines. It is responsible for calling the transform_iter() method of its 
        constituent transformers (except the last one) on batches of records, and the index() method on the last transformer.
        """
        from more_itertools import chunked
        
        if len(self._transformers) > 2:
            #this compose could have > 2 models. we need a composite transform() on all but the last
            prev_transformer = ComposedPipeline(*self._transformers[0:-1])
        else:
            prev_transformer = self._transformers[0]
        last_transformer = self._transformers[-1]
        
        def gen():
            for batch in chunked(iter, batch_size):
                batch_df = prev_transformer.transform_iter(batch)
                for row in batch_df.itertuples(index=False):
                    yield row._asdict()
        return last_transformer.index(gen()) 

    def transform(self, inp):
        out = inp
        for m in self._transformers:
            out = m.transform(out)
        return out

    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va=None, qrels_va=None):
        """
        This is a default implementation for fitting a pipeline. The assumption is that
        all EstimatorBase be composed with a TransformerBase. It will execute any pre-requisite
        transformers BEFORE executing the fitting the stage.
        """
        for m in self._transformers:
            if isinstance(m, Estimator):
                m.fit(topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va)
            else:
                topics_or_res_tr = m.transform(topics_or_res_tr)
                # validation is optional for some learners
                if topics_or_res_va is not None:
                    topics_or_res_va = m.transform(topics_or_res_va)

    def __repr__(self):
        return '(' + ' >> '.join([str(t) for t in self._transformers]) + ')'

    def compile(self) -> Transformer:
        """
            Returns a new transformer that fuses adjacent transformers where possible.
        """
        out = deque()
        inp = deque([t.compile() for t in self._transformers])
        while inp:
            right = inp.popleft()
            if out and isinstance(out[-1], SupportsFuseRight) and (fused := out[-1].fuse_right(right)) is not None:
                out.pop()
                inp.appendleft(fused)
            elif out and isinstance(right, SupportsFuseLeft) and (fused := right.fuse_left(out[-1])) is not None:
                out.pop()
                inp.appendleft(fused)
            else:
                out.append(right)
        if len(out) == 1:
            return out[0]
        return ComposedPipeline(*out)


@runtime_checkable
class SupportsFuseLeft(Protocol):
    def fuse_left(self, left: 'Transformer') -> Optional['Transformer']:
        """Fuses this transformer with a transformer that immediately precedes this one in a pipeline.

        The new transformer should have the same effect as performing the two transformers in sequence, i.e.,
        `pipeline_unfused` and `pipeline_fused` in the following example should provide the same results for
        any input.

        ```
        >>> pipeline_unfused = left >> self
        >>> pipeline_fused = self.fuse_left(left)
        ```

        A fused transformer should be more efficient than the unfused version. For instance, a retriever
        followed by a rank cutoff can be fused to perform the rank cutoff during retrieval.

        Returns:
            A new transformer that is the result of merging this transformer with the left transformer,
            or none if the merge is not possible.
        """


@runtime_checkable
class SupportsFuseRight(Protocol):
    def fuse_right(self, right: 'Transformer') -> Optional['Transformer']:
        """Fuses this transformer with a transformer that immediately follows this one in a pipeline.

        The new transformer should have the same effect as performing the two transformers in sequence, i.e.,
        `pipeline_unfused` and `pipeline_fused` in the following example should provide the same results for
        any input.

        ```
        >>> pipeline_unfused = self >> right
        >>> pipeline_fused = self.fuse_right(right)
        ```

        A fused transformer should be more efficient than the unfused version. For instance, a retriever
        followed by a rank cutoff can be fused to perform the rank cutoff during retrieval.

        Returns:
            A new transformer that is the result of merging this transformer with the right transformer,
            or none if the merge is not possible.
        """


@runtime_checkable
class SupportsFuseRankCutoff(Protocol):
    def fuse_rank_cutoff(self, k: int) -> Optional['Transformer']:
        """Fuses this transformer with a following RankCutoffTransformer.

        This method should return a new transformer that applies the new rank cutoff value k.

        Note that if the transformer currently applies a stricter rank cutoff than the one provided, it should not be
        relaxed. In this case, it is preferred to return `self`.

        If the fusion is not possible, `None` should be returned.
        """


@runtime_checkable
class SupportsFuseFeatureUnion(Protocol):
    def fuse_feature_union(self, other: 'Transformer', is_left: bool) -> Optional['Transformer']:
        """Fuses this transformer with another one that provides features.

        This method should return a new transformer that is equivalent to performing self ** other, or `None`
        if the fusion is not possible.

        is_left is True if self's features are to the left of other's. Otherwise, self's features are to the right.
        """
