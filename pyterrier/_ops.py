from .transformer import Transformer, Estimator, get_transformer, SupportsFuseFeatureUnion, SupportsFuseRankCutoff, SupportsFuseRight, SupportsFuseLeft
from .model import add_ranks
from collections import deque
from warnings import warn
from typing import Optional, Iterable, Tuple, Iterator
from itertools import chain
import pandas as pd
import pyterrier as pt

class NAryTransformerBase(Transformer):
    """
        A base class for all operator transformers that can combine the input of 2 or more transformers. 
    """
    def __init__(self, *transformers: Transformer):
        assert len(transformers) > 0
        # Flatten out multiple layers of the same NAryTransformer into one
        transformers = _flatten(transformers, type(self))
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

    def __iter__(self) -> Iterator[Transformer]:
        """
            Returns an iterator over the transformers in this pipeline.
        """
        return iter(self._transformers)


def _flatten(transformers: Iterable[Transformer], cls: type) -> Tuple[Transformer, ...]:
    return tuple(chain.from_iterable(
        (t._transformers if isinstance(t, cls) else [t]) # type: ignore[attr-defined]
        for t in transformers
    ))

class SetUnion(Transformer):
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

    schematic = {'label': 'SetUnion |'}

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

class SetIntersection(Transformer):
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

    schematic = {'label': 'SetIntersection &'}

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)  
        
        on_cols = ["qid", "docno"]
        rtr = res1.merge(res2, on=on_cols, suffixes=('','_y'))
        rtr.drop(columns=["score", "rank", "score_y", "rank_y", "query_y"], inplace=True, errors='ignore')
        for col in rtr.columns:
            if '_y' not in col:
                continue
            new_name = col.replace('_y', '')
            if new_name in rtr.columns:
                # duplicated column, drop
                rtr.drop(columns=[col], inplace=True)
                continue
            # column only from RHS, keep, but rename by removing '_y' suffix
            rtr.rename(columns={col:new_name}, inplace=True)

        return rtr

class Sum(Transformer):
    """
        Adds the scores of documents from two different retrieval transformers.
        Documents not present in one transformer are given a score of 0.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    schematic = {'label': 'Sum +'}

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

class Concatenate(Transformer):
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

class ScalarProduct(Transformer):
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
    
    def schematic(self, *, input_columns = None): 
        return {'label': f'* {self.scalar}'}

    def __repr__(self):
        return f'ScalarProduct({self.scalar!r})'

class RankCutoff(Transformer):
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
        return f'RankCutoff({self.k!r})'
    
    def schematic(self, *, input_columns = None): 
        return {'label': f'% {self.k}'}

    def fuse_left(self, left: Transformer) -> Optional[Transformer]:
        # If the preceding component supports a native rank cutoff (via fuse_rank_cutoff), apply it.
        if isinstance(left, SupportsFuseRankCutoff):
            return left.fuse_rank_cutoff(self.k)
        return None

class FeatureUnion(NAryTransformerBase):
    """
        Implements the feature union operator.

        Example::
            cands = pt.terrier.Retriever(index wmodel="BM25")
            pl2f = pt.terrier.Retriever(index wmodel="PL2F")
            bm25f = pt.terrier.Retriever(index wmodel="BM25F")
            pipe = cands >> (pl2f ** bm25f)
    """
    schematic = {'inner_pipelines_mode': 'linked', 'label': 'FeatureUnion **'}

    def transform(self, inputRes):
        pt.validate.result_frame(inputRes)

        num_results = len(inputRes)
        import numpy as np

        # a parent could be a feature union, but it still passes the inputRes directly, so inputRes should never have a features column
        if "features" in inputRes.columns:
            raise ValueError("FeatureUnion operates as a re-ranker. They can be nested, but input "
                "should not contain a features column; found columns were %s" %  str(inputRes.columns))
        
        all_results = []

        for i, m in enumerate(self._transformers):
            # IMPORTANT this .copy() is important, in case an operand transformer changes inputRes
            results = m.transform(inputRes.copy())
            if len(results) == 0 and num_results != 0:
                raise ValueError("Got no results from %s, expected %d" % (repr(m), num_results) )
            assert "features_x" not in results.columns 
            assert "features_y" not in results.columns
            all_results.append( results )

    
        for i, (m, res) in enumerate(zip(self._transformers, all_results)):
            # IMPORTANT: dont do this BEFORE calling subsequent feature unions
            if "features" not in res.columns:
                if "score" not in res.columns:
                    raise ValueError("Results from %s did not include either score or features columns, found columns were %s" % (repr(m), str(res.columns)) )

                if len(res) != num_results:
                    warn(
                        "Got number of results different expected from %s, expected %d received %d, feature scores for any "
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
        assert "features" not in inputRes.columns

        # final merge - this brings us the score attribute from any previous transformer
        both_cols = set(inputRes.columns) & set(final_DF.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        final_DF = inputRes.merge(final_DF, on=["qid", "docno"])
        final_DF.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
        final_DF.drop(columns=["%s_y" % col for col in both_cols], inplace=True)
        # remove the duplicated columns
        #final_DF = final_DF.loc[:,~final_DF.columns.duplicated()]
        assert "features_x" not in final_DF.columns 
        assert "features_y" not in final_DF.columns 
        return final_DF

    def compile(self) -> Transformer:
        """
            Returns a new transformer that fuses feature unions where possible.
        """
        out : deque = deque()
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
        return FeatureUnion(*out)

    def __repr__(self):
        return '(' + ' ** '.join([str(t) for t in self._transformers]) + ')'

class Compose(NAryTransformerBase):
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
    name = "Compose"

    def index(self, iter : pt.model.IterDict, batch_size=None):
        """
        This methods implements indexing pipelines. It is responsible for calling the transform_iter() method of its 
        constituent transformers (except the last one) on batches of records, and the index() method on the last transformer.
        """
        from more_itertools import chunked
        
        prev_transformer = Compose(*self._transformers[0:-1])
        last_transformer = self._transformers[-1]

        # guess a good batch size from the batch_size of individual components earlier in the pipeline
        if batch_size is None:
            batch_size = 100 # default to 100 as a reasonable minimum (and fallback if no batch sizes found)
            for tr in prev_transformer:
                if hasattr(tr, 'batch_size') and isinstance(tr.batch_size, int) and tr.batch_size > batch_size:
                    batch_size = tr.batch_size

        def gen():
            for batch in chunked(iter, batch_size):
                yield from prev_transformer.transform_iter(batch)
        return last_transformer.index(gen()) 

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        out = inp
        for transformer in self._transformers:
            out = transformer.transform_iter(out)
        return out
    
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
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

    def compile(self, verbose: bool = False) -> Transformer:
        """Returns a new transformer that iteratively fuses adjacent transformers to form a more efficient pipeline."""
        # compile constituent transformers (flatten allows compile() to return Compose pipelines)
        inp = deque(_flatten((t.compile() for t in self._transformers), Compose))
        out : deque = deque()
        counter = 1
        while inp:
            if verbose:
                print(counter, list(out), list(inp))
            counter +=1 
            right = inp.popleft()
            if out and isinstance(out[-1], SupportsFuseRight) and (fused := out[-1].fuse_right(right)) is not None:
                if verbose:
                    print(f"  fuse_right {out[-1]} >> {right}  == {fused}")
                out.pop()
                # add the fused pipeline to the start of the input queue so it will be processed next (must be done in reverse due to how extendleft works)
                inp.extendleft(reversed(_flatten([fused], Compose)))
            elif out and isinstance(right, SupportsFuseLeft) and (fused := right.fuse_left(out[-1])) is not None:
                if verbose:
                    print(f"  fuse_left {out[-1]} >> {right}  == {fused}")
                out.pop()
                # add the fused pipeline to the start of the input queue so it will be processed next (must be done in reverse due to how extendleft works)
                inp.extendleft(reversed(_flatten([fused], Compose)))
            else:
                out.append(right)
            if counter == MAX_COMPILE_ITER:
                raise OverflowError()
        if len(out) == 1:
            return out[0]
        return Compose(*out)

    def transform_inputs(self):
        # The first transformer in the pipeline may accept multiple input configurations, but not all of these
        # may work for the rest of the pipeline. So find out which (if any) of the input configurations work, and
        # prioritise those.
        io_configurations = [
            {
                'input_columns': input_columns,
                'output_columns': input_columns,
            }
            for input_columns in pt.inspect.transformer_inputs(self._transformers[0])
        ]
        for transformer in self:
            for configuration in io_configurations:
                if configuration['output_columns'] is None:
                    continue
                configuration['output_columns'] = pt.inspect.transformer_outputs(transformer, configuration['output_columns'], strict=False)
        return [io_cfg['input_columns'] for io_cfg in sorted(io_configurations, key=lambda x: x['output_columns'] is None)]

    def transform_outputs(self, input_columns):
        # Figure out the output columns for the given input columns. This is a more direct and robust way of getting the outputs
        # for a composed pipeline than using inspect's default implementation (running an empty dataframe through the whole pipeline)
        # since it can leverage `transform_outputs` implementation of each transformer.
        output_columns = input_columns
        for transformer in self:
            output_columns = pt.inspect.transformer_outputs(transformer, output_columns)
        return output_columns

    def schematic(self, *, input_columns):
        pipeline = []
        columns = input_columns
        for transformer in self:
            schematic = pt.schematic.transformer_schematic(transformer, input_columns=columns)
            pipeline.append(schematic)
            columns = schematic['output_columns']
        return {
            'type': 'pipeline',
            'input_columns': pipeline[0]['input_columns'] if pipeline else None,
            'output_columns': pipeline[-1]['output_columns'] if pipeline else None,
            'title': None,
            'transformers': pipeline,
        }


MAX_COMPILE_ITER = 10_000
