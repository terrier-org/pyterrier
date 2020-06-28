
import types
from matchpy import ReplacementRule, Wildcard, Symbol, Operation, Arity, replace_all, Pattern, CustomConstraint

LAMBDA = lambda:0
def is_lambda(v):
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
       
def is_transformer(v):
    if isinstance(v, TransformerBase):
        return True
    if 'transform' in dir(v):
        return True
    return False

def get_transformer(v):
    ''' 
        Used to coerce functions, lambdas etc into transformers 
    '''

    if isinstance(v, Wildcard):
        # get out of jail for matchpy
        return v
    if is_transformer(v):
        return v
    if is_lambda(v):
        return LambdaPipeline(v)
    if isinstance(v, types.FunctionType):
        return LambdaPipeline(v)
    raise ValueError("Passed parameter %s cannot be coerced into a transformer", str(v))

rewrites_setup = False
rewrite_rules = []

def setup_rewrites():
    from .batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    #three arbitrary "things".
    x = Wildcard.dot('x')
    xs = Wildcard.plus('xs')
    y = Wildcard.dot('y')
    z = Wildcard.dot('z')
    # two different match retrives
    _br1 = Wildcard.symbol('_br1', BatchRetrieve)
    _br2 = Wildcard.symbol('_br2', BatchRetrieve)
    _fbr = Wildcard.symbol('_fbr', FeaturesBatchRetrieve)
    
    # batch retrieves for the same index
    BR_index_matches = CustomConstraint(lambda _br1, _br2: _br1.indexref == _br2.indexref)
    BR_FBR_index_matches = CustomConstraint(lambda _br1, _fbr: _br1.indexref == _fbr.indexref)
    
    # rewrite nested binary feature unions into one single polyadic feature union
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(x, FeatureUnionPipeline(y,z)) ),
        lambda x, y, z: FeatureUnionPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(FeatureUnionPipeline(x,y), z) ),
        lambda x, y, z: FeatureUnionPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(FeatureUnionPipeline(x,y), xs) ),
        lambda x, y, xs: FeatureUnionPipeline(*[x,y]+list(xs))
    ))

    # rewrite nested binary compose into one single polyadic compose
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(x, ComposedPipeline(y,z)) ),
        lambda x, y, z: ComposedPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(ComposedPipeline(x,y), z) ),
        lambda x, y, z: ComposedPipeline(x,y,z)
    ))
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(ComposedPipeline(x,y), xs) ),
        lambda x, y, xs: ComposedPipeline(*[x,y]+list(xs))
    ))

    # rewrite batch a feature union of BRs into an FBR
    rewrite_rules.append(ReplacementRule(
        Pattern(FeatureUnionPipeline(_br1, _br2), BR_index_matches),
        lambda _br1, _br2: FeaturesBatchRetrieve(_br1.indexref, ["WMODEL:" + _br1.controls["wmodel"], "WMODEL:" + _br2.controls["wmodel"]])
    ))

    def push_fbr_earlier(_br1, _fbr):
        #TODO copy more attributes
        _fbr.controls["wmodel"] = _br1.controls["wmodel"]
        return _fbr

    # rewrite a BR followed by a FBR into a FBR
    rewrite_rules.append(ReplacementRule(
        Pattern(ComposedPipeline(_br1, _fbr), BR_FBR_index_matches),
        push_fbr_earlier
    ))

    global rewrites_setup
    rewrites_setup = True


class Scalar(Symbol):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

class TransformerBase:
    '''
        Base class for all transformers. Implements the various operators >> + * | & 
        as well as the compile() for rewriting complex pipelines into more simples ones.
    '''

    def transform(self, topics_or_res):
        '''
            Abstract method for all transformations. Typically takes as input a Pandas
            DataFrame, and also returns one also.
        '''
        pass

    def compile(self):
        '''
            Rewrites this pipeline by applying of the Matchpy rules in rewrite_rules.
        '''
        if not rewrites_setup:
            setup_rewrites()
        print("Applying %d rules" % len(rewrite_rules))
        return replace_all(self, rewrite_rules)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def __rshift__(self, right):
        return ComposedPipeline(self, right)

    def __rrshift__(self, left):
        return ComposedPipeline(left, self)

    def __add__(self, right):
        return CombSumTransformer(self, right)

    def __pow__(self, right):
        return FeatureUnionPipeline(self, right)

    def __mul__(self, rhs):
        assert isinstance(rhs, int) or isinstance(rhs, float)
        return ScalarProductTransformer(self, rhs)

    def __rmul__(self, lhs):
        assert isinstance(lhs, int) or isinstance(lhs, float)
        return ScalarProductTransformer(self, lhs)

    def __or__(self, right):
        return SetUnionTransformer(self, right)

    def __and__(self, right):
        return SetIntersectionTransformer(self, right)

    def __mod__(self, right):
        assert isinstance(right, int)
        return RankCutoffTransformer(self, right)

    def __xor__(self, right):
        return ConcatenateTransformer(self, right)

    def __invert__(self):
        from .cache import ChestCacheTransformer
        return ChestCacheTransformer(self)
    
class EstimatorBase(TransformerBase):
    '''
        This is a base class for things that can be fitted.
    '''
    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va):
        pass

class IdentityTransformer(TransformerBase, Operation):
    arity = Arity.nullary

    def __init__(self, *args, **kwargs):
        super(IdentityTransformer, self).__init__(*args, **kwargs)
    
    def transform(self, topics):
        return topics

# this class is useful for testing. it returns a copy of the same
# dataframe each time transform is called
class UniformTransformer(TransformerBase, Operation):
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        super().__init__(operands=[], **kwargs)
        self.operands=[]
        self.rtr = rtr[0]
    
    def transform(self, topics):
        rtr = self.rtr.copy()
        return rtr

class BinaryTransformerBase(TransformerBase,Operation):
    arity = Arity.binary

    def __init__(self, operands, **kwargs):
        assert 2 == len(operands)
        super().__init__(operands=operands,  **kwargs)
        self.left = operands[0]
        self.right = operands[1]

class NAryTransformerBase(TransformerBase,Operation):
    arity = Arity.polyadic

    def __init__(self, operands, **kwargs):
        super().__init__(operands=operands, **kwargs)
        models = operands
        self.models = list( map(lambda x : get_transformer(x), models) )

    def __getitem__(self, number):
        '''
            Allows access to the ith transformer.
        '''
        return self.models[number]

    def __len__(self):
        '''
            Returns the number of transformers in the operator.
        '''
        return len(self.models)

class SetUnionTransformer(BinaryTransformerBase):
    name = "Union"

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        import pandas as pd
        assert isinstance(res1, pd.DataFrame)
        assert isinstance(res2, pd.DataFrame)
        rtr = pd.concat([res1, res2])
        rtr = rtr.drop_duplicates(subset=["qid", "docno"])
        rtr = rtr.sort_values(by=['qid', 'docno'])
        rtr = rtr.drop(columns=["score"])
        return rtr

class SetIntersectionTransformer(BinaryTransformerBase):
    name = "Intersect"
    
    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        # NB: there may be other duplicate columns
        rtr = res1.merge(res2, on=["qid", "docno"]).drop(columns=["score_x", "score_y"])
        return rtr

class CombSumTransformer(BinaryTransformerBase):
    name = "Sum"

    def transform(self, topics_and_res):
        res1 = self.left.transform(topics_and_res)
        res2 = self.right.transform(topics_and_res)
        merged = res1.merge(res2, on=["qid", "docno"])
        merged["score"] = merged["score_x"] + merged["score_y"]
        merged = merged.drop(columns=['score_x', 'score_y'])
        return merged

class ConcatenateTransformer(BinaryTransformerBase):
    name = "Concat"
    epsilon = 0.0001

    def transform(self, topics_and_res):
        import pandas as pd
        # take the first set as the top of the ranking
        res1 = self.left.transform(topics_and_res)
        # identify the lowest score for each query
        last_scores = res1.groupby('qid').min()[['score']].rename(columns={"score" : "_lastscore"})

        # the right hand side will provide the rest of the ranking        
        res2 = self.right.transform(topics_and_res)

        
        intersection = pd.merge(res1[["qid", "docno"]], res2[["qid", "docno"]].reset_index())
        remainder = res2.drop(intersection["index"])

        # we will append documents from remainder to res1
        # but we need to offset the score from each remaining document based on the last score in res1
        # explanation: remainder["score"] - remainder["_firstscore"] - self.epsilon ensures that the
        # first document in remainder has a score of -epsilon; we then add the score of the last document
        # from res1
        first_scores = remainder.groupby('qid').max()[['score']].rename(columns={"score" : "_firstscore"})

        remainder = remainder.merge(last_scores, on=["qid"]).merge(first_scores, on=["qid"])
        remainder["score"] = remainder["score"] - remainder["_firstscore"] + remainder["_lastscore"] - self.epsilon
        remainder = remainder.drop(columns=["_lastscore",  "_firstscore"])

        # now bring together and re-sort
        # this sort should match trec_eval
        rtr = pd.concat([res1, remainder]).sort_values(["qid", "score", "docno"], ascending=[True, False, True]) 

        # recompute the ranks
        rtr = rtr.drop(columns=["rank"])
        rtr["rank"] = rtr.groupby("qid").rank(ascending=False)["score"].astype(int)

        return rtr

class ScalarProductTransformer(BinaryTransformerBase):
    '''
    multiplies the retrieval score by a scalar
    '''
    arity = Arity.binary
    name = "ScalarProd"

    def __init__(self, operands, **kwargs):
        super().__init__(operands, **kwargs)
        self.transformer = operands[0]
        self.scalar = operands[1]

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        res["score"] = self.scalar * res["score"]
        return res

class RankCutoffTransformer(BinaryTransformerBase):
    '''
    applies a rank cutoff for each query
    '''
    arity = Arity.binary
    name = "RankCutoff"

    def __init__(self, operands, **kwargs):
        operands = [operands[0], Scalar(str(operands[1]), operands[1])] if isinstance(operands[1], int) else operands
        super().__init__(operands, **kwargs)
        self.transformer = operands[0]
        self.cutoff = operands[1]

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        if not "rank" in res.columns:
            assert False, "require rank to be present in the result set"

        res = res[res["rank"] <= self.cutoff.value]
        return res

class LambdaPipeline(TransformerBase):
    """
    This class allows pipelines components to be written as functions or lambdas

    :Example:
    >>> # this pipeline would remove all but the first two documents from a result set
    >>> lp = LambdaPipeline(lambda res : res[res["rank"] < 2])

    """

    def __init__(self, lambdaFn,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = lambdaFn

    def transform(self, inputRes):
        fn = self.fn
        return fn(inputRes)

class FeatureUnionPipeline(NAryTransformerBase):
    name = "FUnion"

    def transform(self, inputRes):
        assert "docno" in inputRes.columns or "docid" in inputRes.columns
        
        all_results = []
        for m in self.models:
            results = m.transform(inputRes).rename(columns={"score" : "features"})
            all_results.append( results )

        def _concat_features(row):
            import numpy as np
            left_features = row["features_x"] if isinstance(row["features_x"], np.ndarray) else [row["features_x"]]
            right_features = row["features_y"] if isinstance(row["features_y"], np.ndarray) else [row["features_y"]]
            return np.concatenate((left_features, right_features))
        
        def _reduce_fn(left, right):
            import pandas as pd
            import numpy as np
            rtr = pd.merge(left, right, on=["qid", "docno"])
            rtr["features"] = rtr.apply(_concat_features, axis=1)
            rtr.drop(columns=["features_x", "features_y"], inplace=True)
            return rtr
        
        from functools import reduce
        final_DF = reduce(_reduce_fn, all_results)
        final_DF = inputRes.merge(final_DF, on=["qid", "docno"])
        return final_DF

class ComposedPipeline(NAryTransformerBase):
    name = "Compose"
    """ 
    This class allows pipeline components to be chained together using the "then" operator.

    :Example:

    >>> comp = ComposedPipeline([ DPH_br, LambdaPipeline(lambda res : res[res["rank"] < 2])])
    >>> # OR
    >>> # we can even use lambdas as transformers
    >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
    >>> # this is equivelent
    >>> # comp = DPH_br >> lambda res : res[res["rank"] < 2]]
    """
    
    def transform(self, topics):
        for m in self.models:
            topics = m.transform(topics)
        return topics

    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va=None, qrels_va=None):
        """
        This is a default implementation for fitting a pipeline. The assumption is that
        all EstimatorBase be composed with a TransformerBase. It will execute any pre-requisite
        transformers BEFORE executing the fitting the stage.
        """
        for m in self.models:
            if isinstance(m, EstimatorBase):
                m.fit(topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va)
            else:
                topics_or_res_tr = m.transform(topics_or_res_tr)
                # validation is optional for some learners
                if topics_or_res_va is not None:
                    topics_or_res_va = m.transform(topics_or_res_va)