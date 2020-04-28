
from matchpy import *


       
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
    # batch retrieves for the same index
    BR_index_matches = CustomConstraint(lambda br1, br2: br1.indexref == br1.indexref)

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
        lambda x, y, z: FeaturesBatchRetrieve(_br1.indexref, ["WMODEL" + _br1.controls["wmodel"], "WMODEL" + _br2.controls["wmodel"]])
    ))


class Scalar(Symbol):
    pass

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
        return replace_all(self, rewrite_rules)

    def __rshift__(self, right):
        return ComposedPipeline(self, right)

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
        self.models = list( map(lambda x : LambdaPipeline(x) if callable(x) else x, models) )

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

# multiplies the retrieval score by a scalar
class ScalarProductTransformer(BinaryTransformerBase):
    arity = Arity.binary
    name = "ScalarProd"

    def __init__(self, operands, **kwargs):
        #mpy_ops = [transformer, Scalar(scalar)] if isinstance(scalar, Scalar) else [transformer, scalar]
        super().__init__(operands, **kwargs)
        self.transformer = operands[0]
        self.scalar = operands[1]

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        res["score"] = self.scalar * res["score"]
        return res

class LambdaPipeline(TransformerBase):
    """
    This class allows pipelines components to be written as functions or lambdas

    :Example:
    >>> #this pipeline would remove all but the first two documents from a result set
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