

class TransformerBase:
    '''
        Base class for all transformers. Implements the various operators >> + * | & 
    '''

    def transform(self, topics_or_res):
        pass

    def __rshift__(self, right):
        return ComposedPipeline(models=[self, right])

    def __add__(self, right):
        return CombSumTransformer(self, right)

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

class IdentityTransformer(TransformerBase):

    def __init__(self):
        super(IdentityTransformer, self).__init__()
    
    def transform(self, topics):
        return topics

# this class is useful for testing. it returns a copy of the same
# dataframe each time transform is called
class UniformTransformer(TransformerBase):

    def __init__(self, rtr):
        super(UniformTransformer, self).__init__()
        self.rtr = rtr
    
    def transform(self, topics):
        return self.rtr.copy()

class BinaryTransformerBase(TransformerBase):

    def __init__(self, left, right):
        super(BinaryTransformerBase, self).__init__()
        self.left = left
        self.right = right

class SetUnionTransformer(BinaryTransformerBase):

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        import pandas as pd
        rtr = pd.concat([res1, res2])
        rtr = rtr.drop_duplicates(subset=["qid", "docno"])
        rtr = rtr.sort_values(by=['qid', 'docno'])
        rtr = rtr.drop(columns=["score"])
        return rtr

class SetIntersectionTransformer(BinaryTransformerBase):
    
    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        # NB: there may be othe other duplicate columns
        rtr = res1.merge(res2, on=["qid", "docno"]).drop(columns=["score_x", "score_y"])
        return rtr

class CombSumTransformer(BinaryTransformerBase):

    def transform(self, topics_and_res):
        res1 = self.left.transform(topics_and_res)
        res2 = self.right.transform(topics_and_res)
        merged = res1.merge(res2, on=["qid", "docno"])
        merged["score"] = merged["score_x"] + merged["score_y"]
        merged = merged.drop(columns=['score_x', 'score_y'])
        return merged

# multiplies the retrieval score by a scalar
class ScalarProductTransformer(TransformerBase):

    def __init__(self, transformer, scalar):
        super(ScalarProductTransformer, self).__init__()
        self.transformer = transformer
        self.scalar = scalar

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

    def __init__(self, lambdaFn):
        super(LambdaPipeline, self).__init__()
        self.fn = lambdaFn

    def transform(self, inputRes):
        fn = self.fn
        return fn(inputRes)

class ComposedPipeline(TransformerBase):
    """ 
    This class allows pipeline components to be chained together.

    :Example:

    >>> comp = ComposedPipeline([ DPH_br, LambdaPipeline(lambda res : res[res["rank"] < 2])])
    >>> OR
    >>>  # we can even use lambdas as transformers
    >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
    
    """
    def __init__(self, models=[]):
        super(ComposedPipeline, self).__init__()
        import types
        self.models = list( map(lambda x : LambdaPipeline(x) if callable(x) else x, models) )
    
    def transform(self, topics):
        for m in self.models:
            topics = m.transform(topics)
        return topics