
import types
from matchpy import ReplacementRule, Wildcard, Symbol, Operation, Arity, replace_all, Pattern, CustomConstraint
from .validation import PipelineError, ValidationError, TRANSFORMER_FAMILY, TYPE_SAFETY_LEVEL, COLUMN_TYPE, COLUMNS_TYPE

from warnings import warn
import pandas as pd
from .model import QUERIES, RANKED_DOCS, RANKED_DOCS_, RETRIEVED_DOCS, RETRIEVED_DOCS_, RETRIEVED_DOCS_FEATURES, RETRIEVED_DOCS_FEATURES_, _last_query, add_ranks
from . import tqdm
import deprecation
from enum import Enum

class Family():
    QUERY_REWRITE = 'queryrewrite'
    RETRIEVAL = 'retrieval'
    QUERY_EXPANSION = 'queryexpansion'
    RERANKING = 'reranking'
    FEATURE_SCORING = 'featurescoring'
from typing import Iterable, List, Sequence, Callable, Iterator, Union

LAMBDA = lambda:0
def is_lambda(v):
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def is_function(v):
    return isinstance(v, types.FunctionType)

def is_transformer(v):
    if isinstance(v, Transformer):
        return True
    return False

def get_transformer(v):
    """ 
        Used to coerce functions, lambdas etc into transformers 
    """

    if isinstance(v, Wildcard):
        # get out of jail for matchpy
        return v
    if is_transformer(v):
        return v
    if is_lambda(v):
        warn('Coercion of a lambda into a transformer is deprecated; use a pt.apply instead')
        return ApplyGenericTransformer(v)
    if is_function(v):
        warn('Coercion of a function into a transformer is deprecated; use a pt.apply instead')
        return ApplyGenericTransformer(v)
    if isinstance(v, pd.DataFrame):
        warn('Coercion of a dataframe into a transformer is deprecated; use a pt.Transformer.from_df() instead')
        return SourceTransformer(v)
    raise ValueError("Passed parameter %s of type %s cannot be coerced into a transformer" % (str(v), type(v)))

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
        _fbr.wmodel = _br1.controls["wmodel"]
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

class Transformer:
    name = "Transformer"
    """
        Base class for all transformers. Implements the various operators ``>>`` ``+`` ``*`` ``|`` ``&`` 
        as well as ``search()`` for executing a single query and ``compile()`` for rewriting complex pipelines into more simples ones.
    """

    def __init__(self, family : str = None, input : COLUMNS_TYPE = None, output : COLUMNS_TYPE = None, **kwargs):
        """
            When initialising a transformer we can pass in certain value in order to allow for type checking and
            pipeline validation
            Arguments:
             - family(str): For common transformers, the family can be passed as a constant, allowing for the default
                    data transformer family types defined in validation.py to be used
             - input([str]): The list of columns a transformer needs for its transform method to work
             - output([str]): The list of columns that a transformer will always output from its transform method
        """

        self.input = None
        self.output = None
        # If family is defined, then we can obtain the minimal input and output from default family mapping
        if family:
            self.input = TRANSFORMER_FAMILY[family]['input']
            self.output = TRANSFORMER_FAMILY[family]['output']

        # setting input or output override the family setting
        if input is not None:
            self.input = input
        if output is not None:
            self.output = output

        # error or warn, as appropriate
        def _raise(msg):
            if TYPE_SAFETY_LEVEL == 1:
                warn(msg)
            elif TYPE_SAFETY_LEVEL > 1:
                raise TypeError(msg)

        # Use type safety level to judge how strictly we need input/output to be defined
        if self.input is None:
            try:
                selfname = repr(self)
            except:
                selfname = self.__class__.__name__
            _raise("No family specified, and minimal input not defined for transformer " + selfname)
        if self.output is None:
            _raise("No family specified and output not defined for transformer " + selfname)

    @staticmethod
    def from_df(input : pd.DataFrame, uniform=False) -> 'Transformer':
        """
        Instantiates a transformer from an input dataframe. Some rows from the input dataframe are returned
        in response to a query on the ``transform()`` method. Depending on the value `uniform`, the dataframe
        passed as an argument to ``transform()`` can affect this selection.

        If `uniform` is True, input will be returned in its entirety each time.
        If `uniform` is False, rows from input that match the qid values from the argument dataframe.
        
        """
        if uniform:
            return UniformTransformer(input)
        return SourceTransformer(input)

    def transform(self, topics_or_res : pd.DataFrame) -> pd.DataFrame:
        """
            Abstract method for all transformations. Typically takes as input a Pandas
            DataFrame, and also returns one.
        """
        pass

    def transform_iter(self, input: Iterable[dict]) -> pd.DataFrame:
        """
            Method that proesses an iter-dict by instantiating it as a dataframe and calling transform().
            Returns the DataFrame returned by transform(). Used in the implementation of index() on a composed 
            pipeline.
        """
        return self.transform(pd.DataFrame(list(input)))

    def transform_gen(self, input : pd.DataFrame, batch_size=1, output_topics=False) -> Iterator[pd.DataFrame]:
        """
            Method for executing a transformer pipeline on smaller batches of queries.
            The input dataframe is grouped into batches of batch_size queries, and a generator
            returned, such that transform() is only executed for a smaller batch at a time. 

            Arguments:
                input(DataFrame): a dataframe to process
                batch_size(int): how many input instances to execute in each batch. Defaults to 1.
            
        """
        docno_provided = "docno" in input.columns
        docid_provided = "docid" in input.columns
        
        if docno_provided or docid_provided:
            queries = input[["qid"]].drop_duplicates()
        else:
            queries = input
        batch=[]      
        for query in queries.itertuples():
            if len(batch) == batch_size:
                batch_topics = pd.concat(batch)
                batch=[]
                res = self.transform(batch_topics)
                if output_topics:
                    yield res, batch_topics
                else:
                    yield res
            batch.append(input[input["qid"] == query.qid])
        if len(batch) > 0:
            batch_topics = pd.concat(batch)
            res = self.transform(batch_topics)
            if output_topics:
                yield res, batch_topics
            else:
                yield res

    def search(self, query : str, qid : str = "1", sort=True) -> pd.DataFrame:
        """
            Method for executing a transformer (pipeline) for a single query. 
            Returns a dataframe with the results for the specified query. This
            is a utility method, and most uses are expected to use the transform()
            method passing a dataframe.

            Arguments:
                query(str): String form of the query to run
                qid(str): the query id to associate to this request. defaults to 1.
                sort(bool): ensures the results are sorted by descending rank (defaults to True)

            Example::

                bm25 = pt.BatchRetrieve(index, wmodel="BM25")
                res = bm25.search("example query")

                # is equivalent to
                queryDf = pd.DataFrame([["1", "example query"]], columns=["qid", "query"])
                res = bm25.transform(queryDf)
            
            
        """
        import pandas as pd
        queryDf = pd.DataFrame([[qid, query]], columns=["qid", "query"])
        rtr = self.transform(queryDf)
        if "qid" in rtr.columns and "rank" in rtr.columns:
            rtr = rtr.sort_values(["qid", "rank"], ascending=[True,True])
        return rtr


    def _calculate_output(self, inputs : Sequence[COLUMN_TYPE]):
        '''
            Method for calculating the output columns of a transformer
            We have 3 possible cases:
             - output is a function - The output columns must be calculated using the input columns
             - output is a list - The output columns are this predefined list
             - output is a list containing ... - The output columns contains at least these columns, along with any other columns in the input
        '''

        if isinstance(self.output, types.FunctionType):
            output = self.output(inputs)
        else:
            output = self.output.copy()
            if Ellipsis in output:
                output.remove(Ellipsis)
                output = list(set(inputs) | set(output))
        return output

    def validate(self, inputs : Union[ List[str], pd.DataFrame]):
        '''
            Default method implementation to validate transformer types. Checks that the input dataframe to the
            transformer has the required attributes, and returns the attributes that will be provided if applicable
        '''
        if type(inputs) == str:
            # If we are given a str, we treat it as a query
            inputs = ["qid", "query"]
        elif isinstance(inputs, pd.DataFrame):
            inputs = inputs.columns.tolist()

        # If we cannot validate a transformer because it's type is not set, then we warn the user
        if self.input is None or self.output is None:
            raise ValidationError(self, inputs)

        # We are validating that the set of input columns is a superset of the set of minimal input columns
        # i.e. all required columns are present
        if set(inputs).issuperset(set(self.input)):
            return self._calculate_output(inputs)
        else:
            raise TypeError("Could not validate transformer %s with given input %s", (str(self), str(inputs)))

    def compile(self) -> 'Transformer':
        """
        Rewrites this pipeline by applying of the Matchpy rules in rewrite_rules. Pipeline
        optimisation is discussed in the `ICTIR 2020 paper on PyTerrier <https://arxiv.org/abs/2007.14271>`_.
        """
        if not rewrites_setup:
            setup_rewrites()
        print("Applying %d rules" % len(rewrite_rules))
        return replace_all(self, rewrite_rules)

    def parallel(self, N : int, backend='joblib') -> 'Transformer':
        """
        Returns a parallelised version of this transformer. The underlying transformer must be "picklable".

        Args:
            N(int): how many processes/machines to parallelise this transformer over. 
            backend(str): which multiprocessing backend to use. Only two backends are supported, 'joblib' and 'ray'. Defaults to 'joblib'.
        """
        from .parallel import PoolParallelTransformer
        return PoolParallelTransformer(self, N, backend)

    # Get and set specific parameter value by parameter's name
    def get_parameter(self, name : str):
        """
            Gets the current value of a particular key of the transformer's configuration state.
            By default, this examines the attributes of the transformer object, using hasattr() and setattr().
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(("Invalid parameter name %s for transformer %s. " + 
                      "Check the list of available parameters") %(str(name), str(self)))

    def set_parameter(self, name : str, value):
        """
            Adjusts this transformer's configuration state, by setting the value for specific parameter.
            By default, this examines the attributes of the transformer object, using hasattr() and setattr().
        """
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError(('Invalid parameter name %s for transformer %s. '+
                    'Check the list of available parameters') %(name, str(self)))

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        """
            Sets up a default method for every transformer, which is aliased to transform(). 
        """
        return self.transform(*args, **kwargs)

    def __rshift__(self, right) -> 'Transformer':
        return ComposedPipeline(self, right)

    def __rrshift__(self, left) -> 'Transformer':
        return ComposedPipeline(left, self)

    def __add__(self, right : 'Transformer') -> 'Transformer':
        return CombSumTransformer(self, right)

    def __pow__(self, right : 'Transformer') -> 'Transformer':
        return FeatureUnionPipeline(self, right)

    def __mul__(self, rhs : Union[float,int]) -> 'Transformer':
        assert isinstance(rhs, int) or isinstance(rhs, float)
        return ScalarProductTransformer(self, rhs)

    def __rmul__(self, lhs : Union[float,int]) -> 'Transformer':
        assert isinstance(lhs, int) or isinstance(lhs, float)
        return ScalarProductTransformer(self, lhs)

    def __or__(self, right : 'Transformer') -> 'Transformer':
        return SetUnionTransformer(self, right)

    def __and__(self, right : 'Transformer') -> 'Transformer':
        return SetIntersectionTransformer(self, right)

    def __mod__(self, right : 'Transformer') -> 'Transformer':
        assert isinstance(right, int)
        return RankCutoffTransformer(self, right)

    def __xor__(self, right : 'Transformer') -> 'Transformer':
        return ConcatenateTransformer(self, right)

    def __invert__(self : 'Transformer') -> 'Transformer':
        from .cache import ChestCacheTransformer
        return ChestCacheTransformer(self)

    def __hash__(self):
        return hash(repr(self))

class TransformerBase(Transformer):
    # this was the older name of Transformer.
    # it will be deprecated in a future release.
    pass

class IterDictIndexerBase(TransformerBase):
    def index(self, iter : Iterable[dict], **kwargs):
        """
            Takes an iterable of dictionaries ("iterdict"), and consumes them. There is no return;
            This method is typically used to implement indexers.
        """
        pass

    
class EstimatorBase(TransformerBase):
    """
        This is a base class for things that can be fitted.
    """
    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va):
        """
            Method for training the transformer.

            Arguments:
                topics_or_res_tr(DataFrame): training topics (usually with documents)
                qrels_tr(DataFrame): training qrels
                topics_or_res_va(DataFrame): validation topics (usually with documents)
                qrels_va(DataFrame): validation qrels
        """
        pass

class IdentityTransformer(TransformerBase, Operation):
    """
        A transformer that returns exactly the same as its input.
    """
    arity = Arity.nullary

    def __init__(self, *args, **kwargs):
        Operation.__init__(self, operands=[])
        TransformerBase.__init__(self, input=[], output=[...], **kwargs)
        
    def transform(self, topics):
        return topics

class SourceTransformer(TransformerBase):
    """
    A Transformer that can be used when results have been saved in a dataframe.
    It will select results on qid.
    If a query column is in the dataframe passed in the constructor, this will override any query
    column in the topics dataframe passed to the transform() method.
    """
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        #Operation.__init__(self, operands=[])
        TransformerBase.__init__(self, input=[], output=rtr.columns.tolist(), **kwargs)
        self.operands=[]
        self.df = rtr#[0]
        self.df_contains_query = "query" in self.df.columns
        assert "qid" in self.df.columns
    
    def transform(self, topics):
        assert "qid" in topics.columns
        columns=["qid"]
        topics_contains_query = "query" in topics.columns
        if not self.df_contains_query and topics_contains_query:
            columns.append("query")
        rtr = topics[columns].merge(self.df, on="qid")
        return rtr

class UniformTransformer(TransformerBase):
    """
        A transformer that returns the same dataframe every time transform()
        is called. This class is useful for testing. 
    """
    arity = Arity.nullary

    def __init__(self, rtr, **kwargs):
        #Operation.__init__(self, operands=[])
        TransformerBase.__init__(self, input=[], output=rtr.columns.tolist(), **kwargs)
        self.operands=[]
        self.rtr = rtr#[0]
    
    def transform(self, topics):
        rtr = self.rtr.copy()
        return rtr

class UnaryTransformerBase(TransformerBase):
    def __init__(self, parent, operand_output=None, **kwargs):
        TransformerBase.__init__(self, input=[], output=[...], **kwargs)
        self.parent = parent
        self.operand_output = operand_output

    def validate(self, inputs):
        # validate left component
        try:
            left_output = self.parent.validate(inputs)
        except TypeError:
            raise PipelineError(self.left, inputs)
        if self.operand_output and not set(self.operand_output).issubset(left_output):
            raise PipelineError(self, left_output, self.operand)

        return self._calculate_output(left_output)


class BinaryTransformerBase(TransformerBase,Operation):
    """
        A base class for all operator transformers that can combine the input of exactly 2 transformers. 
    """
    arity = Arity.binary

    def __init__(self, operands, operand_output=None, **kwargs):
        assert 2 == len(operands)        
        Operation.__init__(self, operands=operands)
        TransformerBase.__init__(self, **kwargs)
        self.left = operands[0]
        self.right = operands[1]
        self.operand_output = operand_output

    def validate(self, inputs):
        # validate left component
        try:
            left_output = self.left.validate(inputs)
        except TypeError:
            raise PipelineError(self.left, inputs)
        if self.operand_output and not set(self.operand_output).issubset(left_output):
            raise PipelineError(self, left_output, self.left)

        right_output=[]
        # validate right component
        if not isinstance(self.right, int) and not isinstance(self.right, float) and not isinstance(self.right, Scalar):
            try:
                right_output = self.right.validate(inputs)
            except TypeError:
                raise PipelineError(self.right, inputs)
            if self.operand_output and not set(self.operand_output).issubset(right_output):
                raise PipelineError(self, right_output, self.right)

        return self._calculate_output(left_output, right_output)

    def _calculate_output(self, left_cols, right_cols):
        pass


class NAryTransformerBase(TransformerBase,Operation):
    """
        A base class for all operator transformers that can combine the input of 2 or more transformers. 
    """
    arity = Arity.polyadic

    def __init__(self, operands, operand_output=[], **kwargs):
        Operation.__init__(self, operands=operands)
        TransformerBase.__init__(self, **kwargs)        
        models = operands
        self.operand_output = operand_output
        self.models = list( map(lambda x : get_transformer(x), models) )

    def __getitem__(self, number):
        """
            Allows access to the ith transformer.
        """
        return self.models[number]

    def __len__(self):
        """
            Returns the number of transformers in the operator.
        """
        return len(self.models)

    def validate(self, inputs):
        # validate each component
        op_outputs=[]
        for m in self.models:
            op_out = None
            try:
                op_out = m.validate(inputs)
            except TypeError:
                raise PipelineError(m, inputs)
            if self.operand_output and not set(self.operand_output).issubset(op_out):
                raise PipelineError(self, op_out, m)
            op_outputs.append(op_out)

        return self._calculate_output(op_outputs)

    def _calculate_output(self, op_output_cols):
        pass

class SetUnionTransformer(BinaryTransformerBase):
    """      
        This operator makes a retrieval set that includes documents that occur in the union (either) of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [1, "text1", doc1, 0.42] and right = [1, "text1", doc2, 0.24]. 
        Then, left | right will be a dataframe with only the columns [qid, query, docno] and two rows = [[1, "text1", doc1], [1, "text1", doc2]].
                
        In case of duplicated both containing (qid, docno), only the first occurrence will be used.
    """
    name = "Union"

    def __init__(self, operands, **kwargs):
        super().__init__(operands, operand_output=RETRIEVED_DOCS, output=RETRIEVED_DOCS, **kwargs)

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

    def _calculate_output(self, left_cols, right_cols):
        rtr = (set(left_cols) | set(right_cols))
        rtr.discard("rank")
        rtr.discard("score")
        return list(rtr)
        

class SetIntersectionTransformer(BinaryTransformerBase):
    """
        This operator makes a retrieval set that only includes documents that occur in the intersection of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [[1, "text1", doc1, 0.42]] (one row) and right = [[1, "text1", doc1, 0.24],[1, "text1", doc2, 0.24]] (two rows).
        Then, left & right will be a dataframe with only the columns [qid, query, docno] and one single row = [[1, "text1", doc1]].
                
        For columns other than (qid, docno), only the left value will be used.
    """
    name = "Intersect"

    def __init__(self, operands, **kwargs):
        super().__init__(operands, operand_output=RETRIEVED_DOCS, output=RETRIEVED_DOCS_, **kwargs)
    
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

    def _calculate_output(self, left_cols, right_cols):
        SetUnionTransformer._calculate_output(self, left_cols, right_cols)

class CombSumTransformer(BinaryTransformerBase):
    """
        Adds the scores of documents from two different retrieval transformers.
        Documents not present in one transformer are given a score of 0.
    """
    name = "Sum"

    def __init__(self, operands, **kwargs):
        super().__init__(operands=operands, **kwargs, operand_output=RANKED_DOCS, output=RANKED_DOCS_)

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

    def _calculate_output(self, left_output_cols, right_output_cols):
        assert set(left_output_cols) == set(right_output_cols), (left_output_cols, right_output_cols)
        return left_output_cols

class ConcatenateTransformer(BinaryTransformerBase):
    name = "Concat"
    epsilon = 0.0001

    def __init__(self, operands, **kwargs):
        super().__init__(operands=operands, **kwargs, operand_output=RANKED_DOCS, output=RANKED_DOCS_)

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

    def _calculate_output(self, left_output_cols, right_output_cols):
        assert set(left_output_cols) == set(right_output_cols), (left_output_cols, right_output_cols)
        return left_output_cols

class ScalarProductTransformer(BinaryTransformerBase):
    """
        Multiplies the retrieval score by a scalar
    """
    arity = Arity.binary
    name = "ScalarProd"

    def __init__(self, operands, **kwargs):
        super().__init__(operands, operand_output=RANKED_DOCS, output=lambda input : input, **kwargs)
        self.transformer = operands[0]
        self.scalar = operands[1]

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        res["score"] = self.scalar * res["score"]
        if self.scalar < 0:
            res = add_ranks(res)
        return res
    
    def _calculate_output(self, left_output_cols, right_output_cols):
        return TransformerBase._calculate_output(self, left_output_cols)

class RankCutoffTransformer(BinaryTransformerBase):
    """
        Applies a rank cutoff for each query
    """
    arity = Arity.binary
    name = "RankCutoff"

    def __init__(self, operands, **kwargs):
        operands = [operands[0], Scalar(str(operands[1]), operands[1])] if isinstance(operands[1], int) else operands
        super().__init__(operands, operand_output=RANKED_DOCS, output=RANKED_DOCS_, **kwargs)
        self.transformer = operands[0]
        self.cutoff = operands[1]
        if self.cutoff.value % 10 == 9:
            warn("Rank cutoff off-by-one bug #66 now fixed, but you used a cutoff ending in 9. Please check your cutoff value. ", DeprecationWarning, 2)

    def transform(self, topics_and_res):
        res = self.transformer.transform(topics_and_res)
        if not "rank" in res.columns:
            assert False, "require rank to be present in the result set"

        # this assumes that the minimum rank cutoff is model.FIRST_RANK, i.e. 0
        res = res[res["rank"] < self.cutoff.value]
        return res

    def _calculate_output(self, left_output_cols, right_output_cols):
        return left_output_cols

class ApplyTransformerBase(TransformerBase):
    """
        A base class for Apply*Transformers
    """
    def __init__(self, fn, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.verbose = verbose

class ApplyForEachQuery(ApplyTransformerBase):
    def __init__(self, fn,  *args, add_ranks=True, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns the new float doument score 
        """
        super().__init__(fn, *args, **kwargs)
        self.add_ranks = add_ranks
    
    def transform(self, res):
        if len(res) == 0:
            return self.fn(res)
        it = res.groupby("qid")
        if self.verbose:
            it = tqdm(it, unit='query')
        try:
            dfs = [self.fn(group) for qid, group in it]
            rtr = pd.concat(dfs)
        except Exception as a:
            raise Exception("Problem applying %s" % self.fn) from a
        if self.add_ranks:
            rtr = add_ranks(rtr)
        return rtr

class ApplyDocumentScoringTransformer(ApplyTransformerBase):
    """
        Implements a transformer that can apply a function to perform document scoring. The supplied function 
        should take as input one row, and return a float for the score of the document.
        
        Usually accessed using pt.apply.doc_score()::

            def _score_fn(row):
                return float(row["url".count("/")])
            
            pipe = pt.BatchRetrieve(index) >> pt.apply.doc_score(_score_fn)
    """
    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns the new float doument score 
        """
        super().__init__(fn, *args, input=RETRIEVED_DOCS, output=RANKED_DOCS_, **kwargs)
    
    def transform(self, inputRes):
        fn = self.fn
        outputRes = inputRes.copy()
        if self.verbose:
            tqdm.pandas(desc="pt.apply.doc_score", unit="d")
            outputRes["score"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["score"] = outputRes.apply(fn, axis=1)
        outputRes = add_ranks(outputRes)
        return outputRes

class ApplyDocFeatureTransformer(ApplyTransformerBase):
    """
        Implements a transformer that can apply a function to perform feature scoring. The supplied function 
        should take as input one row, and return a numpy array for the features of the document.
        
        Usually accessed using pt.apply.doc_features()::

            def _feature_fn(row):
                return numpy.array([len(row["url"], row["url".count("/")])
            
            pipe = pt.BatchRetrieve(index) >> pt.apply.doc_features(_feature_fn) >> pt.LTRpipeline(xgBoost())
    """
    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing that document, and returns a new numpy array representing the features of that document
        """
        super().__init__(fn, *args, input=RETRIEVED_DOCS, output=RETRIEVED_DOCS_FEATURES_, **kwargs)

    def transform(self, inputRes):
        fn = self.fn
        outputRes = inputRes.copy()
        if self.verbose:
            tqdm.pandas(desc="pt.apply.doc_features", unit="d")
            outputRes["features"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["features"] = outputRes.apply(fn, axis=1)
        return outputRes

class ApplyQueryTransformer(ApplyTransformerBase):
    """
        Implements a query rewriting transformer by passing a function to perform the rewriting. The function should take
        as input one row, and return the string form of the new query.
        
        Usually accessed using pt.apply.query() passing it the function::

            def _rewriting_fn(row):
                return row["query"] + " extra words"
            
            pipe = pt.apply.query(_rewriting_fn) >> pt.BatchRetrieve(index)

        Similarly, a lambda function can also be used::

            pipe = pt.apply.query(lambda row: row["query"] + " extra words") >> pt.BatchRetrieve(index)

        In the resulting dataframe, the previous query for each row can be found in the query_0 column.

    """
    def __init__(self, fn, *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda Series for a row representing a query, and returns the new string query 
             - verbose (bool): Display a tqdm progress bar for this transformer
        """
        super().__init__(fn, 
            *args, 
            input = QUERIES,
            output = lambda input : input + [ "query_%d" % (_last_query(input)+1) ],
            **kwargs
            )

    def transform(self, inputRes):
        from .model import push_queries
        fn = self.fn
        outputRes = push_queries(inputRes.copy(), inplace=True, keep_original=True)
        if self.verbose:
            tqdm.pandas(desc="pt.apply.query", unit="d")
            outputRes["query"] = outputRes.progress_apply(fn, axis=1)
        else:
            outputRes["query"] = outputRes.apply(fn, axis=1)
        return outputRes

class ApplyGenericTransformer(ApplyTransformerBase):
    """
    Allows arbitrary pipelines components to be written as functions. The function should take as input
    a dataframe, and return a new dataframe. The function should abide by the main contracual obligations,
    e.g. updating then "rank" column.

    This class is normally accessed through pt.apply.generic()

    If you are scoring, query rewriting or calculating features, it is advised to use one of the other
    variants.

    Example::
        
        # this pipeline would remove all but the first two documents from a result set
        lp = ApplyGenericTransformer(lambda res : res[res["rank"] < 2])

        pipe = pt.BatchRetrieve(index) >> lp

    """

    def __init__(self, fn,  *args, **kwargs):
        """
            Arguments:
             - fn (Callable): Takes as input a panda DataFrame, and returns a new Pandas DataFrame 
        """
        super().__init__(fn, *args, **kwargs)

    def transform(self, inputRes):
        fn = self.fn
        return fn(inputRes)

class FeatureUnionPipeline(NAryTransformerBase):
    """
        Implements the feature union operator.

        Example::
            cands = pt.BatchRetrieve(index wmodel="BM25")
            pl2f = pt.BatchRetrieve(index wmodel="PL2F")
            bm25f = pt.BatchRetrieve(index wmodel="BM25F")
            pipe = cands >> (pl2f ** bm25f)
    """
    name = "FUnion"

    def __init__(self, operands, **kwargs):
        super().__init__(operands=operands, input=[], operand_output=RETRIEVED_DOCS, output=RETRIEVED_DOCS_FEATURES, **kwargs)

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

        for i, m in enumerate(self.models):
            #IMPORTANT this .copy() is important, in case an operand transformer changes inputRes
            results = m.transform(inputRes.copy())
            if len(results) == 0:
                raise ValueError("Got no results from %s, expected %d" % (repr(m), num_results) )
            assert not "features_x" in results.columns 
            assert not "features_y" in results.columns
            all_results.append( results )

    
        for i, (m, res) in enumerate(zip(self.models, all_results)):
            #IMPORTANT: dont do this BEFORE calling subsequent feature unions
            if not "features" in res.columns:
                if not "score" in res.columns:
                    raise ValueError("Results from %s did not include either score or features columns, found columns were %s" % (repr(m), str(res.columns)) )

                if len(res) != num_results:
                    warn("Got number of results different expected from %s, expected %d received %d, feature scores for any "
                        "missing documents be 0, extraneous documents will be removed" % (repr(m), num_results, len(res)))
                    all_results[i] = res = inputRes[["qid", "docno"]].merge(res, on=["qid", "docno"], how="left")
                    res["score"] = res["score"].fillna(value=0)

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
            rtr["features"] = rtr.apply(_concat_features, axis=1)
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

    def _calculate_output(self, op_output_cols):
        first = op_output_cols[0]
        for i, cols in enumerate(op_output_cols):
            assert set(first) == set(cols), "columns from operand %d returned columns %s, but we expected %s " % (i, str(cols), str(first))
        return op_output_cols[0]

class ComposedPipeline(NAryTransformerBase):
    """ 
        This class allows pipeline components to be chained together using the "then" operator.

        :Example:

        >>> comp = ComposedPipeline([ DPH_br, ApplyGenericTransformer(lambda res : res[res["rank"] < 2])])
        >>> # OR
        >>> # we can even use lambdas as transformers
        >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
        >>> # this is equivelent
        >>> # comp = DPH_br >> lambda res : res[res["rank"] < 2]]
    """
    name = "Compose"

    def index(self, iter : Iterable[dict], batch_size=100):
        """
        This methods implements indexing pipelines. It is responsible for calling the transform_iter() method of its 
        constituent transformers (except the last one) on batches of records, and the index() method on the last transformer.
        """
        from more_itertools import chunked
        
        if len(self.models) > 2:
            #this compose could have > 2 models. we need a composite transform() on all but the last
            prev_transformer = ComposedPipeline(self.models[0:-1])
        else:
            prev_transformer = self.models[0]
        last_transformer = self.models[-1]
        
        def gen():
            for batch in chunked(iter, batch_size):
                batch_df = prev_transformer.transform_iter(batch)
                for row in batch_df.itertuples(index=False):
                    yield row._asdict()
        return last_transformer.index(gen())

    def validate(self, inputs):
        # We validate the first transformer in the pipeline
        try:
            next_input = self.models[0].validate(inputs)
        except TypeError:
            raise PipelineError(self.models[0], inputs)

        # In the case where the first transformer of an nary transformer must return a certain type, regardless of
        # further transformers
        if self.input:
            #print("%s had required inputs %s" % (repr(self), self.input))
            try:
                super().validate(next_input)
            except TypeError:
                raise PipelineError(self, next_input, self.models[0])
        #print("after %s next_input is %s" % (repr(self.models[0]), str(next_input)))

        
        # We can then validate the other transformer in pipeline, with using the previous transformer output
        for i in range(len(self)-1):
            try:
                next_input = self.models[i+1].validate(next_input)
            except TypeError:
                raise PipelineError(self.models[i+1], next_input, self.models[i])

        return next_input

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
