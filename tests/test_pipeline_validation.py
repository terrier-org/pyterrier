import pyterrier as pt
import unittest
import pandas as pd
from pyterrier.model import QUERIES
from pyterrier.validation import PipelineError, ValidationError
from pyterrier.pipelines import PerQueryMaxMinScoreTransformer#, ExperimentError
from pyterrier.transformer import TransformerBase, Family

from .base import BaseTestCase


class TestPipelineValidation(BaseTestCase):

    @staticmethod
    def get_index_and_queries():
        vaswani_dataset = pt.datasets.get_dataset("vaswani")
        indexref = vaswani_dataset.get_index()
        queries = vaswani_dataset.get_topics()
        return indexref, queries

    @staticmethod
    def get_transformer_family_instance(indexref):
        # Create an instance of each transformer family
        rewrite = pt.rewrite.SDM()
        retrieval = pt.BatchRetrieve(indexref, wmodel="BM25")
        expansion = pt.rewrite.QueryExpansion(indexref)
        reranking = pt.pipelines.PerQueryMaxMinScoreTransformer()
        featurescoring = pt.FeaturesBatchRetrieve(indexref, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])

        return rewrite, retrieval, expansion, reranking, featurescoring

    def test_invalid_pipeline_composition_fails(self):
        indexref, queries = TestPipelineValidation.get_index_and_queries()

        # We instantiate a member of each transformer family, then we test that each invalid pipeline composition is
        # caught
        rewrite, retrieval, expansion, reranking, featurescoring = TestPipelineValidation.get_transformer_family_instance(indexref)

        rewrite_into_expansion = rewrite >> expansion
        self.assertRaises(PipelineError, rewrite_into_expansion.validate, queries)

        rewrite_into_reranking = rewrite >> reranking
        self.assertRaises(PipelineError, rewrite_into_reranking.validate, queries)

        ranked_docs = retrieval(queries)

        expansion_into_expansion = expansion >> expansion
        self.assertRaises(PipelineError, expansion_into_expansion.validate, ranked_docs)

        expansion_into_reranking = expansion >> reranking
        self.assertRaises(PipelineError, expansion_into_reranking.validate, ranked_docs)

    def test_invalid_pipeline_operators_fails(self):
        indexref, queries = TestPipelineValidation.get_index_and_queries()

        # We instantiate a member of some transformer families that we know the minimal input/output for, and use them
        # to test that invalid transformer operations are caught
        rewrite = pt.rewrite.SDM()

        set_union = rewrite | rewrite
        self.assertRaises(PipelineError, set_union.validate, queries)

        set_intersection = rewrite & rewrite
        self.assertRaises(PipelineError, set_intersection.validate, queries)

        feature_union = rewrite ** rewrite
        self.assertRaises(PipelineError, feature_union.validate, queries)

        comb_sum = rewrite + rewrite
        self.assertRaises(PipelineError, comb_sum.validate, queries)

        concatenate = rewrite ^ rewrite
        self.assertRaises(PipelineError, concatenate.validate, queries)

        ranked_cutoff = rewrite % 10
        self.assertRaises(PipelineError, ranked_cutoff.validate, queries)

    def test_valid_pipelines_pass(self):
        indexref, queries = TestPipelineValidation.get_index_and_queries()

        # We instantiate the possible valid pipelines according to the transformer families and validate them
        rewrite, retrieval, expansion, reranking, featurescoring = TestPipelineValidation.get_transformer_family_instance(indexref)
        ranked_docs = retrieval(queries)

        # Matching minimal input and minimal output
        rewrite_into_retrieval = rewrite >> retrieval
        rewrite_into_retrieval.validate(queries)

        rewrite_into_feature_scoring = rewrite >> featurescoring
        rewrite_into_feature_scoring.validate(queries)

        expansion_into_retrieval = expansion >> retrieval
        expansion_into_retrieval.validate(ranked_docs)

        expansion_into_feature_scoring = expansion >> featurescoring
        expansion_into_feature_scoring.validate(ranked_docs)

        retrieval_into_expansion = retrieval >> expansion
        retrieval_into_expansion.validate(queries)

        retrieval_into_reranking = retrieval >> reranking
        retrieval_into_reranking.validate(queries)

        reranking_into_expansion = reranking >> expansion
        reranking_into_expansion.validate(ranked_docs)

        reranking_into_reranking = reranking >> reranking
        reranking_into_reranking.validate(ranked_docs)

        # minimal output is superset of minimal input
        retrieval_into_rewrite = retrieval >> rewrite
        retrieval_into_rewrite.validate(queries)

        retrieval_into_retrieval = retrieval >> retrieval
        retrieval_into_retrieval.validate(queries)

        reranking_into_rewrite = reranking >> rewrite
        reranking_into_rewrite.validate(ranked_docs)

        reranking_into_retrieval = reranking >> retrieval
        reranking_into_retrieval.validate(ranked_docs)

    @unittest.SkipTest
    def test_feature_scoring_validates(self):
        indexref, queries = TestPipelineValidation.get_index_and_queries()

        # We instantiate the possible valid pipelines according to the transformer families and validate them
        rewrite, retrieval, expansion, reranking, featurescoring = TestPipelineValidation.get_transformer_family_instance(indexref)
        ranked_docs = retrieval(queries)

        feature_scoring_into_rewrite = featurescoring >> rewrite
        feature_scoring_into_rewrite.validate(queries)

        feature_scoring_into_retrieval = featurescoring >> retrieval
        feature_scoring_into_retrieval.validate(queries)

        feature_scoring_into_expansion = featurescoring >> expansion
        feature_scoring_into_expansion.validate(ranked_docs)

        feature_scoring_into_reranking = featurescoring >> reranking
        feature_scoring_into_reranking.validate(ranked_docs)

    def test_validate_returns_correct_columns(self):
        # For some example pipelines given in the PyTerrier docs, we test that when validating the correct columns are
        # returned
        dataset = pt.datasets.get_dataset("vaswani")
        indexref = dataset.get_index()

        BM25 = pt.BatchRetrieve(indexref, wmodel="BM25")
        PL2 = pt.BatchRetrieve(indexref, wmodel="PL2")
        topics = dataset.get_topics()

        for p in [
            BM25,
            BM25 | PL2,
            BM25 % 100,
            ((BM25 % 100 >> PerQueryMaxMinScoreTransformer()) ^ BM25) % 1000,
            0.75 * BM25 + 0.25 * PL2
        ]:
            validate_output = p.validate(topics)
            self.assertIsNotNone(validate_output, p)
            transform_output = p.transform(topics)
            self.assertSetEqual(set(validate_output), set(transform_output.columns), p)


        # TODO: Test feature union pipeline on newest version
        # pipe = BM25 >> (TF_IDF ** PL2)
        # validate_output = pipe.validate("chemical end:2")
        # transform_output = pipe.transform("chemical end:2")
        # self.assertSetEqual(set(validate_output), set(transform_output.columns))

    def test_validation_error_raised(self):
        # If we create a transformer that does not have the necessary attributes for validation, a validation error
        # should be raised
        indexref, queries = TestPipelineValidation.get_index_and_queries()

        generic = pt.apply.generic(lambda res: res[res["rank"] < 2])
        br = pt.BatchRetrieve(indexref, wmodel="BM25")

        pipe = br >> generic
        self.assertRaises(ValidationError, pipe.validate, queries)

    def test_custom_transformer_validate(self):
        # A custom transformer should by default use minimal output as its true output, gotten from family
        class CustomTransformer1(TransformerBase):
            def __init__(self):
                super().__init__(family=Family.RETRIEVAL)

            def transform(self, t):
                return pd.DataFrame([["q1", "q1", "d1", "s1", "r1"]], columns=["qid", "query", "docno", "score", "rank"])

        # A custom transformer should use minimal output as true output, if defined literally
        class CustomTransformer2(TransformerBase):
            def __init__(self):
                super().__init__(input=QUERIES, output=QUERIES + ["url"])

            def transform(self, t):
                t["url"] = "location." + t.qid
                return t

        # A transformer should be able to take a function as its minimal output, and use that to validate
        class CustomTransformer3(TransformerBase):
            def __init__(self):
                def outputfn(inputs):
                    if isinstance(inputs, pd.DataFrame):
                        inputs = inputs.columns
                    return list({"qid", "query"} | set(inputs))

                super().__init__(input=["qid", "query"], output=outputfn)

            def transform(self, t):
                return t

        topics = pd.DataFrame([["q1", "q1"], ["q2", "q1"]], columns=["qid", "query"])
        custom_transformer1 = CustomTransformer1()
        self.assertSetEqual(set(custom_transformer1.validate(topics)), set(custom_transformer1.transform(topics)))

        custom_transformer2 = CustomTransformer2()
        self.assertSetEqual(set(custom_transformer2.validate(topics)), set(custom_transformer2.transform(topics)))

        topics = pd.DataFrame([["q1", "q1", "url1", "d1"], ["q2", "q1", "url1", "d1"]],
                               columns=["qid", "query", "url", "docno"])

        custom_transformer3 = CustomTransformer3()
        validate_output = custom_transformer3.validate(topics)
        transform_output = custom_transformer3.transform(topics)
        self.assertSetEqual(set(validate_output), set(transform_output.columns))

    def test_experiment_auto_validates(self):
        indexref, queries = TestPipelineValidation.get_index_and_queries()
        qrels = queries[["qid"]]
        qrels["docno"] = "d1"
        qrels["label"] = 1
        

        generic = pt.apply.generic(lambda res: res[res["rank"] < 2])
        rewrite = pt.rewrite.SDM()
        invalid_pipeline = rewrite | rewrite

        # A pipeline that cannot be validated should raise ValidationError if validate is True
        self.assertRaises(ValidationError, pt.Experiment, [generic], queries, qrels, ["map"], validate=True)

        # A given invalid pipeline should raise PipelineError if validate is True
        self.assertRaises(PipelineError, pt.Experiment, [invalid_pipeline], queries, qrels, ["map"], validate=True)

        # A given pipeline that cannot be experimented on should raise ExperimentError by default
        self.assertRaises(TypeError, pt.Experiment, [rewrite], queries, qrels, ["map"], validate=True)


if __name__ == "__main__":
    unittest.main()