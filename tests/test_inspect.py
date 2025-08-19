import pandas as pd
import unittest
import pyterrier as pt
import inspect
from .base import BaseTestCase

class TestInspect(BaseTestCase):

    def test_terrier_retriever(self):
        JIR = pt.java.autoclass('org.terrier.querying.IndexRef')
        indexref = JIR.of(self.here + "/fixtures/index/data.properties")
        retr = pt.terrier.Retriever(indexref)
        textl = pt.text.get_text(pt.get_dataset('irds:vaswani'), "text")

        inputs = pt.inspect.transformer_inputs(retr)
        self.assertEqual([['qid', 'query'],  ['qid', 'query', 'docno', 'score', 'rank']], inputs)
        # with a text loader:
        inputs = pt.inspect.transformer_inputs(retr >> textl)
        self.assertEqual([['qid', 'query'],  ['qid', 'query', 'docno', 'score', 'rank']], inputs)

        # as a retriever
        rank_cols = pt.inspect.transformer_outputs(retr, ["qid", "query"])
        example_res = retr.search("what are chemical reactions")
        self.assertEqual(rank_cols, example_res.columns.tolist())

        # as a reranker
        rerank_cols = pt.inspect.transformer_outputs(retr, rank_cols)
        example_res = retr.transform(example_res)
        self.assertEqual(rerank_cols, example_res.columns.tolist())

        # with prf
        rm3 = pt.terrier.rewrite.RM3(indexref)
        prf_cols = pt.inspect.transformer_outputs(rm3, rank_cols)
        example_res = rm3.transform(example_res)
        self.assertEqual(prf_cols, example_res.columns.tolist())

        # as a feature pipeline
        feat_pipe = retr >> (retr ** retr)
        feat_cols = pt.inspect.transformer_outputs(feat_pipe, ["qid", "query"])
        example_res = feat_pipe.search("what are chemical reactions")
        self.assertEqual(feat_cols, example_res.columns.tolist())

        # with a text loader:
        textl_pipe = retr >> textl
        text_cols = pt.inspect.transformer_outputs(textl_pipe, ["qid", "query"])
        self.assertEqual(rank_cols + ["text"], text_cols)

        # in a linear combination
        linear_pipe = retr + retr
        linear_cols = pt.inspect.transformer_outputs(linear_pipe, ["qid", "query"])
        example_res = linear_pipe.search("what are chemical reactions")
        self.assertEqual(linear_cols, example_res.columns.tolist())

        # in a set combination
        intersect_pipe = retr & retr
        intersect_cols = pt.inspect.transformer_outputs(intersect_pipe, ["qid", "query"])
        example_res = intersect_pipe.search("what are chemical reactions")
        self.assertEqual(sorted(intersect_cols), sorted(example_res.columns.tolist()))

        subtransformers = pt.inspect.subtransformers(retr)
        self.assertEqual({}, subtransformers)

        attributes = pt.inspect.transformer_attributes(retr)
        self.assertEqual([
            pt.inspect.TransformerAttribute('index_location', retr.indexref),
            pt.inspect.TransformerAttribute('num_results', 1000),
            pt.inspect.TransformerAttribute('metadata', ['docno']),
            pt.inspect.TransformerAttribute('wmodel', 'DPH'),
            pt.inspect.TransformerAttribute('threads', 1),
            pt.inspect.TransformerAttribute('verbose', False),
            pt.inspect.TransformerAttribute('terrierql', 'on'),
            pt.inspect.TransformerAttribute('parsecontrols', 'on'),
            pt.inspect.TransformerAttribute('parseql', 'on'),
            pt.inspect.TransformerAttribute('applypipeline', 'on'),
            pt.inspect.TransformerAttribute('localmatching', 'on'),
            pt.inspect.TransformerAttribute('filters', 'on'),
            pt.inspect.TransformerAttribute('decorate', 'on'),
            pt.inspect.TransformerAttribute('decorate_batch', 'on'),
            pt.inspect.TransformerAttribute('querying.processes', 'terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,context_wmodel:org.terrier.python.WmodelFromContextProcess,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess,decorate:SimpleDecorateProcess'),
            pt.inspect.TransformerAttribute('querying.postfilters', 'decorate:SimpleDecorate,site:SiteFilter,scope:Scope'),
            pt.inspect.TransformerAttribute('querying.default.controls', 'wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on'),
            pt.inspect.TransformerAttribute('querying.allowed.controls', 'scope,qe,qemodel,start,end,site,scope,applypipeline'),
            pt.inspect.TransformerAttribute('termpipelines', 'Stopwords,PorterStemmer'),
        ], attributes)

        new_retr = pt.inspect.transformer_apply_attributes(retr, num_results=10)
        self.assertEqual(new_retr.controls['end'], '9')

        subtransformers = pt.inspect.subtransformers(retr >> textl)
        self.assertEqual({'transformers': [retr, textl]}, subtransformers)

        attributes = pt.inspect.transformer_attributes(retr >> textl)
        self.assertEqual(attributes, [
            pt.inspect.TransformerAttribute('transformers', (retr, textl), init_parameter_kind=inspect.Parameter.VAR_POSITIONAL)
        ], attributes)


    def test_ltr_pipeline(self):
        import numpy as np

        feat_res = pd.DataFrame([
            ["q1", "a", np.array([1,2])],
            ["q2", "a", np.array([2, 5])],
            ["q1", "b", np.array([3, 5])],
            ["q2", "b", np.array([4, 4])],
            ["q1", "c", np.array([5, 0])]
        ], columns=["qid", "docno", "features"])
        qrels = pd.DataFrame([
            ["q1", "a", 1],
            ["q1", "b", 0],
            ["q2", "a", 1],
            ["q2", "b", 0]
        ], columns=["qid", "docno", "label"])

        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        

        pipelines = [
            pt.ltr.apply_learned_model(rf),
            pt.ltr.ablate_features(1) >> pt.ltr.apply_learned_model(rf),
            pt.ltr.keep_features(0) >> pt.ltr.apply_learned_model(rf)
        ]
        try:
            import xgboost as xgb
            xgparams = {
                'objective': 'rank:ndcg',
                'learning_rate': 0.1,
                'gamma': 1.0, 'min_child_weight': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            pipelines.append(pt.ltr.apply_learned_model(xgb.XGBRanker(**xgparams), form="ltr"))
        except Exception as e:
            print("xgboost not installed, skipping xgboost pipeline test", e)
            pass

        try:
            import fastrank
            train_request = fastrank.TrainRequest.coordinate_ascent()
            params = train_request.params
            params.init_random = True
            params.normalize = True
            params.seed = 1234567
            pipelines.append(pt.ltr.apply_learned_model(train_request, form="fastrank"))
        except Exception as e: 
            print("fastrank not installed, skipping xgboost pipeline test", e)
            pass

        print("testing %d pipelines" % len(pipelines))
        for pipeline in pipelines:
            with self.subTest(pipeline=pipeline):
                print("Testing pipeline:", pipeline)
                pipeline.fit(feat_res, qrels, feat_res, qrels)
                result_res = pipeline.transform(feat_res)

                ltr_cols = pt.inspect.transformer_outputs(pipeline, feat_res.columns.tolist())
                self.assertEqual(ltr_cols, result_res.columns.tolist())

    def test_rewrite_query(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
        })
        t = pt.apply.query(lambda x: x["query"] + " context")
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertEqual(cols, t(df).columns.tolist())

    def test_rename(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'context': ['context1', 'context2']
        })
        br0 = pt.Transformer.from_df(df[['qid', 'query']], uniform=True)
        br1 = pt.Transformer.from_df(df, uniform=True)

        cols = pt.inspect.transformer_outputs(br1 >> pt.apply.rename({'context' : 'prompt'}), ["qid", "query", "context"])
        self.assertEqual(cols, ['qid', 'query', 'prompt'])

        with self.assertRaises(pt.validate.InputValidationError):
            pt.inspect.transformer_outputs(br0 >> pt.apply.rename({'context' : 'prompt'}), ["qid", "query"])

    def test_rename_nocols(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'context': ['context1', 'context2']
        })
        br0 = pt.Transformer.from_df(df[['qid', 'query']], uniform=True)
        br1 = pt.Transformer.from_df(df, uniform=True)

        def _rename_context(df):
            if len(df) == 0:
                raise ValueError("Empty DataFrame")
            return df.rename(columns={'context' : 'prompt'})

        with self.assertRaises(pt.inspect.InspectError):
            pt.inspect.transformer_outputs(br1 >> pt.apply.generic(_rename_context), ["qid", "query", "context"])

        with self.assertRaises(pt.inspect.InspectError):
            pt.inspect.transformer_outputs(br0 >> pt.apply.generic(_rename_context), ["qid", "query"])

if __name__ == "__main__":
    unittest.main()
        
