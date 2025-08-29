import pandas as pd
import unittest
import pyterrier as pt
import inspect
from .base import BaseTestCase
from functools import partial

class TestInspect(BaseTestCase):

    def assertSortedEquals(self, arr1, arr2):
        self.assertEqual(sorted(arr1), sorted(arr2))

    def assertInNoOrder(self, member, container):
        container_set = [set(item) for item in container]
        self.assertIn(set(member), container_set)     

    def test_inspection_mincols(self):
        op_input = [[['qid', 'docno']]]
        sub_inputs = [
            [['qid', 'docno', 'query']],
            [['qid', 'docno', 'query']]
        ]
        plausible_configs = pt.inspect._minimal_inputs(op_input + sub_inputs)
        self.assertInNoOrder(['qid', 'docno', 'query'], plausible_configs)


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
        self.assertSortedEquals(rank_cols, example_res.columns.tolist())

        # as a reranker
        rerank_cols = pt.inspect.transformer_outputs(retr, rank_cols)
        example_res = retr.transform(example_res)
        self.assertSortedEquals(rerank_cols, example_res.columns.tolist())

        # with prf
        rm3 = pt.terrier.rewrite.RM3(indexref)
        prf_cols = pt.inspect.transformer_outputs(rm3, rank_cols)
        example_res = rm3.transform(example_res)
        self.assertSortedEquals(prf_cols, example_res.columns.tolist())

        # as a feature pipeline
        feat_pipe = retr >> (retr ** retr)
        feat_cols = pt.inspect.transformer_outputs(feat_pipe, ["qid", "query"])
        example_res = feat_pipe.search("what are chemical reactions")
        self.assertSortedEquals(feat_cols, example_res.columns.tolist())

        # with a text loader:
        textl_pipe = retr >> textl
        text_cols = pt.inspect.transformer_outputs(textl_pipe, ["qid", "query"])
        self.assertSortedEquals(rank_cols + ["text"], text_cols)

        # in a linear combination
        linear_pipe = retr + retr
        linear_cols = pt.inspect.transformer_outputs(linear_pipe, ["qid", "query"])
        example_res = linear_pipe.search("what are chemical reactions")
        self.assertSortedEquals(linear_cols, example_res.columns.tolist())

        # in a set combination
        intersect_pipe = retr & retr
        intersect_cols = pt.inspect.transformer_outputs(intersect_pipe, ["qid", "query"])
        example_res = intersect_pipe.search("what are chemical reactions")
        self.assertSortedEquals(intersect_cols, example_res.columns.tolist())

        subtransformers = pt.inspect.subtransformers(retr)
        self.assertEqual({}, subtransformers)

        attributes = pt.inspect.transformer_attributes(retr)
        self.assertEqual([
            pt.inspect.TransformerAttribute('index_location', retr.indexref),
            pt.inspect.TransformerAttribute('num_results', 1000),
            pt.inspect.TransformerAttribute('metadata', retr.metadata),
            pt.inspect.TransformerAttribute('wmodel', retr.controls['wmodel']),
            pt.inspect.TransformerAttribute('threads', retr.threads),
            pt.inspect.TransformerAttribute('verbose', retr.verbose),
            pt.inspect.TransformerAttribute('terrierql', retr.controls['terrierql']),
            pt.inspect.TransformerAttribute('parsecontrols', retr.controls['parsecontrols']),
            pt.inspect.TransformerAttribute('parseql', retr.controls['parseql']),
            pt.inspect.TransformerAttribute('applypipeline', retr.controls['applypipeline']),
            pt.inspect.TransformerAttribute('localmatching', retr.controls['localmatching']),
            pt.inspect.TransformerAttribute('filters', retr.controls['filters']),
            pt.inspect.TransformerAttribute('decorate', retr.controls['decorate']),
            pt.inspect.TransformerAttribute('decorate_batch', retr.controls['decorate_batch']),
            pt.inspect.TransformerAttribute('querying.processes', retr.properties['querying.processes']),
            pt.inspect.TransformerAttribute('querying.postfilters', retr.properties['querying.postfilters']),
            pt.inspect.TransformerAttribute('querying.default.controls', retr.properties['querying.default.controls']),
            pt.inspect.TransformerAttribute('querying.allowed.controls', retr.properties['querying.allowed.controls']),
            pt.inspect.TransformerAttribute('termpipelines', retr.properties['termpipelines']),
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
                self.assertSortedEquals(ltr_cols, result_res.columns.tolist())

    def test_rewrite_query(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
        })
        t = pt.apply.query(lambda x: x["query"] + " context")
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertSortedEquals(cols, t(df).columns.tolist())

    def test_apply_docscore(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'docno': ['doc1', 'doc2']
        })
        t = pt.apply.doc_score(lambda x: 1)
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('score', cols)
        self.assertIn('rank', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

    def test_apply_docfeatures(self):
        import numpy as np
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'docno': ['doc1', 'doc2']
        })
        t = pt.apply.doc_features(lambda x: np.array([1.0, 2.0]))
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('features', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

    def test_apply_mkcol(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'docno': ['doc1', 'doc2']
        })
        t = pt.apply.newcol(lambda x: x['query'] + " context")
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('newcol', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

    def test_apply_dropcol(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'docno': ['doc1', 'doc2']
        })
        t = pt.apply.docno(drop=True)
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertNotIn('docno', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

    def test_apply_generic(self):
        self._apply_generic(pt.apply.generic)

    def test_apply_byquery(self):
        self._apply_generic(partial(pt.apply.by_query, add_ranks=False))

    def _apply_generic(self, applyfn):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'docno': ['doc1', 'doc2']
        })

        # Define a generic function that adds a new column based on existing columns
        # this assugnment works with an empty DataFrame
        def _generic_func(df):
            return df.assign(newcol=df['query'] + " context")

        t = applyfn(_generic_func)
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('newcol', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

        def _generic_func_noempty(df):
            if len(df) == 0:
                return df # bad behavior, but we want to empty dfs to have the same return columns
            return df.assign(newcol=df['query'] + " context")

        t = applyfn(_generic_func_noempty)
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertNotIn('newcol', cols)
        
        # undesired behavior: raises error on empty DataFrame - cannot be inspected
        def _generic_func_empty(df):
            if len(df) == 0:
                raise ValueError("Empty DataFrame")
            return df.assign(newcol=df['query'] + " context")
        
        t = applyfn(_generic_func_empty)
        with self.assertRaises(pt.inspect.InspectError):
            cols = pt.inspect.transformer_outputs(t, df.columns.tolist())

        # now check we can specify the columns when specifying transform_outputs
        t = applyfn(_generic_func_empty, transform_outputs=lambda cols: cols + ['newcol'])
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('newcol', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

        # same for the erroring function
        t = applyfn(_generic_func_noempty, transform_outputs=lambda cols: cols + ['newcol'])
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('newcol', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())

    def test_apply_generic_iter(self):
        self._apply_generic_iter(pt.apply.generic)

    def test_apply_byquery_iter(self):
        self._apply_generic_iter(pt.apply.by_query)

    def _apply_generic_iter(self, applyfn):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'docno': ['doc1', 'doc2']
        })

        # Define a generic function that adds a new column based on existing columns
        def _generic_func_iter(iter):
            for r in iter:
                r["newcol"] = r['query'] + " context"
                yield r

        # an iter-only transformer is not inspectable
        t = applyfn(_generic_func_iter, iter=True)
        with self.assertRaises(pt.inspect.InspectError) as ie:
            cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('not inspectable', str(ie.exception.__cause__))

        t = applyfn(_generic_func_iter, iter=True, transform_outputs=lambda cols: cols + ['newcol'])
        cols = pt.inspect.transformer_outputs(t, df.columns.tolist())
        self.assertIn('newcol', cols)
        self.assertSortedEquals(cols, t(df).columns.tolist())
    
    def test_rename(self):
        df = pd.DataFrame({
            'qid': ['1', '2'],
            'query': ['query1', 'query2'],
            'context': ['context1', 'context2']
        })
        br0 = pt.Transformer.from_df(df[['qid', 'query']], uniform=True)
        br1 = pt.Transformer.from_df(df, uniform=True)

        cols = pt.inspect.transformer_outputs(br1 >> pt.apply.rename({'context' : 'prompt'}), ["qid", "query", "context"])
        self.assertSortedEquals(cols, ['qid', 'query', 'prompt'])

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

    def test_transformer_type(self):
        class A(pt.Transformer):
            def transform(self, inp):
                pass
        class B(pt.Indexer):
            def index(self, inp):
                pass
        class C(pt.Indexer):
            def transform(self, inp):
                pass
            def index(self, inp):
                pass

        self.assertEqual(pt.inspect.transformer_type(A()), pt.inspect.TransformerType.transformer)
        self.assertEqual(pt.inspect.transformer_type(B()), pt.inspect.TransformerType.indexer)
        self.assertEqual(pt.inspect.transformer_type(C()), pt.inspect.TransformerType.transformer | pt.inspect.TransformerType.indexer)
        self.assertEqual(pt.inspect.transformer_type(object()), pt.inspect.TransformerType(0))

if __name__ == "__main__":
    unittest.main()
        
