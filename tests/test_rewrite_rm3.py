import pandas as pd
import pyterrier as pt
import os
from matchpy import *
from .base import TempDirTestCase
import pytest

if not pt.java.started():
    terrier_version = os.environ.get("TERRIER_VERSION", None)
    terrier_helper_version = os.environ.get("TERRIER_HELPER_VERSION", None)
    pt.java.set_log_level('DEBUG')
    pt.terrier.set_version(terrier_version)
    pt.terrier.set_helper_version(terrier_helper_version)
    pt.terrier.set_prf_version('rm_tiebreak-SNAPSHOT')
    pt.java.init() # optional, forces java initialisation
    TERRIER_PRF_ON_CLASSPATH = True
else:
    TERRIER_PRF_ON_CLASSPATH = False


def normalize_term_weights(term_weights, digits=7):
    ret = ''
    for i in term_weights.split():
        if '^' in i:
            i = i.split('^')
            i = i[0] + '^' + i[1][:digits]
        ret += ' ' + i
    return ret.strip()

class TestRewriteRm3(TempDirTestCase):
    """This is a set of unit tests for RM3 that can currently not run in the complete test suite, as the "com.github.terrierteam:terrier-prf:-SNAPSHOT" would have to be added to the boot classpath.
    As workaround, the RM3 tests that can not be executed within the complete test suite are added to this dedicated file, so that they can be executed in isolation by running pytest tests/test_rewrite_rm3.py.
    """

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_rm3_expansion_for_query_compact_on_tf_idf(self):
        # top-retrieval results of TF-IDF and BM25 below change, so the RM3 weights differ
        expected = 'applypipeline:off equip^0.037346367 ferrit^0.027371584 modul^0.027371584 suppli^0.037346367 design^0.056739070 microwav^0.027371584 anod^0.037346367 unit^0.037346367 compact^0.674414337 stabil^0.037346367'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.RM3(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='TF_IDF')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(normalize_term_weights(expected), normalize_term_weights(actual.iloc[0]["query"]))

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_rm3_expansion_for_query_compact_on_bm25(self):
        # top-retrieval results of BM25 and TF-IDF above change, so the RM3 weights differ
        expected = 'applypipeline:off equip^0.040264644 ferrit^0.025508024 modul^0.025508024 suppli^0.040264644 design^0.051008239 microwav^0.025508024 anod^0.040264644 unit^0.040264644 compact^0.671144485 stabil^0.040264644'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.RM3(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='BM25')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(normalize_term_weights(expected), normalize_term_weights(actual.iloc[0]["query"]))

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_axiomatic_qe_expansion_for_query_compact_on_bm25(self):
        # just ensure that AxiomaticQE results do not change
        expected = 'applypipeline:off compact^1.000000000'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.AxiomaticQE(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='BM25')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(expected, actual.iloc[0]["query"])

    def test_kl_qe_expansion_for_query_compact_on_bm25(self):
        # just ensure that KLQueryExpansion results do not change
        expected = 'applypipeline:off compact^1.840895333 design^0.348370740 equip^0.000000000 purpos^0.000000000 instrument^0.000000000 ferrit^0.000000000 anod^0.000000000 aircraft^0.000000000 microwav^0.000000000 sideband^0.000000000'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.KLQueryExpansion(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='BM25')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(expected, actual.iloc[0]["query"])

    def test_bo1_qe_expansion_for_query_compact_on_bm25(self):
        # just ensure that Bo1QueryExpansion results do not change
        expected = 'applypipeline:off compact^1.822309726 design^0.287992096 equip^0.000000000 purpos^0.000000000 instrument^0.000000000 ferrit^0.000000000 anod^0.000000000 aircraft^0.000000000 microwav^0.000000000 sideband^0.000000000'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.Bo1QueryExpansion(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='BM25')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(expected, actual.iloc[0]["query"])

    def test_dfr_qe_expansion_for_query_compact_on_bm25(self):
        # just ensure that DFRQueryExpansion results do not change
        expected = 'applypipeline:off compact^1.822309726 design^0.287992096 equip^0.000000000 purpos^0.000000000 instrument^0.000000000 ferrit^0.000000000 anod^0.000000000 aircraft^0.000000000 microwav^0.000000000 sideband^0.000000000'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.DFRQueryExpansion(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='BM25')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(expected, actual.iloc[0]["query"])

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_rm3_end_to_end(self):
        """An end-to-end test, contrasting the smaller tests (that fail faster) from above.
        """
        dataset = pt.datasets.get_dataset("vaswani")
        indexref = dataset.get_index()

        qe = pt.rewrite.RM3(indexref)
        br = pt.BatchRetrieve(indexref)

        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])
        res = br.transform(queriesIn)

        queriesOut = qe.transform(res)
        self.assertEqual(len(queriesOut), 1)
        query = queriesOut.iloc[0]["query"]
        #self.assertTrue("compact^1.82230972" in query)
        self.assertTrue("applypipeline:off " in query)
        
        pipe = br >> qe >> br

        # lets go faster, we only need 18 topics. qid 16 had a tricky case
        t = dataset.get_topics().head(18)

        all_qe_res = pipe.transform(t)
        map_pipe = pt.Evaluate(all_qe_res, dataset.get_qrels(), metrics=["map"])["map"]

        br_qe = pt.BatchRetrieve(indexref, 
            controls={"qe":"on"},
            properties={"querying.processes" : "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,"\
                    +"parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,"\
                    +"sd:DependenceModelPreProcess,localmatching:LocalManager$ApplyLocalMatching,qe:RM3,"\
                    +"labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess"})
        map_qe = pt.Evaluate(br_qe.transform(t), dataset.get_qrels(), metrics=["map"])["map"]

        self.assertAlmostEqual(map_qe, map_pipe, places=2)

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_scoring_rm3_qe(self):
        expected = 'applypipeline:off fox^0.600000024'
        input = pd.DataFrame([["q1", "fox", "d1", "all the fox were fox", 3], ["q1", "fox", "d2", "brown fox jumps", 2]], columns=["qid", "query", "docno", "body", "score"])
        scorer = pt.terrier.retriever.TextIndexProcessor(pt.rewrite.RM3, takes="docs", returns="queries")
        rtr = scorer(input)
        self.assertTrue("qid" in rtr.columns)
        self.assertTrue("query" in rtr.columns)
        self.assertTrue("docno" not in rtr.columns)
        self.assertTrue(expected, rtr.iloc[0]["query"])
