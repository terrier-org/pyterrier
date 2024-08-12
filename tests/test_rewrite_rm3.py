import pandas as pd
import pyterrier as pt
import os
from matchpy import *
from .base import TempDirTestCase
import pytest

if not pt.started():
    terrier_version = os.environ.get("TERRIER_VERSION", None)
    terrier_helper_version = os.environ.get("TERRIER_HELPER_VERSION", None)
    pt.init(version=terrier_version, logging="DEBUG", helper_version=terrier_helper_version, boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    TERRIER_PRF_ON_CLASSPATH = True
else:
    TERRIER_PRF_ON_CLASSPATH = False


class TestRewriteRm3(TempDirTestCase):
    """This is a set of unit tests for RM3 that can currently not run in the complete test suite, as the "com.github.terrierteam:terrier-prf:-SNAPSHOT" would have to be added to the boot classpath.
    As workaround, the RM3 tests that can not be executed within the complete test suite are added to this dedicated file, so that they can be executed in isolation by running pytest tests/test_rewrite_rm3.py.
    """

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_rm3_expansion_for_query_compact_on_tf_idf(self):
        # top-retrieval results of TF-IDF and BM25 below change, so the RM3 weights differ
        expected = 'applypipeline:off equip^0.032653060 sideband^0.028571429 ferrit^0.028571429 modul^0.028571429 suppli^0.032653060 design^0.070748292 unit^0.032653060 anod^0.032653060 compact^0.680272102 stabil^0.032653060'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.RM3(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='TF_IDF')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(expected, actual.iloc[0]["query"])

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
    def test_rm3_expansion_for_query_compact_on_bm25(self):
        # top-retrieval results of BM25 and TF-IDF above change, so the RM3 weights differ
        expected = 'applypipeline:off equip^0.032653060 sideband^0.028571429 ferrit^0.028571429 modul^0.028571429 suppli^0.032653060 design^0.070748292 unit^0.032653060 anod^0.032653060 compact^0.680272102 stabil^0.032653060'
        
        indexref = pt.datasets.get_dataset("vaswani").get_index()
        queriesIn = pd.DataFrame([["1", "compact"]], columns=["qid", "query"])

        qe = pt.rewrite.RM3(indexref)
        br = pt.BatchRetrieve(indexref, wmodel='BM25')

        actual = qe.transform(br.transform(queriesIn))

        self.assertEqual(len(actual), 1)
        self.assertEqual(expected, actual.iloc[0]["query"])

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

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
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

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
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

    @pytest.mark.skipif(not TERRIER_PRF_ON_CLASSPATH, reason="This test only works in isolation when terrier-prf is on the jnius classpath.")
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