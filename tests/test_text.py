import pandas as pd
import pyterrier as pt
from .base import BaseTestCase


class TestText(BaseTestCase):

    def test_scorer(self):
        dfIn = pd.DataFrame(
            [
                ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
                ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
            ], columns=["qid", "query", "docno", "text"])
        dfOut = pt.text.scorer(body_attr="text", wmodel="Tf").transform(dfIn)
        self.assertTrue("rank" in dfOut.columns)
        self.assertTrue("score" in dfOut.columns)
        self.assertEqual(1.0, dfOut.iloc[0].score)
        self.assertEqual(1.0, dfOut.iloc[1].score)

    def test_scorer_rerank(self):
        #checks that the rank attribute is updated.
        dfIn = pd.DataFrame(
            [
                ["q1", "chemical reactions", "d1", 1.0, 0, "professor protor poured the chemicals"],
                ["q1", "chemical reactions", "d2", 0.9, 1, "chemical chemical chemical brothers turned up the beats"],
            ], columns=["qid", "query", "docno", "score", "rank", "text"])
        dfOut = pt.text.scorer(body_attr="text", wmodel="Tf").transform(dfIn)
        self.assertTrue("rank" in dfOut.columns)
        self.assertTrue("score" in dfOut.columns)
        self.assertEqual("d1", dfOut.iloc[0].docno)
        self.assertEqual(1.0, dfOut.iloc[0].score)
        self.assertEqual(3.0, dfOut.iloc[1].score)
        self.assertEqual(0, dfOut.iloc[1]["rank"])
        self.assertEqual(1, dfOut.iloc[0]["rank"])
        print(dfOut)

    def test_snippets(self):
        br = pt.BatchRetrieve.from_dataset("vaswani", "terrier_stemmed_text", metadata=["docno", "text"])
        psg_scorer = ( 
            pt.text.sliding(text_attr='text', length=5, prepend_attr=None) 
            >> pt.text.scorer(body_attr="text", wmodel='Tf')
            >> pt.debug.print_rows(jupyter=False)
        )
        pipe = br >> pt.text.snippets(psg_scorer)
        dfOut = pipe.search("chemical reactions")
        print(dfOut)
        self.assertTrue("rank" in dfOut.columns)
        self.assertTrue("score" in dfOut.columns)
        self.assertTrue("summary" in dfOut.columns)


    def test_fetch_text_docno(self):
        dfinput = pd.DataFrame([["q1", "a query", "1"]], columns=["qid", "query", "docno"])
        #directory, indexref, str, Index
        for indexlike in [
            pt.get_dataset("vaswani").get_index(), 
            pt.IndexRef.of(pt.get_dataset("vaswani").get_index()),
            pt.IndexRef.of(pt.get_dataset("vaswani").get_index()).toString(),
            pt.IndexFactory.of(pt.get_dataset("vaswani").get_index())
        ]:
            textT = pt.text.get_text(indexlike, "docno")
            self.assertTrue(isinstance(textT, pt.transformer.TransformerBase))
            dfOut = textT.transform(dfinput)
            self.assertTrue(isinstance(dfOut, pd.DataFrame))
            self.assertTrue("docno" in dfOut.columns)
        
    def test_fetch_text_docid(self):
        dfinput = pd.DataFrame([["q1", "a query", 1]], columns=["qid", "query", "docid"])
        #directory, indexref, str, Index
        for indexlike in [
            pt.get_dataset("vaswani").get_index(), 
            pt.IndexRef.of(pt.get_dataset("vaswani").get_index()),
            pt.IndexRef.of(pt.get_dataset("vaswani").get_index()).toString(),
            pt.IndexFactory.of(pt.get_dataset("vaswani").get_index())
        ]:
            textT = pt.text.get_text(indexlike, "docno")
            self.assertTrue(isinstance(textT, pt.transformer.TransformerBase))
            dfOut = textT.transform(dfinput)
            self.assertTrue(isinstance(dfOut, pd.DataFrame))
            self.assertTrue("docno" in dfOut.columns)

    def test_fetch_text_irds(self):
        dfinput = pd.DataFrame([
            ["q1", "a query", "4"],
            ["q1", "a query", "1"],
            ["q1", "a query", "4"],
            ], columns=["qid", "query", "docno"])
        textT = pt.text.get_text(pt.get_dataset('irds:vaswani'), "text")
        self.assertTrue(isinstance(textT, pt.transformer.TransformerBase))
        dfOut = textT.transform(dfinput)
        self.assertTrue(isinstance(dfOut, pd.DataFrame))
        self.assertTrue("text" in dfOut.columns)
        self.assertTrue("the british computer society  report of a conference held in cambridge\njune\n" in dfOut.iloc[0].text)
        self.assertTrue("compact memories have flexible capacities  a digital data storage\nsystem with capacity up to bits and random and or sequential access\nis described\n" in dfOut.iloc[1].text)
        self.assertTrue("the british computer society  report of a conference held in cambridge\njune\n" in dfOut.iloc[2].text)

    def test_passager_title(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "title", "body sentence"]], columns=["qid", "query", "docno", "title", "body"])
        passager = pt.text.sliding(length=1, stride=1)
        dfoutput = passager(dfinput)
        self.assertEqual(2, len(dfoutput))
        self.assertEqual("doc1%p0", dfoutput["docno"][0])
        self.assertEqual("doc1%p1", dfoutput["docno"][1])
        
        self.assertEqual("title body", dfoutput["body"][0])
        self.assertEqual("title sentence", dfoutput["body"][1])

    def test_passager(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "body sentence"]], columns=["qid", "query", "docno", "body"])
        passager = pt.text.sliding(length=1, stride=1, prepend_attr=None)
        dfoutput = passager(dfinput)
        self.assertEqual(2, len(dfoutput))
        self.assertEqual("doc1%p0", dfoutput["docno"][0])
        self.assertEqual("doc1%p1", dfoutput["docno"][1])
        
        self.assertEqual("body", dfoutput["body"][0])
        self.assertEqual("sentence", dfoutput["body"][1])

    def test_depassager(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "title", "body sentence"]], columns=["qid", "query", "docno", "title", "body"])
        passager = pt.text.sliding(length=1, stride=1)
        dfpassage = passager(dfinput)
        #     qid    query    docno            body                                         
        # 0  q1  a query  doc1%p0      title body
        # 1  q1  a query  doc1%p1  title sentence
        dfpassage["score"] = [1, 0]

        dfmax = pt.text.max_passage()(dfpassage)
        self.assertEqual(1, len(dfmax))
        self.assertEqual(1, dfmax["score"][0])
        self.assertTrue("query" in dfmax.columns)
        self.assertTrue("body" in dfmax.columns)
        self.assertFalse("pid" in dfmax.columns)
        self.assertFalse("olddocno" in dfmax.columns)
        self.assertEqual("a query", dfmax["query"][0])

        dffirst = pt.text.first_passage()(dfpassage)
        self.assertEqual(1, len(dffirst))
        self.assertEqual(1, dffirst["score"][0])
        self.assertTrue("body" in dffirst.columns)
        self.assertFalse("pid" in dffirst.columns)
        self.assertFalse("olddocno" in dffirst.columns)

        dfmean = pt.text.mean_passage()(dfpassage)
        self.assertEqual(1, len(dfmean))
        self.assertEqual(0.5, dfmean["score"][0])
        self.assertFalse("pid" in dfmean.columns)
        self.assertFalse("olddocno" in dfmax.columns)

        dfmeanK = pt.text.kmaxavg_passage(2)(dfpassage)
        self.assertEqual(1, len(dfmeanK))
        self.assertEqual(0.5, dfmeanK["score"][0])
        self.assertFalse("pid" in dfmeanK.columns)
        self.assertFalse("olddocno" in dfmeanK.columns)