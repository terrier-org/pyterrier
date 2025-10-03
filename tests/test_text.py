import pandas as pd
import pyterrier as pt
from .base import BaseTestCase
import re


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

        # now check that text scorer can also work on an empty dataframe
        dfEmpty = pt.text.scorer(body_attr="text", wmodel="Tf").transform(dfIn.head(0))
        self.assertEqual(0, len(dfEmpty))
        self.assertIn("rank", dfEmpty.columns)
        self.assertIn("score", dfEmpty.columns)

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

    def test_snippets(self):
        br = pt.terrier.Retriever(self.here + "/fixtures/index/data.properties") >> pt.text.get_text(pt.get_dataset('irds:vaswani'), "text")
        #br = pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed_text", metadata=["docno", "text"])
        psg_scorer = ( 
            pt.text.sliding(text_attr='text', length=25, stride=12, prepend_attr=None) 
            >> pt.text.scorer(body_attr="text", wmodel='Tf', takes='docs')
        )
        pipe = br >> pt.text.snippets(psg_scorer)
        dfOut = pipe.search("chemical reactions")
        self.assertTrue("rank" in dfOut.columns)
        self.assertTrue("score" in dfOut.columns)
        self.assertTrue("summary" in dfOut.columns)
        #.count() checks number of non-NaN values
        #so lets count how many are NaN
        self.assertEqual(0, len(dfOut) - dfOut.summary.count())


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
            self.assertTrue(isinstance(textT, pt.transformer.Transformer))
            dfOut = textT.transform(dfinput)
            self.assertTrue(isinstance(dfOut, pd.DataFrame))
            self.assertTrue("docno" in dfOut.columns)
        
    def test_fetch_text_docid(self):
        dfinput = pd.DataFrame([["q1", "a query", 1]], columns=["qid", "query", "docid"])
        df_empty = dfinput.head(0)
        #directory, indexref, str, Index
        for indexlike in [
            pt.get_dataset("vaswani").get_index(), 
            pt.IndexRef.of(pt.get_dataset("vaswani").get_index()),
            pt.IndexRef.of(pt.get_dataset("vaswani").get_index()).toString(),
            pt.IndexFactory.of(pt.get_dataset("vaswani").get_index())
        ]:
            textT = pt.text.get_text(indexlike, "docno")
            self.assertTrue(isinstance(textT, pt.Transformer))
            dfOut = textT.transform(dfinput)
            self.assertTrue(isinstance(dfOut, pd.DataFrame))
            self.assertTrue("docno" in dfOut.columns)
            self.assertEqual('object', dfOut['docno'].dtype)

            dfOut2 = textT.transform(df_empty)
            self.assertTrue(isinstance(dfOut2, pd.DataFrame))
            self.assertTrue("docno" in dfOut2.columns)
            self.assertEqual('object', dfOut2['docno'].dtype)

    def test_fetch_text_irds(self):
        dfinput = pd.DataFrame([
            ["q1", "a query", "4"],
            ["q1", "a query", "1"],
            ["q1", "a query", "4"],
            ], columns=["qid", "query", "docno"])
        df_empty = dfinput.head(0)
        textT = pt.text.get_text(pt.get_dataset('irds:vaswani'), "text")
        self.assertTrue(isinstance(textT, pt.Transformer))
        dfOut = textT.transform(dfinput)
        self.assertTrue(isinstance(dfOut, pd.DataFrame))
        self.assertTrue("text" in dfOut.columns)
        self.assertEqual('object', dfOut['docno'].dtype)
        self.assertTrue("the british computer society  report of a conference held in cambridge\njune\n" in dfOut.iloc[0].text)
        self.assertTrue("compact memories have flexible capacities  a digital data storage\nsystem with capacity up to bits and random and or sequential access\nis described\n" in dfOut.iloc[1].text)
        self.assertTrue("the british computer society  report of a conference held in cambridge\njune\n" in dfOut.iloc[2].text)

        dfOut2 = textT.transform(df_empty)
        self.assertTrue(isinstance(dfOut2, pd.DataFrame))
        self.assertTrue("text" in dfOut2.columns)
        self.assertEqual('object', dfOut2['docno'].dtype)

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

    def test_passager_custom_tokenizer(self):
        class MockTokenizer:
            def tokenize(self, input_str):
                rx = r"\w+(?:\"\w+)?|[^\w\s]"
                return re.findall(rx, input_str)
            
            def convert_tokens_to_string(self, input_toks):
                return ' '.join(input_toks)
    
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "it's a sample document!"]], columns=["qid", "query", "docno", "body"])
        passager = pt.text.sliding(length=4, stride=3, prepend_attr=None, tokenizer=MockTokenizer())
        dfoutput = passager(dfinput)
        self.assertEqual(2, len(dfoutput))
        self.assertEqual("doc1%p0", dfoutput["docno"][0])
        self.assertEqual("doc1%p1", dfoutput["docno"][1])

        self.assertEqual("it ' s a", dfoutput["body"][0])
        self.assertEqual("a sample document !", dfoutput["body"][1])

    def test_passager_HGF(self):
        try: 
            from transformers import AutoTokenizer
        except:
            self.skipTest("`transformers` not installed")
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "it's a sample document!"]], columns=["qid", "query", "docno", "body"])
        passager = pt.text.sliding(length=4, stride=3, prepend_attr=None, tokenizer=tokenizer)
        dfoutput = passager(dfinput)
        self.assertEqual(2, len(dfoutput))
        self.assertEqual("doc1%p0", dfoutput["docno"][0])
        self.assertEqual("doc1%p1", dfoutput["docno"][1])

        self.assertEqual("it ' s a", dfoutput["body"][0])
        self.assertEqual("a sample document!", dfoutput["body"][1])


    def test_depassager(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", 1, "title", "body sentence"]], columns=["qid", "query", "docno", "docid", "title", "body"])
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
        self.assertFalse("docid" in dfmax.columns)
        self.assertFalse("pid" in dfmax.columns)
        self.assertFalse("olddocno" in dfmax.columns)
        self.assertEqual("a query", dfmax["query"][0])

        dffirst = pt.text.first_passage()(dfpassage)
        self.assertEqual(1, len(dffirst))
        self.assertEqual(1, dffirst["score"][0])
        self.assertTrue("body" in dffirst.columns)
        self.assertFalse("pid" in dffirst.columns)
        self.assertFalse("docid" in dffirst.columns)
        self.assertFalse("olddocno" in dffirst.columns)

        dfmean = pt.text.mean_passage()(dfpassage)
        self.assertEqual(1, len(dfmean))
        self.assertEqual(0.5, dfmean["score"][0])
        self.assertFalse("pid" in dfmean.columns)
        self.assertFalse("docid" in dfmean.columns)
        self.assertFalse("olddocno" in dfmax.columns)

        dfmeanK = pt.text.kmaxavg_passage(2)(dfpassage)
        self.assertEqual(1, len(dfmeanK))
        self.assertEqual(0.5, dfmeanK["score"][0])
        self.assertFalse("pid" in dfmeanK.columns)
        self.assertFalse("docid" in dfmeanK.columns)
        self.assertFalse("olddocno" in dfmeanK.columns)

        dfscores = pd.DataFrame([
            ["0", "doc1%p0", 3, pt.model.FIRST_RANK],
            ["0", "doc1%p1", 2, pt.model.FIRST_RANK+1],
            ["0", "doc1%p2", 1, pt.model.FIRST_RANK+2],
        ], columns=['qid', 'docno', 'score', 'rank'])

        dfmeanK2 = pt.text.kmaxavg_passage(2)(dfscores)
        print(dfmeanK2)
        self.assertEqual(1, len(dfmeanK2))
        self.assertEqual(2.5, dfmeanK2["score"][0])

        dfmeanK3 = pt.text.kmaxavg_passage(3)(dfscores)
        self.assertEqual(1, len(dfmeanK3))
        self.assertEqual(2, dfmeanK3["score"][0])

@pt.testing.transformer_test_class
def test_text_sliding():
    return pt.text.sliding()

@pt.testing.transformer_test_class
def test_max_passage():
    return pt.text.max_passage()
