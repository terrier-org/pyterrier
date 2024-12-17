import unittest
from .base import BaseTestCase
import pyterrier as pt
import ir_datasets

class TestTokenisers(BaseTestCase):
    def test_english(self):
        tokeniser = pt.terrier.tokeniser.EnglishTokeniser.tokenise
        self._test_english(tokeniser)
        self.assertEqual(tokeniser("a\u0133a"), ["a", "a"])

    def test_UTF(self):
        import html
        tokeniser = pt.terrier.tokeniser.UTFTokeniser.tokenise
        self._test_english(tokeniser)
        # all examples come from https://github.com/terrier-org/terrier-core/blob/5.x/modules/tests/src/test/java/org/terrier/indexing/tokenisation/TestUTFTokeniser.java

        self.assertEqual(tokeniser("a\u0133a"), ["a\u0133a"])
        self.assertEqual(tokeniser("\u00C0\u00C8\u00CC"), ["\u00C0\u00C8\u00CC".lower()])

        word = html.unescape("&#2327;&#2369;")
        self.assertEqual(tokeniser(word), [word])
        words = ["&#2327;&#2369;&#2332;&#2381;&#2332;&#2352;&#2379;&#2306;",
				"&#2324;&#2352;&#2350;&#2368;&#2339;&#2366;",
				"&#2360;&#2350;&#2369;&#2342;&#2366;&#2351;",
				"&#2325;&#2375;&#2348;&#2368;&#2330;",
				"&#2360;&#2306;&#2328;&#2352;&#2381;&#2359;"]
        words = [html.unescape(w) for w in words]
        self.assertEqual(tokeniser(" ".join(words)), words)

    def _test_english(self, tokeniser):
        # all these examples come from https://github.com/terrier-org/terrier-core/blob/5.x/modules/tests/src/test/java/org/terrier/indexing/tokenisation/TestEnglishTokeniser.java
        
        self.assertEqual(tokeniser("hello"), ["hello"])
        self.assertEqual(tokeniser("a"), ["a"])

        self.assertEqual(tokeniser(""), [])

        self.assertEqual(tokeniser("a bb c"), ["a", "bb", "c"])
        self.assertEqual(tokeniser("a bbb c"), ["a", "bbb", "c"])
        self.assertEqual(tokeniser("a bbbb c"), ["a", "c"])

        self.assertEqual(tokeniser("hello there"), ["hello", "there"])
        self.assertEqual(tokeniser("hello there    "), ["hello", "there"])
        self.assertEqual(tokeniser("a very      big  hello there"), ["a", "very", "big", "hello", "there"])
        self.assertEqual(tokeniser("hello there mr wolf thisisareallylongword aye"), ["hello", "there", "mr", "wolf", "aye"])
        for c in ['.', ',', '-', ':', '\\', '/', '?', '<', '>']:
            self.assertEqual(tokeniser("a"+c+"b"), ["a", "b"])
        self.assertEqual(tokeniser("a.b  "), ["a", "b"])
        self.assertEqual(tokeniser("a;b  -"), ["a", "b"])
        self.assertEqual(tokeniser("a;b  -"), ["a", "b"])
        self.assertEqual(tokeniser("...   a;b ?"), ["a", "b"])

    @unittest.skip("too expensive to run normally")
    def test_python_java_faithfulness(self):
        if not pt.java.started():
            pt.java.init()
        for tokenizer in ['utf', 'english']:
            py_tokenizer = {
                'utf': pt.terrier.tokeniser.UTFTokeniser.tokenise,
                'english': pt.terrier.tokeniser.EnglishTokeniser.tokenise,
            }[tokenizer]
            java_tokenizer = {
                'utf': pt.java.autoclass('org.terrier.indexing.tokenisation.UTFTokeniser')().getTokens,
                'english': pt.java.autoclass('org.terrier.indexing.tokenisation.EnglishTokeniser')().getTokens,
            }[tokenizer]
            for doc in ir_datasets.load('msmarco-passage').docs[:1_000_000]:
                with self.subTest(tokenizer=tokenizer, doc_id=doc.doc_id):
                    self.assertEqual(py_tokenizer(doc.text), java_tokenizer(doc.text))
