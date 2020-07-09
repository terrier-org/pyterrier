import pyterrier as pt
import unittest
from .base import BaseTestCase
import os
import pandas as pd

class TestTopicsParsing(BaseTestCase):

    def testSingleLine(self):
        topics = pt.Utils.parse_singleline_topics_file(
            os.path.dirname(os.path.realpath(__file__)) + "/fixtures/singleline.topics")
        self.assertEqual(2, len(topics))
        self.assertTrue("qid" in topics.columns)
        self.assertTrue("query" in topics.columns)
        self.assertEqual(topics["qid"][0], "1")
        self.assertEqual(topics["qid"][1], "2")
        self.assertEqual(topics["query"][0], "one")
        self.assertEqual(topics["query"][1], "two words")

    def test_parse_trec_topics_file_T(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.trec"
        exp_result = pd.DataFrame([["1", "light"], ["2", "radiowave"], ["3", "sound"]], columns=['qid', 'query'])
        result = pt.Utils.parse_trec_topics_file(input)
        self.assertTrue(exp_result.equals(result))

    def test_parse_trec_topics_file_D(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.trec"
        exp_result = pd.DataFrame([["1", "lights"], ["2", "radiowaves"], ["3", "sounds"]], columns=['qid', 'query'])
        result = pt.Utils.parse_trec_topics_file(input, whitelist=["DESC"], blacklist=["TITLE"])
        self.assertTrue(exp_result.equals(result))