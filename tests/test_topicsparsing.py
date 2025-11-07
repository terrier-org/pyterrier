import os

import pandas as pd

import pyterrier as pt
from pandas.testing import assert_frame_equal
from .base import BaseTestCase


class TestTopicsParsing(BaseTestCase):
    def testSingleLine(self):
        topics = pt.io.read_topics(
            os.path.dirname(os.path.realpath(__file__)) + "/fixtures/singleline.topics",
            format="singleline",
        )
        self.assertEqual(2, len(topics))
        self.assertTrue("qid" in topics.columns)
        self.assertTrue("query" in topics.columns)
        self.assertEqual(topics["qid"][0], "1")
        self.assertEqual(topics["qid"][1], "2")
        self.assertEqual(topics["query"][0], "one")
        self.assertEqual(topics["query"][1], "two words")

    def test_parse_trec_topics_file_T(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.trec"
        exp_result = pd.DataFrame(
            [["1", "light"], ["2", "radiowave"], ["3", "sound"]],
            columns=["qid", "query"],
        )
        result = pt.io.read_topics(input)
        assert_frame_equal(exp_result, result)

    def test_parse_trec_topics_file_TREC1(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.51-100.gz"
        results = pt.io.read_topics(input)
        print(results)
        self.assertEqual(results.query("qid == '51'").iloc[0]["query"], "Airbus Subsidies")

    def test_parse_trec_topics_file_D(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.trec"
        exp_result = pd.DataFrame(
            [["1", "lights"], ["2", "radiowaves"], ["3", "sounds"]],
            columns=["qid", "query"],
        )
        result = pt.io.read_topics(
            input, format="trec", whitelist=["desc"], blacklist=["title"]
        )
        assert_frame_equal(exp_result, result)

    def test_parse_trecxml_topics_file(self):
        input = os.path.dirname(os.path.realpath(__file__)) + "/fixtures/topics.trecxml"
        result = pt.io.read_topics(input, format="trecxml", tags=["title"])
        exp_result = pd.DataFrame(
            [["1", "lights"], ["2", "radiowaves"], ["3", "sounds"]],
            columns=["qid", "query"],
        )
        assert_frame_equal(exp_result, result)
