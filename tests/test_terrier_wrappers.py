import pandas as pd
import pyterrier as pt
import os
import unittest
from .base import BaseTestCase
import warnings

class TestTerrierWrappers(BaseTestCase):

    def test_stemming(self):
        stemmer = pt.TerrierStemmer.porter
        TESTS = {
            "abandoned": "abandon",
    		"abandon": "abandon",
	    	"abergavenny": "abergavenni"
        }
        for i,j in TESTS.items():
            self.assertEqual(j, stemmer.stem(i))

        stemmer = pt.TerrierStemmer.none
        TESTS = {
            "abandoned": "abandon",
    		"abandon": "abandon",
	    	"abergavenny": "abergavenni"
        }
        for i,j in TESTS.items():
            self.assertEqual(i, stemmer.stem(i))

if __name__ == "__main__":
    unittest.main()
