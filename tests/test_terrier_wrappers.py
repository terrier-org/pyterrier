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
        for i,j in TESTS.items():
            self.assertEqual(i, stemmer.stem(i))

        stemmer = pt.TerrierStemmer.portugese
        for i,j in TESTS.items():
            self.assertTrue(len(stemmer.stem(i)) > 0)

if __name__ == "__main__":
    unittest.main()
