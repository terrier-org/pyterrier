import pandas as pd
import unittest
import os
import pyterrier as pt
from .base import BaseTestCase
from .test_fbr import TestFeaturesBatchRetrieve
import warnings

class TestDVFeaturesBatchRetrieve(TestFeaturesBatchRetrieve):

    def __init__(self, *args):
        super().__init__(*args)
        self.method = 'dv'

    def check_version(self):
        if not pt.terrier.check_version("5.10"):
            self.skipTest("Terrier 5.10 is required")

if __name__ == "__main__":
    unittest.main()
