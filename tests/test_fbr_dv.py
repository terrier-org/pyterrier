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

if __name__ == "__main__":
    unittest.main()
