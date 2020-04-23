import unittest
import os
import pyterrier as pt

class BaseTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BaseTestCase, self).__init__(*args, **kwargs)
        terrier_version = os.environ.get("TERRIER_VERSION", None)
        if terrier_version is not None:
            print("Testing with Terrier version" + terrier_version)
        if not pt.started():
            pt.init(version=terrier_version)
        self.here = os.path.dirname(os.path.realpath(__file__))