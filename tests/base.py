import unittest
import os
import pyterrier as pt

import tempfile
import shutil
import os

class BaseTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BaseTestCase, self).__init__(*args, **kwargs)
        terrier_version = os.environ.get("TERRIER_VERSION", None)
        terrier_helper_version = os.environ.get("TERRIER_HELPER_VERSION", None)
        if not pt.started():

            # display for debugging what is being used
            if terrier_version is not None:
                print("Testing with Terrier version " + terrier_version)
            if terrier_helper_version is not None:
                print("Testing with Terrier Helper version " + terrier_helper_version)
            
            pt.init(version=terrier_version, logging="DEBUG", helper_version=terrier_helper_version)
            # jvm_opts=['-ea'] can be added here to ensure that all Java assertions are met
        self.here = os.path.dirname(os.path.realpath(__file__))


    def skip_windows(self):
        if BaseTestCase.is_windows():
            self.skipTest("Test disabled on Windows")

    @staticmethod
    def is_windows() -> bool:
        import platform
        return platform.system() == 'Windows'

class TempDirTestCase(BaseTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass

    
