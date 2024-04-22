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
        if terrier_version is not None:
            print("Testing with Terrier version " + terrier_version)
        terrier_helper_version = os.environ.get("TERRIER_HELPER_VERSION", None)
        if terrier_helper_version is not None:
            print("Testing with Terrier Helper version " + terrier_helper_version)
        if not pt.started():
            pt.init(version=terrier_version, logging="DEBUG", helper_version=terrier_helper_version)
            # jvm_opts=['-ea'] can be added here to ensure that all Java assertions are met
        self.here = os.path.dirname(os.path.realpath(__file__))
        assert "version" in pt.init_args
        assert pt.init_args["version"] == terrier_version


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

    