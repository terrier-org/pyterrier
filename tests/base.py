import unittest
import os
import tempfile
import shutil
import pyterrier as pt

class BaseTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BaseTestCase, self).__init__(*args, **kwargs)
        self.here = os.path.dirname(os.path.realpath(__file__))
        if not pt.java.started():
            terrier_version = os.environ.get("TERRIER_VERSION", None)
            terrier_helper_version = os.environ.get("TERRIER_HELPER_VERSION", None)

            # display for debugging what is being used
            if terrier_version is not None:
                print("Testing with Terrier version " + terrier_version)
            if terrier_helper_version is not None:
                print("Testing with Terrier Helper version " + terrier_helper_version)
            
            pt.terrier.set_version(terrier_version)
            pt.terrier.set_helper_version(terrier_helper_version)
            pt.java.set_log_level("DEBUG")
            # pt.java.add_option('-ea') can be added here to ensure that all Java assertions are met
            pt.java.init()

    def skip_windows(self):
        if pt.utils.is_windows():
            self.skipTest("Test disabled on Windows")


class TempDirTestCase(BaseTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass

    
parallel_test = unittest.skipIf(os.environ.get("PARALLEL_TESTING") is None, "Parallel test disabled, enable with PARALLEL_TESTING=1")
