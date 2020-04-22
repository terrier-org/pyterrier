
from pyterrier import mavenresolver
import unittest
import shutil
import tempfile
from os import path


class TestMaven(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMaven, self).__init__(*args, **kwargs)

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testVersion(self):
        ver = mavenresolver.latest_version_num("org.terrier", "terrier-core")
        self.assertGreaterEqual(5.2, float(ver))

    def testDownload(self):
        ver = mavenresolver.latest_version_num("org.terrier", "terrier-python-helper")
        jar = mavenresolver.downloadfile("org.terrier", "terrier-python-helper", ver, self.test_dir)
        self.assertTrue(path.exists(jar))

if __name__ == "__main__":
    unittest.main()
