
from pyterrier import mavenresolver
import unittest
import shutil
import tempfile
from os import path

class TestMaven(unittest.TestCase):
    ''' This doesn't rely on Terrier, so doesn't inherit from BaseTestCase '''

    def __init__(self, *args, **kwargs):
        super(TestMaven, self).__init__(*args, **kwargs)

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass

    def testVersion(self):
        ver = mavenresolver.latest_version_num("org.terrier", "terrier-core")
        self.assertGreaterEqual(float(ver), 5.3)

    def testDownload(self):
        ver = mavenresolver.latest_version_num("org.terrier", "terrier-python-helper")
        jar = mavenresolver.downloadfile("org.terrier", "terrier-python-helper", ver, self.test_dir)
        self.assertTrue(path.exists(jar))

    def testJitpack(self):
        jar = mavenresolver.downloadfile("com.github.terrierteam", "terrier-ciff", "-SNAPSHOT", self.test_dir)
        self.assertTrue(path.exists(jar))

if __name__ == "__main__":
    unittest.main()
