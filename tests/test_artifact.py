import unittest
import tempfile
import pyterrier as pt
from .base import BaseTestCase


class TestArtifact(BaseTestCase):
    def test_from_hf_and_build_package(self):
        # this test covers a lot of stuff:
        #  - mapping of hf URLs (via from_hf)
        #  - downloading and validating package
        #  - loading correct artifact type
        #  - build_package
        index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
        retr = index.bm25(num_results=10)
        self.assertEqual(10, len(retr.search('chemical reactions')))
        with tempfile.TemporaryDirectory() as d:
            index.build_package(d+'/artifact.tar.lz4')


if __name__ == "__main__":
    unittest.main()
