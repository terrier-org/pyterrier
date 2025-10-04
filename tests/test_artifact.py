import unittest
import tempfile
import urllib
import pyterrier as pt
from .base import BaseTestCase


class TestArtifact(BaseTestCase):
    def test_from_hf_and_build_package(self):
        # this test covers a lot of stuff:
        #  - mapping of hf URLs (via from_hf)
        #  - downloading and validating package
        #  - loading correct artifact type
        #  - build_package
        for mem in [True, False]:
            try:
                index = pt.Artifact.from_hf('pyterrier/vaswani.terrier', memory=mem)
            except urllib.error.HTTPError as ex:
                if ex.code != 429: # too many requests ... can just ignore
                    raise
            retr = index.bm25(num_results=10)
            self.assertEqual(10, len(retr.search('chemical reactions')))
        with tempfile.TemporaryDirectory() as d:
            index.build_package(d+'/artifact.tar.lz4')


if __name__ == "__main__":
    unittest.main()
