import unittest
import os
import pyterrier as pt
class AnseriniTestCase(unittest.TestCase):

    def skip_pyserini(self):
        try:
            import pyserini.setup
            pt.anserini._init_anserini()
        except BaseException as e:
            if os.environ.get("ANSERINI_TESTING", None) is not None:
                raise e
            else:
                self.skipTest("Test disabled due to lack of Pyserini")

    def __init__(self, *args, **kwargs):
        super(AnseriniTestCase, self).__init__(*args, **kwargs)
        anserini_version = os.environ.get("ANSERINI_VERSION", "0.9.4")
        terrier_version = os.environ.get("TERRIER_VERSION", None)
        if terrier_version is not None:
            print("Testing with Terrier version " + terrier_version)
        if not pt.started():
            pt.init(version=terrier_version, logging="DEBUG", boot_packages=["io.anserini:anserini:%s:fatjar" % anserini_version])
        self.here = os.path.dirname(os.path.realpath(__file__))
        assert "version" in pt.init_args
        assert pt.init_args["version"] == terrier_version

    def test_anserini_vaswani(self):
        self.skip_pyserini()
        dest_index = os.path.join(self.here, "..", "fixtures", "anserini_index")
        ret = pt.anserini.AnseriniBatchRetrieve(dest_index)
        dataset = pt.get_dataset("vaswani")
        df = pt.Experiment([ret], dataset.get_topics(), dataset.get_qrels(), ["map"])
        self.assertEqual(0.2856489577946, df.iloc[0]["map"])