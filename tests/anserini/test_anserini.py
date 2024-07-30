import unittest
import os
import pyterrier as pt
ANSERINI_VERSION="0.22.0"

class AnseriniTestCase(unittest.TestCase):

    def skip_pyserini(self):
        if not pt.anserini.is_installed():
            if os.environ.get("ANSERINI_TESTING", None) is not None:
                raise RuntimeError('pyserini not installed')
            else:
                self.skipTest("Test disabled due to lack of Pyserini")

    def __init__(self, *args, **kwargs):
        super(AnseriniTestCase, self).__init__(*args, **kwargs)
        if pt.anserini.is_installed():
            anserini_version = os.environ.get("ANSERINI_VERSION", ANSERINI_VERSION)
            terrier_version = os.environ.get("TERRIER_VERSION", None)
            if terrier_version is not None:
                print("Testing with Terrier version " + terrier_version)
            if not pt.started():
                pt.init(version=terrier_version, logging="DEBUG", boot_packages=["io.anserini:anserini:%s:fatjar" % anserini_version])
            self.here = os.path.dirname(os.path.realpath(__file__))

    def test_anserini_vaswani(self):
        self.skip_pyserini()
        dest_index = os.path.join(self.here, "..", "fixtures", "anserini_index")
        bm25 = pt.anserini.AnseriniBatchRetrieve(dest_index)
        qld = pt.anserini.AnseriniBatchRetrieve(dest_index, wmodel='QLD')
        tf_idf = pt.anserini.AnseriniBatchRetrieve(dest_index, wmodel='TFIDF')
        dataset = pt.get_dataset("vaswani")
        df = pt.Experiment([
                bm25,
                qld,
                tf_idf                
            ], 
            dataset.get_topics(), 
            dataset.get_qrels(), 
            ["map"])
        self.assertEqual(0.2856564466226712, df.iloc[0]["map"])
        for i in df['map']:
            self.assertGreater(i, 0)
        
        # check re-ranking works too
        resIn = tf_idf.search("chemical reactions") 
        resOut = (tf_idf >> bm25).search("chemical reactions")
        self.assertEqual(len(resIn), len(resOut))