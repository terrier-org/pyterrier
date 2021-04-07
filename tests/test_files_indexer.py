
from .base import BaseTestCase
import warnings
import tempfile
import shutil
import os
import pyterrier as pt

class TestFilesIndexer(BaseTestCase):

    def test_2_docs(self):
        files = pt.io.find_files(os.path.join(self.here, "fixtures", "sample_docs"))
        indexer = pt.FilesIndexer(self.test_dir)
        indexref = indexer.index(files)
        index = pt.IndexFactory.of(indexref)
        self.assertEqual(2, index.getCollectionStatistics().getNumberOfDocuments())

    def test_2_docs_title_meta(self):
        sample_dir = os.path.join(self.here, "fixtures", "sample_docs")
        files = pt.io.find_files(sample_dir)
        indexer = pt.FilesIndexer(self.test_dir, meta={"docno" : 20, "filename" : 512, "title" : 20}, meta_tags={"title":"title"})
        indexref = indexer.index(files)
        index = pt.IndexFactory.of(indexref)
        
        # check index size
        self.assertEqual(2, index.getCollectionStatistics().getNumberOfDocuments())
        print(index.getMetaIndex().getAllItems(0))

        html_file = os.path.join(sample_dir, "a.html")
        html_pos = files.index(html_file)
        self.assertTrue(html_pos < len(files))
        # test filename -> docid lookup
        self.assertEqual(html_pos, index.getMetaIndex().getDocument("filename", html_file))
        # test docid -> filename lookup
        self.assertEqual(html_file, index.getMetaIndex().getItem("filename", html_pos))
        # test title has been recorded in metaindex
        self.assertEqual("test title", index.getMetaIndex().getItem("title", html_pos))


    def setUp(self):
            # Create a temporary directory
            self.test_dir = tempfile.mkdtemp()
            print("Created " + self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        print("Deleting " + self.test_dir)