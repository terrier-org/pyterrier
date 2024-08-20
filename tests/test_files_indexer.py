
from .base import TempDirTestCase
import warnings
import tempfile
import shutil
import os
import pyterrier as pt

class TestFilesIndexer(TempDirTestCase):

    def test_2_docs(self):
        files = pt.io.find_files(os.path.join(self.here, "fixtures", "sample_docs"))
        indexer = pt.FilesIndexer(self.test_dir)
        indexref = indexer.index(files)
        index = pt.IndexFactory.of(indexref)
        self.assertEqual(2, index.getCollectionStatistics().getNumberOfDocuments())

    def test_2_docs_title_body_meta(self):
        sample_dir = os.path.join(self.here, "fixtures", "sample_docs")
        files = pt.io.find_files(sample_dir)
        indexer = pt.FilesIndexer(
            self.test_dir, 
            meta={"docno" : 20, "filename" : 512, "title" : 20, "body" : 40}, 
            meta_tags={"title":"title", "body" : "ELSE"})
        indexref = indexer.index(files)
        index = pt.IndexFactory.of(indexref)
        
        # check index size
        self.assertEqual(2, index.getCollectionStatistics().getNumberOfDocuments())
        print(index.getMetaIndex().getAllItems(0))
        print(index.getMetaIndex().getAllItems(1))

        # determine file locations as docids
        html_file = os.path.join(sample_dir, "a.html")
        html_pos = files.index(html_file)
        txt_file = os.path.join(sample_dir, "b.txt")
        txt_pos = files.index(txt_file)        

        self.assertTrue(html_pos < len(files))
        # test filename -> docid lookup
        self.assertEqual(html_pos, index.getMetaIndex().getDocument("filename", html_file))
        # test docid -> filename lookup
        self.assertEqual(html_file, index.getMetaIndex().getItem("filename", html_pos))
        # test title has been recorded in metaindex
        self.assertEqual("test title", index.getMetaIndex().getItem("title", html_pos))

        if not pt.terrier.check_version("5.5"):
            return

        # test bodies have been recorded in metaindex
        self.assertEqual("test body", index.getMetaIndex().getItem("body", html_pos))
        self.assertEqual("empty text document",index.getMetaIndex().getItem("body", txt_pos))