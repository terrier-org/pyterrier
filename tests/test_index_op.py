import pyterrier as pt
import tempfile
import unittest
from .base import TempDirTestCase
import os
import pandas as pd
import shutil
import tempfile


class TestIndexOp(TempDirTestCase):

    def test_index_add_write(self):
        # inspired by https://github.com/terrier-org/pyterrier/issues/390
        documents = [
            {'text': 'Creates a Function that returns a Function or returns the value of the given property .'},
            {'text': 'Returns the URL of the occupants .'},
            {'text': 'Exit the timer with the default values .'},
        ]

        # Create new index from pandas dataframe
        pd_indexer = pt.DFIndexer(tempfile.mkdtemp(), blocks=True)
        df = pd.DataFrame(documents)
        df['docno'] = df.index.astype(str)
        indexref = pd_indexer.index(df['text'], df['docno'])
        index1 = pt.IndexFactory.of(indexref)

        # Create new index from pandas dataframe
        pd_indexer = pt.DFIndexer(tempfile.mkdtemp(), blocks=True)
        df = pd.DataFrame(documents)
        df['docno'] = df.index.astype(str)
        indexref = pd_indexer.index(df['text'], df['docno'])
        index2 = pt.IndexFactory.of(indexref)

        # Merge indexes
        comb_index = index1 + index2

        new_disk_index_loc = tempfile.mkdtemp()
        self.assertEqual(len(index1) + len(index2), len(comb_index))
        
        # Instantiate writer object and write merged index to disk
        writer = pt.autoclass("org.terrier.structures.indexing.DiskIndexWriter")(new_disk_index_loc, "data")
        writer.write(comb_index)

        new_disk_index = pt.IndexFactory.of(new_disk_index_loc)

        self.assertEqual(len(index1) + len(index2), len(new_disk_index))