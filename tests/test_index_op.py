import pyterrier as pt
import tempfile
import unittest
from .base import TempDirTestCase
import os
import pandas as pd
import shutil
import tempfile


class TestIndexOp(TempDirTestCase):

    def test_index_corpus_iter(self):
        documents = [
            {'docno' : 'd1', 'text': 'stemming stopwords stopwords'},
        ]
        index = pt.IndexFactory.of( pt.IterDictIndexer(tempfile.mkdtemp(), stopwords=None, stemmer=None).index(documents) )
        self.assertEqual(1, len(index))
        self.assertEqual(2, index.getCollectionStatistics().getNumberOfUniqueTerms())
        self.assertEqual(3, index.getCollectionStatistics().getNumberOfTokens())

        # check that get_corpus_iter() contains the correct information 
        iter = index.get_corpus_iter()
        first_doc = next(iter)
        self.assertTrue(first_doc is not None)
        self.assertIn('docno', first_doc)
        self.assertIn('toks', first_doc)
        self.assertIn('stemming', first_doc['toks'])
        self.assertIn('stopwords', first_doc['toks'])
        self.assertEqual(1, first_doc['toks']['stemming'])
        self.assertEqual(2, first_doc['toks']['stopwords'])
        with(self.assertRaises(StopIteration)):
            next(iter)

        #  now check that a static pruning pipe can operate as expected. this example comes from terrier-index-api.rst
        index_pipe = (
            # update the toks column for each document, keeping only terms with frequency > 1
            pt.apply.toks(lambda row: { t : row['toks'][t] for t in row['toks'] if row['toks'][t] > 1 } ) 
            >> pt.IterDictIndexer(tempfile.mkdtemp(), pretokenised=True)
        )
        new_index_ref = index_pipe.index( index.get_corpus_iter())
        pruned_index = pt.IndexFactory.of(new_index_ref)
        self.assertEqual(1, len(pruned_index))
        self.assertEqual(1, pruned_index.getCollectionStatistics().getNumberOfUniqueTerms())
        self.assertEqual(2, pruned_index.getCollectionStatistics().getNumberOfTokens())

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