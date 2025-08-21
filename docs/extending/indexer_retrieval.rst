Building in PyTerrier Support for Indexing and Retrieval Backends
=================================================================

Aim: To provide guidance for how to make a indexing and retrieval
backends availble through PyTerrier.

Motivations
-----------

The PyTerrier ecosystem allow you to use your indexer and retriever with
state-of-the-art plugins.

For instance, to index using doc2query (with
`pyterrier_doc2query <https://github.com/terrierteam/pyterrier_doc2query>`__)
on a collection of your choice:

.. code:: python

   import pyterrier_doc2query
   doc2query = pyterrier_doc2query.Doc2Query(append=True)
   pipeline = doc2query >> MyIndexer()
   pipeline.index(corpus)

Or re-rank the results of your system using monoT5:

.. code:: python

   from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
   monoT5 = MonoT5ReRanker() # loads castorini/monot5-base-msmarco by default
   duoT5 = DuoT5ReRanker() # loads castorini/duot5-base-msmarco by default

   mono_pipeline = MyIndex.bm25() >> pt.text.get_text(dataset, "text") >> monoT5
   duo_pipeline = mono_pipeline % 50 >> duoT5 # apply a rank cutoff of 50 from monoT5 since duoT5

And evaluate these pipelines using ``pt.Experiment``:

.. code:: python


   pt.Expertiment(
    [mono_pipeline %50, duo_pipeline],
    dataset.get_topics(),
    dataset.get_qrels(),
    [ MRR@10, "mrt" ]
   )

Quickstart
----------

PyTerrier has two formats for expressing data inputs and outputs of Transformers.
The first is Pandas dataframes, while the second is a iteratable of dicts, In both
cases, the expected columns are as defined in the :doc:`PyTerrier Data Model <../datamodel>`.
In the following, we mainly focus upon the DataFrame format, but the iter-dict format is
identically supported.

For retrieval, at the very least you need to define a function that
takes a dataframe with two columns ``['qid', 'query']``, and returns a
dataframe with the following columns: ``['qid', 'docno', 'score']``.
This can be made into a pt.Transformer instance using a `pt.apply
wrapper <https://pyterrier.readthedocs.io/en/latest/apply.html>`__, or
by directly extending pt.Transformer and naming your function as
``transform()``. The types of the qid and docno columns are strings.

For indexing, you need to make a class that inherits from pt.Indexer
with an ``index()`` function that consumes an iterator of dicts, where each dict
contains information for one document, for instance:

.. code:: python

   [
     { 'docno' : 'd1', 'text' : 'Hello there'},
     { 'docno' : 'd2', 'text' : 'Nice to meet you'}
   ]

If your indexer is for learned sparse retrieval, the tokens are
typically at pre-tokenised, as follows:

.. code:: python

   [
     { 'docno' : 'd1', 'toks' : {'hello' : 1, 'there' : 1 }],
     { 'docno' : 'd2', 'toks' : {'nice' : 1, 'to; : 1, 'meet' : 1, 'you' : 1 }
   ]

For such “pre-tokenised” settings at retrieval time, the query
dataframe should be expected to have qid and query_toks columns, where
query_toks has the same format as the toks column for indexing (but
float query weights are typically supported here):

.. code:: python

   [
     { 'qid' : 'q1', 'query' : 'hello hello there'  'query_toks' : {'hello' : 2.0, 'there' : 1.0 }],
   ]

Examples of learned sparse integrations are available at
`pyt_splade <https://github.com/cmacdonald/pyt_splade>`__.

Note that PyTerrier assumes docnos are strings - if you internally use
an integer-based scheme, your indexer and retrieval classes should
record a id->docno mapping file. Many of our PyTerrier plugins use
the `npids <https://github.com/seanmacavaney/npids>`__ package for this.

Example implementation
----------------------

.. code:: python

   import pyterrier as pt
   from collections.abc import Iterable
   import pandas as pd

   class MyIndexer(pt.Indexer):

     def __init__(self, indexpath : str):
       pass
     
     def index(self, iterdict : Iterable[dict]):
       """
         Consume the documents in the iterator, assuming that it has keys
         docno (string) and text (string)
       """
       return index # return your Index class here.
       

   class MyIndex:
     """
     The index class is used as a factory to allow easy access to different retriever implementations
     """

     def __init__(self, indexpath : str):
       # open your index, initialise etc
       pass
       
     def bm25(self) -> pt.Transformer:
       def _retr_fn(single_query_df : pd.DataFrame) -> pd.DataFrame
       
         qid = single_query_df.iloc[0]["qid"]
         query = single_query_df.iloc[0]["query"]
         # populate a results dataframe with columns ['qid', 'docno', 'score']
         return pt.model.add_ranks(results) # adds rank column
         
       return pt.apply.by_query(_retr_fn)
       
       
   # NB: You can merge these two classes into a single one. PyTerrier DR and PyTerrier PISA both use this scheme.

Optionally, your MyIndex class can extend :doc:`pt.Artifact <../ext/pyterrier-alpha/artifact>`-
this allows your index to be easily shared as an Artifact on Huggingface/Zenodo etc. 

Other Examples:
---------------

-  PyTerrier ColBERT:
   `ColBERTFactory <https://github.com/terrierteam/pyterrier_colbert/blob/5772d080bba50955f589ac3f87e9933f2122a126/pyterrier_colbert/ranking.py#L492>`__
   is the "MyIndex" class;
   `CoLBERTIndexer <https://github.com/terrierteam/pyterrier_colbert/blob/main/pyterrier_colbert/indexing.py#L261>`__
   is the indexer.
-  PyTerrier DR: Index classes are both Indexers and retrievers (and also pt.Artefact),
   e.g. `NumpyIndex <https://github.com/terrierteam/pyterrier_dr/blob/master/pyterrier_dr/indexes.py#L123>`__.
-  PyTerrier Pisa:
   `PisaIndex <https://github.com/terrierteam/pyterrier_pisa/blob/main/src/pyterrier_pisa/__init__.py#L98>`__
   is the pt.Indexer and the index factory (and also pt.Artefact).
