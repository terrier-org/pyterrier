.. _pt.text:

Working with Document Texts
---------------------------

Many modern retrieval techniques are concerned with operating directly on the text of documents. PyTerrier supports these
forms of interactions.

Indexing and Retrieval of Text in Terrier indices
=================================================

If you are using a Terrier index for your first-stage ranking, you will want to record the text
of the documents in the MetaIndex. The following configuration demonstrates saving the title
and remainder of the documents separately in the Terrier index MetaIndex when indexing a 
TREC-formatted corpus::

    files = []  # list of filenames to be indexed
    indexer = pt.TRECCollectionIndexer(INDEX_DIR, 
        # record that we save additional document metadata called 'text'
        meta= {'docno' : 26, 'text' : 2048},
        # The tags from which to save the text. ELSE is special tag name, which means anything not consumed by other tags.
        meta_tags = {'text' : 'ELSE'}
        verbose=True)
    indexref = indexer.index(files)
    index = pt.IndexFactory.of(indexref)

On the other-hand, for a TSV-formatted corpus such as MSMARCO passages, indexing is easier
using IterDictIndexer::

    def msmarco_generate():
        dataset = pt.get_dataset("trec-deep-learning-passages")
        with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
            for l in corpusfile:
                docno, passage = l.split("\t")
                yield {'docno' : docno, 'text' : passage}

    iter_indexer = pt.IterDictIndexer("./passage_index")
    indexref = iter_indexer.index(msmarco_generate(), meta={'docno' : 20, 'text': 4096})


During retrieval you will need to have the text stored as an attribute in your dataframes.

This can be achieved in one of several ways:
 - requesting document metadata when using `Retriever`
 - adding document metadata later using `get_text()`

Retriever accepts a `metadata` keyword-argument which allows for additional metadata attributes to be retrieved.

Alternatively, the `pt.text.get_text()` transformer can be used, which can extract metadata from a Terrier index
or IRDSDataset for documents already retrieved. The main advantage of using IRDSDataset is that it supports
all document fields, not just those that were included as meta fields when indexing.

Examples::

    # the following pipelines are equivalent
    pipe1 = pt.terrier.Retriever(index, metadata=["docno", "body"])

    pipe2 = pt.terrier.Retriever(index) >> pt.text.get_text(index, "body")

    dataset = pt.get_dataset('irds:vaswani')
    pipe3 = pt.terrier.Retriever(index) >> pt.text.get_text(dataset, "text")

.. autofunction:: pyterrier.text.get_text()

.. schematic::
    :input_columns: qid,query,docno,text

    pt.text.get_text(pt.get_dataset('irds:vaswani'))

Scoring query/text similarity
==============================

.. autofunction:: pyterrier.text.scorer()

One pipeline could be retrieve documents, get their text, and then re-score them using a text-based scorer such as
`BM25 <https://en.wikipedia.org/wiki/Okapi_BM25>`_ or even MonoT5 from `pyterrier_t5`.

.. schematic::
    :input_columns: qid,query

    pt.terrier.TerrierIndex.example().bm25() >> pt.text.get_text(pt.get_dataset('irds:vaswani')) >> pt.text.scorer(body_attr="text")

.. schematic::
    :input_columns: qid,query

    from pyterrier_t5 import MonoT5ReRanker
    pt.terrier.TerrierIndex.example().bm25() % 100 >> pt.text.get_text(pt.get_dataset('irds:vaswani')) >> MonoT5ReRanker()

Other text scorers are available in the form of neural re-rankers - separate to PyTerrier, see :ref:`neural`.


Working with Passages rather than Documents
===========================================

As documents are long, relevant content may only be found in a small portion of the document. Moreover, some models are more suited
to operating on small parts of the document. For this reason, passage-based retrieval techniques have been conceived. PyTerrier supports
the creation of passages from longer documents, and for the aggregation of scores from these passages.

.. autofunction:: pyterrier.text.sliding()


Example Inputs and Outputs:

Consider the following dataframe with one or more documents:

+-------+---------+-----------------+
+  qid  +  docno  +  text           +
+=======+=========+=================+
|  q1   | d1      +  a b c d        +
+-------+---------+-----------------+

The result of applying `pyterrier.text.sliding(length=2, stride=1, prepend_title=False)` would be:

+-------+---------+-----------------+
+  qid  +  docno  +  text           +
+=======+=========+=================+
|  q1   | d1%p1   +  a b            +
+-------+---------+-----------------+
|  q1   | d1%p2   +  b c            +
+-------+---------+-----------------+
|  q1   | d1%p3   +  c d            +
+-------+---------+-----------------+


.. autofunction:: pyterrier.text.max_passage()

.. autofunction:: pyterrier.text.first_passage()

.. autofunction:: pyterrier.text.mean_passage()

.. autofunction:: pyterrier.text.kmaxavg_passage()


Examples
~~~~~~~~

Assuming that a retrieval pipeline such as `sliding()` followed by `scorer()` could return a dataframe that looks like this:

+-------+---------+--------+--------+
+  qid  +  docno  +  rank  + score  +
+=======+=========+========+========+
|  q1   | d1%p5   +  0     + 5.0    +
+-------+---------+--------+--------+
|  q1   | d2%p4   +  1     + 4.0    +
+-------+---------+--------+--------+
|  q1   | d1%p3   +  2     + 3.0    +
+-------+---------+--------+--------+
|  q1   | d1%p1   +  3     + 1.0    +
+-------+---------+--------+--------+

The output of the `max_passage()` transformer would be:

+-------+---------+--------+--------+
+  qid  +  docno  +  rank  + score  +
+=======+=========+========+========+
|  q1   | d1      +  0     + 5.0    +
+-------+---------+--------+--------+
|  q1   | d2      +  1     + 4.0    +
+-------+---------+--------+--------+


The output of the `mean_passage()` transformer would be:

+-------+---------+--------+--------+
+  qid  +  docno  +  rank  + score  +
+=======+=========+========+========+
|  q1   | d1      +  0     + 4.5    +
+-------+---------+--------+--------+
|  q1   | d2      +  1     + 4.0    +
+-------+---------+--------+--------+

The output of the `first_passage()` transformer would be:

+-------+---------+--------+--------+
+  qid  +  docno  +  rank  + score  +
+=======+=========+========+========+
|  q1   | d2      +  0     + 4.0    +
+-------+---------+--------+--------+
|  q1   | d1      +  1     + 1.0    +
+-------+---------+--------+--------+


Finally, the output of the `kmaxavg_passage(2)` transformer would be:

+-------+---------+--------+--------+
+  qid  +  docno  +  rank  + score  +
+=======+=========+========+========+
|  q1   | d2      +  1     + 4.0    +
+-------+---------+--------+--------+
|  q1   | d1      +  0     + 1.0    +
+-------+---------+--------+--------+

Example Pipelines
~~~~~~~~~~~~~~~~~

A typical passage-based retrieval pipeline might look like this::

    from pyterrier_t5 import MonoT5ReRanker
    index = pt.terrier.TerrierIndex.from_hf('pyterrier/vaswani.terrier')
    bm25 = index.bm25()
    passage_pipeline = (
        bm25 % 100 >> 
        pt.text.get_text(pt.get_dataset('irds:vaswani'), "text") >> 
        pt.text.sliding(length=100, stride=50, text_attr='text', prepend_attr=None) >> 
        MonoT5ReRanker() >> 
        pt.text.max_passage()
    )

.. schematic::
    :input_columns: qid,query

    from pyterrier_t5 import MonoT5ReRanker
    index = pt.terrier.TerrierIndex.from_hf('pyterrier/vaswani.terrier')
    bm25 = index.bm25()
    passage_pipeline = (
        bm25 % 100 >> 
        pt.text.get_text(pt.get_dataset('irds:vaswani'), "text") >> 
        pt.text.sliding(length=100, stride=50, text_attr='text', prepend_attr=None) >> 
        MonoT5ReRanker() >> 
        pt.text.max_passage()
    )
    passage_pipeline

So while the index retrievers documents, MonoT5 is applied to passages, and then the passage scores are aggregated back to document scores using ``pt.text.max_passage()``.

Alternatively you can apply passing at indexing time, and then use passage-level retrieval followed by aggregation::

    from pyterrier_t5 import MonoT5ReRanker
    indexer = pt.text.sliding() >> pt.IterDictIndexer("./index")
    indexer.index(document_corpus)
    passage_index = pt.terrier.TerrierIndex("./index")
    passage_pipeline = (
        index.bm25() % 100 >> 
        MonoT5ReRanker() >> 
        pt.text.max_passage()
    )

where ``passage_pipeline`` returns documents rather than passages. Experiments on TREC Robust 2004 have shown that passage indexing and retrieval does not benefit 
effectiveness compared to document-level indexing and retrieval, when using a strong re-ranker such as MonoT5 .. cite.dblp:`journals/tweb/WangMTO23`.

Query-biased Summarisation (Snippets)
=====================================

.. autofunction:: pyterrier.text.snippets()

Examples of Sentence-Transformers
=================================

Here we demonstrate the use of `pt.apply.doc_score( , batch_size=128)` to allow an easy application of `Sentence Transformers <https://www.sbert.net/>`_ for reranking BM25 results:: 

    import pandas as pd
    from sentence_transformers import CrossEncoder, SentenceTransformer
    crossmodel = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
    bimodel = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def _crossencoder_apply(df : pd.DataFrame):
        return crossmodel.predict(list(zip(df['query'].values, df['text'].values)))

    cross_encT = pt.apply.doc_score(_crossencoder_apply, batch_size=128)

    def _biencoder_apply(df : pd.DataFrame):
        from sentence_transformers.util import cos_sim
        query_embs = bimodel.encode(df['query'].values)
        doc_embs = bimodel.encode(df['text'].values)
        scores =  cos_sim(query_embs, doc_embs)
        return scores[0]

    bi_encT = pt.apply.doc_score(_biencoder_apply, batch_size=128)

    pt.Experiment(
        [ bm25, bm25 >> bi_encT, bm25 >> cross_encT ],
        dataset.get_topics(),
        dataset.get_qrels(),
        ["map"],
        names=["BM25", "BM25 >> BiEncoder", "BM25 >> CrossEncoder"]
    )

You can `browse the whole notebook <https://github.com/terrier-org/pyterrier/blob/master/examples/notebooks/sentence_transformers.ipynb>`_ or `try it yourself it on Colab <https://colab.research.google.com/terrier-org/pyterrier/blob/master/examples/notebooks/sentence_transformers.ipynb>`_

References
==========

.. cite.dblp:: conf/trec/ChenHSCH020
.. cite.dblp:: conf/sigir/DaiC19
