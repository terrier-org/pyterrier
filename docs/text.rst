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
 - requesting document metadata when using `BatchRetrieve`
 - adding document metadata later using `get_text()`

BatchRetrieve accepts a `metadata` keyword-argument which allows for additional metadata attributes to be retrieved.

Alternatively, the `pt.text.get_text()` transformer can be used, which can extract metadata from a Terrier index
or IRDSDataset for documents already retrieved. The main advantage of using IRDSDataset is that it supports
all document fields, not just those that were included as meta fields when indexing.

Examples::

    # the following pipelines are equivalent
    pipe1 = pt.BatchRetrieve(index, metadata=["docno", "body"])

    pipe2 = pt.BatchRetrieve(index) >> pt.text.get_text(index, "body")

    dataset = pt.get_dataset('irds:vaswani')
    pipe3 = pt.BatchRetrieve(index) >> pt.text.get_text(dataset, "text")


.. autofunction:: pyterrier.text.get_text()


Scoring query/text similarity
==============================

.. autofunction:: pyterrier.text.scorer()

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


References
==========

 - [Chen2020] ICIP at TREC-2020 Deep Learning Track, X. Chen et al. Procedings of TREC 2020.
 - [Dai2019] Deeper Text Understanding for IR with Contextual Neural Language Modeling. Z. Dai & J. Callan. Proceedings of SIGIR 2019.
