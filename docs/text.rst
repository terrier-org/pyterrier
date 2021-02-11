Working with Document Texts
---------------------------

Many modern retrieval techniques are concerned with operating directly on the text of documents. PyTerrier supports these
forms of interactions.

Retrieving the Text of Documents
================================

When operating on the text of documents, you will need to have the text stored as an attribute in your dataframes.

This can be achieved in one of several ways:
 - requesting document metadata when using `BatchRetrieve`
 - adding document metadata later using `get_text()`

BatchRetrieve accepts a `metadata` keyword-argument which allows for additional metadata attributes to be retrieved.
Alternatively, the `pt.text.get_text()` transformer can be used, which can extract metadata from a Terrier index for
documents already retrieved.

Examples::

    # the following pipelines are equivalent
    pipe1 = pt.BatchRetrieve(index, metadata=["docno", "body"])

    pipe2 = pt.BatchRetrieve(index) >> pt.text.get_text(index, "body")


.. autofunction:: pyterrier.text.get_text()


Scoring query/text similarity
==============================

.. autofunction:: pyterrier.text.scorer()

Other text scorers are available in the form of neural re-rankers - separate to PyTerrier. See `Neural Rankers and Rerankers <neural.html>`_.


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
