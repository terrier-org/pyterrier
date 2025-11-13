Terrier Quick Start Tutorial
-------------------------------------------------

.. related:: pyterrier.terrier.TerrierIndex.bm25

Terrier is an open-source search engine that allows for efficient indexing and retrieval of documents.

In this tutorial, you will:

- Index a small collection of web text using Terrier
- Retrieve over that text using BM25
- Build a simple retrieval pipeline

To complete this tutorial, you first need to:

- Have :doc:`PyTerrier installed <../installation>`.

Tutorial
==================================================

In this tutorial, you will index and retrieve from ANTIQUE [#]_, which is a collection of
around 400,000 web documents from question-answering forums. ANTIQUE is built into PyTerrier's Dataset API,
so it is downloaded for you automatically when you first need it.

.. admonition:: Optional: Exploring the ANTIQUE Collection
    :class: tip, dropdown

    You can check out what the collection looks like by loading it into a DataFrame:

    .. code-block:: python

        import pandas as pd
        import pyterrier as pt
        dataset = pt.get_dataset('irds:antique')
        corpus = pd.DataFrame(dataset.get_corpus_iter()) # :footnote: This loads the entire ANTIQUE corpus into a Pandas DataFrame for exploration. Most collections will be too large to load into memory like this, but ANTIQUE is small enough to do so.

    The resulting ``corpus`` dataframe will look something like this:

    +----+-----------+----------------------------------------------------+
    |    |     docno | text                                               |
    +====+===========+====================================================+
    |  0 | 2020338_0 | A small group of politicians believed strongly ... |
    +----+-----------+----------------------------------------------------+
    |  1 | 2020338_1 | Because there is a lot of oil in Iraq.             |
    +----+-----------+----------------------------------------------------+
    |  2 | 2020338_2 | It is tempting to say that the US invaded Iraq ... |
    +----+-----------+----------------------------------------------------+
    |  3 | 2020338_3 | I think Yuval is pretty spot on. It's a proving... |
    +----+-----------+----------------------------------------------------+
    |  4 | 2874684_0 | Call an area apiarist.  They should be able to ... |
    +----+-----------+----------------------------------------------------+
    | ...| ...       | ...                                                |
    +----+-----------+----------------------------------------------------+

    Go ahead and play around with it to get a feel for the data! You can try answering the following questions:

    - How many documents are in the collection?
    - Are there any documents that are particularly long or short?
    - Can you find any interesting patterns or themes in the text?

We will start by building an index of the collection. This constructs data structures that allow for
efficient retrieval of documents based on their content.

To index a collection using Terrier, you first need to create a :class:`~pyterrier.terrier.TerrierIndex` object.
Since Terrier indexes are stored on disk, you need to provide a path where the index will be stored when constructing it.
To add documents to the index, you can call ``index()``, passing in the corpus.

.. code-block:: python
    :caption: Creating a Terrier index

    import pyterrier as pt
    my_index = pt.terrier.TerrierIndex('my_index.terrier') # :footnote: You can specify any path you like here. We typically use the ``.terrier`` extension to indicate that it is a Terrier index, but this isn't required.
    dataset = pt.get_dataset('irds:antique')
    my_index.index(dataset.get_corpus_iter())

This step may take a minute or two to download the dataset and index it, but once it is done,
you will have a Terrier index stored at the specified location.

Once indexing is complete, we can retrieve documents. Terrier has a variety of ways to
retrieve documents, but we will use the popular BM25 retrieval model. To retrieve
documents using BM25, we can use the ``bm25()`` method of the :class:`~pyterrier.terrier.TerrierIndex`
object. This method returns a retriever object that can be used to perform retrieval.

.. code-block:: python
    :caption: Retrieving documents using BM25

    bm25_retriever = my_index.bm25() # :footnote: This creates a BM25 transformer object that can be used to perform retrieval over ``my_index``.
    results = bm25_retriever.search("capital of Germany") # :footnote: This performs retrieval for the given query and returns the results as a DataFrame.

You should get results that look like this:

+-----+-----+--------+------------+------+-----------+--------------------+
|     | qid |  docid |      docno | rank |     score |              query |
+=====+=====+========+============+======+===========+====================+
| 0   |   1 | 218864 |   846016_7 |    0 | 22.357888 | capital of Germany |
+-----+-----+--------+------------+------+-----------+--------------------+
| 1   |   1 |  42629 |  4034012_0 |    1 | 21.672244 | capital of Germany |
+-----+-----+--------+------------+------+-----------+--------------------+
| 2   |   1 | 347695 |   58580_10 |    2 | 17.453893 | capital of Germany |
+-----+-----+--------+------------+------+-----------+--------------------+
| 3   |   1 |  92087 | 4255880_12 |    3 | 16.887855 | capital of Germany |
+-----+-----+--------+------------+------+-----------+--------------------+
| ... | ... |    ... |        ... |  ... |       ... |                ... |
+-----+-----+--------+------------+------+-----------+--------------------+

We can see that retrieval worked, returning documents for our query. However, we do not see the contents
of the documents, only their unique identifier (``docno``). We can build a simple pipeline to load the
document text so we can see what was retrieved.

.. code-block:: python
    :caption: Building a retrieval pipeline that loads document text

    retrieval_pipeline = my_index.bm25() >> dataset.text_loader() # :footnote: Here, we build a pipeline that first retrieves documents using BM25, then loads the document text using the dataset's text loader.
    results = retrieval_pipeline.search("capital of Germany")

Now, when we run the retrieval pipeline, we get results that include the document text:

+----+-----+--------+------------+------+-----------+--------------------+---------------------------------------------------+
|    | qid |  docid |      docno | rank |     score |              query |                                              text |
+====+=====+========+============+======+===========+====================+===================================================+
| 0  |   1 | 218864 |   846016_7 |    0 | 22.357888 | capital of Germany | Why can't you just be glad that Hamburg isn't ... |
+----+-----+--------+------------+------+-----------+--------------------+---------------------------------------------------+
| 1  |   1 |  42629 |  4034012_0 |    1 | 21.672244 | capital of Germany | Berlin is the Capital of Germany.. . It as als... |
+----+-----+--------+------------+------+-----------+--------------------+---------------------------------------------------+
| 2  |   1 | 347695 |   58580_10 |    2 | 17.453893 | capital of Germany | I go to school in the U.S. and they don't real... |
+----+-----+--------+------------+------+-----------+--------------------+---------------------------------------------------+
| 3  |   1 |  92087 | 4255880_12 |    3 | 16.887855 | capital of Germany | American - Capitol Amber (Madison, Wisconsin).... |
+----+-----+--------+------------+------+-----------+--------------------+---------------------------------------------------+
| ...| ... |    ... |        ... |  ... |       ... |                ... |                                               ... |
+----+-----+--------+------------+------+-----------+--------------------+---------------------------------------------------+

Although not all the results are relevant, we can see that we have the answer to our question (Berlin is the Capital of Germany)
in row 1.

References
==================================================

.. [#]
    .. cite.dblp:: conf/ecir/HashemiAZC20

