Terrier How-To Guides
============================================================

.. _terrier:index-standard:
How do I index a standard corpus?
------------------------------------------------------------

.. code-block:: python
    :caption: Indexing a standard corpus with Terrier

    import pyterrier as pt
    dataset = pt.datasets.get_dataset("irds:msmarco-passage") # (1)
    my_index = pt.terrier.TerrierIndex('/path/to/index/location.terrier') # [2]
    my_index.index(dataset.get_corpus_iter()) # [3]

.. .. code-annotations::
..     1. Select your dataset here. If the corpus is not available in PyTerrier datasets, see :ref:`terrier:index-custom`

``[1]`` Select your dataset here. If the corpus is not available in PyTerrier datasets, see :ref:`terrier:index-custom`

``[2]`` Specify the location where you want to store the Terrier index. The location must not yet exist. We recommend using the ``.terrier`` extension, though this is not required.

``[3]`` This performs indexing with default settings. If you need more control over the indexing settings, see :meth:`~pyterrier.terrier.TerrierIndex.indexer` and :class:`~pyterrier.terrier.TerrierIndexer` for advanced options.



.. _terrier:index-custom:
How do I index a custom collection?
------------------------------------------------------------

.. code-block:: python
    :caption: Indexing a custom collection with Terrier

    import pyterrier as pt
    my_collection = [ # [1]
        {"docno": "doc1", "title": "This is the text of document one.", "body": "This is the body of document one."},
        {"docno": "doc2", "title": "This is the text of document two.", "body": "This is the body of document two."},
        {"docno": "doc3", "title": "This is the text of document three.", "body": "This is the body of document three."}
    ]
    my_index = pt.terrier.TerrierIndex('/path/to/index/location.terrier') # [2]
    indexer = my_index.indexer(fields=["title", "body"]) # [3]
    indexer.index(my_collection)

.. [1] Each document should be a dictionary with ``docno`` (a unique identifier) and additional text fields. Your collection can be any iterable type (list, generator, etc.).
.. [2] Specify the location where you want to store the Terrier index. The location must not yet exist. We recommend using the ``.terrier`` extension, though this is not required.
.. [3] ``fields`` lets you specify which fields to index.


.. _terrier:langs:
How do I index and retrieve languages other than English?
-----------------------------------------------------------------------------

Terrier provides built-in support for several other languages (see list in :class:`~pt.terrier.TerrierStemmer`).
If your target language is supported, you just need to be sure to set appropriate tokenisation,
stemming, and stopword removal options during indexing. Here is an example for German:

.. code-block:: python
    :caption: Indexing German text with Terrier

    import pyterrier as pt
    my_collection = [
        {"docno": "doc1", "text": "Dies ist der Text von Dokument eins."},
        {"docno": "doc2", "text": "Dies ist der Text von Dokument zwei."},
        {"docno": "doc3", "text": "Dies ist der Text von Dokument drei."}
    ]
    my_index = pt.terrier.TerrierIndex('/pfad/zum/indexort.terrier')

    # Indexing
    indexer = my_index.indexer(
        tokeniser=pt.terrier.TerrierTokeniser.utf, # [1]
        stopwords=pt.terrier.TerrierStopwords.none, # [1]
        stemmer=pt.terrier.TerrierStemmer.german, # [2]
    )
    indexer.index(my_collection)

    # Retrieval
    retriever = my_index.bm25()
    retriever.search('Dokumente')

.. [1] Be sure to specify :attr:`pyterrier.terrier.TerrierTokeniser.utf` and :attr:`pyterrier.terrier.TerrierStopwords.none` for non-English text -- the default English settings do not work well for other languages.
.. [2] Specify the appropriate stemmer for your target language.


If your target language does not have built-in support, you can applie custom pre-processing
steps in the pipeline. Here is an example using `Spacy <https://spacy.io/>`__ for Czech:

.. code-block:: python
    :caption: Indexing Czech text with Terrier

    import spacy
    import pyterrier as pt

    nlp = spacy.blank("cs")
    def cs_preprocess(text): # [1]
        doc = nlp(text)
        toks = [str(token) for token in doc if not token.is_stop]
        return ' '.join(toks) # combine toks back into a string

    my_collection = [
        {"docno": "doc1", "text": "Toto je text prvního dokumentu."},
        {"docno": "doc2", "text": "Toto je text druhého dokumentu."},
        {"docno": "doc3", "text": "Toto je text třetího dokumentu."}
    ]
    my_index = pt.terrier.TerrierIndex('/cesta/k/indexu/umístění.terrier')

    # Indexing
    indexer = my_index.indexer(
        tokeniser=pt.terrier.TerrierTokeniser.utf,
        stopwords=pt.terrier.TerrierStopwords.none, # [2]
        stemmer=pt.terrier.TerrierStemmer.none, # [2]
    )
    indexer_pipeline = pt.apply.text(lambda d: cs_preprocess(d['text'])) >> indexer [3]
    indexer_pipeline.index(my_collection)

    # Retrieval
    retriever = my_index.bm25()
    retriever_pipeline = pt.apply.query(lambda d: cs_preprocess(d['query'])) >> retriever [3]
    retriever_pipeline.search('dokumentu')

.. [1] Here we define a function that performs the necessary pre-procesisng steps (in this case, Czech tokenization and stopword removal).
.. [2] Since we are applying custom pre-processing, we disable stopword removal and stemming in Terrier by setting them to :attr:`pyterrier.terrier.TerrierStopwords.none` and :attr:`pyterrier.terrier.TerrierStemmer.none`.
.. [3] Include the pre-processing steps as stages of the retrieval and indexing pipelines.
