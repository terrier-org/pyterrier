Terrier How-To Guides
============================================================

This page provides a set of how-to guides for common tasks when using Terrier with PyTerrier.

-----------------------------------------------------------------------------

.. _terrier:how-to:index-standard:

How do I index a standard corpus?
------------------------------------------------------------

.. related:: pyterrier.terrier.TerrierIndex.index

.. code-block:: python
    :caption: Indexing a standard corpus with Terrier

    import pyterrier as pt
    dataset = pt.datasets.get_dataset("irds:msmarco-passage") # :footnote: Select your dataset here. If the corpus is not available in PyTerrier datasets, see :ref:`terrier:how-to:index-custom`
    my_index = pt.terrier.TerrierIndex('/path/to/index/location.terrier') # :footnote: Specify the location where you want to store the Terrier index. The location must not yet exist. We recommend using the ``.terrier`` extension, though this is not required.
    my_index.index(dataset.get_corpus_iter()) # :footnote: This performs indexing with default settings. If you need more control over the indexing settings, see :meth:`~pyterrier.terrier.TerrierIndex.indexer` and :class:`~pyterrier.terrier.IterDictIndexer` for advanced options.


-----------------------------------------------------------------------------

.. _terrier:how-to:index-custom:

How do I index a custom collection?
------------------------------------------------------------

.. related:: pyterrier.terrier.TerrierIndex.indexer

.. code-block:: python
    :caption: Indexing a custom collection with Terrier

    import pyterrier as pt
    my_collection = [ # :footnote: Each document should be a dictionary with ``docno`` (a unique identifier) and additional text fields. Your collection can be any iterable type (list, generator, etc.).
        {"docno": "doc1", "title": "This is the text of document one.", "body": "This is the body of document one."},
        {"docno": "doc2", "title": "This is the text of document two.", "body": "This is the body of document two."},
        {"docno": "doc3", "title": "This is the text of document three.", "body": "This is the body of document three."}
    ]
    my_index = pt.terrier.TerrierIndex('/path/to/index/location.terrier') # :footnote: Specify the location where you want to store the Terrier index. The location must not yet exist. We recommend using the ``.terrier`` extension, though this is not required.
    indexer = my_index.indexer(fields=["title", "body"]) # :footnote: ``fields=...`` lets you specify which fields to index. The ``"text"`` field is the default.
    indexer.index(my_collection)


-----------------------------------------------------------------------------

.. _terrier:how-to:langs:

How do I index and retrieve languages other than English?
-----------------------------------------------------------------------------

.. related:: pyterrier.terrier.TerrierTokeniser
.. related:: pyterrier.terrier.TerrierStopwords
.. related:: pyterrier.terrier.TerrierStemmer

Terrier provides built-in support for several other languages (see list in :class:`~pyterrier.terrier.TerrierStemmer`).
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
        tokeniser=pt.terrier.TerrierTokeniser.utf, # :footnote: Be sure to specify :attr:`pyterrier.terrier.TerrierTokeniser.utf` and :attr:`pyterrier.terrier.TerrierStopwords.none` for non-English text -- the default English settings do not work well for other languages.
        stopwords=pt.terrier.TerrierStopwords.none,
        stemmer=pt.terrier.TerrierStemmer.german, # :footnote: Specify the appropriate stemmer for your target language.
    )
    indexer.index(my_collection)

    # Retrieval
    retriever = my_index.bm25()
    retriever.search('Dokumente')

If your target language does not have built-in support, you can applie custom pre-processing
steps in the pipeline. Here is an example using `Spacy <https://spacy.io/>`__ for Czech:

.. code-block:: python
    :caption: Indexing Czech text with Terrier

    import spacy
    import pyterrier as pt

    nlp = spacy.blank("cs")
    def cs_preprocess(text): # :footnote: Here we define a function that performs the necessary pre-procesisng steps (in this case, Czech tokenization and stopword removal).
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
        stopwords=pt.terrier.TerrierStopwords.none, # :footnote: Since we are applying custom pre-processing, we disable stopword removal and stemming in Terrier by setting them to :attr:`pyterrier.terrier.TerrierStopwords.none` and :attr:`pyterrier.terrier.TerrierStemmer.none`.
        stemmer=pt.terrier.TerrierStemmer.none,
    )
    indexer_pipeline = pt.apply.text(lambda d: cs_preprocess(d['text'])) >> indexer [3]
    indexer_pipeline.index(my_collection)

    # Retrieval
    retriever = my_index.bm25()
    retriever_pipeline = pt.apply.query(lambda d: cs_preprocess(d['query'])) >> retriever # :footnote: Include the pre-processing steps as stages of the retrieval and indexing pipelines.
    retriever_pipeline.search('dokumentu')


-----------------------------------------------------------------------------

.. _terrier:how-to:loop-docs:

How can I loop over all documents in an index?
-----------------------------------------------------------------------------

.. related:: pyterrier.terrier.TerrierIndex.get_corpus_iter

:meth:`TerrierIndex.get_corpus_iter() <pyterrier.terrier.TerrierIndex.get_corpus_iter>` provides an iterator over all documents in a Terrier index.

.. code-block:: python
    :caption: Looping over all documents in a Terrier index

    import pyterrier as pt
    index = pt.terrier.TerrierIndex('/path/to/index/location.terrier')
    for doc in index.get_corpus_iter(): # :footnote: This creates an iterator over all documents in the specified Terrier index.
        print(doc)
        # do something with doc

-----------------------------------------------------------------------------

.. _terrier:how-to:access-lexicon:

How can I access the terms in an index?
-----------------------------------------------------------------------------

.. related:: pyterrier.terrier.TerrierIndex.lexicon

:meth:`TerrierIndex.lexicon() <pyterrier.terrier.TerrierIndex.lexicon>` provides access to the Lexicon of a Terrier index.

.. code-block:: python
    :caption: Accessing the Lexicon of a Terrier index

    import pyterrier as pt
    index = pt.terrier.TerrierIndex('/path/to/index/location.terrier')
    lexicon = index.lexicon()
    for term, entry in lexicon: # :footnote: You can iterate over all terms in the Lexicon
        print(term)
        print(entry.getDocumentFrequency()) # :footnote: Lexicon provides low-level API access through Java bindings. ``getDocumentFrequency()`` is defined in the Java `LexiconEntry <http://terrier.org/docs/current/javadoc/org/terrier/structures/LexiconEntry.html>`__ class.
    
    print("frequency for 'chemic':", lexicon["chemic"].getDocumentFrequency()) # :footnote: You can also access statistics for a specific term
