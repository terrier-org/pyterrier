Accessing Terrier's Index API
-----------------------------


Once a Terrier index has been built, PyTerrier provides a number of ways to access it. 
In doing so, we access the standard Terrier index API, however, some types are patched by PyTerrier
to make them easier to use.

NB: Examples in this document are also available as a Jupyter notebook:
 - GitHub: https://github.com/terrier-org/pyterrier/blob/master/examples/notebooks/index_api.ipynb
 - Google Colab: https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/index_api.ipynb

Loading an Index
================

Terrier has `IndexRef <http://terrier.org/docs/current/javadoc/org/terrier/querying/IndexRef.html>`_ and 
`Index <http://terrier.org/docs/current/javadoc/org/terrier/structures/Index.html>`_ objects, along 
with an `IndexFactory <http://terrier.org/docs/current/javadoc/org/terrier/structures/IndexFactory.html>`_ 
class that allows an Index to be obtained from the IndexRef.

IndexRef is essentially a String that tells Terrier where the index is located. Typically it is a file location, pointing to a data.properties file::

    indexref = pt.IndexRef.of("/path/to/data.properties")

IndexRefs can also be obtained from a PyTerrier dataset::

    indexref = dataset.get_index()

IndexRef objects can be directly passed to Retriever::

    pt.terrier.Retriever(indexref).search("chemical reactions")

If you want to access the underlying data structures, you need to use IndexFactory, using the indexref, or the string location:: 
    
    index = pt.IndexFactory.of(indexref)
    #or
    index = pt.IndexFactory.of("/path/to/data.properties")

NB: Retriever will accept anything "index-like", i.e. a string location of an index, an IndexRef or an Index.

We can also ask for the index to be loaded into memory::

    index = pt.IndexFactory.of("/path/to/data.properties", mem=True)

.. autoclass:: pyterrier.IndexFactory
    :members: of

Whats in an Index
=================

An index has several data structures:

 - the `CollectionStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/CollectionStatistics.html>`_ - the salient global statistics of the index (number of documents, etc).
 - the `Lexicon <http://terrier.org/docs/current/javadoc/org/terrier/structures/Lexicon.html>`_ -  consists of an entry for each unique term in the index, which contains the corresponding statistics of each term (frequency etc), and a pointer to the inverted index posting list for that term.
 - the inverted index (a `PostingIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/PostingIndex.html>`_) - contains the posting list for each term, which records the documents that a given term appears in, and with what frequency for each document.
 - the `DocumentIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/DocumentIndex.html>`_ - contains the length of the document (and other field lengths).
 - the `MetaIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/MetaIndex.html>`_ - contains document metadata, such as the docno, and optionally the raw text and the URL of each document.
 - the direct index (also a PostingIndex) - contains a posting list for each document, detailing which terms occur in that document and with which frequency. The presence of the direct index depends on the IndexingType that has been applied - single-pass and some memory indices do not provide a direct index.

Each of these objects is available from the Index using a get method, e.g. `index.getCollectionStatistics()`. For instance, we can easily view the CollectionStatistics::

    print(index.getCollectionStatistics())
    Number of documents: 11429
    Number of terms: 7756
    Number of postings: 224573
    Number of fields: 0
    Number of tokens: 271581
    Field names: []
    Positions:   false

In this example, the indexed collection had 11429 documents, which contained 271581 word occurrences. 7756 unique words were identified. The total number of postings in the inverted index is 224573.
This index did not record fields during indexing (which can be useful for models such as BM25F). Similarly, positions, which are used for phrasal queries or proximity models were not recorded.

We can check what metadata is recorded::

    print(index.getMetaIndex().getKeys())

Usually, this will respond with `['docno']` - indeed docno is by convention the unique identifier for each document.

NB: Terrier's Index API is just that, an API of interfaces and abstract classes - depending on the indexing configuration, the exact implementation you will receive will differ.

Using a Terrier index in your own code
======================================

How many documents does term X occur in?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the Lexicon object, particularly the getLexiconEntry(String) method. However, PyTerrier aliases this, so
lookup can be done like accessing a dictionary::

    index.getLexicon()["chemic"].getDocumentFrequency()

As our index is stemmed, we used the stemmed form of the word 'chemical' which is 'chemic'.

How can I see all terms in an index?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can iterate over a Lexicon. Like calling the ``iterator()`` method of 
`Lexicon <http://terrier.org/docs/current/javadoc/org/terrier/structures/Lexicon.html>`_, 
in Java, each iteration obtains a ``Map.Entry<String,LexiconEntry>``. This can be decoded, 
so we can iterate over each term and LexiconEntry (which provides access to the statistics 
of each term) contained within the Lexicon::

    for term, le in index.getLexicon():
        print(term)
        print(le.getFrequency())

What is the un-smoothed probability of term Y occurring in the collection?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we again use the Lexicon of the underlying Terrier index. We check that the term occurs in the lexicon (to prevent a KeyError). 
The Lexicon returns a `LexiconEntry <http://terrier.org/docs/current/javadoc/org/terrier/structures/LexiconEntry.html>`_, which allows us access to the number of occurrences of the term in the index.

Finally, we use the CollectionStatistics object to determine the total number of occurrences of all terms in the index::

    index.getLexicon()["chemic"].getFrequency() / index.getCollectionStatistics().getNumberOfTokens() if "chemic" in index.getLexicon() else 0

What terms occur in the 11th document?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use the direct index. We need a `Pointer <http://terrier.org/docs/current/javadoc/org/terrier/structures/Pointer.html>`_ into 
the direct index, which we obtain from the DocumentIndex.
`PostingIndex.getPostings() <http://terrier.org/docs/current/javadoc/org/terrier/structures/PostingIndex.html#getPostings(org.terrier.structures.Pointer)>`_
is our method to get a posting list. Indeed, it returns an `IterablePosting <http://terrier.org/docs/current/javadoc/org/terrier/structures/postings/IterablePosting.html>`_.
Note that IterablePosting can be used in Python for loops::

    di = index.getDirectIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    docid = 10 #docids are 0-based
    #NB: postings will be null if the document is empty
    for posting in di.getPostings(doi.getDocumentEntry(docid)):
        termid = posting.getId()
        lee = lex.getLexiconEntry(termid)
        print("%s with frequency %d" % (lee.getKey(),posting.getFrequency()))

What documents does term "Z" occur in?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use the inverted index (also a PostingIndex). The Pointer this time comes from the Lexicion,
in that the LexiconEntry implements Pointer. Finally, we use the `MetaIndex <http://terrier.org/docs/current/javadoc/org/terrier/structures/MetaIndex.html>`_ 
to lookup the docno corresponding to the docid::

    meta = index.getMetaIndex()
    inv = index.getInvertedIndex()

    le = lex.getLexiconEntry( "chemic" )
    # the lexicon entry is also our pointer to access the inverted index posting list
    for posting in inv.getPostings( le ): 
        docno = meta.getItem("docno", posting.getId())
        print("%s with frequency %d " % (docno, posting.getFrequency()))

What are the PL2 weighting model scores of documents that "Y" occurs in?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use of a WeightingModel class needs some setup, namely the `EntryStatistics <http://terrier.org/docs/current/javadoc/org/terrier/structures/EntryStatistics.html>`_ 
of the term (obtained from the Lexicon, in the form of the LexiconEntry), as well as the CollectionStatistics (obtained from the index)::

    inv = index.getInvertedIndex()
    meta = index.getMetaIndex()
    lex = index.getLexicon()
    le = lex.getLexiconEntry( "chemic" )
    wmodel = pt.autoclass("org.terrier.matching.models.PL2")()
    wmodel.setCollectionStatistics(index.getCollectionStatistics())
    wmodel.setEntryStatistics(le);
    wmodel.setKeyFrequency(1)
    wmodel.prepare()
    for posting in inv.getPostings(le):
        docno = meta.getItem("docno", posting.getId())
        score = wmodel.score(posting)
        print("%s with score %0.4f"  % (docno, score))

Note that using Retriever or similar is probably an easier prospect for such a use case.

Can I get the index as a corpus_iter()?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A corpus_iter can be obtained from an Index object, which allows for instance:
 - indexing the pre-tokenised Terrier index directly in another indexing pipeline
 - extracting document metadata for ingestion into another indexing pipeline

Metadata Example::

    iter = index.get_corpus_iter(return_toks=False)
    next(iter)
    # would display {'docno' : 'd1', 'text' : 'This document contains ...' }
    # assuming that index has been built with metadata=['docno', 'text']

Pre-tokenised Example::

    iter = index.get_corpus_iter()
    next(iter)
    # would display {'docno' : 'd1', 'toks' : {'a' : 1, 'the' : 2}}

Document Pruning Example::

    index_pipe = (
        # update the toks column for each document, keeping only terms with frequency > 1
        pt.apply.toks(lambda row: { t : row['toks'][t] for t in row['toks'] if row['toks'][t] > 1 } ) 
        >> pt.IterDictIndexer("./pruned_index", pretokenised=True)
    )
    new_index_ref = index_pipe.index( index.get_corpus_iter())
