Accessing Terrier's Index API
-----------------------------

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
