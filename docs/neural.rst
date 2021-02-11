.. _neural:

Neural Rankers and Rerankers
----------------------------

PyTerrier is designed with for ease of integration with neural ranking models, such as BERT.
In short, neural re-rankers that can take the text of the query and the text of a document
can be easily expressed using an :ref:`pyterrier.apply` transformer. 

More complex rankers (for instance, that can be trained within PyTerrier, or that can take
advantage of batching to speed up GPU operations) typically require more complex integrations.
We have separate repositories with integrations of well-known neural re-ranking plaforms 
(`CEDR <https://github.com/Georgetown-IR-Lab/cedr>`_, `ColBERT <https://github.com/stanford-futuredata/ColBERT>`_). 

Indexing and Retrieving Text in Terrier indices
===============================================

If you are using a Terrier index for your first-stage ranking, you will want to record the text
of the documents in the MetaIndex. The following configuration demonstrates saving the title
and remainder of the documents separately in the Terrier index MetaIndex when indexing a 
TREC-formatted corpus::

    files = []  # list of filenames to be indexed
    properties = {
        "TaggedDocument.abstracts":"title,text",
        # The tags from which to save the text. ELSE is special tag name, which means anything not consumed by other tags.
        "TaggedDocument.abstracts.tags":"title,ELSE",
        # Should the tags from which we create abstracts be case-sensitive?
        "TaggedDocument.abstracts.tags.casesensitive":"false",
        # The max lengths of the abstracts. Abstracts will be cropped to this length. Defaults to empty.
        "TaggedDocument.abstracts.lengths":"256,4096",

        # We also need to tell the indexer to store the abstracts generated
        # In addition to the docno, we also need to move the 'title' and 'text' tezxt generated to the meta index
        "indexer.meta.forward.keys":"docno,title,text",
        # The maximum lengths for the meta index entries - we store upto 4KB of document text
        "indexer.meta.forward.keylens":"26,256,4096",
        # We will not be doing reverse lookups using the abstracts and so they are not listed here.
        "indexer.meta.reverse.keys":"docno"
    }
    indexer = pt.TRECCollectionIndexer(INDEX_DIR, verbose=True)
    indexer.setProperties(**properties)
    indexref = indexer.index(files)
    index = pt.IndexFactory.of(indexref)

On the other-hand, for a TSV-formatted corpus such as MSMARCO passages, indexing is easier
using IterDictIndexer::

    def msmarco_generate():
        dataset = pt.get_dataset(("trec-deep-learning-passages")
        with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
            for l in corpusfile:
                docno, passage = l.split("\t")
                yield {'docno' : docno, 'text' : passage}

    iter_indexer = pt.IterDictIndexer("./passage_index")
    indexref = iter_indexer.index(msmarco_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096])


For retrieval, the text can be easily included in the dataframe using::

    DPH_br_body = pt.BatchRetrieve(
        index, 
        wmodel="DPH",
        metadata=["docno", "body"])

If you already have a dataframe of retrieved documents, a simple apply function can 
be used to retrieve the text from the MetaIndex::

    def _add_body_title(res):
        res = res.copy()
        meta = index.getMetaIndex()
        res["body"] = res.apply(lambda row : meta.getItem("body", row["docid"]), axis=1)
        return res

    add_text = pt.apply.generic(_add_body_title)


Available Neural Re-ranking Integrations
========================================

 - The separate `pyterrier_bert <https://github.com/cmacdonald/pyterrier_bert>`_ repository includes `CEDR <https://github.com/Georgetown-IR-Lab/cedr>`_ and `ColBERT <https://github.com/stanford-futuredata/ColBERT>`_ re-ranking integrations.
 - An initial `BERT-QE <https://github.com/cmacdonald/BERT-QE>`_ integration is available.

The following gives an example ranking pipeline using ColBERT for re-ranking documents in PyTerrier.
Long documents are broken up into passages using a sliding-window operation. The final score for each
document is the maximum of any consitutent passages::

    from pyterrier_bert.colbert import ColBERTPipeline

    pipeline = DPH_br_body >> \
        pt.text.sliding() >> \
        ColBERTPipeline("/path/to/checkpoint") >> \
        pt.text.max_passage()

Outlook
=======

We continue to work on improving the integration of neural rankers and re-rankers within PyTerrier. We foresee:
 - easier indexing of text.
 - first-stage dense retrieval transformers. 