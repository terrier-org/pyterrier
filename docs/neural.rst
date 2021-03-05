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

Indexing, Retrieval and Scoring of Text using Terrier
=====================================================

If you are using a Terrier index for your first-stage ranking, you will want to record the text
of the documents in the MetaIndex. More of PyTerrier's support for operating on text is documented
in :ref:`pt.text`.

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
 - first-stage dense retrieval transformers.
 - more neural integration via `OpenNIR <https://opennir.net/>`_.