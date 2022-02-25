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

Available Neural Dense Retrieval and Re-ranking Integrations
============================================================

 - `OpenNIR <https://opennir.net/>`_ has integration with PyTerrier - see its `notebook examples <https://github.com/Georgetown-IR-Lab/OpenNIR/tree/master/examples>`_.
 - `PyTerrier_ColBERT <https://github.com/terrierteam/pyterrier_colbert>`_ contains a `ColBERT <https://github.com/stanford-futuredata/ColBERT>`_ integration, including both a text-scorer and a end-to-end dense retrieval.
 - `PyTerrier_ANCE <https://github.com/terrierteam/pyterrier_ance>`_ contains an `ANCE <https://github.com/microsoft/ANCE/>`_ integration for end-to-end dense retrieval.
 - `PyTerrier_T5 <https://github.com/terrierteam/pyterrier_t5>`_ contains a `monoT5 <https://arxiv.org/pdf/2101.05667.pdf>`_ integration.
 - `PyTerrier_doc2query <https://github.com/terrierteam/pyterrier_doc2query>`_ contains a `docT5query <https://github.com/castorini/docTTTTTquery>`_ integration.
 - `PyTerrier_DeepCT <https://github.com/terrierteam/pyterrier_deepct>`_ contains a `DeepCT <https://github.com/AdeDZY/DeepCT>`_ integration.
 - The separate `PyTerrier_BERT <https://github.com/cmacdonald/pyterrier_bert>`_ repository includes `CEDR <https://github.com/Georgetown-IR-Lab/cedr>`_ integration (including "vanilla" BERT models), as well as an earlier ColBERTPipeline integration.
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

We continue to work on improving the integration of neural rankers and re-rankers within PyTerrier.