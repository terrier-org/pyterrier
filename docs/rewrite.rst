Query Rewriting & Expansion
---------------------------

Query rewriting refers to changing the formulation of the query in order to improve the effectiveness of the
search ranking. PyTerrier supplies a number of query rewriting transformers designed to work with BatchRetrieve.

Firstly, we differentiate between two forms of query rewriting:

 - `Q -> Q`: this rewrites the query, for instance by adding/removing extra query terms. Examples might 
   be a WordNet- or Word2Vec-based QE; The input dataframes contain only `["qid", "docno"]` columns. The 
   output dataframes contain `["qid", "query", "query_0"]` columns, where `"query"` contains the reformulated
   query, and `"query_0"` contains the previous formulation of the query.

 - `R -> Q`: these class of transformers rewrite a query by making use of an associated set of documents.
   This is typically exemplifed by pseudo-relevance feedback. Similarly the output dataframes contain 
   `["qid", "query", "query_0"]` columns.

The previous formulation of the query can be restored using `pt.rewrite.reset()`, discussed below.

SequentialDependence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class implements Metzler and Croft's sequential dependence model, designed to boost the scores of
documents where the query terms occur in close proximity. Application of this transformer rewrites 
each input query such that:

 - pairs of adjacent query terms are added as `#1` and `#uw8` complex query terms, with a low weight.
 - the full query is added as `#uw12` complex query term, with a low weight.
 - all terms are weighted by a proximity model, either Dirichlet LM or pBiL2.

For example, the query `pyterrier IR platform` would become `pyterrier IR platform #1(pyterrier IR) #1(IR platform) #uw8(pyterrier IR) #uw8(IR platform) #uw12(pyterrier IR platform)`.
NB: Acutally, we have simplified the rewritten query - in practice, we also (a) set the weight of the proximity terms to be low using a `#combine()`
operator and (b) set a proximity term weighting model.

This transfomer is only compatible with BatchRetrieve, as Terrier supports the `#1` and `#uwN` complex query terms operators. The Terrier index must have blocks (positional information) recorded in the index.

.. autoclass:: pyterrier.rewrite.SequentialDependence
    :members: transform

Example::

    sdm = pt.rewrite.SequentialDependence()
    dph = pt.BatchRetrieve(index, wmodel="DPH")
    pipeline = sdm >> dph

References:
 - A Markov Random Field Model for Term Dependencies. Donald Metzler and W. Bruce Croft. In Proceedings of SIGIR 2005. 
 - Incorporating Term Dependency in the DFR Framework. Jie Peng, Craig Macdonald, Ben He, Vassilis Plachouras, Iadh Ounis. In Proceedings of SIGIR 2007. July 2007. Amsterdam, the Netherlands. 2007.

Bo1QueryExpansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class applies the Bo1 Divergence from Randomess query expansion model to rewrite the
query based on the occurences of terms in the feedback documents provided for each
query. In this way, it takes in a dataframe with columns `["qid", "query", "docno", "score", "rank"]`
and returns a dataframe with  `["qid", "query"]`.

.. autoclass:: pyterrier.rewrite.Bo1QueryExpansion
    :members: transform 

Example::

    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    dph = pt.BatchRetrieve(index, wmodel="DPH")
    pipelineQE = dph >> bo1 >> dph

**Alternative Formulations**

Note that it is also possible to configure BatchRetrieve to perform QE directly using controls,
which will result in identical retrieval effectiveness::

    pipelineQE = pt.BatchRetrieve(index, wmodel="DPH", controls={"qemodel" : "Bo1", "qe" : "on"})

However, using `pt.rewrite.Bo1QueryExpansion` is preferable as:

 - the semantics of retrieve >> rewrite >> retrieve are clearly visible.
 - the complex control configuration of Terrier need not be learned. 
 - the rewritten query is visible outside, and not hidden inside Terrier.

References:
 - Amati, Giambattista (2003) Probability models for information retrieval based on divergence from randomness. PhD thesis, University of Glasgow.

KLQueryExpansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to Bo1, this class deploys a Divergence from Randomess query expansion model based on Kullback Leibler divergence.

.. autoclass:: pyterrier.rewrite.KLQueryExpansion
    :members: transform 

References:
 - Amati, Giambattista (2003) Probability models for information retrieval based on divergence from randomness. PhD thesis, University of Glasgow.


RM3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.rewrite.RM3
    :members: transform 

References:
 - Nasreen Abdul-Jaleel, James Allan, W Bruce Croft, Fernando Diaz, Leah Larkey, Xiaoyan Li, Mark D Smucker, and Courtney Wade. UMass at TREC 2004: Novelty and HARD. In Proceedings of TREC 2004.


AxiomaticQE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.rewrite.AxiomaticQE
    :members: transform 

References:
 - Hui Fang, Chang Zhai.: Semantic term matching in axiomatic approaches to information retrieval. In: Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 115–122. SIGIR 2006. ACM, New York (2006).
 - Peilin Yang and Jimmy Lin, Reproducing and Generalizing Semantic Term Matching in Axiomatic Information Retrieval. In Proceedings of ECIR 2019.

Combining Query Formulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyterrier.rewrite.linear

Resetting the Query Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The application of any query rewriting operation, including the apply transformer, `pt.apply.query()`, will return a dataframe
that includes the *input* formulation of the query in the `query_0` column, and the new reformulation in the `query` column. The
previous query reformulation can be obtained by inclusion of a reset transformer in the pipeline.

.. autofunction:: pyterrier.rewrite.reset()


Stashing the Documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you want to apply a query rewriting function as a re-ranker, but your rewriting function uses a different document ranking.
In this case, you can use `pt.rewrite.stash_results()` to stash the documents.

.. autofunction:: pyterrier.rewrite.stash_results()

.. autofunction:: pyterrier.rewrite.reset_results()

Example: Query Expansion as a re-ranker

Some papers advocate for the use of query expansion (PRF) as a re-ranker. This can be attained in PyTerrier through use of
`stash_results()` and `reset_results()`::

    # index: the corpus you are ranking

    dph = pt.BatchRetrieve(index)
    Pipe = dph 
        >> pt.rewrite.stash_results(clear=False)
        >> pt.rewrite.RM3(index)
        >> pt.rewrite.reset_results()
        >> dph

Summary of dataframe types:

+--------------+--------------------+------------------------------------------+
|output of     |dataframe contents  |actual columns                            |
+==============+====================+==========================================+
|  dph         | R                  |qid, query, docno, score                  |
+--------------+--------------------+------------------------------------------+
|save_docs     |R + "stashed_docs_0"|qid, query, docno, score, stashed_docs_0  |
+--------------+--------------------+------------------------------------------+
|RM3           |Q + "stashed_docs_0"|qid, query, query_0, stashed_docs_0       |
+--------------+--------------------+------------------------------------------+
|restore_docs  |R                   |qid, query, docno, score, query_0         |
+--------------+--------------------+------------------------------------------+
|dph           |R                   |qid, query, docno, score, query_0         |
+--------------+--------------------+------------------------------------------+
        

Example: Collection Enrichment as a re-ranker::

    # index: the corpus you are ranking
    # wiki_index: index of Wikipedia, used for enrichment

    dph = pt.BatchRetrieve(index)
    Pipe = dph 
        >> pt.rewrite.save_docs()          
        >> pt.BatchRetrieve(wiki_index)
        >> pt.rewrite.RM3(wiki_index)
        >> pt.rewrite.reset_docs()
        >> dph

In general, collection enrichment describes conducting a PRF query expansion process on an external corpus (often Wikipedia), 
before applying the reformulated query to the main corpus. Collection enrichment can be used for improving a first pass 
retrieval (`pt.BatchRetrieve(wiki_index) >> pt.rewrite.RM3(wiki_index) >> pt.BatchRetrieve(main_index)`). Instead, the particular 
example shown above applies collection enrichment as a re-ranker.

Summary of dataframe types:

+--------------+------------------+----------------------------------------+
|output of     |dataframe contents|actual columns                          |
+==============+==================+========================================+
|  dph         | R                |qid, query, docno, score                |
+--------------+------------------+----------------------------------------+
|save_docs     |Q + "saved_docs_0"|qid, query, saved_docs_0                |
+--------------+------------------+----------------------------------------+
|BatchRetrieve |R + "saved_docs_0"|qid, query, docno, score, stashed_docs_0|
+--------------+------------------+----------------------------------------+
|RM3           |Q + "saved_docs_0"|qid, query, query_0, stashed_docs_0     |
+--------------+------------------+----------------------------------------+
|restore_docs  |R                 |qid, query, docno, score, query_0       |
+--------------+------------------+----------------------------------------+
|dph           |R                 |qid, query, docno, score, query_0       |
+--------------+------------------+----------------------------------------+
