Terrier Query Rewriting & Expansion
------------------------------------------

Query rewriting refers to changing the formulation of the query in order to improve the effectiveness of the
search ranking. PyTerrier supplies a number of query rewriting transformers designed to work with Retriever.

Firstly, we differentiate between two forms of query rewriting:

- `Q -> Q`: this rewrites the query, for instance by adding/removing extra query terms. Examples might 
  be a WordNet- or Word2Vec-based QE; The input dataframes contain only `["qid", "docno"]` columns. The 
  output dataframes contain `["qid", "query", "query_0"]` columns, where `"query"` contains the reformulated
  query, and `"query_0"` contains the previous formulation of the query. An example is the sequential dependence
  model, discussed below.

.. schematic::

    pt.rewrite.SequentialDependence()

- `R -> Q`: these class of transformers rewrite a query by making use of an associated set of documents, in a formulation
  typically referred to as pseudo-relevance feedback. Similarly the output dataframes contain 
  `["qid", "query", "query_0"]` columns. This is typically used in a pipeline such as ``Retriever >> Rewriter >> Retriever``, 
  as shown below. Examples of include RM3, Bo1 and KL QE, discussed below.

.. schematic::

    index = pt.terrier.TerrierIndex.example()
    dph = index.dph()
    dph >> pt.rewrite.RM3(index.index_obj()) >> dph

If needed, the previous formulation of the query can be restored using ``pt.rewrite.reset()``, discussed below.

Sequential Dependence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.TerrierIndex.sdm

:meth:`~pyterrier.terrier.TerrierIndex.sdm` provides the sequential dependence model of Metzler and Croft,
designed to boost the scores of documents where the query terms occur in close proximity. Application of this
transformer rewrites  each input query such that:

- pairs of adjacent query terms are added as `#1` and `#uw8` complex query terms, with a low weight.
- the full query is added as `#uw12` complex query term, with a low weight.
- all terms are weighted by a proximity model, either Dirichlet LM or pBiL2.

For example, the query `pyterrier IR platform` would become `pyterrier IR platform #1(pyterrier IR) #1(IR platform) #uw8(pyterrier IR) #uw8(IR platform) #uw12(pyterrier IR platform)`.
NB: Acutally, we have simplified the rewritten query - in practice, we also (a) set the weight of the proximity terms to be low using a `#combine()`
operator and (b) set a proximity term weighting model.

This transfomer is only compatible with Retriever, as Terrier supports the `#1` and `#uwN` complex query terms operators. The Terrier index must have blocks (positional information) recorded in the index.

Example:

.. schematic::
    :show_code:

    index = pt.terrier.TerrierIndex.example()
    # FOLD
    pipeline = index.sdm() >> index.dph()


.. tip::

    The SDM query transformation does not technically depend on the index. It's :meth:`TerrierIndex.sdm() <pyterrier.terrier.TerrierIndex.sdm>` is available,
    however, to first check that the index has the positional information necessary to perform SDM. This helps avoid
    errors that can crop up once executed.

.. cite.dblp:: conf/sigir/MetzlerC05
.. cite.dblp:: conf/sigir/PengMHPO07

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
    dph = pt.terrier.Retriever(index, wmodel="DPH")
    pipelineQE = dph >> bo1 >> dph

View the expansion terms::

    pipelineDisplay = dph >> bo1
    pipelineDisplay.search("chemical reactions")
    # will return a dataframe with ['qid', 'query', 'query_0'] columns
    # the reformulated query can be found in the 'query' column, 
    # while the original query is in the 'query_0' columns

**Alternative Formulations**

Note that it is also possible to configure Retriever to perform QE directly using controls,
which will result in identical retrieval effectiveness::

    pipelineQE = pt.terrier.Retriever(index, wmodel="DPH", controls={"qemodel" : "Bo1", "qe" : "on"})

However, using `pt.rewrite.Bo1QueryExpansion` is preferable as:

- the semantics of retrieve >> rewrite >> retrieve are clearly visible.
- the complex control configuration of Terrier need not be learned. 
- the rewritten query is visible outside, and not hidden inside Terrier.

.. cite.dblp:: phd/ethos/Amati03

KLQueryExpansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to Bo1, this class deploys a Divergence from Randomess query expansion model based on Kullback Leibler divergence.

.. autoclass:: pyterrier.rewrite.KLQueryExpansion
    :members: transform 

.. cite.dblp:: phd/ethos/Amati03


RM3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier.rewrite.RM3
    :members: transform 

.. cite.dblp:: conf/trec/JaleelACDLLSW04

Resetting the Query Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.rewrite.reset

The application of any query rewriting operation, including the apply transformer, ``pt.apply.query()``, will return a dataframe
that includes the *input* formulation of the query in the `query_0` column, and the new reformulation in the `query` column. The
previous query reformulation can be obtained by inclusion of a :func:`~pyterrier.terrier.rewrite.reset` transformer in the pipeline.

This is useful if, for instance, you want to use a PRF pipeline to retrieve more relevant documents, but then want to
revert to the original query formulation for a final ranking step such as MonoT5. For example:

.. schematic::
    :show_code:

    from pyterrier_t5 import MonoT5ReRanker
    index = pt.terrier.TerrierIndex.example()
    dph = index.dph()
    monoT5 = MonoT5ReRanker()
    # FOLD
    pipeline = index.dph() >> index.rm3() >> index.dph() >> pt.rewrite.reset() >> pt.get_dataset('irds:vaswani').text_loader() >> monoT5

Tokenising the Query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.rewrite.tokenise

Sometimes your query can include symbols that aren't compatible with how your retriever parses the query.
In this case, a custom tokeniser can be applied as part of the retrieval pipeline. using :meth:`pt.terrier.rewrite.tokenise <pyterrier.terrier.rewrite.tokenise>`.

Advanced: Combining Query Formulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.rewrite.linear

In some cases, you may want to combine multiple query formulations into a single query.
This can be achieved using :meth:`~pyterrier.terrier.rewrite.linear`, which allows you to linearly combine multiple query columns
into a single query column.

Advanced: Stashing the Documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. related:: pyterrier.terrier.rewrite.stash_results
.. related:: pyterrier.terrier.rewrite.reset_results

Very rarely, you might want to apply a query rewriting function as a re-ranker, but your rewriting function uses a different document ranking.
In this case, you can use :meth:`~pyterrier.terrier.rewrite.stash_results` to stash the retrieved documents for each query, so they can be recovered and 
re-ranked later using your rewritten query formulation. :meth:`~pyterrier.terrier.rewrite.reset_results` can then be used later to restore the stashed documents.

Example: Query Expansion as a re-ranker

.. cite.dblp::     conf/ictir/Diaz15

Some papers advocate for the use of query expansion (PRF) as a re-ranker. 
This can be attained in PyTerrier through use of ``stash_results()`` and ``reset_results()``:

.. code-block:: python

    # index: the corpus you are ranking
    pipeline = (
        index.dph()
        >> pt.terrier.rewrite.stash_results(clear=False)
        >> index.rm3()
        >> pt.terrier.rewrite.reset_results()
        >> index.dph()
    )


Summary of dataframe types:

+--------------+------------------------+---------------------------------------------+
|output of     |dataframe contents      |actual columns                               |
+==============+========================+=============================================+
|  dph         | R                      |qid, query, docno, score                     |
+--------------+------------------------+---------------------------------------------+
|stash_results |R + "stashed_results_0" |qid, query, docno, score, stashed_results_0  |
+--------------+------------------------+---------------------------------------------+
|RM3           |Q + "stashed_results_0" |qid, query, query_0, stashed_results_0       |
+--------------+------------------------+---------------------------------------------+
|reset_results |R                       |qid, query, docno, score, query_0            |
+--------------+------------------------+---------------------------------------------+
|dph           |R                       |qid, query, docno, score, query_0            |
+--------------+------------------------+---------------------------------------------+
        
Indeed, as we need RM3 to have the initial ranking of documents as input, we use `clear=False` as the kwarg
to stash_results().

Example: Collection Enrichment as a re-ranker:

.. code-block:: python

    # index: the corpus you are ranking
    # wiki_index: index of Wikipedia, used for enrichment

    pipeline = (
        index.dph()
        >> pt.terrier.rewrite.stash_results()          
        >> wiki_index.dph()
        >> wiki_index.rm3()
        >> pt.terrier.rewrite.reset_results()
        >> index.dph()
    )

In general, collection enrichment describes conducting a PRF query expansion process on an external corpus (often Wikipedia), 
before applying the reformulated query to the main corpus. Collection enrichment can be used for improving a first pass 
retrieval (``wiki_index.dph() >> wiki_index.rm3() >> main_index.dph()``). Instead, the particular 
example shown above applies collection enrichment as a re-ranker.


Summary of dataframe types:

+--------------+-----------------------+-------------------------------------------+
|output of     |dataframe contents     |actual columns                             |
+==============+=======================+===========================================+
|  dph         | R                     |qid, query, docno, score                   |
+--------------+-----------------------+-------------------------------------------+
|stash_results |Q + "stashed_results_0"|qid, query, saved_docs_0                   |
+--------------+-----------------------+-------------------------------------------+
|Retriever     |R + "stashed_results_0"|qid, query, docno, score, stashed_results_0|
+--------------+-----------------------+-------------------------------------------+
|RM3           |Q + "stashed_results_0"|qid, query, query_0, stashed_results_0     |
+--------------+-----------------------+-------------------------------------------+
|reset_results |R                      |qid, query, docno, score, query_0          |
+--------------+-----------------------+-------------------------------------------+
|dph           |R                      |qid, query, docno, score, query_0          |
+--------------+-----------------------+-------------------------------------------+

In this example, we have a Retriever instance executed on the wiki_index before RM3, so we clear the
document ranking columns when using ``stash_results()``.
