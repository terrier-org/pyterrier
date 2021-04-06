Parallelisation
---------------

With large datasets, retrieval can sometimes take some time. To address this, PyTerrier Transformers can parallelised.

Each Transformer has a `.parallel()` method, which parallelises the transformer.  Two backends are supported:

 - `'joblib'` - uses multiple processes on your current machine. Resources such as indices will be opened multiple times on your machine. Joblib is the default backend for parallelisation in PyTerrier.
 - `'ray'` - uses multiple processes on your machine or on other machines in the same cluster. Large indices will be reopened on each machine.

Parallelisation occurs by partitioning dataframes and separating them across different processes. Partitioning depends on the type
of the input dataframe:

 - queries: partitioned by qid
 - documents: partitioned by docno
 - ranked documents: partitioned by qid

Parallelisation using Joblib
============================

A transformer pipeline can be parallelised by using the .parallel() transformer method::

    dph = pt.BatchRetrieve(index, wmodel="DPH")
    dph_fast = dph.parallel(2)

In this way, any set of queries passed to dph_fast will be separated into two partitions based on qid.

Parallelisation using Ray
=========================

`Ray <https://ray.io>`_ is a framework for distributing Python tasks across multiple machines. For using it in PyTerrier,
setup your Ray cluster by following the Ray documentation.  Thereafter parallelisation over Ray can be used in PyTerrier in 
a similar way as for joblib::

    import ray
    ray.init() #configure Ray as per your cluster setup
    dph = pt.BatchRetrieve(index, wmodel="DPH")
    dph_fast = dph.parallel(2, backend='ray)

In particular, `ray.init()` must have been called before calling `.parallel()`.

What to Parallelise
===================

Only transformers that can be `pickled <https://docs.python.org/3/library/pickle.html>`_. Transformers that use native code
may not be possible to pickle. Some standard PyTerrier transformers have additional support for parallelisation:

 - Terrier retrieval: pt.BatchRetrieve(), pt.FeaturesBatchRetrieve()
 - Anserini retrieval: pt.AnseriniBatchRetrieve()

Pure python transformers, such as `pt.text.sliding()` are picklable. However, parallelising only `pt.text.sliding()` may not produce
efficiency gains, due to the overheads of shuffling data back and forward. 

Entire transformer pipelines (i.e. combined using operators) can be pickled and parallelised. In general, you should only parallelise 
the most inefficient component of your process, while also minimising the amount of data being transferred between processes. For instance,
consider the following pipeline::

    pipe = pt.BatchRetrieve(index, metadata=["docno", "text"] >> pt.text.sliding() >> pt.text.scorer() >> pt.text.max_passage()

While BatchRetrieve might represent the slowest component of the pipeline, it might make sense to parallelise pipe as a whole,
rather than just BatchRetrieve, as then only the queries and final results  need to be passed betwene processes. Indeed among the
following semantically equivalent pipelines, we expect `parallel_pipe0`  and `parallel_pipe2`  to be faster than `parallel_pipe1`::

    parallel_pipe0 = pt.BatchRetrieve(index, metadata=["docno", "text"]).parallel() >> pt.text.sliding() >> pt.text.scorer() >> pt.text.max_passage()
    parallel_pipe1 = ( pt.BatchRetrieve(index, metadata=["docno", "text"]).parallel() >> pt.text.sliding() ).parallel(2)  >> pt.text.max_passage()
    parallel_pipe2 = pipe.parallel(2)


There are of course overheads on paralllelisation - for instance, the Terrier index has to be loaded for *each* separate process, 
so your machine(s) require enough memory. 

Finally, we do not recommend parallelisation in resource constrained containerised environments such as Google Colab.

If you find PyTerrier transformers that do not parallelise and you think it should, please `raise an issue on the PyTerrier github repository <https://github.com/terrier-org/pyterrier/issues>`_.

Outlook
=======

We expect to integate parallelisation at different parts of the PyTerrier platform, such as for conducting a gridearch. Moreover, we hope that proper integration of multi-threaded retrieval in pt.BatchRetrieve() (while requires upstream improvements in the underlying Terrier platform) will reduce the need for this form of parallelisation.
 
