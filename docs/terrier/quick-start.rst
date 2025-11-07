Terrier Quick Start Guide
-------------------------------------------------

.. related:: pyterrier.terrier.TerrierIndex.bm25

This guide provides a brief introduction to using the Terrier integration in PyTerrier.

In this guide, you will:
 - Index a small collection of web text using Terrier
 - Perform retrieval using BM25
 - Apply query expansion techniques to reformulate the query

Prerequisites
==================================================

You first need to have :doc:`PyTerrier installed <installation>`.

You will also need a little bit of storage (around 200MB) on your machine for the dataset and the index.

Indexing a Collection
==================================================

.. cite.dblp:: conf/ecir/HashemiAZC20

To index a collection using Terrier, you first need to create a :class:`~pyterrier.terrier.TerrierIndex` object
pointing to the location where you want to store the index.

.. code-block:: python
    :caption: Creating a Terrier index

    import pyterrier as pt
    my_index = pt.terrier.TerrierIndex('/path/to/index/location.terrier')

