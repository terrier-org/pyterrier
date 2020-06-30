![Python package](https://github.com/terrier-org/pyterrier/workflows/Python%20package/badge.svg) [![PyPI version](https://badge.fury.io/py/python-terrier.svg)](https://badge.fury.io/py/python-terrier)

# Pyterrier

A Python API for Terrier

# Installation

Easiest way to get started with Pyterrier is to use one of our Colab notebooks - look for the ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) badges below.

### Linux or Google Colab
1. `pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier`
2. You may need to set JAVA_HOME environment variable if Pyjnius cannot find your Java installation.

### macOS

1. You need to hava Java installed. Pyjnius/PyTerrier will pick up the location automatically.
2. `pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier`

### Windows
Pyterrier is not available for Windows because [pytrec_eval](https://github.com/cvangysel/pytrec_eval) [isn't available for Windows](https://github.com/cvangysel/pytrec_eval/issues/19). If you can compile & install pytrec_eval youself, it should work fine.

# Indexing

PyTerrier has a number of useful classes for creating indices:

 - You can create an index from TREC formatted collection using TRECCollectionIndexer.    
 - For TXT, PDF, Microsoft Word files, etc files you can use FilesIndexer.
 - For Pandas Dataframe you can use DFIndexer.
 - For any abitrary iterable dictionaries, you can use IterDictIndexer.

See examples in the [indexing notebook](examples/notebooks/indexing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/indexing.ipynb)

# Retrieval and Evaluation

```python
topics = pt.Utils.parse_trec_topics_file(topicsFile)
qrels = pt.Utils.parse_qrels(qrelsFile)
BM25_br = pt.BatchRetrieve(index, controls={"wmodel": "BM25"})
res = BM25_br.transform(topics)
pt.Utils.evaluate(res, qrels, metrics = ['map'])
```

There is a worked example in the [retrieval and evaluation notebook](examples/notebooks/retrieval_and_evaluation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/retrieval_and_evaluation.ipynb)

# Experiment - Perform Retrieval and Evaluation with a single function
Pyterrier provides an experiment object, which allows to compare multiple retrieval approaches on the same queries & relevance assessments:

```python
pt.Experiment([BM25_br, PL2_br], topics, qrels, ["map", "ndcg"])
```

There is a worked example in the [experiment notebook](examples/notebooks/experiment.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/experiment.ipynb)

# Pipelines

Pyterrier makes it easy to develop complex [retrieval pipelines](pipelines.md) using Python operators to combine different retrieval approaches. Our [example pipelines](pipeline_examples.md) show how to conduct various common use cases. Each retrieval approach is a transformer, having one key method, `transform()`, which takes a single Pandas dataframe as input, and returns another dataframe. For more information, see the [Pyterrier data model](datamodel.md).

# Learning to Rank

Complex learning to rank pipelines, including for learning-to-rank, can be constructed using Pyterrier's operator language. There are several worked examples in the [learning-to-rank notebook](examples/notebooks/ltr.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/ltr.ipynb)

# Dataset API

Pyterrier allows simple access to standard information retrieval test collections through its dataset API, which can download the topics, qrels, corpus or, for some test collections, a ready-made Terrier index.

```python
topics = pt.datasets.get_dataset("trec-robust-2004").get_topics()
qrels = pt.datasets.get_dataset("trec-robust-2004").get_qrels()
pt.Experiment(topics, [BM25_br, PL2_br], eval_metrics, qrels)
```

You can use `pt.datasets.list_datasets()` to see available test collections - if your favourite test collection is missing, [you can submit a Pull Request](https://github.com/terrier-org/pyterrier/pulls).

# Index API

All of the standard Terrier Index API can be access easily from Pyterrier. 

For instance, accessing term statistics is a single call on an index:
```python
index.getLexicon()["circuit"].getDocumentFrequency()
```

There are lots of examples in the [index API notebook](examples/notebooks/index_api.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/index_api.ipynb)

## Credits

 - Alex Tsolov, University of Glasgow
 - Craig Macdonald, University of Glasgow
 - Nicola Tonellotto, University of Pisa
 - Arthur Câmara, Delft University
 - Alberto Ueda, Federal University of Minas Gerais
 - Sean MacAvaney, Georgetown University
