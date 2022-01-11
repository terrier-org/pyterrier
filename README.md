![Python package](https://github.com/terrier-org/pyterrier/workflows/Python%20package/badge.svg) 
[![PyPI version](https://badge.fury.io/py/python-terrier.svg)](https://badge.fury.io/py/python-terrier)
[![Documentation Status](https://readthedocs.org/projects/pyterrier/badge/?version=latest)](https://pyterrier.readthedocs.io/en/latest/)


# PyTerrier

A Python API for Terrier - v.0.7

# Installation

The easiest way to get started with PyTerrier is to use one of our Colab notebooks - look for the ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) badges below.

### Linux or Google Colab
1. `pip install python-terrier`
2. You may need to set JAVA_HOME environment variable if Pyjnius cannot find your Java installation.

### macOS

1. You need to hava Java installed. Pyjnius/PyTerrier will pick up the location automatically.
2. `pip install python-terrier`

### Windows 🆕
1. `pip install python-terrier`
2. You may need to set JAVA_HOME environment variable if Pyjnius cannot find your Java installation.

# Indexing

PyTerrier has a number of useful classes for creating indices:

 - You can create an index from TREC formatted collection using TRECCollectionIndexer.    
 - For TXT, PDF, Microsoft Word files, etc files you can use FilesIndexer.
 - For Pandas Dataframe you can use DFIndexer.
 - For any abitrary iterable dictionaries, you can use IterDictIndexer.

See the [indexing documentation](https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html), or the examples in the [indexing notebook](examples/notebooks/indexing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/indexing.ipynb)

# Retrieval and Evaluation

```python
topics = pt.io.read_topics(topicsFile)
qrels = pt.io.read_qrels(qrelsFile)
BM25_br = pt.BatchRetrieve(index, wmodel="BM25")
res = BM25_br.transform(topics)
pt.Utils.evaluate(res, qrels, metrics = ['map'])
```

See also the [retrieval documentation](https://pyterrier.readthedocs.io/en/latest/terrier-retrieval.html), or the worked example in the [retrieval and evaluation notebook](examples/notebooks/retrieval_and_evaluation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/retrieval_and_evaluation.ipynb)

# Experiment - Perform Retrieval and Evaluation with a single function
PyTerrier provides an [Experiment](https://pyterrier.readthedocs.io/en/latest/experiments.html) function, which allows to compare multiple retrieval approaches on the same queries & relevance assessments:

```python
pt.Experiment([BM25_br, PL2_br], topics, qrels, ["map", "ndcg"])
```

There is a worked example in the [experiment notebook](examples/notebooks/experiment.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/experiment.ipynb)

# Pipelines

PyTerrier makes it easy to develop complex retrieval pipelines using Python operators such as `>>` to chain different retrieval components. Each retrieval approach is a [transformer](https://pyterrier.readthedocs.io/en/latest/transformer.html), having one key method, `transform()`, which takes a single Pandas dataframe as input, and returns another dataframe. Two examples might encapsulate applying the sequential dependence model, or a query expansion process:
```python
sdm_bm25 = pt.rewrite.SDM() >> pt.BatchRetrieve(indexref, wmodel="BM25")
bo1_qe = BM25_br >> pt.rewrite.Bo1QueryExpansion() >> BM25_br
```

There is documentation on [transformer operators](https://pyterrier.readthedocs.io/en/latest/operators.html) as well as [example pipelines](https://pyterrier.readthedocs.io/en/latest/pipeline_examples.html) show other common use cases. For more information, see the [PyTerrier data model](https://pyterrier.readthedocs.io/en/latest/datamodel.html).

# Neural Reranking and Dense Retrieval

PyTerrier has additional plugins for BERT (through OpenNIR), T5, ColBERT, ANCE, DeepCT and doc2query.

 - OpenNIR: [[Github](https://github.com/Georgetown-IR-Lab/OpenNIR)] [[Documentation](https://opennir.net/)]
 - PyTerrier_ANCE: [[Github](https://github.com/terrierteam/pyterrier_ance)] - dense retrieval
 - PyTerrier_ColBERT: [[Github](https://github.com/terrierteam/pyterrier_colbert)] - dense retrieval and/or neural reranking
 - PyTerrier_T5: [[Github](https://github.com/terrierteam/pyterrier_t5)] - neural reranking
 - PyTerrier_doc2query: [[Github](https://github.com/terrierteam/pyterrier_doc2query)] - neural augmented indexing
 - PyTerrier_DeepCT: [[Github](https://github.com/terrierteam/pyterrier_deepct)] - neural augmented indexing

You can see examples of how to use these, including notebooks that run on Google Colab, in the contents of our [ECIR 2021 tutorial](https://github.com/terrier-org/ecir2021tutorial).

# Learning to Rank

Complex learning to rank pipelines, including for learning-to-rank, can be constructed using PyTerrier's operator language. For example, to combine two features and make them available for learning, we can use the `**` operator.
```python
two_features = BM25_br >> ( 
  pt.BatchRetrieve(indexref, wmodel="DirichletLM") ** 
  pt.BatchRetrieve(indexref, wmodel="PL2") 
)
```

See also the [learning to rank documentation](https://pyterrier.readthedocs.io/en/latest/ltr.html), as well as the worked examples in the [learning-to-rank notebook](examples/notebooks/ltr.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/ltr.ipynb). Some pipelines can be automatically optimised - more detail about pipeline optimisation are included in our ICTIR 2020 paper.

# Dataset API

PyTerrier allows simple access to standard information retrieval test collections through its [dataset API](https://pyterrier.readthedocs.io/en/latest/datasets.html), which can download the topics, qrels, corpus or, for some test collections, a ready-made Terrier index.

```python
topics = pt.get_dataset("trec-robust-2004").get_topics()
qrels = pt.get_dataset("trec-robust-2004").get_qrels()
pt.Experiment([BM25_br, PL2_br], topics, qrels, eval_metrics)
```

You can index datasets that include a corpus using IterDictIndexer and get_corpus_iter:

```python
dataset = pt.get_dataset('irds:cord19/trec-covid')
indexer = pt.IterDictIndexer('./cord19-index')
index_ref = indexer.index(dataset.get_corpus_iter(), fields=('title', 'abstract'))
```

You can use `pt.list_datasets()` to see available test collections - if your favourite test collection is missing, [you can submit a Pull Request](https://github.com/terrier-org/pyterrier/pulls).

All datasets from the [ir_datasets package](https://github.com/allenai/ir_datasets) are available
under the `irds:` prefix. E.g., use `pt.datasets.get_dataset("irds:medline/2004/trec-genomics-2004")`
to get the TREC Genomics 2004 dataset. A full catalogue of ir_datasets is available [here](https://ir-datasets.com/all.html).

# Index API

All of the standard Terrier Index API can be access easily from Pyterrier. 

For instance, accessing term statistics is a single call on an index:
```python
index.getLexicon()["circuit"].getDocumentFrequency()
```

There are lots of examples in the [index API notebook](examples/notebooks/index_api.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/index_api.ipynb)

# Documentation

More documentation for PyTerrier is available at https://pyterrier.readthedocs.io/en/latest/.

# Open Source Licence

PyTerrier is subject to the terms detailed in the Mozilla Public License Version 2.0. The Mozilla Public License can be found in the file [LICENSE.txt](LICENSE.txt). By using this software, you have agreed to the licence.

# Citation Licence

The source and binary forms of PyTerrier are subject to the following citation license: 

By downloading and using PyTerrier, you agree to cite at the undernoted paper describing PyTerrier in any kind of material you produce where PyTerrier was used to conduct search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation licence.

[Declarative Experimentation in Information Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020.](https://arxiv.org/abs/2007.14271)

```bibtex
@inproceedings{pyterrier2020ictir,
    author = {Craig Macdonald and Nicola Tonellotto},
    title = {Declarative Experimentation inInformation Retrieval using PyTerrier},
    booktitle = {Proceedings of ICTIR 2020},
    year = {2020}
}

```

# Credits

 - Alex Tsolov, University of Glasgow
 - Craig Macdonald, University of Glasgow
 - Nicola Tonellotto, University of Pisa
 - Arthur Câmara, Delft University
 - Alberto Ueda, Federal University of Minas Gerais
 - Sean MacAvaney, Georgetown University/University of Glasgow
 - Chentao Xu, University of Glasgow
 - Sarawoot Kongyoung, University of Glasgow
 - Zhan Su, Copenhagen University
