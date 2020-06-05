![Python package](https://github.com/terrier-org/pyterrier/workflows/Python%20package/badge.svg)

# Pyterrier

A Python API for Terrier

# Installation

### Linux or Google Colab
1. `pip install python-terrier`
2. You may need to set JAVA_HOME environment variable if we cannot find your Java installation.

### macOS

1. You need to hava Java installed. Pyjnius/PyTerrier will pick up the location automatically.
2. `pip install python-terrier`

### Windows
Pyterrier is not available for Windows because [pytrec_eval](https://github.com/cvangysel/pytrec_eval) [isn't available for Windows](https://github.com/cvangysel/pytrec_eval/issues/19). If you can compile & install pytrec_eval youself, it should work fine.

# Indexing

### Indexing TREC formatted collections

You can create an index from TREC formatted collection using TRECCollectionIndexer.    
For TXT, PDF, Microsoft Word files, etc files you can use FilesIndexer.    
For Pandas Dataframe you can use DFIndexer.

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
We provide an experiment object, which allows to compare multiple retrieval approaches on the same queries & relevance assessments:

```python
pt.Experiment(topics, [BM25_br, PL2_br], eval_metrics, qrels)
```

There is a worked example in the [experiment notebook](examples/notebooks/experiment.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](examples/notebooks/experiment.ipynb)

# Learning to Rank
First create a `FeaturesBatchRetrieve(index, features)` object with the desired features.

Call the `transform(topics_set)` function with the train, validation and test topic sets to get dataframes with the feature scores and use them to train your chosen model.

Use your trained model to predict the score of the test_topics and evaluate the result with `pt.Utils.evaluate()`.

```python
BM25_with_features_br = pt.FeaturesBatchRetrieve(index, ["WMODEL:BM25F", "WMODEL:PL2F"], controls={"wmodel" : "BM25"})
```

## LTR_pipeline

Create a LTR_pipeline object with arguments:

1. Index reference or path to index on disc
2. Weighting model name
3. Features list
4. Qrels
5. LTR model

Call the `fit()` method on the created object with the training topics.

Evaluate the results with the Experiment function by using the test topics

```python
pt.LTR_pipeline(index, model, features, qrels, LTR)
```

More learning to rank examples are provided at:
https://colab.research.google.com/drive/1KwHoahx_i0vax9fnCZpLP-JmI9jvSoey


## Credits

 - Alex Tsolov, University of Glasgow
 - Craig Macdonald, University of Glasgow
 - Nicola Tonellotto, University of Pisa
 - Arthur Câmara, Delft University
