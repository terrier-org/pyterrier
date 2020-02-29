# Pyterrier

## Terrier Python API

# Installation

### Linux
1. Make sure that JAVA_HOME environment variable is set to your java directory
2. `pip install python-terrier`

### Windows
Pyterrier is not available for Windows because pytrec_eval isn't available for Windows.

### Colab notebooks
```
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"    
!pip install python-terrier
```

# Indexing

### Indexing TREC formatted collections

You can create an index from TREC formatted collection using TRECCollectionIndexer.    
For TXT, PDF, Microsoft Word files, etc files you can use FilesIndexer.    
For Pandas Dataframe you can use DFIndexer.

See examples at:    
https://colab.research.google.com/drive/17WpzhtlMj1U2UJku-RaO2axNsUFhPI6z

# Retrieval and Evaluation

See examples at:
https://colab.research.google.com/drive/1yime_0D21Q-KzFD4IbsRzTvjRbo9vz4I

# Experiment - Perform Retrieval and Evaluation with a single function
We provide an experiment object, which allows to compare multiple retrieval approaches on the same queries & relevance assessments:

```
pt.Experiment(topics, retr_systems, eval_metrics, qrels, perquery=False, dataframe=True)
```

More examples are provided at:
https://colab.research.google.com/drive/15oG7HwyYCBFuborjmfYglea0VLkUjyK-


