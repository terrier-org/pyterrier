# Pyterrier

## Terrier Python API

# Installation

### Linux
1. Make sure that JAVA_HOME environment variable is set to your java directory
2. pip install python-terrier

### Windows
Pyterrier is not available for windows because pytrec_eval isn't available for windows.

### Colab notebooks
1.os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"    
2.!pip install python-terrier

# Indexing

### Indexing TREC formatted collections

You can create an index from TREC formatted collection using TRECIndex.    
For pdf,word,txt or etc files you can use FilesIndex.    
For pandas Dataframe you can use DFIndex.

See examples at:    
https://colab.research.google.com/drive/17WpzhtlMj1U2UJku-RaO2axNsUFhPI6z

# Retrieval and Evaluation

See examples at:
https://colab.research.google.com/drive/1yime_0D21Q-KzFD4IbsRzTvjRbo9vz4I

# Experiment - Perform Retriaval and Evaluation with a single function
pt.Experiment(topics,retr_systems,eval_metrics,qrels, perquery=False, dataframe=True):

See examples at:
https://colab.research.google.com/drive/15oG7HwyYCBFuborjmfYglea0VLkUjyK-


