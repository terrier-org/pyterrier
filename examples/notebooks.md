
# Examples Notebooks for PyTerrier

This page summarises the available notebooks for PyTerrier.

## PyTerrier Functionality


|    Notebook      |   On Colab?     | Description                      |  Read More 
| ---------------- | --------------- | -------------------------------- | --------------- | 
| [indexing.ipynb](notebooks/indexing.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/indexing.ipynb) | How to index a corpus using PyTerrier | [Indexing documentation](https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html)
| [index_api.ipynb](notebooks/index_api.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/index_api.ipynb) | How to access Terrier's index structures from PyTerrier |  | 
| [retrieval_and_evaluation.ipynb](notebooks/retrieval_and_evaluation.ipynb)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/retrieval_and_evaluation.ipynb) | How to perform retrieval and evaluation using PyTerrier | [Retrieval documentation](https://pyterrier.readthedocs.io/en/latest/terrier-retrieval.html)  | 
| [experiment.ipynb](notebooks/experiment.ipynb)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/experiment.ipynb) | How to use PyTerrier's declarative Experiment formulation | [Experiment documentation](https://pyterrier.readthedocs.io/en/latest/experiments.html) | 
| [ltr.ipynb](notebooks/ltr.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/ltr.ipynb) | How to perform learning-to-rank using PyTerrier |  [Learning to Rank documentation](https://pyterrier.readthedocs.io/en/latest/ltr.html)
| [non_en_retrieval.ipynb](notebooks/non_en_retrieval.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/non_en_retrieval.ipynb) | How to configure PyTerrier for languages other than English. |  

## Experiments using PyTerrier

|    Notebook      |   On Colab?     | Description                      |   
| ---------------- | --------------- | -------------------------------- |
| [Robust04.ipynb](experiments/Robust04.ipynb)   |                 | Demonstration of [query expansion](https://pyterrier.readthedocs.io/en/latest/rewrite.html) effectiveness on TREC Robust04 |
| [uogTrBaseDPH.ipynb](https://github.com/cmacdonald/pyterrier-msmarco-document-leaderboard-runs/blob/master/uogTrBaseDPH.ipynb)  |  |   DPH run submitted to MSMARCO  Document Ranking Task leaderboard  |
| [uogTrBaseDPHQ.ipynb](https://github.com/cmacdonald/pyterrier-msmarco-document-leaderboard-runs/blob/master/uogTrBaseDPHQ.ipynb)  |  |   DPH + Bo1 query expansion run submitted to MSMARCO  Document Ranking Task leaderboard  |

## PyTerrier Neural Re-Ranking

|    Notebook      |   On Colab?     | Description                      |  Read More | 
| ---------------- | --------------- | -------------------------------- | ---------- |
| [pt_indexed_epic.ipynb](https://github.com/Georgetown-IR-Lab/OpenNIR/blob/master/examples/pt_indexed_epic.ipynb)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Georgetown-IR-Lab/OpenNIR/blob/master/examples/pt_indexed_epic.ipynb) | Demonstration of pre-computing EPIC vectors and using them for second-stage scoring using PyTerrier on the TREC COVID benchmark | [OpenNIR repository](https://github.com/Georgetown-IR-Lab/OpenNIR)


## PyTerrier Dense Retrieval

|    Notebook      |   On Colab?     | Description                      |  Read More | 
| ---------------- | --------------- | -------------------------------- | ---------- |
| [pyterrier_ance_vaswani.ipynb](https://github.com/terrierteam/pyterrier_ance/blob/main/pyterrier_ance_vaswani.ipynb)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_ance/blob/main/pyterrier_ance_vaswani.ipynb) | Demonstration of ANCE dense indexing and retrieval using PyTerrier on the Vaswani NPL corpus | [PyTerrier_ance repository](https://github.com/terrierteam/pyterrier_ance)


