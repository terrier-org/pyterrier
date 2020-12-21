# TREC Robust 2004

This document gives a flavour of indexing and obtaining retrieval baselines on the TREC Robust04 test collections. 
You can run these experiments for yourself by using the [associated provided notebook](https://github.com/terrier-org/pyterrier/blob/master/examples/experiments/Robust04.ipynb).

You need to have obtain the TREC Disks 4 & 5 corpora [from NIST](https://trec.nist.gov/data/cd45/index.html).

Topics and Qrels are provided through the `"trec-robust-2004"` PyTerrier dataset.


## Indexing

Indexing is fairly simply. We apply a filter to remove files that shouldn't be indexed, including the Congressional Record.
Indexing on a reasonable machine using a single-thread takes around 7 minutes.

```python
DISK45_PATH="/path/to/disk45"
INDEX_DIR="/path/to/create/the/index"

files = pt.io.find_files(DISK45_PATH)
# no-one indexes the congressional record in directory /CR/
# indeed, recent copies from NIST dont contain it
# we also remove some of the other unneeded files
bad = ['/CR/', '/AUX/', 'READCHG', 'READFRCG']
for b in bad:
    files = list(filter(lambda f: b not in f, files))
indexer = pt.TRECCollectionIndexer(INDEX_DIR, verbose=True)
indexref = indexer.index(files)
```

## Retrieval - Simple Weighting Models

Here we define and evaluate standard weighting models.

```python

BM25 = pt.BatchRetrieve(index, wmodel="BM25")
DPH  = pt.BatchRetrieve(index, wmodel="DPH")
PL2  = pt.BatchRetrieve(index, wmodel="PL2")
DLM  = pt.BatchRetrieve(index, wmodel="DirichletLM")

pt.Experiment(
    [BM25, DPH, PL2, DLM],
    pt.get_dataset("trec-robust-2004").get_topics(),
    pt.get_dataset("trec-robust-2004").get_qrels(),
    eval_metrics=["map", "P_10", "P_20", "ndcg_cut_20"],
    names=["BM25", "DPH", "PL2", "Dirichlet QL"]
)

```

Results are as follows:

|    | name         |      map |     P_10 |     P_20 |   ndcg_cut_20 |
|---:|:-------------|---------:|---------:|---------:|--------------:|
|  0 | BM25         | 0.241763 | 0.426104 | 0.349398 |      0.408061 |
|  1 | DPH          | 0.251307 | 0.44739  | 0.361446 |      0.422524 |
|  2 | PL2          | 0.229386 | 0.420884 | 0.343775 |      0.402179 |
|  3 | Dirichlet QL | 0.236826 | 0.407631 | 0.337952 |      0.39687  |

## Retrieval - Query Expansion

Here we define and evaluate standard weighting models on top of DPH and BM25, respectively.
We use the default Terrier parameters for query expansion, namely:
 - 10 expansion terms
 - 3 documents
 - For RM3, a lambda value of 0.5


```python
Bo1 = pt.rewrite.Bo1QueryExpansion(index)
KL = pt.rewrite.KLQueryExpansion(index)
RM3 = pt.rewrite.RM3(index)
pt.Experiment(
    [
            BM25, 
            BM25 >> Bo1 >> BM25, 
            BM25 >> KL >> BM25, 
            BM25 >> RM3 >> BM25, 
    ],
    pt.get_dataset("trec-robust-2004").get_topics(),
    pt.get_dataset("trec-robust-2004").get_qrels(),
    eval_metrics=["map", "P_10", "P_20", "ndcg_cut_20"],
    names=["BM25", "+Bo1", "+KL", "+RM3"]
    )

pt.Experiment(
    [
            DPH, 
            DPH >> Bo1 >> DPH, 
            DPH >> KL >> DPH, 
            DPH >> RM3 >> DPH, 
    ],
    pt.get_dataset("trec-robust-2004").get_topics(),
    pt.get_dataset("trec-robust-2004").get_qrels(),
    eval_metrics=["map", "P_10", "P_20", "ndcg_cut_20"],
    names=["DPH", "+Bo1", "+KL", "+RM3"]
    )
```

Results are as follows:

|    | name   |      map |     P_10 |     P_20 |   ndcg_cut_20 |
|---:|:-------|---------:|---------:|---------:|--------------:|
|  0 | BM25   | 0.241763 | 0.426104 | 0.349398 |      0.408061 |
|  1 | +Bo1   |*0.279458*| 0.448996 | 0.378916 |     *0.436533*|
|  2 | +KL    | 0.279401 | 0.444177 | 0.378313 |      0.435196 |
|  3 | +RM3   | 0.276544 |*0.453815*|*0.379518*|      0.430367 |
|----|--------|----------|----------|----------|---------------|
|  0 | DPH    | 0.251307 | 0.447390 | 0.361446 |      0.422524 |
|  1 | +Bo1   | 0.285334 | 0.458635 | 0.387952 |     *0.444528*|
|  2 | +KL    |*0.285720*| 0.458635 | 0.386948 |      0.442636 |
|  3 | +RM3   | 0.281796 |*0.461044*|*0.389960*|      0.441863 |