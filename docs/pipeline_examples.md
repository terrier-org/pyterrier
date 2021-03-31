# Examples of Retrieval Pipelines

## Query Rewriting 

### Sequential Dependence Model


```python
pt.rewrite.SDM() >> pt.BatchRetrieve(indexref, wmodel="BM25")
```

Note that the SDM() rewriter has a number of constructor parameters:
 - `remove_stopwords` - defines if stopwords should be removed from the query
 - `prox_model` - change the proximity model. For true language modelling, you should set `prox_model` to "org.terrier.matching.models.Dirichlet_LM"


### Divergence from Randomness Query Expansion

A simple QE transformer can be achieved using
```python
pt.BatchRetrieve(indexref, wmodel="BM25", controls={"qe" : "on"})
```

As this is pseudo-relevance feedback in nature, it identifies a set of documents, extracts informative term in the top-ranked documents, and re-exectutes the query.

However, more control can be achieved by using the QueryExpansion transformer separately, as thus:
```python
pt.BatchRetrieve(indexref, wmodel="BM25") >> \
    pt.rewrite.QueryExpansion(indexref) >> \
    pt.BatchRetrieve(indexref, wmodel="BM25")
```

The QueryExpansion() object has the following constructor parameters:
 - `index_like` - which index you are using to obtain the contents of the documents. This should match the preceeding BatchRetrieve. 
 - `fb_docs` - number of feedback documents to examine
 - `fb_terms` - number of feedback terms to add to the query

Note that different indexes can be used to achieve query expansion using an external collection (sometimes called collection enrichment or external feedback).  For example, to expand queries using Wikipedia as an external resource, in order to get higher quality query re-weighted queries, would look like this:

```python
pt.BatchRetrieve(wikipedia_index, wmodel="BM25") >> \
    pt.rewrite.QueryExpansion(wikipedia_index) >> \
    pt.BatchRetrieve(local_index, wmodel="BM25")
```

### RM3 Query Expansion

We also provide RM3 query expansion, by virtue of an external plugin to Terrier called [terrier-prf](https://github.com/terrierteam/terrier-prf). This needs to be load at initialisation time.

```python
pt.init(boot_packages=["org.terrier:terrier-prf:0.0.1-SNAPSHOT"])
pt.BatchRetrieve(indexref, wmodel="BM25") >> \
    pt.rewrite.RM3(indexref) >> \
    pt.BatchRetrieve(indexref, wmodel="BM25")
```
## Combining Rankings

Sometimes we have good retrieval approaches and we wish to combine these in a unsupervised manner. We can do that using the linear combination operator:
```python
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")
dph = pt.BatchRetrieve(indexref, wmodel="DPH")
linear = bm25_cands + dph_cands
```

Of course, some weighting can help:
```python
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")
dph = pt.BatchRetrieve(indexref, wmodel="DPH")
linear = bm25_cands + 2* dph_cands
```

However, if the score distributions are not similar, finding a good weight can be tricky. Normalisation of retrieval scores can be advantagous in this case. We provide PerQueryMaxMinScoreTransformer() to make easy normalisation.

```python
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25") >> pt.pipelines.PerQueryMaxMinScoreTransformer()
dph = pt.BatchRetrieve(indexref, wmodel="DPH" >> pt.pipelines.PerQueryMaxMinScoreTransformer()
linear = 0.75 * bm25_cands + 0.25 * dph_cands
```


## Learning to Rank

Having shown some of the main formulations, lets show how to build different formulations into a LTR model.
 - Some authors report that it is useful to take a union of different retrieval mechanisms in order to build a good candidate set. We use the set-union operator here to combine the rankings of BM25 and DPH weighting models.
 - We then score each of the retrieved documents 

```python
bm25_cands = pt.BatchRetrieve(indexref, wmodel="BM25")
dph_cands = pt.BatchRetrieve(indexref, wmodel="DPH")
all_cands = bm25_cands | dph_cands

all_features = all_cands >> (  
    pt.BatchRetrieve(indexref, wmodel="BM25F") **
    pt.rewrite.SDM() >> pt.BatchRetrieve(indexref, wmodel="BM25")
    )

import xgboost as xgb
params = {'objective': 'rank:ndcg', 
          'learning_rate': 0.1, 
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6,
          'verbose': 2,
          'random_state': 42 
         }
lambdamart = pt.ltr.apply_learned_model(xgb.sklearn.XGBRanker(**params), form='ltr')
final_pipe = all_features >> lambdamart
final_pipe.fit(tr_topics, tr_qrels, va_topics, va_qrels)

```

