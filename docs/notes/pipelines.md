# Pipelines and Operators

Part of the power of PyTerrier comes in the ability to make complex retrieval pipelines. This is made possible by the operators available on Pyterrier's transformer objects. The following table summarises the available operators

| Operator   | Meaning                         | Implemented |
|------------|---------------------------------|-------------|
|    `>>`    |      Then - chaining pipes      | [x]         |
|    `&`     |    Document Set Intersection    | [x]         |
|    `\|`    |        Document Set Union       | [x]         |
|    `**`    |          Features Union         | [x]         |
|    `+`     |   Linear combination of scores  | [x]         |
|    `*`     |    Scalar factoring of scores   | [x]         |
|    `%`     |        Apply rank cutoff        | [x]         |
|    `^`     |  Concatenate run with another   | [x]         |
|    `~`     |     Cache transformer result    | [x]         |

## Definitions

### Then:

Apply one transformation followed by another
```python

#rewrites topics to include #uw2 etc
sdm = RewriteSDM()
br = BatchRetrieve(index, "DPH")

res = br.transform( sdm.transform(topics))

```

We use `>>` as shorthand for then.

```python
res = (sdm >> br).transform(topics)
```

### Linear Combine:

This allows the scores of different retrieval systems to be linearly combined.
(with weights)

Instead of the following Python:
```python
br_DPH = BatchRetrieve(index, "DPH")
br_BM25 = BatchRetrieve(index, "BM25")


res1 = br_DPH.trasnform(topics)
res2 = br_BM25.trasnform(topics)
res = res1.merge(res2, on=["qid", "docno"])
res["score"] = 2 * res["score_x"] + res["score_y"]

```
we use binary + and * operators. This is natural, as it is intuitive to combine weighted retrieval functions using + and *.

```python
br_DPH = BatchRetrieve(index, "DPH")
br_BM25 = BatchRetrieve(index, "BM25")
res = (2* br_DPH + br_BM25).transform(topics)
```

### Feature Union:

Here we take one system, DPH, to get an initial candidate set, then add more systems as features.

The Python would have looked like:
```python
sample_br = BatchRetrieve(index, "DPH")
BM25F_br = BatchRetrieve(index, "BM25F")
PL2F_br = BatchRetrieve(index, "PL2F")

sampleRes = sample_br.transform(topics)
#assumes sampleRes contains the queries
BM25F_res = BM25F_br.transform(sampleRes)
PL2F_res = PL2F_br.transform(sampleRes)

final_res = BM25F_res.join(PL2F_res, on=["qid", "docno"])
final_res["features"] = np.stack(final_res["features_x"], final_res["features_y"])

```

Instead, we use ** to denote feature-union:


```python

sample_br = BatchRetrieve(index, "DPH")
BM25F_br = BatchRetrieve(index, "BM25F")
PL2F_br = BatchRetrieve(index, "PL2F")

# ** is the feature union operator. It requires a candidate document set as input 
(BM25F_br ** PL2F_br)).transform(sample_br.transform(topics))
# or combined with the then operator, >>
(sample_br >> (BM25F_br ** PL2F_br)).transform(topics)

```

### Set Intersection

Make a retrieval set that only includes documents that occur in the intersection of both retrieval sets. Scores are undefined. Normally, these documents would be re-scored.

```python
BM25_br = BatchRetrieve(index, "BM25")
PL2_br = BatchRetrieve(index, "PL2")

(BM25_br & PL2_br).transform(topics)
```

### Set Union

Make a retrieval set that includes documents that occur in the union (either) of both retrieval sets. Scores are undefined. Normally, these documents would be re-scored.

```python
BM25_br = BatchRetrieve(index, "BM25")
PL2_br = BatchRetrieve(index, "PL2")

(BM25_br | PL2_br).transform(topics)
```

### Rank Cutoff

This limits the nuimber of results for each query. 

```python
pipe1 = pt.BatchRetrieve(index, "BM25") % 10

```


### Concatenate

Sometimes, we may only want to apply an expensive retrieval process on a few top-ranked documents, and fill up the rest of the ranking with the rest of the documents (removing duplicates). We can do that using the concatenate operator. Concretely, in the example below, `alldocs` is our candidate set, of say 1000 documents per query. We re-rank the top 10 documents for each query using `ExpensiveReranker()`, in a pipeline called `topdocs`. We then use the concatenate operator (^) to append the 990 documents, such that they have scores and ranks descending after the documents obtained from the `topdocs` pipeline.

```python
alldocs = BatchRetrieve(index, "BM25")
topdocs = alldocs % 10 >> ExpensiveReranker()
finaldocs = topdocs ^ alldocs
```

## Caching

Some transformers are expensive to apply, particularly initial retrievals. For instance, we might find ourselves repeatedly running our BM25 baseline. We can request Pyterrier to _cache_ the outcome of a transformer for a given qid by using the unary `~` operator.

Consider the following example:
```python
from pyterrier import BatchRetrieve, Experiment
firstpass = BatchRetrieve(index, "BM25")
reranker = ~firstpass >> BatchRetrieve(index, "BM25F")
Experiment([~firstpass, ~reranker], topics, qrels)
```
In this example, `firstpass` is cached when it is used in the Experiment evaluation, as well as when it is used in the reranker. We also cache the outcome of the Experiment, so that another evaluation will be faster.

By default, Pyterrier caches results to `~/.pyterrier/transformer_cache/`.

## Optimisation

Some operators applied to transformer can be optimised by the underlying search engine - for instance, cutting a ranking earlier. So while the following two pipelines are semantically equivalent, the latter might be more efficient:
```python
pipe1 = BatchRetrieve(index, "BM25") % 10
pipe2 = pipe1.compile()
```

## Fitting

When fit() is called on a pipeline, all estimators (transformer that also have a fit()) method, as specified by EstimatorBase) are fitted in turn.
