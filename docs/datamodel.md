# PyTerrier Data Model

Pyterrier allows the chaining of different transformers in different manners. Each transformer has a `transform()` method, which takes as input a Pandas dataframe, and returns a dataframe also.

Different transformers change the dataframes in different ways - for instance, retrieving documents, or rewriting queries.

We define different dataframe "types" of data frame - the "primary key" is emphasised.

| Data Type         | Required Columns                                  | Description                                 | 
|-------------------|---------------------------------------------------|---------------------------------------------|
|    Q              | `["qid", "query"]`                                | A set of queries                            |
|    D              | `["docno"]`                                       | A set of documents                          |
|    R              | `["qid", "docno", "score", "rank"]`               | A ranking of documents for each query       |
|    R w/ features  | `["qid", "docno", "score", "rank", "features"]`   | A numpy array of features for each document |

## 1. Queries (Q)

A dataframe with two columns, uniquely identified by qid:
 - _qid_: A unique identified for each query (primary key)
 - query: The textual form of each query

Different transformers may have additional columns, but none are currently implemented.

An example dataframe with one query might be constructed as:
```python
pd.DataFrame([["q1", "a query"]], columns=["qid", "query")
```
or `pt.new.ranked_documents()`
```python
pt.new.queries(["a query"], qid=["q1"])
```

When a query has been rewritten, for instance by applying the sequential dependence model or
query expansion, the previous formulation of the query is available under the "query_0" attribute.

## 2. Set of documents (D)

A dataframe with columns:
 - _docno_: The unique idenitifier for each document (primary key)

There might be other attributes such as text

## 3. Ranked Documents (R)

A dataframe representing which documents are retrieved and scored for a given query. This dataframe type has various columns that clearly inspired by the [TREC](https://trec.nist.gov/) results format:
 - _qid_: A unique identifier for each query  (primary key)
 - query: The textual form of each query
 - _docno_: The unique idenitifier for each document  (primary key)
 - score: The score for each document for that query
 - rank: The rank of the document for that query

Note that rank is computed by sorting by qid ascending, then score descending. The first rank for each query is 0. The `pyterrier.model.add_rank()` function is used for adding the rank column. 

Optional columns might support additional transformers, such as text (for the contents of the documents), url or title columns. Their presence can facilitate more advanced transformers, such as BERT-based transformers which operate on the raw text of the documents. For instance, if the Terrier index has additional metadata attributes, these can be included by BatchRetrieve using the `metadata` kwarg, i.e. `pt.terrier.Retrieve(index, metadata=["docno", "title", "body"])`. 

Note that the retrieved documents is a subset of the cartesian product of documents and queries; it is important that the query (text) attribute is present for at least ONE document rather than all documents for a given query.

An example dataframe with two documents might be constructed as:

```python
pd.DataFrame([["q1", "a query", "d5", 5.2, 9], ["q1", None, "d10", 4.9, 1]], columns=["qid", "query", "docno", "score", "rank")
```
or using `pt.new.ranked_documents()`:
```python
pt.new.ranked_documents([[5.2, 4.9]], qid=["q1"], docno=[["d5", "d10"]])
```

## 4. Set of documents with features

A dataframe with more columns, clearly inspired by the TREC results format:
 - _qid_: A unique identifier for each query  (primary key)
 - query: The textual form of each query
 - _docno_: The unique idenitifier for each document  (primary key)
 - features: A Numpy array of feature scores for each document

These may optionally have been ranked by a score attribute.