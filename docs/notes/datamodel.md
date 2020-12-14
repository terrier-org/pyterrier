# PyTerrier Data Model

Pyterrier allows the chaining of different transformers in different manners. Each transformer has a `transform()` method, which takes as input a Pandas dataframe, and returns a dataframe also.

Different transformers change the dataframes in different ways - for instance, retrieving documents, or rewriting queries.

We define different dataframe "types" of data frame - the "primary key" is emphasised.

| Data Type         | Required Columns                                | Description                                 | 
|-------------------|-------------------------------------------------|---------------------------------------------|
|    Q              | ["qid", "query"]                                | A set of queries                            |
|    D              | ["docno"]                                       | A set of documents                          |
|    R              | ["qid", "docno", "score", "rank"]               | A ranking of documents for each query       |
|    R w/ features  | ["qid", "docno", "score", "rank", "features"]   | A numpy array of features for each document |

1. Queries (Q)

A dataframe with two columns:
 - _qid_: A unique identified for each query
 - query: The textual form of each query

Different transformers may have additional columns, but none are currently implemented.

An example dataframe with one query might be constructed as:
```python
pd.DataFrame([["q1", "a query"]], columns=["qid", "query")
```

2. Set of documents (D)

A dataframe with columns:
 - _docno_: The unique idenitifier for each document

There might be other attributes such as text

3. Ranked Documents (R)

A dataframe with more columns, clearly inspired by the TREC results format:
 - _qid_: A unique identifier for each query
 - query: The textual form of each query
 - _docno_: The unique idenitifier for each document
 - score: The score for each document for that query
 - rank: The rank of the document for that query

Optional columns might support additional transformers, such as text (for the contents of the documents), url or title columns.

Note that the retrieved documents is a subset of the cartesian product of documents and queries; it is important that the query (text) attribute is present for at least ONE document rather than all documents for a given query.

An example dataframe with two documents might be constructed as:

```python
pd.DataFrame([["q1", "a query", "d5", 5.2, 1], ["q1", None, "d10", 4.9, 2]], columns=["qid", "query", "docno", "score", "rank")
```

4. Set of documents with features

A dataframe with more columns, clearly inspired by the TREC results format:
 - _qid_: A unique identifier for each query
 - query: The textual form of each query
 - _docno_: The unique idenitifier for each document
 - features: A Numpy array of feature scores for each document

These may optionally have been ranked by a score attribute.