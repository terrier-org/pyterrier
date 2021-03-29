
def _wrap(text, width):
    return text
    #return '\\\n_'.join(textwrap.wrap(text, width=width))

def _get_text(row, name, width):
    value = row.get(name)
    if type(value) != list:
        return value
    if len(value) == 1:
        return value
    return '[' + _wrap(', '.join(value), width=width) + ']'

def fix_width(df, name, width=20):
    df[name] = df.apply(lambda row: _get_text(row, name, width), axis=1)
    return df


import pyterrier as pt

def setup():
    import os
    if not os.path.exists("_includes"):
        os.makedirs("_includes")

def dataset_include():
    df = pt.list_datasets()
    df = fix_width(df, "qrels")
    df = fix_width(df, "topics")
    df = fix_width(df, "corpus")
    df = df[["dataset", "corpus", "index", "topics", "qrels", "info_url"]]
    table = df.set_index('dataset').to_markdown(tablefmt="rst")

    with open("_includes/datasets-list-inc.rst", "wt") as f:
        f.write(table)

def experiment_includes():
    dataset = pt.get_dataset("vaswani")
    # vaswani dataset provides an index, topics and qrels

    # lets generate two BRs to compare
    tfidf = pt.BatchRetrieve(dataset.get_index(), wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(dataset.get_index(), wmodel="BM25")

    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"]
    ).to_markdown(tablefmt="rst")

    with open("_includes/experiment-basic.rst", "wt") as f:
        f.write(table)

    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"]
    ).to_markdown(tablefmt="rst")

    with open("_includes/experiment-names.rst", "wt") as f:
        f.write(table)

    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["official"],
        names=["TF_IDF", "BM25"]
    ).to_markdown(tablefmt="rst")

    with open("_includes/experiment-official.rst", "wt") as f:
        f.write(table)

    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        round={"map" : 4, "recip_rank" : 3},
        names=["TF_IDF", "BM25"]
    ).to_markdown(tablefmt="rst")

    with open("_includes/experiment-round.rst", "wt") as f:
        f.write(table)

    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"],
        baseline=0
    ).to_markdown(tablefmt="rst")
    with open("_includes/experiment-sig.rst", "wt") as f:
        f.write(table)

    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map"],
        names=["TF_IDF", "BM25"],
        baseline=0, correction='bonferroni'
    ).to_markdown(tablefmt="rst")
    with open("_includes/experiment-sig-corr.rst", "wt") as f:
        f.write(table)

    import statsmodels.stats.multitest
    import pandas as pd
    rows=[]
    for (shortcut, name), aliases in zip ( statsmodels.stats.multitest.multitest_methods_names.items(), statsmodels.stats.multitest._alias_list ):
        rows.append([aliases, name])
    table = pd.DataFrame(rows, columns=["Aliases", "Correction Method"]).to_markdown(tablefmt="rst")
    with open("_includes/experiment-corr-methods.rst", "wt") as f:
        f.write(table)

    
    table = pt.Experiment(
        [tfidf, bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"],
        perquery=True
    ).head().to_markdown(tablefmt="rst")
    with open("_includes/experiment-perq.rst", "wt") as f:
        f.write(table)
