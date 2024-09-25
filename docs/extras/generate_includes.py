
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
    print("Regenerating experiments includes - to skip, use QUICK=1 make html")
    dataset = pt.get_dataset("vaswani")
    # vaswani dataset provides an index, topics and qrels

    # lets generate two BRs to compare
    try:
        indexref = dataset.get_index()
    except ValueError:
        import os, tempfile 
        # if data.terrier.org is down, build the index
        indexref = pt.IterDictIndexer(
                os.path.join(tempfile.gettempdir(), "vaswani_index")
            ).index(pt.get_dataset('vaswani').get_corpus_iter())

    tfidf = pt.terrier.Retriever(indexref, wmodel="TF_IDF")
    bm25 = pt.terrier.Retriever(indexref, wmodel="BM25")

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


def artifact_list_include():
    table = [
        {'class': 'pyterrier.terrier.TerrierIndex', 'package': 'python-terrier', 'package_url': 'https://github.com/terrier-org/pyterrier', 'type': 'sparse_index', 'format': 'terrier'},
        {'class': 'pyterrier_pisa.PisaIndex', 'package': 'pyterrier-pisa', 'package_url': 'https://github.com/terrierteam/pyterrier_pisa', 'type': 'sparse_index', 'format': 'pisa'},
        {'class': 'pyterrier_anserini.AnseriniIndex', 'package': 'pyterrier-anserini', 'package_url': 'https://github.com/seanmacavaney/pyterrier-anserini', 'type': 'sparse_index', 'format': 'anserini'},
        {'class': 'pyterrier_adaptive.corpus_graph.NpTopKCorpusGraph', 'package': 'pyterrier-adaptive', 'package_url': 'https://github.com/terrierteam/pyterrier-adaptive', 'type': 'corpus_graph', 'format': 'np_topk'},
        {'class': 'pyterrier_ciff.CiffIndex', 'package': 'pyterrier-ciff', 'package_url': 'https://github.com/seanmacavaney/pyterrier-ciff', 'type': 'sparse_index', 'format': 'ciff'},
        {'class': 'pyterrier_dr.FlexIndex', 'package': 'pyterrier-dr', 'package_url': 'https://github.com/terrierteam/pyterrier_dr', 'type': 'dense_index', 'format': 'flex'},
        {'class': 'pyterrier_quality.QualCache', 'package': 'pyterrier-quality', 'package_url': 'https://github.com/terrierteam/pyterrier-quality', 'type': 'quality_score_cache', 'format': 'numpy'},
        {'class': 'pyterrier_caching.Lz4PickleIndexerCache', 'package': 'pyterrier-caching', 'package_url': 'https://github.com/seanmacavaney/pyterrier-caching', 'type': 'indexer_cache', 'format': 'lz4pickle'},
        {'class': 'pyterrier_caching.DbmRetrieverCache', 'package': 'pyterrier-caching', 'package_url': 'https://github.com/seanmacavaney/pyterrier-caching', 'type': 'retriever_cache', 'format': 'dbm.dumb'},
        {'class': 'pyterrier_caching.Hdf5ScorerCache', 'package': 'pyterrier-caching', 'package_url': 'https://github.com/seanmacavaney/pyterrier-caching', 'type': 'scorer_cache', 'format': 'hdf5'},
    ]
    with open("_includes/artifact_list.rst", "wt") as f:
        f.write('''
.. list-table::
   :header-rows: 1

   * - Class
     - Package
     - Type / Format
     - Links
''')
        for rec in table:
            f.write('''
   * - :class:`~{class}`
     - `{package} <{package_url}>`_
     - ``{type}``/``{format}``
     - `HuggingFace <https://huggingface.co/datasets?other=pyterrier-artifact.{type}.{format}>`__
'''.format(**rec))
