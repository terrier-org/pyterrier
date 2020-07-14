import pandas as pd


SUPPORTED_TOPIC_FORMATS = ["trec", "trecxml", "singleline"]

def autoopen(filename, mode='rb'):
    if filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        import bz2
        return bz2.open(filename, mode)
    return open(filename, mode)

def read_results(filename, format="trec", **kwargs):
    if format == "trec" or format is None:
        return _read_results_trec(filename)
    elif format == "letor":
        return _read_results_letor(filename, **kwargs)
    else:
        raise ValueError("Format %s not known, possible values are trec or letor" % format)

def _read_results_letor(filename, labels=False):

    def _parse_line(l):
            # $line =~ s/(#.*)$//;
            # my $comment = $1;
            # my @parts = split /\s+/, $line;
            # my $label = shift @parts;
            # my %hash = map {split /:/, $_} @parts;
            # return ($label, $comment, %hash);
        import re
        import numpy as np
        line, comment = l.split("#")
        line = line.strip()
        parts = re.split(r'\s+|:', line)
        label = parts.pop(0)
        m = re.search(r'docno\s?=\s?(\S+)', comment)
        docno = m.group(1)
        kv = {}
        qid = None
        for i, k in enumerate(parts):
            if i % 2 == 0:
                if k == "qid":
                    qid = parts[i+1]
                else:
                    kv[int(k)] = float(parts[i+1])
        features = np.array([kv[i] for i in sorted(kv.keys())])
        return (label, qid, docno, features)       

    with autoopen(filename, 'rt') as f:
        rows = []
        for line in f:
            if line.startswith("#"):
                continue
            (label, qid, docno, features) = _parse_line(line)
            if labels:
                rows.append([qid, docno, features, label])
            else:
                rows.append([qid, docno, features])
        return pd.DataFrame(rows, columns=["qid", "docno", "features", "label"] if labels else ["qid", "docno", "features"])

def _read_results_trec(filename):
    results = []
    df = pd.read_csv(filename, sep=r'\s+', names=["qid", "iter", "docno", "rank", "score", "name"])
    df = df.drop(columns="iter")
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)
    return df

def write_results(res, filename, format="trec", **kwargs):
    if format == "trec" or format is None:
        _write_results_trec(res, filename, **kwargs)
    elif format == "letor":
        _write_results_letor(res, filename, **kwargs)
    else:
        raise ValueError("Format %s not known, possible values are trec or letor")

def _write_results_trec(res, filename, run_name="pyterrier"):
        res_copy = res.copy()[["qid", "docno", "rank", "score"]]
        res_copy.insert(1, "Q0", "Q0")
        res_copy.insert(5, "run_name", run_name)
        res_copy.to_csv(filename, sep=" ", header=False, index=False)

def _write_results_letor(res, filename, qrels=None, default_label=0):
    if qrels is not None:
        res = res.merge(qrels, on=['qid', 'docno'], how='left').fillna(default_label)
    with autoopen(filename, "wt") as f:
        for row in res.itertuples():
            values = row.features
            label = row.label if qrels is not None else default_label
            feat_str = ' '.join( [ '%i:%f' % (i+1,values[i]) for i in range(len(values)) ] )
            f.write("%d qid:%s %s # docno=%s\n" % (label, row.qid, feat_str, row.docno))

def read_topics(filename, format="trec", **kwargs):
    if format == "trec" or format is None:
        return _read_topics_trec(filename, **kwargs)
    elif format == "trec":
        return _read_topics_trecxml(filename, **kwargs)    
    elif format == "singleline":
        return _read_topics_singleline(filename, **kwargs)
    else:
        raise ValueError("Format %s not known, possible values are trec, trecxml or singleline")

def _read_topics_trec(file_path, doc_tag="TOP", id_tag="NUM", whitelist=["TITLE"], blacklist=["DESC","NARR"]):
    """
    Parse a file containing topics in standard TREC format

    Args:
        file_path(str): The path to the topics file

    Returns:
        pandas.Dataframe with columns=['qid','query']
        both columns have type string
    """
    from jnius import autoclass
    from . import check_version
    assert check_version("5.3")
    trecquerysource = autoclass('org.terrier.applications.batchquerying.TRECQuery')
    tqs = trecquerysource(
        [file_path], doc_tag, id_tag, whitelist, blacklist,
        # help jnius select the correct constructor 
        signature="([Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)V")
    topics_lst=[]
    while(tqs.hasNext()):
        topic = tqs.next()
        qid = tqs.getQueryId()
        topics_lst.append([qid,topic])
    topics_dt = pd.DataFrame(topics_lst,columns=['qid','query'])
    return topics_dt

def _read_topics_trecxml(filename, tags=["query", "question", "narrative"], tokenise=True):
    """
    Parse a file containing topics in TREC-like XML format

    Args:
        filename(str): The path to the topics file

    Returns:
        pandas.Dataframe with columns=['qid','query']
    """
    import xml.etree.ElementTree as ET
    import pandas as pd
    tags=set(tags)
    topics=[]
    tree = ET.parse(filename)
    root = tree.getroot()
    from jnius import autoclass
    tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    for child in root.iter('topic'):
        qid = child.attrib["number"]
        query = ""
        for tag in child:
            if tag.tag in tags:
                query_text = tag.text
                if tokenise:
                    query_text = " ".join(tokeniser.getTokens(query_text))
                query += " " + query_text
        topics.append((str(qid), query)) 
    return pd.DataFrame(topics, columns=["qid", "query"])

def _read_topics_singleline(filepath, tokenise=True):
    """
    Parse a file containing topics, one per line

    Args:
        file_path(str): The path to the topics file
        tokenise(bool): whether the query should be tokenised, using Terrier's standard Tokeniser.
            If you are using matchop formatted topics, this should be set to False.

    Returns:
        pandas.Dataframe with columns=['qid','query']
    """
    rows = []
    from jnius import autoclass
    from . import check_version
    assert check_version("5.3")
    slqIter = autoclass("org.terrier.applications.batchquerying.SingleLineTRECQuery")(filepath, tokenise)
    for q in slqIter:
        rows.append([slqIter.getQueryId(), q])
    return pd.DataFrame(rows, columns=["qid", "query"])

def read_qrels(file_path):
    """
    Parse a file containing qrels

    Args:
        file_path(str): The path to the qrels file

    Returns:
        pandas.Dataframe with columns=['qid','docno', 'label']
        with column types string, string, and int
    """
    df = pd.read_csv(file_path, sep=r'\s+', names=["qid", "iter", "docno", "label"])
    df = df.drop(columns="iter")
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    return df