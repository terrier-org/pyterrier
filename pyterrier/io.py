import pandas as pd

def autoopen(filename, mode='rb'):
    """
    A drop-in for open() that applies automatic compression for .gz and .bz2 file extensions
    """

    if filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        import bz2
        return bz2.open(filename, mode)
    return open(filename, mode)

def read_results(filename, format="trec", **kwargs):
    """
    Reads a file into a results dataframe.

    Parameters:
        filename (str): The filename of the file to be read. Compressed files are handled automatically.
        format (str): The format of the results file: one of "trec", "letor", "minimal". Default is "trec"
        **kwargs (dict): Other arguments for the internal method
    
    Returns:
        dataframe with usual qid, docno, score columns etc
    """
    if format is None:
        format = "trec"
    if not format in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % str(SUPPORTED_RESULTS_FORMATS.keys()))
    return SUPPORTED_RESULTS_FORMATS[format][0](filename, **kwargs)

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
    """
    Write a results dataframe to a file.

    Parameters:
        res (DataFrame): A results dataframe, with usual columns of qid, docno etc 
        filename (str): The filename of the file to be written. Compressed files are handled automatically.
        format (str): The format of the results file: one of "trec", "letor", "minimal"
        **kwargs (dict): Other arguments for the internal method

    Supported Formats:
        * "trec" -- output columns are $qid Q0 $docno $rank $score $runname
        * "letor" -- This follows the LETOR and MSLR datasets, in that output columns are $label qid:$qid [$fid:$value]+ # docno=$docno
        * "minimal": output columns are $qid $docno $rank.
    
    """
    if format is None:
        format = "trec" 
    if not format in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % str(SUPPORTED_RESULTS_FORMATS.keys()))
    return SUPPORTED_RESULTS_FORMATS[format][1](res, filename, **kwargs)

def _write_results_trec(res, filename, run_name="pyterrier"):
        res_copy = res.copy()[["qid", "docno", "rank", "score"]]
        res_copy.insert(1, "Q0", "Q0")
        res_copy.insert(5, "run_name", run_name)
        res_copy.to_csv(filename, sep=" ", header=False, index=False)

def _write_results_minimal(res, filename, run_name="pyterrier"):
        res_copy = res.copy()[["qid", "docno", "rank"]]
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
    """
    Reads a file containing topics 

    Parameters:
        filename(str): The filename of the topics file
        format(str): One of "trec", "trecxml" or "singleline". Default is "trec" 

    Returns:
        pandas.Dataframe with columns=['qid','query']
        both columns have type string

    Supported Formats:
        * "trec" -- an SGML-formatted TREC topics file. Delimited by TOP tags, each having NUM and TITLE tags; DESC and NARR tags are skipped by default. Control using whitelist and blacklist kwargs
        * "trecxml" -- a more modern XML formatted topics file. Delimited by topic tags, each having nunber tags. query, question and narrative tags are parsed by default. Control using tags kwarg.
        * "singeline" -- one query per line, preceeded by a space or colon. Tokenised by default, use tokenise=False kwargs to prevent tokenisation.
    """
    if format is None:
        format = "trec"
    if not format in SUPPORTED_TOPICS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % str(SUPPORTED_TOPICS_FORMATS.keys()))
    return SUPPORTED_TOPICS_FORMATS[format](filename, **kwargs)

def _read_topics_trec(file_path, doc_tag="TOP", id_tag="NUM", whitelist=["TITLE"], blacklist=["DESC","NARR"]):
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
    Reads a file containing qrels (relevance assessments)

    Parameters:
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

SUPPORTED_TOPICS_FORMATS = {
    "trec" : _read_topics_trec,
    "trecxml" : _read_topics_trecxml,
    "singleline": _read_topics_singleline
}

SUPPORTED_RESULTS_FORMATS = {
    "trec" : (_read_results_trec, _write_results_trec),
    "letor" : (_read_results_letor, _write_results_letor),
    "minimal" : (None, _write_results_minimal)
}