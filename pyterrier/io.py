import os
import pandas as pd
from contextlib import contextmanager


def coerce_dataframe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    import types
    if isinstance(obj, types.GeneratorType):
        #its a generator, lets assume it generates dataframes
        rtr=[]
        for x in obj:
            assert isinstance(x, pd.DataFrame)
            rtr.append(x)
        return pd.concat(rtr)

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

def find_files(dir):
    """
    Returns all the files present in a directory and its subdirectories

    Args:
        dir(str): The directory containing the files

    Returns:
        paths(list): A list of the paths to the files
    """
    import os
    lst = []
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir, followlinks=True):
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    return sorted(files)


@contextmanager
def _finalized_open_base(path, mode, open_fn):
    assert mode in ('b', 't') # must supply either binary or text mode
    path_tmp = '{}.tmp{}'.format(*os.path.splitext(path)) # add tmp before extension (needed for autoopen)
    # adapted from <https://github.com/allenai/ir_datasets/blob/master/ir_datasets/util/__init__.py#L34>
    try:
        with open_fn(path_tmp, f'x{mode}') as f: # open in exclusive write mode (raises error if already exists)
            yield f
        os.replace(path_tmp, path) # on success, move temp file to original path
    except:
        try:
            os.remove(path_tmp)
        except:
            pass # edge case: removing temp file failed. Ignore and just raise orig error
        raise


def finalized_open(path: str, mode: str):
    """
    Opens a file for writing, but reverts it if there was an error in the process.

    Args:
        path(str): Path of file to open
        mode(str): Either t or b, for text or binary mode

    Example:
        Returns a contextmanager that provides a file object, so should be used in a "with" statement. E.g.::

            with pt.io.finalized_open("file.txt", "t") as f:
                f.write("some text")
            # file.txt exists with contents "some text"

        If there is an error when writing, the file is reverted::

            with pt.io.finalized_open("file.txt", "t") as f:
                f.write("some other text")
                raise Exception("an error")
            # file.txt remains unchanged (if existed, contents unchanged; if didn't exist, still doesn't)
    """
    return _finalized_open_base(path, mode, open)


def finalized_autoopen(path: str, mode: str):
    """
    Opens a file for writing with ``autoopen``, but reverts it if there was an error in the process.

    Args:
        path(str): Path of file to open
        mode(str): Either t or b, for text or binary mode

    Example:
        Returns a contextmanager that provides a file object, so should be used in a "with" statement. E.g.::

            with pt.io.finalized_autoopen("file.gz", "t") as f:
                f.write("some text")
            # file.gz exists with contents "some text"

        If there is an error when writing, the file is reverted::

            with pt.io.finalized_autoopen("file.gz", "t") as f:
                f.write("some other text")
                raise Exception("an error")
            # file.gz remains unchanged (if existed, contents unchanged; if didn't exist, still doesn't)
    """
    return _finalized_open_base(path, mode, autoopen)

def ok_filename(fname) -> bool:
    """
    Checks to see if a filename is valid.
    """
    BAD_CHARS = ':"%/<>^|?' + os.sep
    for c in BAD_CHARS:
        if c in fname:
            return False
    return True

def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    """
    Eqiuvalent to touch command on linux.
    Implementation from https://stackoverflow.com/a/1160227
    """
    import os
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd, **kwargs)


def read_results(filename, format="trec", **kwargs):
    """
    Reads a file into a results dataframe.

    Parameters:
        filename (str): The filename of the file to be read. Compressed files are handled automatically. A URL is also supported for the "trec" format.
        format (str): The format of the results file: one of "trec", "letor". Default is "trec".
        **kwargs (dict): Other arguments for the internal method

    Returns:
        dataframe with usual qid, docno, score columns etc
    """
    if not format in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_RESULTS_FORMATS.keys())))
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

def write_results(res, filename, format="trec", append=False, **kwargs):
    """
    Write a results dataframe to a file.

    Parameters:
        res (DataFrame): A results dataframe, with usual columns of qid, docno etc
        filename (str): The filename of the file to be written. Compressed files are handled automatically.
        format (str): The format of the results file: one of "trec", "letor", "minimal"
        append (bool): Append to an existing file. Defaults to False.
        **kwargs (dict): Other arguments for the internal method

    Supported Formats:
        * "trec" -- output columns are $qid Q0 $docno $rank $score $runname, space separated
        * "letor" -- This follows the LETOR and MSLR datasets, in that output columns are $label qid:$qid [$fid:$value]+ # docno=$docno
        * "minimal": output columns are $qid $docno $rank, tab-separated. This is used for submissions to the MSMARCO leaderboard.

    """
    if not format in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_RESULTS_FORMATS.keys())))
    # convert generators to results
    res = coerce_dataframe(res)
    return SUPPORTED_RESULTS_FORMATS[format][1](res, filename, append=append, **kwargs)

def _write_results_trec(res, filename, run_name="pyterrier", append=False):
        res_copy = res.copy()[["qid", "docno", "rank", "score"]]
        res_copy.insert(1, "Q0", "Q0")
        res_copy.insert(5, "run_name", run_name)
        res_copy.to_csv(filename, sep=" ", mode='a' if append else 'w', header=False, index=False)

def _write_results_minimal(res, filename, run_name="pyterrier", append=False):
        res_copy = res.copy()[["qid", "docno", "rank"]]
        res_copy.to_csv(filename, sep="\t", mode='a' if append else 'w', header=False, index=False)

def _write_results_letor(res, filename, qrels=None, default_label=0, append=False):
    if qrels is not None:
        res = res.merge(qrels, on=['qid', 'docno'], how='left').fillna(default_label)
    mode='wa' if append else 'wt'
    with autoopen(filename, mode) as f:
        for row in res.itertuples():
            values = row.features
            label = row.label if qrels is not None else default_label
            feat_str = ' '.join( [ '%i:%f' % (i+1,values[i]) for i in range(len(values)) ] )
            f.write("%d qid:%s %s # docno=%s\n" % (label, row.qid, feat_str, row.docno))

def read_topics(filename, format="trec", **kwargs):
    """
    Reads a file containing topics.

    Parameters:
        filename(str): The filename of the topics file. A URL is supported for the "trec" and "singleline" formats.
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
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_RESULTS_FORMATS.keys())))
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
    Parse a file containing topics, one per line. This function uses Terrier, so supports reading direct from URLs.

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
        file_path(str): The path to the qrels file.  A URL is also supported.

    Returns:
        pandas.Dataframe with columns=['qid','docno', 'label']
        with column types string, string, and int
    """
    df = pd.read_csv(file_path,
                     sep=r'\s+',
                     names=["qid", "iter", "docno", "label"],
                     dtype={"qid": "str", "docno": "str"})
    df = df.drop(columns="iter")
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
