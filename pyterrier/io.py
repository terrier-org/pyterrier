import re
import os
import io
import tempfile
import shutil
import urllib
from types import GeneratorType
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from hashlib import sha256
from contextlib import contextmanager
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as eps
import pyterrier as pt
from typing import Optional, Tuple

DEFAULT_CHUNK_SIZE = 16_384 # 16kb

def coerce_dataframe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, GeneratorType):
        #its a generator, lets assume it generates dataframes
        rtr=[]
        for x in obj:
            assert isinstance(x, pd.DataFrame)
            rtr.append(x)
        return pd.concat(rtr)

def autoopen(filename, mode='rb'):
    """
    A drop-in for open() that applies automatic compression for .gz, .bz2 and .lz4 file extensions
    """

    if filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        import bz2
        return bz2.open(filename, mode)
    elif filename.endswith(".lz4"):
        import lz4.frame
        return lz4.frame.open(filename, mode)
    return open(filename, mode)

def find_files(dir):
    """
    Returns all the files present in a directory and its subdirectories

    Args:
        dir(str): The directory containing the files

    Returns:
        paths(list): A list of the paths to the files
    """
    lst = []
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir, followlinks=True):
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    return sorted(files)


@contextmanager
def _finalized_open_base(path: str, mode: str, open_fn) -> io.IOBase:
    assert mode in ('b', 't') # must supply either binary or text mode
    prefix = f'.{os.path.basename(path)}.tmp.'
    dirname = os.path.dirname(path)
    try:
        fd, path_tmp = tempfile.mkstemp(prefix=prefix, dir=dirname)
        os.close(fd) # mkstemp returns a low-level file descriptor... Close it and re-open the file the normal way
        with open_fn(path_tmp, f'w{mode}') as fout:
            yield fout
        os.chmod(path_tmp, 0o666) # default file umask
    except:
        try:
            os.remove(path_tmp)
        except:
            raise
        raise

    os.replace(path_tmp, path)


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


@contextmanager
def finalized_directory(path: str) -> str:
    prefix = f'.{os.path.basename(path)}.tmp.'
    dirname = os.path.dirname(path)
    try:
        path_tmp = tempfile.mkdtemp(prefix=prefix, dir=dirname)
        yield path_tmp
        os.chmod(path_tmp, 0o777) # default directory umask
    except:
        try:
            shutil.rmtree(path_tmp)
        except:
            raise
        raise

    os.replace(path_tmp, path)


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
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd, **kwargs)


def read_results(filename, format="trec", topics=None, dataset=None, **kwargs):
    """
    Reads a file into a results dataframe.

    Parameters:
        filename (str): The filename of the file to be read. Compressed files are handled automatically. A URL is also supported for the "trec" format.
        format (str): The format of the results file: one of "trec", "letor". Default is "trec".
        topics (None or pandas.DataFrame): If provided, will merge the topics to merge into the results. This is helpful for providing query text. Cannot be used in conjunction with dataset argument.
        dataset (None, str or pyterrier.datasets.Dataset): If provided, loads topics from the dataset (or dataset ID) and merges them into the results. This is helpful for providing query text. Cannot be used in conjunction with dataset topics.
        **kwargs (dict): Other arguments for the internal method

    Returns:
        dataframe with usual qid, docno, score columns etc

    Examples::

        # a dataframe of results can be used directly in a pt.Experiment
        pt.Experiment(
            [ pt.io.read_results("/path/to/baselines-results.res.gz") ],
            topics,
            qrels,
            ["map"]
        )

        # make a transformer from a results dataframe, include the query text
        first_pass = pt.Transformer.from_df( pt.io.read_results("/path/to/results.gz", topics=topics) )
        # make a max_passage retriever based on a previously saved results
        max_passage = (first_pass 
            >> pt.text.get_text(dataset)
            >> pt.text.sliding()
            >> pt.text.scorer()
            >> pt.text.max_passage()
        )

    """
    if not format in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_RESULTS_FORMATS.keys())))
    results = SUPPORTED_RESULTS_FORMATS[format][0](filename, **kwargs)
    if dataset is not None:
        assert topics is None, "Cannot provide both dataset and topics"
        if isinstance(dataset, str):
            dataset = pt.get_dataset(dataset)
        topics = dataset.get_topics()
    if topics is not None:
        results = pd.merge(results, topics, how='left', on='qid')
    return results

def _read_results_letor(filename, labels=False):

    def _parse_line(l):
            # $line =~ s/(#.*)$//;
            # my $comment = $1;
            # my @parts = split /\s+/, $line;
            # my $label = shift @parts;
            # my %hash = map {split /:/, $_} @parts;
            # return ($label, $comment, %hash);
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
    df = pd.read_csv(filename, sep=r'\s+', names=["qid", "iter", "docno", "rank", "score", "name"], dtype={'qid': str, 'docno': str, 'rank': int, 'score': float}) 
    df = df.drop(columns="iter")
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
        * "trecxml" -- a more modern XML formatted topics file. Delimited by topic tags, each having number tags. query, question and narrative tags are parsed by default. Control using tags kwarg.
        * "singleline" -- one query per line, preceeded by a space or colon. Tokenised by default, use tokenise=False kwargs to prevent tokenisation.
    """
    if format is None:
        format = "trec"
    if not format in SUPPORTED_TOPICS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_TOPICS_FORMATS.keys())))
    return SUPPORTED_TOPICS_FORMATS[format](filename, **kwargs)

@pt.java.required
def _read_topics_trec(file_path, doc_tag="TOP", id_tag="NUM", whitelist=["TITLE"], blacklist=["DESC","NARR"]):
    assert pt.terrier.check_version("5.3")
    trecquerysource = pt.java.autoclass('org.terrier.applications.batchquerying.TRECQuery')
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

@pt.java.required
def _read_topics_trecxml(filename, tags=["query", "question", "narrative"], tokenise=True):
    """
    Parse a file containing topics in TREC-like XML format

    Args:
        filename(str): The path to the topics file

    Returns:
        pandas.Dataframe with columns=['qid','query']
    """
    tags=set(tags)
    topics=[]
    tree = ET.parse(filename)
    root = tree.getroot()
    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    for child in root.iter('topic'):
        try:
            qid = child.attrib["number"]
        except KeyError:
            qid = child.find("number").text
        query = ""
        for tag in child:
            if tag.tag in tags:
                query_text = tag.text
                if tokenise:
                    query_text = " ".join(tokeniser.getTokens(query_text))
                query += " " + query_text
        topics.append((str(qid), query.strip()))
    return pd.DataFrame(topics, columns=["qid", "query"])

@pt.java.required
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
    assert pt.terrier.check_version("5.3")
    slqIter = pt.java.autoclass("org.terrier.applications.batchquerying.SingleLineTRECQuery")(filepath, tokenise)
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



def download(url: str, path: str, *, expected_sha256: str = None, verbose: bool = True) -> None:
    with finalized_open(path) as fout, \
         download_stream(url, expected_sha256=expected_sha256, verbose=verbose) as fin:
        while chunk := fin.read1():
            fout.write(chunk)


@contextmanager
def download_stream(url: str, *, expected_sha256: Optional[str] = None, verbose: bool = True) -> io.IOBase:
    with urllib.request.urlopen(url) as fin:
        if fin.status == 200:
            total = None
            if 'Content-Length' in fin.headers:
                total = int(fin.headers['Content-Length'])
            try:
                tqdm_sha256_in = TqdmSha256BufferedReader(fin, total, desc=url, enabled=verbose)
                yield tqdm_sha256_in
            finally:
                tqdm_sha256_in.close()
            if expected_sha256 is not None:
                if not expected_sha256.lower() == tqdm_sha256_in.sha256.hexdigest().lower():
                    raise ValueError(f'Corrupt download of {url}: expected_sha256={expected_sha256} '
                                     f'but found {tqdm_sha256_in.sha256.hexdigest()}')
        else:
            raise OSError(f'Unhandled status code: {fin.status}')


@contextmanager
def open_or_download_stream(
    path_or_url: str,
    *,
    expected_sha256: Optional[str] = None,
    verbose: bool = True
) -> io.IOBase:
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        with download_stream(path_or_url, expected_sha256=expected_sha256, verbose=verbose) as fin:
            yield fin
    elif os.path.isfile(path_or_url):
        total = os.path.getsize(path_or_url)
        with open(path_or_url, 'rb') as fin:
            tqdm_sha256_in = TqdmSha256BufferedReader(fin, total, desc=path_or_url, enabled=verbose)
            yield tqdm_sha256_in
    else:
        raise OSError(f'path or url {path_or_url!r} not found')


class TqdmSha256BufferedReader(io.BufferedIOBase):
    def __init__(self, reader: io.IOBase, total: int, desc: str, enabled: bool = True):
        self.reader = reader
        self.pbar = pt.tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024, disable=not enabled)
        self.seek = self.reader.seek
        self.tell = self.reader.tell
        self.seekable = self.reader.seekable
        self.readable = self.reader.readable
        self.writable = self.reader.writable
        self.flush = self.reader.flush
        self.isatty = self.reader.isatty
        self.sha256 = sha256()

    def read1(self, size: int = -1) -> bytes:
        if size == -1:
            size = DEFAULT_CHUNK_SIZE
        chunk = self.reader.read1(min(size, DEFAULT_CHUNK_SIZE))
        self.pbar.update(len(chunk))
        self.sha256.update(chunk)
        return chunk

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = DEFAULT_CHUNK_SIZE
        chunk = self.reader.read(min(size, DEFAULT_CHUNK_SIZE))
        self.pbar.update(len(chunk))
        self.sha256.update(chunk)
        return chunk

    def close(self) -> None:
        self.pbar.close()
        self.reader.close()


class TqdmSha256BufferedSequenceReader(io.BufferedIOBase):
    def __init__(self, readers):
        self.readers = readers
        self.x = next(self.readers)
        self.reader = self.x.__enter__()
        self.pbar = self.reader.pbar
        self.seek = self.reader.seek
        self.tell = self.reader.tell
        self.seekable = self.reader.seekable
        self.readable = self.reader.readable
        self.writable = self.reader.writable
        self.flush = self.reader.flush
        self.isatty = self.reader.isatty
        self.close = self.reader.close
        self.sha256 = sha256()

    def read1(self, size: int = -1) -> bytes:
        if size == -1:
            size = DEFAULT_CHUNK_SIZE
        chunk = self.reader.read1(min(size, DEFAULT_CHUNK_SIZE))
        if len(chunk) == 0:
            self.reader.close()
            try:
                self.x = next(self.readers)
            except StopIteration:
                return chunk
            self.reader = self.x.__enter__()
            self.pbar = self.reader.pbar
            self.seek = self.reader.seek
            self.tell = self.reader.tell
            self.seekable = self.reader.seekable
            self.readable = self.reader.readable
            self.writable = self.reader.writable
            self.flush = self.reader.flush
            self.isatty = self.reader.isatty
            self.close = self.reader.close
            chunk = self.reader.read1(min(size, DEFAULT_CHUNK_SIZE))
        self.sha256.update(chunk)
        return chunk

    def read(self, size: int = -1) -> bytes:
        chunk = b''
        if size == -1:
            size = DEFAULT_CHUNK_SIZE
        while len(chunk) < size and self.reader is not None:
            chunk += self.reader.read(size - len(chunk))
            if len(chunk) < size:
                self.reader.close()
                try:
                    self.x = next(self.readers)
                except StopIteration:
                    self.x = None
                    self.reader = None
                    return chunk
                self.reader = self.x.__enter__()
                self.pbar = self.reader.pbar
                self.seek = self.reader.seek
                self.tell = self.reader.tell
                self.seekable = self.reader.seekable
                self.readable = self.reader.readable
                self.writable = self.reader.writable
                self.flush = self.reader.flush
                self.isatty = self.reader.isatty
                self.close = self.reader.close
        self.sha256.update(chunk)
        return chunk


class Sha256BufferedWriter(io.BufferedIOBase):
    def __init__(self, writer: io.IOBase):
        self.writer = writer
        self.seek = self.writer.seek
        self.tell = self.writer.tell
        self.seekable = self.writer.seekable
        self.readable = self.writer.readable
        self.writable = self.writer.writable
        self.flush = self.writer.flush
        self.isatty = self.writer.isatty
        self.close = self.writer.close
        self.sha256 = sha256()

    def write(self, content: bytes) -> None:
        self.writer.write(content)
        self.sha256.update(content)


def path_is_under_base(path: str, base: str) -> bool:
    return os.path.realpath(os.path.abspath(os.path.join(base, path))).startswith(os.path.realpath(base))


def byte_count_to_human_readable(byte_count: float) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    while byte_count > 1024 and len(units) > 1:
        byte_count /= 1024
        units = units[1:]
    if units[0] == 'B':
        return f'{byte_count:.0f} {units[0]}'
    return f'{byte_count:.1f} {units[0]}'


def entry_points(group: str) -> Tuple[EntryPoint, ...]:
    try:
        return tuple(eps(group=group))
    except TypeError:
        return tuple(eps().get(group, tuple()))


def pyterrier_home() -> str:
    """
    Returns pyterrier's home directory. By default this is ~/.pyterrier, but it can also be set with the PYTERRIER_HOME
    env variable.
    """
    if "PYTERRIER_HOME" in os.environ:
        home = os.environ["PYTERRIER_HOME"]
    else:
        home = os.path.expanduser('~/.pyterrier')
    if not os.path.exists(home):
        os.makedirs(home)
    return home
