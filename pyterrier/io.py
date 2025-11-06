import re
import os
import io
import shutil
import tempfile
import urllib
import typing
from typing import Callable, Iterable, Optional, Generator, ContextManager, Union, Dict, Literal, Tuple, List
from types import GeneratorType
from contextlib import ExitStack, contextmanager
from abc import ABC, abstractmethod
from hashlib import sha256
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pyterrier as pt

if typing.TYPE_CHECKING:
    from collections.abc import Buffer # type: ignore[attr-defined]
else:
    Buffer = Union[bytes, bytearray, memoryview]


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

def autoopen(filename, mode='rb', **kwargs):
    """
    A drop-in for open() that applies automatic compression for .gz, .bz2 and .lz4 file extensions
    """

    if filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode, **kwargs)
    elif filename.endswith(".bz2"):
        import bz2
        return bz2.open(filename, mode, **kwargs)
    elif filename.endswith(".lz4"):
        import lz4.frame
        return lz4.frame.open(filename, mode, **kwargs)
    return open(filename, mode, **kwargs)

def find_files(dir):
    """
    Returns all the files present in a directory and its subdirectories

    :param dir: The directory containing the files

    :return: A list of the paths to the files
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir, followlinks=True):
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    return sorted(files)


@contextmanager
def _finalized_open_base(path: str, mode: str, open_fn: Callable) -> Generator[io.BufferedIOBase, None, None]:
    assert mode in ('b', 't') # must supply either binary or text mode
    dirname = os.path.dirname(path)
    prefix, suffix = os.path.splitext(os.path.basename(path))
    prefix = f'.{prefix}.'
    suffix = f'.tmp{suffix}' # last part of the suffix needed for autoopen
    try:
        fd, path_tmp = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dirname)
        os.close(fd) # mkstemp returns a low-level file descriptor... Close it and re-open the file the normal way
        with open_fn(path_tmp, f'w{mode}') as fout:
            yield fout
        os.chmod(path_tmp, 0o666) # default file umask
    except:
        try:
            os.remove(path_tmp)
        except Exception:
            pass # edge case: removing temp file failed. Ignore and just raise orig error
        raise

    os.replace(path_tmp, path)


def finalized_open(path: str, mode: str) -> ContextManager[io.BufferedIOBase]:
    """
    Opens a file for writing, but reverts it if there was an error in the process.

    :param path: Path of file to open
    :param mode: Either t or b, for text or binary mode

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


def finalized_autoopen(path: str, mode: str) -> ContextManager[io.BufferedIOBase]:
    """
    Opens a file for writing with ``autoopen``, but reverts it if there was an error in the process.

    :param path: Path of file to open
    :param mode: Either t or b, for text or binary mode

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

def ok_filename(fname : str) -> bool:
    """
    Checks to see if a filename is valid.
    """
    BAD_CHARS = ':"%/<>^|?' + os.sep
    for c in BAD_CHARS:
        if c in fname:
            return False
    return True

def touch(fname : str, mode=0o666, dir_fd=None, **kwargs):
    """
    Eqiuvalent to touch command on linux.
    Implementation from https://stackoverflow.com/a/1160227
    """
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd, **kwargs)


def read_results(filename : str, format="trec", topics : Optional[pd.DataFrame] = None, dataset : Optional[pt.datasets.Dataset] =None, **kwargs) -> pd.DataFrame:
    """
    Reads a file into a results dataframe.

    :param filename: The filename of the file to be read. Compressed files are handled automatically. A URL is also supported for the "trec" format.
    :param format: The format of the results file: one of "trec", "letor". Default is "trec".
    :param topics: If provided, will merge the topics to merge into the results. This is helpful for providing query text. Cannot be used in conjunction with dataset argument.
    :param dataset: If provided, loads topics from the dataset (or dataset ID) and merges them into the results. This is helpful for providing query text. Cannot be used in conjunction with dataset topics.
    :param kwargs: Other arguments for the internal method

    :return: dataframe with usual qid, docno, score columns etc

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
    if format not in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_RESULTS_FORMATS.keys())))
    if SUPPORTED_RESULTS_FORMATS[format][0] is None:
        raise ValueError("Format %s does not support reading" % format)
    results = SUPPORTED_RESULTS_FORMATS[format][0](filename, **kwargs) # type: ignore
    if dataset is not None:
        assert topics is None, "Cannot provide both dataset and topics"
        if isinstance(dataset, str):
            dataset = pt.get_dataset(dataset)
        topics = dataset.get_topics()
    if topics is not None:
        results = pd.merge(results, topics, how='left', on='qid')
    return results

def _read_results_letor(filename, labels=False):

    def _parse_line(line):
            # $line =~ s/(#.*)$//;
            # my $comment = $1;
            # my @parts = split /\s+/, $line;
            # my $label = shift @parts;
            # my %hash = map {split /:/, $_} @parts;
            # return ($label, $comment, %hash);
        line, comment = line.split("#")
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
    df = pd.read_csv(filename, sep=r'\s+', names=["qid", "iter", "docno", "rank", "score", "name"], dtype={'qid': str, 'docno': str, 'rank': int, 'score': float}) 
    df = df.drop(columns="iter")
    return df

def write_results(res : pd.DataFrame, filename : str, format : Literal['trec', 'letor', 'minimal'] ="trec", append=False, **kwargs):
    """
    Write a results dataframe to a file.

    :param res: A results dataframe, with usual columns of qid, docno etc
    :param filename: The filename of the file to be written. Compressed files are handled automatically.
    :param format: The format of the results file: one of "trec", "letor", "minimal"
    :param append: Append to an existing file. Defaults to False.
    :param kwargs: Other arguments for the internal method

    Supported Formats:
        * "trec" -- output columns are `$qid Q0 $docno $rank $score $runname, space separated`
        * "letor" -- This follows the LETOR and MSLR datasets, in that output columns are `$label qid:$qid [$fid:$value]+ # docno=$docno`
        * "minimal": output columns are `$qid $docno $rank`, tab-separated. This is used for submissions to the MSMARCO leaderboard.

    """
    if format not in SUPPORTED_RESULTS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_RESULTS_FORMATS.keys())))
    if SUPPORTED_RESULTS_FORMATS[format][1] is None:
        raise ValueError("Format %s does not support writing" % format)
    # convert generators to results
    res = coerce_dataframe(res)
    return SUPPORTED_RESULTS_FORMATS[format][1](res, filename, append=append, **kwargs)  # type: ignore

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

def read_topics(filename : str, format :Literal['trec', 'trecxml', 'singleline'] = "trec", **kwargs) -> pd.DataFrame:
    """
    Reads a file containing topics.

    :param filename: The filename of the topics file. A URL is supported for the "trec" and "singleline" formats.
    :param format: One of "trec", "trecxml" or "singleline". Default is "trec"

    :return: pandas.Dataframe with columns=['qid','query'], where both columns have type str.

    Supported Formats:
        * "trec" -- an SGML-formatted TREC topics file. Delimited by TOP tags, each having NUM and TITLE tags; DESC and NARR tags are skipped by default. Control using whitelist and blacklist kwargs
        * "trecxml" -- a more modern XML formatted topics file. Delimited by topic tags, each having number tags. query, question and narrative tags are parsed by default. Control using tags kwarg.
        * "singleline" -- one query per line, preceeded by a space or colon. Tokenised by default, use tokenise=False kwargs to prevent tokenisation.
    """
    if format is None:
        format = "trec"
    if format not in SUPPORTED_TOPICS_FORMATS:
        raise ValueError("Format %s not known, supported types are %s" % (format, str(SUPPORTED_TOPICS_FORMATS.keys())))
    return SUPPORTED_TOPICS_FORMATS[format](filename, **kwargs)

@pt.java.required
def _read_topics_trec(file_path, doc_tag="TOP", id_tag="NUM", whitelist=["TITLE"], blacklist=["DESC","NARR"], tokenise=True) -> pd.DataFrame:
    assert pt.terrier.check_version("5.3")
    assert tokenise, "Tokenisation is always performed for TREC SGML-formatted topics; set tokenise=True, or try using irds to obtain topics"
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

@typing.no_type_check
@pt.java.required
def _read_topics_trecxml(filename : str, tags : List[str] = ["query", "question", "narrative"], tokenise=False) -> pd.DataFrame:
    """
    Parse a file containing topics in TREC-like XML format

    Args:
        filename(str): The path to the topics file

    Returns:
        pandas.Dataframe with columns=['qid','query']
    """
    _tags=set(tags)
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
            if tag.tag in _tags:
                query_text = tag.text
                if tokenise:
                    query_text = " ".join(tokeniser.getTokens(query_text))
                query += " " + query_text
        topics.append((str(qid), query.strip()))
    return pd.DataFrame(topics, columns=["qid", "query"])

def _read_topics_singleline(filepath, contains_qid=True, tokenise=False) -> pd.DataFrame:
    """
    Parse a file containing topics, one per line. Supports reading direct from URLs.
    Uses Terrier's parser if tokenise=True.

    Args:
        file_path(str): The path to the topics file
        tokenise(bool): whether the query should be tokenised, using Terrier's standard Tokeniser.
            If you are using matchop formatted topics, this should be set to False.

    Returns:
        pandas.Dataframe with columns=['qid','query']
    """
    if tokenise:
        return _read_topics_singleline_tokenise(filepath, tokenise=True)
    qid_counter = 0
    
    def _open(filepath):
        if filepath.startswith('http://') or filepath.startswith('https://'):
            return pt.io.download_stream(filepath)
        return pt.io.autoopen(filepath, 'rt')
    
    with _open(filepath) as f:
        rows = []
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if len(line) == 0:
                continue
            if contains_qid:
                m = re.match(r'^([^:\s]+)[\s:]+(.*)$', line)
                if m:
                    qid = m.group(1)
                    query = m.group(2)
            else:
                qid_counter += 1
                qid = str(qid_counter)
            rows.append([qid, query])
        return pd.DataFrame(rows, columns=["qid", "query"])

@pt.java.required
def _read_topics_singleline_tokenise(filepath, tokenise=True) -> pd.DataFrame:
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

def read_qrels(file_path : str) -> pd.DataFrame:
    """
    Reads a file containing qrels (relevance assessments)

    :param file_path: The path to the qrels file.  A URL is also supported.

    :return: pandas.Dataframe with columns=['qid','docno', 'label']
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

SUPPORTED_RESULTS_FORMATS : Dict[str, Tuple[Optional[Callable[..., pd.DataFrame]], Optional[Callable[..., None]]]] = {
    "trec" : (_read_results_trec, _write_results_trec),
    "letor" : (_read_results_letor, _write_results_letor),
    "minimal" : (None, _write_results_minimal)
}


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


@contextmanager
def finalized_directory(path: str) -> Generator[str, None, None]:
    """Creates a directory, but reverts it if there was an error in the process."""
    dirname = os.path.dirname(path)
    prefix, suffix = os.path.splitext(os.path.basename(path))
    prefix = f'.{prefix}.'
    suffix = f'.tmp{suffix}' # keep final suffix/extension
    try:
        path_tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dirname)
        yield path_tmp
        os.chmod(path_tmp, 0o777) # default directory umask
    except:
        try:
            shutil.rmtree(path_tmp)
        except:
            raise
        raise

    os.replace(path_tmp, path)


def download(url: str, path: str, *, expected_sha256: Optional[str] = None, verbose: bool = True, headers={}) -> None:
    """Downloads a file from a URL to a local path."""
    with finalized_open(path, 'b') as fout, \
         download_stream(url, expected_sha256=expected_sha256, verbose=verbose, headers = headers) as fin:
        while chunk := fin.read1():
            fout.write(chunk)


@contextmanager
def download_stream(
    url: str,
    *,
    expected_sha256: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Generator[io.BufferedIOBase, None, None]:
    """Downloads a file from a URL to a stream."""
    with ExitStack() as stack:
        request = urllib.request.Request(url, headers=headers or {})
        fin = stack.enter_context(urllib.request.urlopen(request))
        if fin.status != 200:
            raise OSError(f'Unhandled status code: {fin.status}')

        if verbose:
            total = int(fin.headers.get('Content-Length', 0)) or None
            fin = stack.enter_context(TqdmReader(fin, total=total, desc=url))

        if expected_sha256 is not None:
            fin = stack.enter_context(HashReader(fin, expected=expected_sha256))

        yield fin


@contextmanager
def open_or_download_stream(
    path_or_url: str,
    *,
    expected_sha256: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Generator[io.BufferedIOBase, None, None]:
    """Opens a file or downloads a file from a URL to a stream."""
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        with download_stream(path_or_url, headers=headers, expected_sha256=expected_sha256, verbose=verbose) as fin:
            yield fin
    elif os.path.isfile(path_or_url):
        with ExitStack() as stack:
            fin = stack.enter_context(open(path_or_url, 'rb'))

            if verbose:
                total = os.path.getsize(path_or_url)
                fin = stack.enter_context(TqdmReader(fin, total=total, desc=path_or_url))

            if expected_sha256 is not None:
                fin = stack.enter_context(HashReader(fin, expected=expected_sha256))

            yield fin
    else:
        raise OSError(f'path or url {path_or_url!r} not found') # error can occur here if protocol entrypoints were not found - try pip install .


class _NosyReader(io.BufferedIOBase, ABC):
    def __init__(self, reader: io.BufferedIOBase):
        self.reader = reader
        self.seek = self.reader.seek # type: ignore[method-assign]
        self.tell = self.reader.tell # type: ignore[method-assign]
        self.seekable = self.reader.seekable # type: ignore[method-assign]
        self.readable = self.reader.readable # type: ignore[method-assign]
        self.writable = self.reader.writable # type: ignore[method-assign]
        self.flush = self.reader.flush # type: ignore[method-assign]
        self.isatty = self.reader.isatty # type: ignore[method-assign]

    @abstractmethod
    def on_data(self, data: bytes) -> None:
        pass

    def read1(self, size: Optional[int] = None) -> bytes:
        if size is None:
            size = io.DEFAULT_BUFFER_SIZE
        chunk = self.reader.read1(size)
        self.on_data(chunk)
        return chunk

    def read(self, size: Optional[int] = None) -> bytes:
        if size is None:
            size = io.DEFAULT_BUFFER_SIZE
        chunk = self.reader.read(size)
        self.on_data(chunk)
        return chunk

    def close(self) -> None:
        self.reader.close()


class _NosyWriter(io.BufferedIOBase, ABC):
    def __init__(self, writer: io.BufferedIOBase):
        self.writer = writer
        self.seek = self.writer.seek # type: ignore[method-assign]
        self.tell = self.writer.tell # type: ignore[method-assign]
        self.seekable = self.writer.seekable # type: ignore[method-assign]
        self.readable = self.writer.readable # type: ignore[method-assign]
        self.writable = self.writer.writable # type: ignore[method-assign]
        self.flush = self.writer.flush # type: ignore[method-assign]
        self.isatty = self.writer.isatty # type: ignore[method-assign]
        self.close = self.writer.close # type: ignore[method-assign]
        self.sha256 = sha256()

    @abstractmethod
    def on_data(self, data: Buffer) -> None:
        pass

    def write(self, data: Buffer) -> int:
        res = self.writer.write(data)
        self.on_data(data)
        return res

    def replace_writer(self, writer: io.BufferedIOBase) -> None:
        self.writer = writer # type: ignore[method-assign]
        self.seek = self.writer.seek # type: ignore[method-assign]
        self.tell = self.writer.tell # type: ignore[method-assign]
        self.seekable = self.writer.seekable # type: ignore[method-assign]
        self.readable = self.writer.readable # type: ignore[method-assign]
        self.writable = self.writer.writable # type: ignore[method-assign]
        self.flush = self.writer.flush # type: ignore[method-assign]
        self.isatty = self.writer.isatty # type: ignore[method-assign]
        self.close = self.writer.close # type: ignore[method-assign]


class HashReader(_NosyReader):
    """A reader that computes the sha256 hash of the data read."""
    def __init__(self, reader: io.BufferedIOBase, *, hashfn: Callable = sha256, expected: Optional[str] = None):
        """Create a HashReader."""
        super().__init__(reader)
        self.hash = hashfn()
        self.expected = expected

    def on_data(self, data: bytes) -> None:
        """Called when data is read."""
        self.hash.update(data)

    def hexdigest(self) -> str:
        """Return the hexdigest of the hash."""
        return self.hash.hexdigest()

    def close(self) -> None:
        """Close the reader and check the hash."""
        self.reader.close()
        if self.expected is not None:
            if self.expected.lower() != self.hexdigest():
                raise ValueError(f'Expected sha256 {self.expected!r} but found {self.hexdigest()!r}')


class HashWriter(_NosyWriter):
    """A writer that computes the sha256 hash of the data written."""
    def __init__(self, writer: io.BufferedIOBase, *, hashfn: Callable = sha256):
        """Create a HashWriter."""
        super().__init__(writer)
        self.hash = hashfn()

    def on_data(self, data: Buffer) -> None:
        """Called when data is written."""
        self.hash.update(data)

    def hexdigest(self) -> str:
        """Return the hexdigest of the hash."""
        return self.hash.hexdigest()


class TqdmReader(_NosyReader):
    """A reader that displays a progress bar."""
    def __init__(self, reader: io.BufferedIOBase, *, total: Optional[int] = None, desc: Optional[str] = None, disable: bool = False):
        """Create a TqdmReader."""
        super().__init__(reader)
        self.pbar = pt.tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024, disable=disable) # type: ignore[misc]

    def on_data(self, data: bytes) -> None:
        """Called when data is read."""
        self.pbar.update(len(data))

    def close(self) -> None:
        """Close the reader and the progress bar."""
        super().close()
        self.reader.close()
        self.pbar.close()


class CallbackReader(_NosyReader):
    """A reader that calls a callback with the data read."""
    def __init__(self, reader: io.BufferedIOBase, callback: Callable):
        """Create a CallbackReader."""
        super().__init__(reader)
        self.callback = callback

    def on_data(self, data: bytes) -> None:
        """Called when data is read."""
        self.callback(data)


class MultiReader(io.BufferedIOBase):
    """A reader that reads from multiple readers in sequence."""
    def __init__(self, readers: Iterable[io.BufferedIOBase]):
        """Create a MultiReader."""
        self.readers = iter(readers)
        self._reader: Optional[io.BufferedIOBase] = next(self.readers)
        self.reader: Optional[io.BufferedIOBase] = self._reader.__enter__()
        self.seek = self.reader.seek # type: ignore[method-assign]
        self.tell = self.reader.tell # type: ignore[method-assign]
        self.seekable = self.reader.seekable # type: ignore[method-assign]
        self.readable = self.reader.readable # type: ignore[method-assign]
        self.writable = self.reader.writable # type: ignore[method-assign]
        self.flush = self.reader.flush # type: ignore[method-assign]
        self.isatty = self.reader.isatty # type: ignore[method-assign]
        self.close = self.reader.close # type: ignore[method-assign]

    def read1(self, size: Optional[int] = None) -> bytes:
        """Read a single chunk of data."""
        if self.reader is None:
            raise RuntimeError('reader is closed')
        if size is None:
            size = io.DEFAULT_BUFFER_SIZE
        chunk = self.reader.read1(min(size, io.DEFAULT_BUFFER_SIZE))
        if len(chunk) == 0:
            self.reader.close()
            try:
                self._reader = next(self.readers)
            except StopIteration:
                self._reader = None
                self.reader = None
                return chunk
            self.reader = self._reader.__enter__()
            if hasattr(self.reader, 'pbar'):
                self.pbar = self.reader.pbar
            self.seek = self.reader.seek # type: ignore[method-assign]
            self.tell = self.reader.tell # type: ignore[method-assign]
            self.seekable = self.reader.seekable # type: ignore[method-assign]
            self.readable = self.reader.readable # type: ignore[method-assign]
            self.writable = self.reader.writable # type: ignore[method-assign]
            self.flush = self.reader.flush # type: ignore[method-assign]
            self.isatty = self.reader.isatty # type: ignore[method-assign]
            self.close = self.reader.close # type: ignore[method-assign]
            chunk = self.reader.read1(min(size, io.DEFAULT_BUFFER_SIZE))
        return chunk

    def read(self, size: Optional[int] = None) -> bytes:
        """Read data."""
        chunk = b''
        if size is None:
            size = io.DEFAULT_BUFFER_SIZE
        while len(chunk) < size and self.reader is not None:
            chunk += self.reader.read(size - len(chunk))
            if len(chunk) < size:
                self.reader.close()
                try:
                    self._reader = next(self.readers)
                except StopIteration:
                    self._reader = None
                    self.reader = None
                    return chunk
                self.reader = self._reader.__enter__()
                if hasattr(self.reader, 'pbar'):
                    self.pbar = self.reader.pbar
                self.seek = self.reader.seek # type: ignore[method-assign]
                self.tell = self.reader.tell # type: ignore[method-assign]
                self.seekable = self.reader.seekable # type: ignore[method-assign]
                self.readable = self.reader.readable # type: ignore[method-assign]
                self.writable = self.reader.writable # type: ignore[method-assign]
                self.flush = self.reader.flush # type: ignore[method-assign]
                self.isatty = self.reader.isatty # type: ignore[method-assign]
                self.close = self.reader.close # type: ignore[method-assign]
        return chunk


def path_is_under_base(path: str, base: str) -> bool:
    """Returns True if the path is under the base directory."""
    return os.path.realpath(os.path.abspath(os.path.join(base, path))).startswith(os.path.realpath(base))
