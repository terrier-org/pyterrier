import os
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, List, Optional, Union, Protocol, runtime_checkable, Dict, Any
from collections import defaultdict
import zipfile
import tarfile
import types
import requests
import urllib
import functools
from warnings import warn
import pyterrier as pt
from pyterrier.transformer import is_lambda


# DATASET_MAP provides legacy support for registering datasets in extensions. It will be removed in a future version
# in favor of using a DatasetProvider registered with an entry point.
DATASET_MAP: Dict[str, Any] = {}


class Dataset:
    """Represents a dataset (test collection) for indexing and/or retrieval.

    A common use case is for an Experiment::

        dataset = pt.get_dataset("trec-robust-2004")
        pt.Experiment([br1, br2], dataset.get_topics(), dataset.get_qrels(), eval_metrics=["map", "recip_rank"])
    """
    def get_corpus(self) -> List[str]:
        """Returns the location of the files to allow indexing the corpus, i.e. it returns a list of filenames.

        .. deprecated::
            Use :meth:`get_corpus_iter` instead.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        raise NotImplementedError(f'{self!r} does not support get_corpus')

    def get_corpus_iter(self, *, verbose: bool = True) -> pt.model.IterDict:
        """Returns an iter of dicts for this collection.

        Args:
            verbose (bool, optional): Whether to print progress bar. Defaults to True.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        raise NotImplementedError(f'{self!r} does not support get_corpus_iter')

    def get_corpus_lang(self) -> Optional[str]:
        """Returns the ISO 639-1 language code for the corpus, or None for multiple/other/unknown."""
        return None

    def get_index(self, variant=None, **kwargs): # TODO: -> pt.Artifact
        """Returns the IndexRef of the index to allow retrieval. Only a few datasets provide indices ready made.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        raise NotImplementedError(f'{self!r} does not support get_index')

    def get_topics(self, variant: Optional[str] = None) -> pd.DataFrame:
        """Returns the topics (if available), as a dataframe, ready for retrieval. 

        Args:
            variant (str, optional): The variant of the topics, such as a sub-dataset of the field. Defaults to None.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        raise NotImplementedError(f'{self!r} does not support get_topics')

    def get_topics_lang(self) -> Optional[str]:
        """Returns the ISO 639-1 language code for the topics, or None for multiple/other/unknown."""
        return None

    def get_qrels(self, variant: Optional[str] = None) -> pd.DataFrame:
        """Returns the qrels, as a dataframe, for evaluation (if available).

        Args:
            variant (str, optional): The variant of the topics, such as a sub-dataset of the field. Defaults to None.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        raise NotImplementedError(f'{self!r} does not support get_qrels')

    def get_topicsqrels(self, variant: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns both the topics and qrels in a tuple. This is useful for pt.Experiment().

        Args:
            variant (str, optional): The variant of the topics, such as a sub-dataset of the field. Defaults to None.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        return (
            self.get_topics(variant=variant),
            self.get_qrels(variant=variant)
        )

    def info_url(self) -> Optional[str]:
        """Returns a url that provides more information about this dataset, or None if URL is unknown."""
        return None

    def get_results(self, variant: Optional[str] = None) -> pd.DataFrame:
        """Returns a standard result set provided by the dataset (if available). This is useful for re-ranking experiments.

        Args:
            variant (str, optional): The variant of the topics, such as a sub-dataset of the field. Defaults to None.

        Raises:
            NotImplementedError: If the dataset does not support this method.
        """
        raise NotImplementedError(f'{self!r} does not support get_results')

    def _describe_component(self, component: str) -> Optional[Union[bool, List[str]]]:
        return None


class DatasetProvider(ABC):
    """ Represents a source of datasets. For instance: the default suite of datasets provided directly by PyTerrier,
    or datasets from the ``ir-datasets`` package.

    Dataset providers should be registred as ``pyterrier.dataset_provider`` entry points.
    """

    @abstractmethod
    def get_dataset(self, name: str) -> Dataset:
        """ Returns a dataset for the provided identifier (``name``).

        Args:
            name (str): The identifier of the dataset.

        Returns:
            Dataset: The dataset.

        Raises:
            KeyError: If the dataset is not found.
        """

    def list_dataset_names(self) -> Iterable[str]:
        """ Returns the names of the datasets that this provider gives access to.

        The primary purpose of this method is to populate the `list of datasets <https://pyterrier.readthedocs.io/en/latest/datasets.html>`__
        in the documentation.

        .. note::

            This method is optional and does not need to return all available datasets for this provider. For instance,
            sometimes the number of datasets is too large to reasonably display, or it may be too expensive to fetch
            all datasets.

        Returns:
            Iterable[str]: An iterable of tuples, where the first element is the name of the dataset and the second
                element is the dataset itself.
        """
        return []


class RemoteDataset(Dataset):

    def __init__(self, name, locations):
        self.locations = locations
        self.name = name
        self.user = None
        self.password = None

    def _configure(self, user: Optional[str] = None, password: Optional[str] = None):
        self.corpus_home = os.path.join(pt.io.pyterrier_home(), "corpora", self.name)
        if user is not None:
            self.user = user
        if password is not None:
            self.password = password

    @staticmethod
    def download(URLs : Union[str,List[str]], filename : str, **kwargs):
        basename = os.path.basename(filename)

        if isinstance(URLs, str):
            URLs = [URLs]
        
        finalattempt=len(URLs)-1
        error = None
        for i, url in enumerate(URLs):            
            try:
                r = requests.get(url, allow_redirects=True, stream=True, **kwargs)
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with pt.io.finalized_open(filename, 'b') as file, pt.tqdm( # type: ignore
                        desc=basename,
                        total=total,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in r.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)
                    break
            except Exception as e:
                if error is not None:
                    e.__cause__ = error # chain errors to show all if fails
                error = e
                if i == finalattempt:
                    raise error
                else:
                    warn(
                        "Problem fetching %s, resorting to next mirror" % url)
            

    def _check_variant(self, component, variant=None):
        name = self.name
        if component not in self.locations:
            raise ValueError("No %s in dataset %s" % (component, name))
        if variant is None:
            if not isinstance(self.locations[component], list):
                raise ValueError("For %s in dataset %s, you must specify a variant. Available are: %s" % (component, name, str(list(self.locations[component].keys()))))
        else:
            if isinstance(self.locations[component], list):
                raise ValueError("For %s in dataset %s, there are no variants, but you specified %s" % (component, name, variant))
            if variant not in self.locations[component]:
                raise ValueError("For %s in dataset %s, there is no variant %s. Available are: %s" % (component, name, variant, str(list(self.locations[component].keys()))))

    def _get_one_file(self, component, variant=None):
        filetype=None
        name=self.name
        self._check_variant(component, variant)
        location = self.locations[component][0] if variant is None else self.locations[component][variant]

        if is_lambda(location) or isinstance(location, types.FunctionType):
            argcount = location.__code__.co_argcount
            if argcount == 0:
                return location()
            elif argcount == 3:
                return location(self, component, variant)
            else:
                raise TypeError("Expected function with 0 or 3 arguments for  %s %s %s" % (component, name, variant))

        local = location[0]
        URL = location[1]
        if len(location) > 2:
            filetype = location[2]

        if not os.path.exists(self.corpus_home):
            os.makedirs(self.corpus_home)

        local = os.path.join(self.corpus_home, local)
        actualURL = URL if isinstance(URL, str) else URL[0]
        if "#" in actualURL and not os.path.exists(local):
            tarname, intarfile = actualURL.split("#")
            assert "/" not in intarfile
            assert ".tar" in tarname or ".tgz" in tarname or ".zip" in tarname
            localtarfile, _ = self._get_one_file("tars", tarname)
            tarobj = tarfile.open(localtarfile, "r")
            tarobj.extract(intarfile, path=self.corpus_home)
            extractor = zipfile.ZipFile if ".zip" in tarname else tarfile.open
            with extractor(localtarfile, "r") as tarobj:
                tarobj.extract(intarfile, path=self.corpus_home)
            os.rename(os.path.join(self.corpus_home, intarfile), local)
            return (local, filetype)

        if not os.path.exists(local):
            try:
                print("Downloading %s %s to %s" % (self.name, component, local))
                kwargs = {}
                if self.user is not None:
                    kwargs["auth"]=(self.user, self.password)
                RemoteDataset.download(URL, local, **kwargs)
            except urllib.error.HTTPError as he:
                raise ValueError("Could not fetch " + URL) from he
        return (local, filetype)

    def _get_all_files(self, component, variant=None, **kwargs):
        if variant is None:
            localDir = os.path.join(self.corpus_home, component)
        else:
            localDir = os.path.join(self.corpus_home, component, variant)

        kwargs = {}
        if self.user is not None:
            kwargs["auth"]=(self.user, self.password)

        direxists = os.path.exists(localDir)
        
        location = self.locations[component]
        if is_lambda(location) or isinstance(location, types.FunctionType):
            # functions are expensive to call, normally another HTTP is needed.
            # just assume we have everthing we need if we have the local directory already
            # and it contains a .complete file.
            if direxists and os.path.exists(os.path.join(localDir, ".complete")):
                return localDir

            # call the function, and get the file list
            file_list = location(self, component, variant, **kwargs)
        else:
            file_list = self.locations[component] if variant is None else self.locations[component][variant]

        if not direxists:
            os.makedirs(localDir)
            print("Downloading %s %s to %s" % (self.name, component, localDir))
        

        # check for how much space is required and available space
        def _totalsize(file_list):
            total = -1
            for f in file_list:
                if len(f) > 2:
                    total += f[2]
            if total != -1:
                total += 1
            return total

        totalsize = _totalsize(file_list)
        if totalsize > 0:
            import shutil
            total, used, free = shutil.disk_usage(localDir)
            if free < totalsize:
                raise ValueError("Insufficient freedisk space at %s to download index" % localDir)
            if totalsize > 2 * 2**30:
                warn(
                    "Downloading index of > 2GB.")

        # all tarfiles that we will need to process        
        tarfiles = defaultdict(list)
        for fileentry in file_list:
            local = fileentry[0]
            URL = fileentry[1]
            assert "/" not in local, "cant handle / in %s, local name is %s" % (local)
            expectedlength = -1
            if len(fileentry) == 3:
                expectedlength = fileentry[2]
            local = os.path.join(localDir, local)
            
            # if file exists and we know length, check if dowload is complete
            fileexists = os.path.exists(local)
            if fileexists and expectedlength >= 0:
                length = os.stat(local).st_size
                if expectedlength != length:
                    warn(
                        "Removing partial download of %s (expected %d bytes, found %d)" % (local, expectedlength, length ))
                    os.remove(local)
                    fileexists = False

            if not fileexists:
                if "#" in URL:
                    tarname, intarfile = URL.split("#")
                    assert ".tar" in tarname or ".tgz" in tarname or ".zip" in tarname, "I dont know how to decompress file %s" % tarname
                    localtarfile, _ = self._get_one_file("tars", tarname)
                    # append intarfile to the list of files to be extracted from localtarfile
                    tarfiles[localtarfile].append((intarfile, local))
                else:
                    try:
                        RemoteDataset.download(URL, local, **kwargs)
                    except urllib.error.HTTPError as he:
                        raise ValueError("Could not fetch " + URL) from he

                    # verify file if exists
                    if expectedlength >= 0:
                        length = os.stat(local).st_size
                        if expectedlength != length:
                            raise ValueError("Failed download of %s to %s (expected %d bytes, found %d)" % (URL, local, expectedlength, length ))

        # now extract all required files from each tar file
        for localtarfile in tarfiles:
            extractor = zipfile.ZipFile if ".zip" in tarname else tarfile.open
            with extractor(localtarfile, "r") as tarobj:
                # 5 is abrtary threshold - if we have lots of files to extract, give a progress bar. alternative would be delay=5?
                iter = pt.tqdm(tarfiles[localtarfile], unit="file", desc="Extracting from " + localtarfile) if len(tarfiles[localtarfile]) > 5 else tarfiles[localtarfile]
                for (intarfile, local) in iter:
                    tarobj.extract(intarfile, path=self.corpus_home)
                    local = os.path.join(self.corpus_home, local)
                    os.rename(os.path.join(self.corpus_home, intarfile), local)
                    #TODO, files /could/ be recompressed here to save space, if not already compressed

        # finally, touch a file signifying that download has been completed
        pt.io.touch(os.path.join(localDir, ".complete"))
        return localDir

    def _describe_component(self, component):
        if component not in self.locations:
            return None
        if isinstance(self.locations[component], list):
            return True
        if isinstance(self.locations[component], dict):
            return list(self.locations[component].keys())
        return True

    def get_corpus(self, **kwargs):
        return list(filter(lambda f : not f.endswith(".complete"), pt.io.find_files(self._get_all_files("corpus", **kwargs))))

    def get_corpus_iter(self, **kwargs):
        if "corpus_iter" not in self.locations:
            raise NotImplementedError(f"{self!r} does not support get_corpus_iter")
        return self.locations["corpus_iter"](self, **kwargs)
        
    def get_corpus_lang(self):
        if 'corpus' in self.locations:
            return 'en' # all are english
        return None

    def get_qrels(self, variant=None):
        filename, type = self._get_one_file("qrels", variant)
        if type == "direct":
            return filename 
        return pt.io.read_qrels(filename)

    def get_topics(self, variant=None, **kwargs):
        file, filetype = self._get_one_file("topics", variant)
        if filetype is None or filetype in pt.io.SUPPORTED_TOPICS_FORMATS:
            return pt.io.read_topics(file, format=filetype, **kwargs)
        elif filetype == "direct":
            return file
        raise ValueError("Unknown filetype %s for %s topics %s"  % (filetype, self.name, variant))
    
    def get_topics_lang(self):
        if 'topics' in self.locations:
            return 'en' # all are english
        return None

    def get_index(self, variant=None, **kwargs):
        if self.name == "50pct" and variant is None:
            variant="ex1"
        thedir = self._get_all_files("index", variant=variant, **kwargs)
        return thedir

    def __repr__(self):
        return f"RemoteDataset({self.name!r}, { {k: ... for k in self.locations}!r})"

    def info_url(self):
        return self.locations.get('info_url')


@runtime_checkable
class HasConfigure(Protocol):
    def _configure(self, **kwargs):
        pass

# This is a temporary decorator to give the option of whether or not to tokenize the queries.
def add_tokenize_query_arg(fn):
    @functools.wraps(fn)
    def _wrapper(variant: Optional[str] = None, tokenise_query: bool = True):
        topics = fn(variant)
        if topics is not None and tokenise_query and 'query' in topics:
            tokeniser = _pt_tokeniser()
            topics['query'] = topics['query'].apply(tokeniser)
        return topics
    _wrapper._has_add_tokenize_query_arg_applied = True
    return _wrapper


@pt.java.required
def _pt_tokeniser():
    tokeniser = pt.terrier.J.Tokenizer.getTokeniser()
    def pt_tokenise(text):
        return ' '.join(tokeniser.getTokens(text))
    return pt_tokenise


_loaded_providers = {}


def _load_all_providers():
    for ep in pt.utils.entry_points('pyterrier.dataset_provider'):
        _loaded_providers[ep.name] = ep.load()()


def _load_provider(name: str) -> Optional[DatasetProvider]:
    if name not in _loaded_providers:
        for ep in pt.utils.entry_points('pyterrier.dataset_provider'):
            if ep.name == name:
                _loaded_providers[name] = ep.load()()
                break
    return _loaded_providers.get(name)


def get_dataset(name: str, **configure_kwargs) -> Dataset:
    """Get a dataset by name.

    The name may provide a provider as a prefix in the format of ``provider:dataset``. In this case, the dataset is
    loaded from that provider (e.g., ``irds``).

    Args:
        name (str): The name of the dataset.
        **configure_kwargs: Additional configuration arguments to pass to the dataset's ``_configure()`` method (if one is available)

    Returns:
        Dataset: The dataset.

    Raises:
        KeyError: If the dataset is not found.
    """
    if name in DATASET_MAP:
        result = DATASET_MAP[name]
    else:
        orig_name = name
        provider_name = 'builtin' # default provider when no provider: prefix is given
        if ':' in name:
            provider_name, name = name.split(':', 1)

        provider = _load_provider(provider_name)

        if provider is None:
            if provider_name == 'builtin':
                # builtin should always be found; an absence of builtin means the pyterrier is not installed properly
                raise KeyError(f'Dataset {orig_name!r} not found due to missing provider {provider_name!r}. '
                               +'You need to pip install python-terrier to ensure dataset providers can be found.')
            # We could provide suggestions here on packages to install for missing entry points in due course
            raise KeyError(f'Dataset {orig_name!r} not found due to missing provider {provider_name!r}. Are you missing a package?')

        try:
            result = provider.get_dataset(name)
        except KeyError as ex:
            raise KeyError(f'Dataset {orig_name!r} not found in provider {provider_name!r}.') from ex

    # Configure
    if isinstance(result, HasConfigure):
        result._configure(**configure_kwargs)
    elif configure_kwargs:
        raise TypeError(f'Unsupported keyword arguments passed to get_dataset: {get_dataset}')

    # Temporary handling of topic tokenization
    if not hasattr(result.get_topics, '_has_add_tokenize_query_arg_applied'):
        result.get_topics = add_tokenize_query_arg(result.get_topics)
    return result


def datasets() -> List[str]:
    """Lists the names of available datasets."""
    _load_all_providers()
    result = list(DATASET_MAP.keys())
    for provider_name in sorted(_loaded_providers.keys()):
        provider = _loaded_providers[provider_name]
        for name in provider.list_dataset_names():
            if provider_name != '':
                name = f'{provider_name}:{name}'
            result.append(name)
    return result


def find_datasets(query, en_only=True):
    """
    A grep-like method to help identify datasets. Filters the output of list_datasets() based on the name containing the query
    """
    datasets = list_datasets(en_only=en_only)
    return datasets[datasets['dataset'].str.contains(query)]


def list_datasets(en_only=True):
    """
        Returns a dataframe of all datasets, listing which topics, qrels, corpus files or indices are available.
        By default, filters to only datasets with both a corpus and topics in English.
    """
    with pt.utils.temp_env("IR_DATASETS_SKIP_DEPRECATED_WARNING", "true"): # hide IRDS deprecated warning when loading all datasets
        rows = []
        for k in datasets():
            dataset = get_dataset(k)
            rows.append([
                k, 
                dataset._describe_component("topics"), 
                dataset.get_topics_lang(), 
                dataset._describe_component("qrels"), 
                dataset._describe_component("corpus"), 
                dataset.get_corpus_lang(), 
                dataset._describe_component("index"), 
                dataset.info_url()
            ])
    
    result = pd.DataFrame(rows, columns=["dataset", "topics", "topics_lang", "qrels", "corpus", "corpus_lang", "index", "info_url"])
    if en_only:
        topics_filter = (result['topics'].isnull()) | (result['topics_lang'] == 'en')
        corpus_filter = (result['corpus'].isnull()) | (result['corpus_lang'] == 'en')
        result = result[topics_filter & corpus_filter]
    return result


def transformer_from_dataset(
    dataset : Union[str, Dataset],
    clz,
    variant: Optional[str] = None,
    version: str = 'latest',        
    **kwargs
) -> pt.Transformer:
    """Returns a Transformer instance of type ``clz`` for the provided index of variant ``variant``."""
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)
    if version != "latest":
        raise ValueError("index versioning not yet supported")
    indexref = dataset.get_index(variant)

    classname = clz.__name__
    classnames = [classname]
    if classname == 'Retriever':
        # we need to look for BatchRetrieve.args.json for legacy support
        classnames.append('BatchRetrieve')
    for c in classnames:
        # now look for, e.g., BatchRetrieve.args.json file, which will define the args for Retriever, e.g. stemming
        indexdir = indexref #os.path.dirname(indexref.toString())
        argsfile = os.path.join(indexdir, classname + ".args.json")
        if os.path.exists(argsfile):
            with pt.io.autoopen(argsfile, "rt") as f:
                args = json.load(f)
                # anything specified in kwargs of this methods overrides the .args.json file
                args.update(kwargs)
                kwargs = args
    return clz(indexref, **kwargs)
