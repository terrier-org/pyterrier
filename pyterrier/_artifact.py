"""PyTerrier artifact module."""
import contextlib
import io
import json
import os
import sys
import tarfile
import tempfile
import requests
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union
from urllib.parse import ParseResult, urlparse

from lz4.frame import LZ4FrameFile

import pyterrier as pt


class Artifact:
    """Base class for PyTerrier artifacts.

    An artifact is a component stored on disk, such as an index.

    Artifacts usually act as factories for transformers that use them. For example, an index artifact
    may provide a `.retriever()` method that returns a transformer that searches the index.
    """
    def __init__(self, path: Union[Path, str]):
        """Initialize the artifact at the provided URL."""
        self.path: Path = Path(path)

    @classmethod
    def load(cls, path: str) -> 'Artifact':
        """Load the artifact from the specified path.

        If invoked on the base class, this method will attempt to find a supporting Artifact
        implementation that can load the artifact at the specified path. If invoked on a subclass,
        it will attempt to load the artifact using the specific implementation.

        Args:
            path: The path of the artifact on disk.

        Returns:
            The loaded artifact.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If no implementation is found that supports the artifact at the specified path.
        """
        if cls is Artifact:
            return _load(path)
        else:
            # called SomeArtifact.load(path), so load this specific artifact
            # TODO: add error message if not loaded
            return cls(path)

    @classmethod
    def from_url(cls, url: str, *, expected_sha256: Optional[str] = None) -> 'Artifact':
        """Load the artifact from the specified URL.

        The artifact at the specified URL will be downloaded and stored in PyTerrier's artifact cache.

        Args:
            url: The URL or file path of the artifact.
            expected_sha256: The expected SHA-256 hash of the artifact. If provided, the downloaded artifact will be
            verified against this hash and an error will be raised if the hash does not match.

        Returns:
            The loaded artifact.

        Raises:
            ValueError: If no implementation is found that supports the artifact at the specified path.
        """
        # Support mapping "protocols" of the URL other than http[s]
        # e.g., "hf:abc/xyz@branch" -> "https://huggingface.co/datasets/abc/xyz/resolve/branch/artifact.tar"
        parsed_url = urlparse(url)
        protocol_entry_points = {ep.name: ep for ep in pt.utils.entry_points('pyterrier.artifact.url_protocol_resolver')}
        while parsed_url.scheme in protocol_entry_points:
            resolver = protocol_entry_points[parsed_url.scheme].load()
            tmp_url = resolver(parsed_url)
            del protocol_entry_points[parsed_url.scheme] # avoid the possiblity of an infinite loop here
            if tmp_url is not None:
                url = tmp_url
                parsed_url = urlparse(tmp_url)

        if parsed_url.scheme == '' and os.path.exists(url) and os.path.isdir(url):
            return cls.load(url) # already resolved to a directory, load this

        # buid local path
        base_path = os.path.join(pt.io.pyterrier_home(), 'artifacts')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = os.path.join(base_path, sha256(url.encode()).hexdigest())

        if not os.path.exists(path):
            # try to load the package metadata file (if it exists)
            download_info = {}
            try:
                with pt.io.open_or_download_stream(f'{url}.json', verbose=False) as fin:
                    download_info = json.load(fin)
                if 'expected_sha256' in download_info:
                    new_expected_sha256 = download_info['expected_sha256'].lower()
                    if expected_sha256 is not None:
                        assert expected_sha256.lower() == new_expected_sha256
                    expected_sha256 = new_expected_sha256
                if 'total_size' in download_info:
                    # TODO: make sure there's available space on the disk for this
                    pass
            except OSError:
                pass


            with contextlib.ExitStack() as stack:
                dout = stack.enter_context(pt.io.finalized_directory(path))

                metadata_out = stack.enter_context(pt.io.finalized_open(f'{path}.json', 't'))
                if parsed_url.scheme == '':
                    json.dump({'path': url}, metadata_out)
                else:
                    json.dump({'url': url}, metadata_out)
                metadata_out.write('\n')

                if 'segments' in download_info:
                    # the file is segmented -- use a sequence reader to stitch them back together
                    fin = stack.enter_context(pt.io.MultiReader(
                        pt.io.open_or_download_stream(f'{url}.{i}')
                        for i in range(len(download_info['segments']))
                    ))
                    if expected_sha256 is not None:
                        fin = stack.enter_context(pt.io.HashReader(fin, expected=expected_sha256))
                else:
                    fin = stack.enter_context(pt.io.open_or_download_stream(url, expected_sha256=expected_sha256))

                cin = fin
                if parsed_url.path.endswith('.lz4'):
                    cin = stack.enter_context(LZ4FrameFile(fin))
                # TODO: support other compressions

                tar_in = stack.enter_context(tarfile.open(fileobj=cin, mode='r|'))

                metadata_out.flush()
                for member in tar_in:
                    if (member.isfile() or member.isdir()) and pt.io.path_is_under_base(member.path, dout):
                        print(f'extracting {member.path} [{pt.utils.byte_count_to_human_readable(member.size)}]')
                        tar_in.extract(member, dout, set_attrs=False)

        return cls.load(path)

    @classmethod
    def from_hf(cls, repo: str, branch: str = None, *, expected_sha256: Optional[str] = None) -> 'Artifact':
        """Load an artifact from Hugging Face Hub.

        Args:
            repo: The Hugging Face repository name.
            branch: The branch or tag of the repository to load. (Default: main). A branch can also be provided directly
            in the repository name using "owner/repo@branch".
            expected_sha256: The expected SHA-256 hash of the artifact. If provided, the downloaded artifact will be
            verified against this hash and an error will be raised if the hash does not match.
        """
        if branch is not None:
            if '@' in repo:
                raise ValueError('Provided branch in both repository name (via @) and as argument to from_hf')
            repo = f'{repo}@{branch}'
        return cls.from_url(f'hf:{repo}', expected_sha256=expected_sha256)

    def to_hf(self, repo: str, *, branch: str = None, pretty_name: Optional[str] = None) -> None:
        """Upload this artifact to Hugging Face Hub.

        Args:
            repo: The Hugging Face repository name.
            branch: The branch or tag of the repository to upload to. (Default: main) A branch can also be provided
            directly in the repository name using "owner/repo@branch".
            pretty_name: The human-readable name of the artifact. (Default: the repository name)
        """
        import huggingface_hub

        if '@' in repo:
            if branch is not None:
                raise ValueError('Provided branch in both repository name (via @) and as argument to to_hf')
            repo, branch = repo.split('@', 1)
        if branch is None:
            branch = 'main'

        with tempfile.TemporaryDirectory() as d:
            # build a package with a maximum individual file size of just under 5GB, the limit for HF datasets
            metadata = {}
            self.build_package(os.path.join(d, 'artifact.tar.lz4'), max_file_size=4.9e9, metadata_out=metadata)
            readme = self._hf_readme(repo=repo, branch=branch, pretty_name=pretty_name, metadata=metadata)
            if readme:
                with open(f'{d}/README.md', 'wt') as fout:
                    fout.write(readme)
            try:
                huggingface_hub.create_repo(repo, repo_type='dataset')
            except huggingface_hub.utils.HfHubHTTPError as e:
                if e.server_message != 'You already created this dataset repo':
                    raise
            try:
                huggingface_hub.create_branch(repo, repo_type='dataset', branch=branch)
            except huggingface_hub.utils.HfHubHTTPError as e:
                if not e.server_message.startswith('Reference already exists:'):
                    raise
            path = huggingface_hub.upload_folder(
                repo_id=repo,
                folder_path=d,
                repo_type='dataset',
                revision=branch,
            )
            sys.stderr.write(f"\nArtifact uploaded to {path}\nConsider editing the README.md to help explain this "
                              "artifact to others.\n")

    @classmethod
    def from_dataset(cls, dataset: str, variant: str, *, expected_sha256: Optional[str] = None) -> 'Artifact':
        """Load an artifact from a PyTerrier dataset.

        Args:
            dataset: The name of the dataset.
            variant: The variant of the dataset.
            expected_sha256: The expected SHA-256 hash of the artifact. If provided, the downloaded artifact will be
            verified against this hash and an error will be raised if the hash does not match.
        """
        return cls.from_hf(
            repo='pyterrier/from-dataset',
            branch=f'{dataset}.{variant}',
            expected_sha256=expected_sha256)

    def to_zenodo(self, *, pretty_name: Optional[str] = None, sandbox: bool = False) -> None:
        """Upload this artifact to Zenodo.

        Args:
            pretty_name: The human-readable name of the artifact.
            sandbox: Whether to perform a test upload to the Zenodo sandbox.
        """
        if sandbox:
            base_url = 'https://sandbox.zenodo.org/api'
        else:
            base_url = 'https://zenodo.org/api'

        access_token = os.environ.get('ZENODO_TOKEN')
        params = {'access_token': access_token}

        with tempfile.TemporaryDirectory() as d:
            r = requests.post(f'{base_url}/deposit/depositions', params=params, json={})
            r.raise_for_status()
            deposit_data = r.json()
            sys.stderr.write("Created {}\n".format(deposit_data['links']['html']))
            try:
                metadata = {}
                sys.stderr.write("Building package.\n")
                self.build_package(os.path.join(d, 'artifact.tar.lz4'), metadata_out=metadata)
                z_meta = {
                    'metadata': self._zenodo_metadata(
                        pretty_name=pretty_name,
                        zenodo_id=deposit_data['id'],
                        metadata=metadata,
                    ),
                }
                r = requests.put(deposit_data['links']['latest_draft'], params=params, json=z_meta)
                r.raise_for_status()
                sys.stderr.write("Uploading...\n")
                for file in sorted(os.listdir(d)):
                    file_path = os.path.join(d, file)
                    with open(file_path, 'rb') as fin, \
                         pt.io.TqdmReader(fin, total=os.path.getsize(file_path), desc=file) as fin:
                        r = requests.put(
                            '{}/{}'.format(deposit_data['links']['bucket'], file),
                            params={'access_token': access_token},
                            data=fin)
                        r.raise_for_status()
            except:
                sys.stderr.write("Discarding {}\n".format(deposit_data['links']['html']))
                requests.post(deposit_data['links']['discard'], params=params, json={})
                raise
            sys.stderr.write("Upload complete. Please complete the form at {} to publish this artifact. (Note that "
                "publishing to Zenodo cannot be undone.)\n".format(deposit_data['links']['html']))

    @classmethod
    def from_zenodo(cls, zenodo_id: str, *, expected_sha256: Optional[str] = None) -> 'Artifact':
        """Load an artifact from Zenodo.

        Args:
            zenodo_id: The Zenodo record ID of the artifact.
            expected_sha256: The expected SHA-256 hash of the artifact. If provided, the downloaded artifact will be
            verified against this hash and an error will be raised if the hash does not match.
        """
        return cls.from_url(f'zenodo:{zenodo_id}', expected_sha256=expected_sha256)

    def _package_files(self) -> Iterator[Tuple[str, Union[str, io.BytesIO]]]:
        has_pt_meta_file = False
        for root, dirs, files in os.walk(self.path):
            rel_root = os.path.relpath(root, start=self.path)
            for file in sorted(files):
                file_full_path = os.path.join(root, file)
                file_rel_path = os.path.join(rel_root, file) if rel_root != '.' else file
                yield file_rel_path, file_full_path
                if file_rel_path == 'pt_meta.json':
                    has_pt_meta_file = True
        if not has_pt_meta_file:
            metadata = self._build_metadata()
            if metadata is not None:
                yield 'pt_meta.json', io.BytesIO(json.dumps(metadata).encode())

    def _build_metadata(self) -> Optional[Dict[str, Any]]:
        metadata = {}

        try:
            metadata['type'], metadata['format'] = pt.inspect.artifact_type_format(self)
        except TypeError:
            pass # couldn't identify type and format

        if hasattr(self, 'ARTIFACT_PACKAGE_HINT'):
            metadata['package_hint'] = self.ARTIFACT_PACKAGE_HINT
        else:
            metadata['package_hint'] = self.__class__.__module__.split('.')[0]

        return metadata

    def _hf_readme(self,
        *,
        repo: str,
        branch: Optional[str] = 'main',
        pretty_name: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        if pretty_name is None:
            title = repo.split('/')[-1]
            pretty_name = '# pretty_name: "" # Example: "MS MARCO Terrier Index"'
        else:
            title = pretty_name
            pretty_name = f'pretty_name: {pretty_name!r}'
        if branch != 'main':
            repo = f'{repo}@{branch}'
        if metadata is None:
            metadata = {}
        tags = ['pyterrier', 'pyterrier-artifact']
        if 'type' in metadata:
            tags.append('pyterrier-artifact.{type}'.format(**metadata))
        if 'type' in metadata and 'format' in metadata:
            tags.append('pyterrier-artifact.{type}.{format}'.format(**metadata))
        tags = '\n- '.join([''] + tags)
        return f'''---
{pretty_name}
tags:{tags}
task_categories:
- text-retrieval
viewer: false
---

# {title}

## Description

*TODO: What is the artifact?*

## Usage

```python
# Load the artifact
import pyterrier as pt
artifact = pt.Artifact.from_hf({repo!r})
# TODO: Show how you use the artifact
```

## Benchmarks

*TODO: Provide benchmarks for the artifact.*

## Reproduction

```python
# TODO: Show how you constructed the artifact.
```

## Metadata

```
{json.dumps(metadata, indent=2)}
```
'''

    def _zenodo_metadata(self, *, zenodo_id: str, pretty_name: Optional[str] = None, metadata: Dict) -> Optional[str]:
        description = f'''
<h2>Description</h2>

<p>
<i>TODO: What is the artifact?</i>
</p>

<h2>Usage</h2>

<pre>
# Load the artifact
import pyterrier as pt
artifact = pt.Artifact.from_zenodo({str(zenodo_id)!r})
# TODO: Show how you use the artifact
</pre>

<h2>Benchmarks</h2>

<p>
<i>TODO: Provide benchmarks for the artifact.</i>
</p>

<h2>Reproduction</h2>

<pre>
# TODO: Show how you constructed the artifact.
</pre>

<h2>Metadata</h2>

<pre>
{json.dumps(metadata, indent=2)}
</pre>
'''
        tags = ['pyterrier', 'pyterrier-artifact']
        if 'type' in metadata:
            tags.append('pyterrier-artifact.{type}'.format(**metadata))
        if 'type' in metadata and 'format' in metadata:
            tags.append('pyterrier-artifact.{type}.{format}'.format(**metadata))
        metadata = {
            'description': description,
            'upload_type': 'other',
            'publisher': 'Zenodo',
            'publication_date': datetime.today().strftime('%Y-%m-%d'),
            'keywords': tags,
        }
        if pretty_name:
            metadata['title'] = pretty_name
        return metadata

    def build_package(
        self,
        package_path: Optional[str] = None,
        *,
        max_file_size: Optional[float] = None,
        metadata_out: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> str:
        """Builds a package for this artifact.

        Packaged artifacts are useful for distributing an artifact as a single file, such as from Artifact.from_url().
        A separate metadata file is also generated, which gives information about the package's contents, including
        file sizes and an expected hash for the package.

        Args:
            package_path: The path of the package to create. Defaults to the artifact path with a .tar.lz4 extension.
            max_file_size: the (approximate) maximum size of each file.
            metadata_out: A dictionary that is updated with the metadata of the artifact (if provided).
            verbose: Whether to display a progress bar when packaging.

        Returns:
            The path of the package created.
        """
        if package_path is None:
            package_path = str(self.path) + '.tar.lz4'

        metadata = {
            'expected_sha256': None,
            'total_size': 0,
            'contents': [],
        }

        chunk_num = 0
        chunk_start_offset = 0
        def manage_maxsize(_: None):
            nonlocal raw_fout
            nonlocal chunk_num
            nonlocal chunk_start_offset
            if max_file_size is not None and raw_fout.tell() >= max_file_size:
                raw_fout.flush()
                chunk_start_offset += raw_fout.tell()
                raw_fout.close()
                if chunk_num == 0:
                    metadata['segments'] = []
                    metadata['segments'].append({'idx': chunk_num, 'offset': 0})
                chunk_num += 1
                if verbose:
                    print(f'starting segment {chunk_num}')
                metadata['segments'].append({'idx': chunk_num, 'offset': chunk_start_offset})
                raw_fout = stack.enter_context(pt.io.finalized_open(f'{package_path}.{chunk_num}', 'b'))
                sha256_fout.replace_writer(raw_fout)

        with contextlib.ExitStack() as stack:
            raw_fout = stack.enter_context(pt.io.finalized_open(f'{package_path}.{chunk_num}', 'b'))
            sha256_fout = stack.enter_context(pt.io.HashWriter(raw_fout))
            lz4_fout = stack.enter_context(LZ4FrameFile(sha256_fout, 'wb'))
            tarout = stack.enter_context(tarfile.open(fileobj=lz4_fout, mode='w'))

            added_dirs = set()
            for rel_path, file in self._package_files():
                path_dir, name = os.path.split(rel_path)
                if path_dir and path_dir not in added_dirs:
                    tar_record = tarfile.TarInfo(path_dir)
                    tar_record.type = tarfile.DIRTYPE
                    tarout.addfile(tar_record)
                    added_dirs.add(path_dir)
                lz4_fout.flush() # flush before each file, allowing seeking directly to this file within the archive
                tar_record = tarfile.TarInfo(rel_path)
                if isinstance(file, io.BytesIO):
                    tar_record.size = file.getbuffer().nbytes
                else:
                    tar_record.size = os.path.getsize(file)
                metadata['contents'].append({
                    'path': rel_path,
                    'size': tar_record.size,
                    'offset': chunk_start_offset + raw_fout.tell(),
                })
                metadata['total_size'] += tar_record.size
                if verbose:
                    print(f'adding {rel_path} [{pt.utils.byte_count_to_human_readable(tar_record.size)}]')

                if isinstance(file, io.BytesIO):
                    file.seek(0)
                    if rel_path == 'pt_meta.json' and metadata_out is not None:
                        metadata_out.update(json.load(file))
                    file.seek(0)
                    with pt.io.CallbackReader(file, manage_maxsize) as fin:
                        tarout.addfile(tar_record, fin)
                else:
                    with open(file, 'rb') as fin, \
                         pt.io.CallbackReader(fin, manage_maxsize) as fin:
                        tarout.addfile(tar_record, fin)
                    if rel_path == 'pt_meta.json' and metadata_out is not None:
                        with open(file, 'rb') as fin:
                            metadata_out.update(json.load(fin))

            tarout.close()
            lz4_fout.close()

            metadata['expected_sha256'] = sha256_fout.hexdigest()

            metadata_outf = stack.enter_context(pt.io.finalized_open(f'{package_path}.json', 't'))
            json.dump(metadata, metadata_outf)
            metadata_outf.write('\n')

        if chunk_num == 0:
            # no chunking was actually done, can use provided name directly
            os.rename(f'{package_path}.{chunk_num}', package_path)

        return package_path


def _load_metadata(path: str) -> Dict:
    """Load the metadata file for the artifact at the specified path.

    Args:
        path: The path of the artifact on disk.

    Returns:
        The loaded metadata file, or an empty dictionary if the metadata file does not exist.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.isdir(path):
        if os.path.isfile(path):
            return {}
        raise FileNotFoundError(f'{path} not found')

    metadata_file = os.path.join(path, 'pt_meta.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rt') as f:
            return json.load(f)

    return {}


def _metadata_adapter(path: str):
    directory_listing = []
    if os.path.isdir(path):
        directory_listing = os.listdir(path)
    for entry_point in pt.utils.entry_points('pyterrier.artifact.metadata_adapter'):
        if (metadata := entry_point.load()(path, directory_listing)) is not None:
            return metadata
    return {}


def _load(path: str) -> Artifact:
    """Load the artifact from the specified path.

    This function attempts to load the artifact using a supporting Artifact implementation.

    Args:
        path: The path of the artifact on disk.

    Returns:
        The loaded artifact.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If no implementation is found that supports the artifact at the specified path.
    """
    metadata = _load_metadata(path)

    if 'type' not in metadata or 'format' not in metadata:
        metadata.update(_metadata_adapter(path))

    if 'type' in metadata and 'format' in metadata:
        entry_points = list(pt.utils.entry_points('pyterrier.artifact'))
        matching_entry_points = [ep for ep in entry_points if ep.name == '{type}.{format}'.format(**metadata)]
        for entry_point in matching_entry_points:
            artifact_cls = entry_point.load()
            return artifact_cls(path)

    # coudn't find an implementation that supports this artifact
    error = [
        f'No implementation found that supports the artifact at {_path_repr(path)}.',
        f'type={metadata.get("type", "(unknown)")}, format={metadata.get("format", "(unknown)")}.'
    ]
    if 'package_hint' in metadata:
        error.append(f'Do you need to `pip install {metadata["package_hint"]}`?')
    raise ValueError('\n'.join(error))


def _path_repr(path: str) -> str:
    """Return a string representation of a path, including information from the artifact's metadata file."""
    artifact_info_path = f'{path}.json'
    if os.path.exists(artifact_info_path):
        with open(artifact_info_path) as fin:
            artifact_info = json.load(fin)
        if 'url' in artifact_info:
            return f'{path!r} <from {artifact_info["url"]!r}>'
        if 'path' in artifact_info:
            return f'{path!r} <from {artifact_info["path"]!r}>'
    return repr(path)


def _hf_url_resolver(parsed_url: ParseResult) -> str:
    # paths like: hf:macavaney/msmarco-passage.terrier
    org_repo = parsed_url.path
    # default to ref=main, but allow user to specify another branch, hash, etc, with abc/xyz@branch
    ref = 'main'
    if '@' in org_repo:
        org_repo, ref = org_repo.split('@', 1)
    return f'https://huggingface.co/datasets/{org_repo}/resolve/{ref}/artifact.tar.lz4'


def _zenodo_url_resolver(parsed_url: ParseResult) -> str:
    # paths like: zenodo:111952
    zenodo_id = parsed_url.path
    return f'https://zenodo.org/records/{zenodo_id}/files/artifact.tar.lz4'
