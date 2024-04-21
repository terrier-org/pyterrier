import contextlib
import json
import os
import tarfile
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Dict, Optional
from urllib.parse import urlparse

from lz4.frame import LZ4FrameFile

import pyterrier as pt


class Artifact(ABC):
    """Base class for PyTerrier artifacts.

    An artifact is a component stored on disk, such as an index.

    Artifacts usually act as factories for transformers that use them. For example, an index artifact
    may provide a `.retriever()` method that returns a transformer that searches the index.
    """

    @classmethod
    @abstractmethod
    def _try_load(cls, path: str, metadata: Dict) -> Optional['Artifact']:
        """Return the artifact loaded with this sublass if it supports the artifact at the specified path.

        Args:
            path: The path of the artifact on disk.
            metadata: The loaded meta.json file of the artifact, if available.

        Returns:
            The loaded artifact, or None if this subclass does not support the artifact at the specified path.
        """
        pass

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
            return load(path)
        else:
            # called SomeArtifact.load(path), so load this specific artifact
            # TODO: add error message if not loaded
            artifact = cls._try_load(path, load_metadata())
            if artifact is not None:
                return artifact
            raise ValueError(f'{cls} does not support the artifact at {path_repr(path)}.')

    @classmethod
    def from_url(cls, url: str, *, expected_sha256: Optional[str] = None) -> 'Artifact':
        """Load the artifact from the specified URL.

        The artifact at the specified URL will be downloaded and stored in PyTerrier's artifact cache.

        Args:
            url: The URL or file path of the artifact.

        Returns:
            The loaded artifact.

        Raises:
            ValueError: If no implementation is found that supports the artifact at the specified path.
        """

        # Support mapping "protocols" of the URL other than http[s]
        # e.g., "hf:abc/xyz@branch" -> "https://huggingface.co/datasets/abc/xyz/resolve/branch/artifact.tar"
        parsed_url = urlparse(url)
        if parsed_url.scheme == 'hf':
            org_repo = parsed_url.path
            # default to ref=main, but allow user to specify another branch, hash, etc, with abc/xyz@branch
            ref = 'main'
            if '@' in org_repo:
                org_repo, ref = org_repo.split('@', 1)
            url = f'https://huggingface.co/datasets/{org_repo}/resolve/{ref}/artifact.tar.lz4'
            parsed_url = urlparse(url)

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
                    # TODO: still need to verify sha256 in this case
                    fin = stack.enter_context(pt.io.TqdmSha256BufferedSequenceReader(
                        (pt.io.open_or_download_stream(f'{url}.{i}') for i in range(len(download_info['segments'])))))
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
                        fin.pbar.write(f'extracting {member.path} [{pt.io.byte_count_to_human_readable(member.size)}]')
                        tar_in.extract(member, dout, set_attrs=False)

        return cls.load(path)

    @classmethod
    def from_dataset(cls, dataset: str, variant: str, *, expected_sha256: Optional[str] = None) -> 'Artifact':
        hf_repo = 'macavaney/pyterrier-from-dataset'
        branch = f'{dataset}.{variant}'
        return cls.from_url(f'hf:{hf_repo}@{branch}', expected_sha256=expected_sha256)


from_url = Artifact.from_url
from_dataset = Artifact.from_dataset


def load_metadata(path: str) -> Dict:
    """
    Load the metadata file for the artifact at the specified path.

    Args:
        path: The path of the artifact on disk.

    Returns:
        The loaded metadata file, or an empty dictionary if the metadata file does not exist.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f'{path} not found')

    metadata_file = os.path.join(path, 'pt_meta.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rt') as f:
            return json.load(f)

    # Hack! Assume some metadata based on the presence of some well-known files
    if os.path.exists(os.path.join(path, 'data.properties')):
        return {
            'type': 'sparse_index',
            'format': 'terrier',
        }

    return {}


def load(path: str) -> Artifact:
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
    metadata = load_metadata(path)

    # called Artifact.load(path), so try to find a supporting artifact
    entry_points = list(pt.io.entry_points('pyterrier.artifact'))

    if 'type' in metadata and 'format' in metadata:
        # if the metadata includes type and format, first try to find the
        # specific artifact class via the entry_point names
        artifact_key = '{type}.{format}'.format(**metadata)
        keyed_entry_points = [ep for ep in entry_points if ep.name == artifact_key]
        for entry_point in keyed_entry_points:
            artifact_cls = entry_point.load()
            artifact = artifact_cls._try_load(path, metadata)
            if artifact is not None:
                return artifact
        # not found via key, fall back on trying other types. No need to try
        # the types we already checked.
        entry_points = [ep for ep in entry_points if ep.name != artifact_key]

    # See which artifact type supports this artifact
    for entry_point in entry_points:
        artifact_cls = entry_point.load()
        artifact = artifact_cls._try_load(path, metadata)
        # TODO: add better logging & error message if nothing ever found
        if artifact is not None:
            return artifact

    # coudn't find an implementation that supports this artifact
    raise ValueError(f'No implementation found that supports the artifact at {path_repr(path)}.')


def build_package(artifact_path: str, package_path: Optional[str] = None, verbose: bool = True) -> str:
    """
    Build a package from the specified artifact path.

    Packaged artifacts are useful for distributing an artifact as a single file, such as from Artifact.from_url().
    A separate metadata file is also generated, which gives information about the package's contents, including
    file sizes and an expected hash for the package.

    Args:
        artifact_path: The path of the artifact to package.
        package_path: The path of the package to create. Defaults to the artifact path with a .tar.lz4 extension.
        verbose: Whether to display a progress bar when packaging.

    Returns:
        The path of the package created.
    """
    if package_path is None:
        package_path = artifact_path + '.tar.lz4'

    metadata = {
        'expected_sha256': None,
        'total_size': 0,
        'contents': [],
    }

    with contextlib.ExitStack() as stack:
        raw_fout = stack.enter_context(pt.io.finalized_open(package_path, 'b'))
        sha256_fout = stack.enter_context(pt.io.Sha256BufferedWriter(raw_fout))
        lz4_fout = stack.enter_context(LZ4FrameFile(sha256_fout, 'wb'))
        tarout = stack.enter_context(tarfile.open(fileobj=lz4_fout, mode='w'))

        for root, dirs, files in os.walk(artifact_path):
            rel_root = os.path.relpath(root, start=artifact_path)
            if rel_root != '.':
                tar_record = tarfile.TarInfo(rel_root)
                tar_record.type = tarfile.DIRTYPE
                tarout.addfile(tar_record)
            for file in sorted(files):
                lz4_fout.flush() # flush before each file, allowing seeking directly to this file within the archive
                file_full_path = os.path.join(root, file)
                file_rel_path = os.path.join(rel_root, file) if rel_root != '.' else file
                tar_record = tarfile.TarInfo(file_rel_path)
                tar_record.size = os.path.getsize(file_full_path)
                metadata['contents'].append({
                    'path': file_rel_path,
                    'size': tar_record.size,
                    'offset': raw_fout.tell(),
                })
                metadata['total_size'] += tar_record.size
                if verbose:
                    print(f'adding {file_rel_path} [{pt.io.byte_count_to_human_readable(tar_record.size)}]')
                with open(file_full_path, 'rb') as fin:
                    tarout.addfile(tar_record, fin)

        metadata['expected_sha256'] = sha256_fout.sha256.hexdigest()

        metadata_out = stack.enter_context(pt.io.finalized_open(f'{package_path}.json', 't'))
        json.dump(metadata, metadata_out)
        metadata_out.write('\n')

    return package_path


def path_repr(path):
    artifact_info_path = f'{path}.json'
    if os.path.exists(artifact_info_path):
        with open(artifact_info_path) as fin:
            artifact_info = json.load(fin)
        if 'url' in artifact_info:
            return f'{path!r} <from {artifact_info["url"]!r}>'
        if 'path' in artifact_info:
            return f'{path!r} <from {artifact_info["path"]!r}>'
    return repr(path)
