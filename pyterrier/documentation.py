"""Tools for working with the PyTerrier documentation."""
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Tuple

import pyterrier as pt
import requests

_BASE_URL = 'https://pyterrier.readthedocs.io/en/latest/'

_cached_objects_inv = None


def _is_old(path: Path, url: str) -> bool:
    file_last_updated = path.stat().st_mtime
    if file_last_updated < datetime.now().timestamp() - 86400:
        # check at most once per day
        return False
    try:
        response = requests.head(url)
        response.raise_for_status()
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')
            is_old = file_last_updated < last_modified_date.timestamp()
            if not is_old:
                path.touch() # reset the st_mtime
            return is_old
    except requests.RequestException:
        pass  # If we can't fetch the URL, assume it's old
    return True


def objects_inv() -> dict:
    """Returns a dictionary mapping objects IDs in the PyTerrier documentation to their documentation pages."""
    global _cached_objects_inv
    if _cached_objects_inv is not None:
        return _cached_objects_inv
    objects_inv_path = Path(pt.io.pyterrier_home()) / 'documentation' / 'objects.inv'
    objects_inv_url = _BASE_URL + 'objects.inv'
    if not objects_inv_path.exists() or _is_old(objects_inv_path, objects_inv_url):
        if not objects_inv_path.parent.exists():
            objects_inv_path.parent.mkdir(parents=True, exist_ok=True)
        pt.io.download(objects_inv_url, str(objects_inv_path), verbose=False, headers={"User-Agent": "curl/7.81.0"})
    # parse the objects.inv file
    with objects_inv_path.open('rb') as f:
        for _ in range(4):
            lineb = f.readline()
            if not lineb.startswith(b'#'):
                raise ValueError(f'Invalid objects.inv file: expected comment line, got {lineb!r}')
        data = f.read()
    lines = zlib.decompress(data).decode('utf-8').splitlines()
    objects : Dict[Tuple[str,str], Tuple[str, str, int]] = {}
    for line in lines:
        if line.startswith('#'):
            continue  # skip comments
        parts = line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue  # skip malformed lines
        name, type_, priority, url, title = parts
        ipriority = int(priority)
        if (type_, name) not in objects or objects[(type_, name)][2] > ipriority:
            objects[type_, name] = (_BASE_URL + url, title, ipriority)
    _cached_objects_inv = objects
    return _cached_objects_inv


def url_for_class(cls: Union[type, object]) -> Optional[str]:
    """Returns the URL of the documentation page for the specified class."""
    objects = objects_inv()
    if not isinstance(cls, type):
        cls = cls.__class__
    cls_name = f'{cls.__module__}.{cls.__name__}'
    key = ('py:class', cls_name)
    if key in objects:
        return objects[key][0]
    return None
