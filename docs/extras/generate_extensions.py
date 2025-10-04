import re
import importlib
from pathlib import Path
from hashlib import sha1

def generate_extensions():
    Path('_includes').mkdir(parents=True, exist_ok=True)

    with Path('_includes/ext_toc.rst').open('wt') as f_ext:
        f_ext.write('''
.. toctree::
   :maxdepth: 1
   :caption: Extensions

    ''')
        for line in open('extensions.txt'):
            if '#' in line:
                pkg, display_name = line.split('#', 1)
            else:
                pkg, display_name = line, line
            pkg = pkg.strip()
            display_name = display_name.strip()
            if '<' in display_name:
                display_name, url = display_name.split('<')
                display_name = display_name.strip()
                url = url.strip('>')
            else:
                url = None

            if pkg == '':
                if not url:
                    print(f'Skipping {line!r} -- must be in the format of "# Package Name <URL>"')
                else:
                    f_ext.write(f'   {display_name} <{url}>\n')
                continue

            try:
                metadata = importlib.metadata.metadata(pkg)
            except importlib.metadata.PackageNotFoundError:
                if not url:
                    print(f'Skipping {line!r} -- package not installed and no fallback URL provided (format: "# package_name <url>"). You may want to run pip -r extensions.txt for a complete documentation build.')
                else:
                    print(f'Falling back on provided url for {pkg} since the package is not installed. You may want to run pip -r extensions.txt for a complete documentation build.')
                    f_ext.write(f'   {display_name} <{url}>\n')
                continue

            pkg_name = metadata['name']
            docs = importlib.resources.files(pkg).joinpath('pt_docs')
            if docs.is_dir():
                # Documentation included in the package, copy it over
                paths = [(docs, Path(f'ext/{pkg_name}'))]
                while paths:
                    src, dest = paths.pop()
                    dest.mkdir(parents=True, exist_ok=True)
                    for path in src.iterdir():
                        if path.is_dir():
                            paths.append((path, dest/path.name))
                        else:
                            if (dest/path.name).exists():
                                source_hash = sha1(path.open('rb').read()).hexdigest()
                                dest_hash = sha1((dest/path.name).open('rb').read()).hexdigest()
                                if source_hash == dest_hash:
                                    continue
                            with path.open('rb') as fin, (dest/path.name).open('wb') as fout:
                                fout.write(fin.read())
                f_ext.write(f'   {display_name} <../ext/{pkg_name}/index.rst>\n')
            elif 'home-page' in metadata:
                # No documentation included in the package, but we can link to the repo
                f_ext.write(f'   {display_name} <{metadata["home-page"]}>\n')
            elif url:
                # No documentation included in the package and no home-page, but still can link to the provided <url>> from the config
                f_ext.write(f'   {display_name} <{url}>\n')
            else:
                print(f'Skipping {line!r} -- No pt_docs in package and no home-page or project-url in metadata"')
