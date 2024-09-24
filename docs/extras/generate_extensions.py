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

            if pkg == '':
                if '<' not in display_name:
                    print(f'Skipping {line!r} -- must be in the format of "# Package Name <URL>"')
                else:
                    f_ext.write(f'   {display_name}\n')
                continue

            metadata = importlib.metadata.metadata(pkg)
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
            else:
                print(f'Skipping {line!r} -- No pt_docs in package and no home-page in metadata"')
