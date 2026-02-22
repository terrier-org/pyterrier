from typing import Optional
import os
import json
import requests
import bibtexparser
import gzip
from pylatexenc.latex2text import LatexNodes2Text
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.domains.std import StandardDomain
from sphinx.util import logging

logger = logging.getLogger(__name__)

latex_converter = LatexNodes2Text()

# Cache file to store BibTeX entries
CACHE_FILE = "dblp_cache.json.gz"


def load_dblp_cache():
    """Load the cache from the file, or create a new one if it doesn't exist."""
    if os.path.exists(CACHE_FILE):
        with gzip.open(CACHE_FILE, "rt") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

dblp_cache = load_dblp_cache()

def save_dblp_cache():
    """Save the cache to the file."""
    with gzip.open(CACHE_FILE, "wt") as f:
        json.dump(dblp_cache, f, indent=4)


class CiteNode(nodes.General, nodes.Element):
    """Custom node to render DBLP citation."""
    def __init__(self, citation_id, citation, link, bibtex):
        super().__init__()
        self['id'] = citation_id
        self['citation'] = citation
        self['link'] = link
        self['bibtex'] = bibtex


class CiteDirective(Directive):
    required_arguments = 1
    has_content = True
    option_spec = {'citation': str, 'link': str, 'blbtex': str}

    def run(self):
        # Create a custom node to hold the BibTeX content and display data
        node = CiteNode(
            self.get_citation_id(),
            self.get_citation(),
            self.get_link(),
            self.get_bibtex()
        )

        # in the future, could store citation information in self.state.document.settings.env
        # to generate a bibliography page.
        if not hasattr(self.state.document.settings.env, 'bibliography'):
            self.state.document.settings.env.bibliography = {}
        self.state.document.settings.env.bibliography.setdefault(node['id'], {
            'id': node['id'],
            'citation': node['citation'],
            'link': node['link'],
            'bibtex': node['bibtex'],
            'refs': {},
        })['refs'][self.state.document.settings.env.docname] = {
            'target': self.state.document.settings.env.docname,
        }

        return [node]

    def get_citation_id(self) -> str:
        return self.arguments[0]

    def get_citation(self) -> Optional[str]:
        return self.options.get('citation')

    def get_link(self) -> Optional[str]:
        return self.options.get('link')

    def get_bibtex(self) -> Optional[str]:
        if self.content:
            return '\n'.join(self.content)
        return None


class CiteDblpDirective(CiteDirective):
    """Directive to fetch and display a citation from DBLP using a DBLP ID."""
    has_content = False
    required_arguments = 1  # The DBLP ID must be provided as an argument.

    def run(self):
        dblp_id = self.arguments[0]

        if dblp_id in dblp_cache:
            bibtex_entry_short, bibtex_entry_full = dblp_cache[dblp_id]
        else:
            # Fetch BibTeX entry from DBLP
            try:
                logger.info(f"Fetching BibTeX for DBLP ID '{dblp_id}' from DBLP...")
                response = requests.get(f"https://dblp.org/rec/{dblp_id}.bib?param=0")
                response.raise_for_status()
                bibtex_entry_short = response.text
                response = requests.get(f"https://dblp.org/rec/{dblp_id}.bib?param=1")
                response.raise_for_status()
                bibtex_entry_full = response.text
                dblp_cache[dblp_id] = [bibtex_entry_short, bibtex_entry_full]  # Save to cache
                save_dblp_cache()  # Persist the cache
            except requests.RequestException as e:
                error_msg = f"Failed to fetch BibTeX for DBLP ID '{dblp_id}': {str(e)}"
                logger.error(error_msg)
                return [nodes.error(None, nodes.Text(error_msg))]
        self.bibtex_entry_short = bibtex_entry_short
        self.bibtex_entry_full = bibtex_entry_full
        return super().run()

    def get_citation_id(self) -> str:
        return 'dblp:' + self.arguments[0]

    def get_citation(self) -> Optional[str]:
        citation = super().get_citation()
        if citation is not None:
            return citation
        res = bibtexparser.loads(self.bibtex_entry_short).entries[0]
        authors = latex_converter.latex_to_text(res.get('author', ''))
        if authors:
            authors = authors.split(' and\n')
            if len(authors) > 2:
                authors = authors[0].split()[-1] + ' et al'
            elif len(authors) == 2:
                authors = authors[0].split()[-1] + ' and ' + authors[1].split()[-1]
            else:
                authors = authors[0].split()[-1]
        title = latex_converter.latex_to_text(res.get('title', ''))
        year = latex_converter.latex_to_text(res.get('year', ''))
        if 'journal' in res and res['journal'] == 'CoRR':
            citation = f'{authors}. {title}. arXiv {year}.'
        elif 'journal' in res:
            journal = latex_converter.latex_to_text(res['journal'])
            citation = f'{authors}. {title}. {journal} {year}.'
        elif 'booktitle' in res:
            conf = latex_converter.latex_to_text(res['booktitle'])
            citation = f'{authors}. {title}. {conf} {year}.'
        else:
            citation = f'{authors}. {title}. {year}.'
        return citation

    def get_link(self) -> Optional[str]:
        link = super().get_link()
        if link is not None:
            return link
        res = bibtexparser.loads(self.bibtex_entry_full).entries[0]
        return latex_converter.latex_to_text(res.get('url', ''))

    def get_bibtex(self) -> Optional[str]:
        bibtex = super().get_bibtex()
        if bibtex is not None:
            return bibtex
        return self.bibtex_entry_full


def get_hierarchical_title(docname, app):
    """
    Get a hierarchical title for a Sphinx document, including its position in the toctree.

    Args:
        docname (str): The docname (e.g., 'something/text').
        app (sphinx.application.Sphinx): The Sphinx application object.

    Returns:
        str: The hierarchical title, with sections separated by " > ".
    """
    def get_doc_title(docname):
        """Get the title for a given docname."""
        title_node = app.env.titles.get(docname)
        return title_node.astext() if title_node else "Untitled"

    def find_parents(docname, toctree):
        """Find the parent hierarchy of a document in the toctree."""
        for node in toctree.traverse():
            if node.tagname == "reference" and node.get("refuri") == app.builder.get_target_uri(docname):
                # Found the current doc in the toctree
                parent_node = node.parent.parent
                if parent_node.tagname == "toctree":
                    parent_docname = parent_node.get("docname")
                    if parent_docname:
                        return find_parents(parent_docname, app.env.tocs[parent_docname]) + [parent_docname]
        return []

    # Find parents by walking up the toctree
    parent_hierarchy = []
    if docname in app.env.tocs:
        parent_hierarchy = find_parents(docname, app.env.tocs[docname])

    # Collect titles for the hierarchy
    hierarchical_titles = [get_doc_title(parent) for parent in parent_hierarchy]
    hierarchical_titles.append(get_doc_title(docname))

    # Join titles with separators
    return " » ".join(hierarchical_titles)


def visit_cite_node_html(self, node):
    """HTML visitor to render the DblpNode."""
    citation = node['citation']
    link = node['link']
    bibtex = node['bibtex']

    if link:
        link_html = f' <a href="{link}" target="_blank">[link]</a>'
    else:
        link_html = ''

    if bibtex:
        body_html = f'''
<details class="dblp-citation">
  <summary><b>{citation}</b>{link_html}</summary>
  <pre class="bibtex">{bibtex.strip()}</pre>
</details>
'''
    else:
        body_html = f'<p><b>{citation}</b>{link_html}</p>'

    self.body.append(f'''
<div class="admonition">
  <p class="admonition-title">Citation</p>
  {body_html}
</div>
    ''')


def depart_cite_node_html(self, node):
    """HTML departure for DblpNode."""
    pass


def build_breadcrumb_titles(env, toc):
    result = {}
    for node in toc.traverse():
        if node.tagname == 'toctree':
            for page_title, page in node['entries']:
                if page_title is None:
                    page_title = env.titles.get(page)
                    page_title = page_title.astext() if page_title else 'Untitled'
                page_title = page_title.strip()
                result[page] = page_title
                if page in env.tocs:
                    for subpage, subpage_title in build_breadcrumb_titles(env, env.tocs[page]).items():
                        result[subpage] = f'{page_title} » {subpage_title}'
    return result

def collect_bibliiography(app):
    if app.builder.embedded or len(app.env.bibliography) == 0:
        # Building embedded (e.g. htmlhelp or ePub) or there were no bibs found
        return []

    breadcrumb_titles = build_breadcrumb_titles(app.env, app.env.tocs['index'])

    for bib in app.env.bibliography.values():
        bib['refs'] = list(bib['refs'].values())
        for ref in bib['refs']:
            ref['url'] = app.builder.get_target_uri(ref['target'])
            ref['title'] = breadcrumb_titles.get(ref['target'], 'Untitled')

    return [(
        'bibliography',
        {'bibliography': sorted(app.env.bibliography.values(), key=lambda x: x['citation'])},
        'bibliography.html',
    )]


def init_bib(app, env):
    if not hasattr(env, 'bibliography'):
        env.bibliography = {}
    return []


def merge_bib(app, env, docnames, other):
    if not hasattr(env, 'bibliography'):
        env.bibliography = {}
    if hasattr(other, 'bibliography'):
        env.bibliography.update(other.bibliography)


def setup(app):
    app.add_node(CiteNode, html=(visit_cite_node_html, depart_cite_node_html))
    app.add_directive('cite', CiteDirective)
    app.add_directive('cite.dblp', CiteDblpDirective)
    app.connect('env-get-updated', init_bib)
    app.connect('env-merge-info', merge_bib)
    app.connect('html-collect-pages', collect_bibliiography)
    StandardDomain._virtual_doc_names['bibliography'] = ('bibliography', 'Bibliography')
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
