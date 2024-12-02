import os
import json
import requests
import bibtexparser
import gzip
from pylatexenc.latex2text import LatexNodes2Text
from docutils import nodes
from docutils.parsers.rst import Directive

latex_converter = LatexNodes2Text()

# Cache file to store BibTeX entries
CACHE_FILE = "dblp_cache.json.gz"


def load_cache():
    """Load the cache from the file, or create a new one if it doesn't exist."""
    if os.path.exists(CACHE_FILE):
        with gzip.open(CACHE_FILE, "rt") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_cache(cache):
    """Save the cache to the file."""
    with gzip.open(CACHE_FILE, "wt") as f:
        json.dump(cache, f, indent=4)


class CiteNode(nodes.General, nodes.Element):
    """Custom node to render DBLP citation."""
    pass


class DblpDirective(Directive):
    """Directive to fetch and display a citation from DBLP using a DBLP ID."""
    has_content = False
    required_arguments = 1  # The DBLP ID must be provided as an argument.

    def run(self):
        dblp_id = self.arguments[0]
        cache = load_cache()

        if dblp_id in cache:
            bibtex_entry_short, bibtex_entry_full = cache[dblp_id]
        else:
            # Fetch BibTeX entry from DBLP
            try:
                response = requests.get(f"https://dblp.org/rec/{dblp_id}.bib?param=0")
                response.raise_for_status()
                bibtex_entry_short = response.text
                response = requests.get(f"https://dblp.org/rec/{dblp_id}.bib?param=1")
                response.raise_for_status()
                bibtex_entry_full = response.text
                cache[dblp_id] = [bibtex_entry_short, bibtex_entry_full]  # Save to cache
                save_cache(cache)  # Persist the cache
            except requests.RequestException as e:
                error_msg = f"Failed to fetch BibTeX for DBLP ID '{dblp_id}': {str(e)}"
                return [nodes.error(None, nodes.Text(error_msg))]

        citation_id = f'dblp:{dblp_id}'
        citation = self.parse_bibtex_citation(bibtex_entry_short)
        link = self.parse_bibtex_link(bibtex_entry_full)
        bibtex = bibtex_entry_full

        # Create a custom node to hold the BibTeX content and display data
        node = CiteNode()
        node['id'] = citation_id
        node['citation'] = citation
        node['link'] = link
        node['bibtex'] = bibtex

        env = self.state.document.settings.env
        if not hasattr(env, "citations"):
            env.citations = {}
        if citation_id not in env.citations:
            env.citations[citation_id] = {
                'id': citation_id,
                'citation': citation,
                'link': link,
                'docnames': []
            }
        env.citations[citation_id]['docnames'].append(env.docname)

        return [node]

    def parse_bibtex_citation(self, bibtex_entry):
        """Parse BibTeX to extract authors, title, and PDF link."""
        res = bibtexparser.loads(bibtex_entry).entries[0]
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

    def parse_bibtex_link(self, bibtex_entry):
        res = bibtexparser.loads(bibtex_entry).entries[0]
        return latex_converter.latex_to_text(res.get('url', ''))

def visit_cite_node_html(self, node):
    """HTML visitor to render the DblpNode."""
    citation = node['citation']
    link = node['link']
    bibtex = node['bibtex']

    self.body.append(f'''
<div class="admonition">
  <p class="admonition-title">Citation</p>
  <details class="dblp-citation">
    <summary><b>{citation}</b> <a href="{link}" target="_blank">[link]</a></summary>
    <pre class="bibtex">{bibtex.strip()}</pre>
  </details>
</div>
    ''')


def depart_cite_node_html(self, node):
    """HTML departure for DblpNode."""
    pass


def setup(app):
    app.add_node(CiteNode, html=(visit_cite_node_html, depart_cite_node_html))
    app.add_directive('cite:dblp', DblpDirective)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
