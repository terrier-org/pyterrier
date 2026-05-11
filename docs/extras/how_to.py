"""
Sphinx extension for creating how-to sections with automatic indexing.

This extension provides a `how-to` directive that formats how-to entries
and automatically generates an index of all how-tos in the documentation.
"""
import uuid
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective
from sphinx.domains.std import StandardDomain
from sphinx.util import logging

logger = logging.getLogger(__name__)


class HowToDirective(SphinxDirective):
    """
    Directive for creating how-to sections.
    
    Usage:
        .. how-to:: Title of the how-to
           
           Content goes here (parsed as RST)
    """
    
    has_content = True
    required_arguments = 1  # The title
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    
    def run(self):
        env = self.env
        
        # Initialize howto_list if it doesn't exist
        if not hasattr(env, 'howto_list'):
            env.howto_list = {}
        
        # Get the title from arguments
        title = ' '.join(self.arguments)
        label = uuid.uuid4().hex
        
        if env.docname not in env.howto_list:
            # Get the actual page title from the document metadata
            titles = []
            node = self.state.parent
            while node:
                if isinstance(node, nodes.section):
                    # Get section ID
                    section_id = (node.get('ids') or [None])[0]
                    
                    # Get section title
                    title_node = node.next_node(nodes.title)
                    section_title = title_node.astext() if title_node else 'Unknown Section'
                    titles.append(section_title)
                node = node.parent
            
            env.howto_list[env.docname] = {
                'docname': env.docname,
                'title': titles[-1] if len(titles) > 0 else env.docname,
                'how_tos': [],
            }
        env.howto_list[env.docname]['how_tos'].append({
            'title': title,
            'label': label,
        })
        
        # Create the output nodes
        result_nodes = []
        
        # Add horizontal rule
        hr_node = nodes.transition()
        result_nodes.append(hr_node)
        
        target_node = nodes.target('', '', ids=[label], names=[label])
        self.state.document.note_explicit_target(target_node)
        result_nodes.append(target_node)

        # Add title as a section
        section_node = nodes.section()
        section_node['classes'].append('howto-section')
        
        # Create title node with proper formatting
        title_nodes, messages = self.state.inline_text(title, self.lineno)
        title_node = nodes.title(title, '', *title_nodes)
        section_node += title_node
        
        self.state.nested_parse(self.content, self.content_offset, section_node)
        
        result_nodes.append(section_node)
        
        return result_nodes


def init_howto_list(app, env):
    """Initialize the howto_list if it doesn't exist."""
    if not hasattr(env, 'howto_list'):
        env.howto_list = {}
    return []


def merge_howto_list(app, env, docnames, other):
    """Merge howto_list from parallel builds."""
    if not hasattr(env, 'howto_list'):
        env.howto_list = {}
    if hasattr(other, 'howto_list'):
        env.howto_list.update(other.howto_list)


def purge_howtos(app, env, docname):
    """Remove how-tos for a document that's being rebuilt."""
    if hasattr(env, 'howto_list') and docname in env.howto_list:
        del env.howto_list[docname]


def collect_howtos(app):
    """Collect all how-tos and generate the index page."""
    env = app.env
    
    if app.builder.embedded or not hasattr(env, 'howto_list') or len(env.howto_list) == 0:
        # Building embedded (e.g. htmlhelp or ePub) or there were no howtos found
        return []

    pages = []
    for page in sorted(env.howto_list.values(), key=lambda x: x['title']):
        # Generate URLs for each how-to
        for howto in sorted(page['how_tos'], key=lambda x: x['title']):
            # Use the document name to get the URI, then append the anchor
            howto['url'] = app.builder.get_target_uri(page['docname']) + '#' + howto['label']
        pages.append(page)

    return [('how-tos', {'pages': pages}, 'how-tos.html')]


def setup(app):
    """Setup function for the Sphinx extension."""
    app.add_directive('how-to', HowToDirective)
    
    # Connect to proper lifecycle events for persistence
    app.connect('env-get-updated', init_howto_list)
    app.connect('env-purge-doc', purge_howtos)
    app.connect('env-merge-info', merge_howto_list)
    app.connect('html-collect-pages', collect_howtos)
    
    StandardDomain._virtual_doc_names['how-tos'] = ('how-tos', "How-To Guides")
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
