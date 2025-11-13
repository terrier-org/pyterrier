from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util import logging
from sphinx import addnodes

logger = logging.getLogger(__name__)


class RelatedDirective(SphinxDirective):
    """
    Directive to mark related items.
    
    Usage:
        .. related:: module.Class.method
           :title: Optional custom title
    """
    required_arguments = 1
    optional_arguments = 0
    option_spec = {
        'title': directives.unchanged,
    }
    has_content = False

    def run(self):
        env = self.env
        target = self.arguments[0]
        
        # Get current document and section info
        docname = env.docname
        
        # Find the current section
        section_id, section_title, page_title = self._get_current_section()
        
        # Store the related reference for later processing
        if not hasattr(env, 'related_items'):
            env.related_items = {}
        
        if target not in env.related_items:
            env.related_items[target] = []
        
        env.related_items[target].append({
            'docname': docname,
            'section_id': section_id,
            'section_title': section_title,
            'page_title': page_title,
        })
        
        # Return empty list - don't output anything where the directive appears
        return []
    
    def _get_current_section(self):
        """Find the current section ID and title."""
        node = self.state.parent
        section_ids = []
        titles = []
        while node:
            if isinstance(node, nodes.section):
                # Get section ID
                section_id = node.get('ids', [None])[0]
                
                # Get section title
                title_node = node.next_node(nodes.title)
                section_title = title_node.astext() if title_node else 'Unknown Section'
                section_ids.append(section_id)
                titles.append(section_title)
            node = node.parent
        
        if section_ids:
            return section_ids[0], titles[0], titles[-1]
        return None, 'Top of Document'


def process_related_items(app, doctree, docname):
    """
    Process related items after doctree is resolved.
    This adds backlinks to the referenced documentation.
    """
    env = app.builder.env
    
    if not hasattr(env, 'related_items'):
        return
    
    # Check if current document is referenced by any related directives
    for obj in doctree.traverse(addnodes.desc):
        # Get the signature node
        sig = obj.next_node(addnodes.desc_signature)
        if not sig:
            continue
        
        # Get the full name of the documented object
        names = sig.get('ids', [])
        
        for name in names:
            if name in env.related_items:
                # Add a "Referenced in" section
                references = env.related_items[name]
                
                # Find the desc_content node (the <dd> element)
                desc_content = obj.next_node(addnodes.desc_content)
                if not desc_content:
                    continue
                
                # Create a seealso admonition
                seealso = nodes.admonition()
                seealso['classes'].append('seealso')
                seealso['classes'].append('related')
                
                # Add title
                title = nodes.title()
                title += nodes.Text('See also')
                seealso += title
                
                for i, ref in enumerate(references):
                    para = nodes.paragraph()
                    
                    # Create reference to the section
                    ref_text = ref['section_title']
                    
                    # Build proper cross-reference
                    if ref['section_id']:
                        refuri = app.builder.get_relative_uri(docname, ref['docname']) + '#' + ref['section_id']
                    else:
                        refuri = app.builder.get_relative_uri(docname, ref['docname'])
                    
                    if ref['page_title'] != ref_text:
                        para += nodes.Text(ref['page_title'] + ' » ')

                    ref_node = nodes.reference(
                        '',
                        ref_text,
                        internal=True,
                        refuri=refuri,
                    )
                    para += ref_node
                    
                    seealso += para
                
                # Append to the existing desc_content
                desc_content += seealso


def purge_related_items(app, env, docname):
    """Remove related items when a document is removed."""
    if not hasattr(env, 'related_items'):
        return
    
    # Remove entries that reference the removed document
    for target, refs in list(env.related_items.items()):
        env.related_items[target] = [
            ref for ref in refs if ref['docname'] != docname
        ]
        if not env.related_items[target]:
            del env.related_items[target]


def merge_related_items(app, env, docnames, other):
    """Merge related items from parallel builds."""
    if not hasattr(other, 'related_items'):
        return
    
    if not hasattr(env, 'related_items'):
        env.related_items = {}
    
    for target, refs in other.related_items.items():
        if target not in env.related_items:
            env.related_items[target] = []
        env.related_items[target].extend(refs)


def setup(app):
    """Setup the extension."""
    app.add_directive('related', RelatedDirective)
    app.connect('doctree-resolved', process_related_items)
    app.connect('env-purge-doc', purge_related_items)
    app.connect('env-merge-info', merge_related_items)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
