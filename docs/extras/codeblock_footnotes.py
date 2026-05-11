from docutils import nodes
from sphinx.directives.code import CodeBlock
from sphinx.application import Sphinx
from docutils.statemachine import StringList
import re


class CodeBlockFootnotes(CodeBlock):
    """
    Enhanced code-block directive that extracts footnotes from comments.
    """
    
    def run(self):
        # Get the code content
        code = '\n'.join(self.content)
        
        # Find all footnotes in the code
        footnotes = []
        footnote_pattern = r'#\s*:footnote:\s*(.+?)(?=\n|$)'
        
        def replace_footnote(match):
            footnote_text = match.group(1).strip()
            footnote_num = len(footnotes) + 1
            if ':nocomment:' in footnote_text:
                footnote_text = footnote_text.replace(':nocomment:', '').strip()
                comment = ''
            else:
                comment = f'# [{footnote_num}]'
            footnotes.append(footnote_text)
            return comment
        
        # Replace footnotes in code with numbered references
        modified_code = re.sub(footnote_pattern, replace_footnote, code)
        
        # Update the content with modified code
        self.content = modified_code.split('\n')
        
        # Run the parent CodeBlock directive to create the code block
        code_nodes = super().run()
        
        # If there are footnotes, add them after the code block
        if footnotes:
            # Create a container for the code block and footnotes
            container = nodes.container()
            container += code_nodes
            
            # Create footnote list
            footnote_list = nodes.enumerated_list()
            footnote_list['classes'].append('code-footnotes')
            
            for footnote_text in footnotes:
                list_item = nodes.list_item()
                
                # Parse the footnote text as reStructuredText
                self.state.nested_parse(
                    StringList([footnote_text], source=''),
                    self.content_offset,
                    list_item
                )
                
                footnote_list += list_item
            
            container += footnote_list
            
            return [container]
        
        return code_nodes


def setup(app: Sphinx):
    """
    Setup function for the Sphinx extension.
    """
    # Override the default code-block directive
    app.add_directive('code-block', CodeBlockFootnotes, override=True)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
