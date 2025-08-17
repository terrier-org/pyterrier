import ast
import pyterrier as pt
import pyterrier_alpha as pta
from docutils import nodes
from docutils.parsers.rst import Directive


def run_and_return_last(code: str, globals_dict=None, locals_dict=None):
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = globals_dict

    # Parse the code into an AST
    parsed = ast.parse(code, mode="exec")

    # If the last node is an expression, we separate it
    *body, last = parsed.body
    if isinstance(last, ast.Expr):
        # Make a new module for the body and compile + exec it
        module_body = ast.Module(body=body, type_ignores=[])
        compiled_body = compile(module_body, filename="<input>", mode="exec")
        exec(compiled_body, globals_dict, locals_dict)

        # Compile just the expression part and eval it
        expr = ast.Expression(last.value)
        compiled_expr = compile(expr, filename="<input>", mode="eval")
        return eval(compiled_expr, globals_dict, locals_dict)
    else:
        # No final expression -> just execute the whole thing
        exec(compile(parsed, filename="<input>", mode="exec"), globals_dict, locals_dict)
        return None


class SchematicDirective(Directive):
    required_arguments = 0
    has_content = True
    option_spec = {'input_columns': str}

    def run(self):
        code = "\n".join(self.content)
        result = run_and_return_last(code, {'pt': pt}, {})
        if not isinstance(result, (dict, pt.Transformer)):
            return [self.state_machine.reporter.error(f"Expected dict or Transformer, got {result!r} (type: {type(result)})", line=self.lineno)]
        html = pta.schematic.draw(result)
        return [nodes.raw('', html, format='html')]


def setup(app):
    app.add_directive('schematic', SchematicDirective)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
