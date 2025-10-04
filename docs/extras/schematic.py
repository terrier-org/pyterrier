import ast
import pyterrier as pt
from docutils import nodes
from docutils.parsers.rst import Directive


def run_and_return_last(code: str, globals_dict=None, locals_dict=None, source=None, lineno=None):
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = globals_dict

    # Parse the code into an AST
    parsed = ast.parse(code, mode="exec")

    if source is not None:
        source = source+f":{lineno}"

    # If the last node is an expression, we separate it
    *body, last = parsed.body
    try:
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
    except Exception as e:
        if source is None:
            raise RuntimeError(f"Error executing code: {code} because {e.args}") from e
        raise RuntimeError(f"Error executing code: {code} from {source} because {e.args}") from e


class SchematicDirective(Directive):
    required_arguments = 0
    has_content = True
    option_spec = {'input_columns': str}

    def run(self):
        # File name of the current .rst document
        source = self.reporter.source  # absolute path to the .rst file

        # Line number where the directive was invoked
        lineno = self.state_machine.abs_line_number()  # Directive base class stores this

        code = "\n".join(self.content)
        result = run_and_return_last(code, {'pt': pt}, {}, source=source, lineno=lineno)
        if not isinstance(result, (dict, pt.Transformer)):
            return [self.state_machine.reporter.error(f"Expected dict or Transformer, got {result!r} (type: {type(result)})", line=self.lineno)]
        # parse any input columns supplied in the directive 
        input_columns = self.options.get('input_columns')
        if input_columns is not None:
            input_columns = [x.strip() for x in input_columns.split(',')]

        html = pt.schematic.draw(result, input_columns=input_columns)
        return [nodes.raw('', html, format='html')]


def setup(app):
    app.add_directive('schematic', SchematicDirective)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
