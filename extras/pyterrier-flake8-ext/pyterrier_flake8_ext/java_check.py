import ast
from typing import Generator, Tuple, Type, Any, List, Optional
import pyterrier_flake8_ext


class JavaCheck:
    """McCabe cyclomatic complexity checker."""
    name = 'javacheck'
    version = pyterrier_flake8_ext.__version__
    _missing_anno_tmpl = "PT100 {} uses java but is not annotated with @pt.java.required"
    _extra_anno_tmpl = "PT101 {} is annotated with @pt.java.required but doesn't use java"

    def __init__(self, tree: ast.AST):
        self.tree = tree

    def run(self) -> Generator[Tuple[int, int, str, Type[Any]], None, None]:
        yield from self._run(self.tree)

    def _run(self, node: ast.AST, context: List[str] = []) -> Generator[Tuple[int, int, str, Type[Any]], None, None]:
        if isinstance(node, ast.FunctionDef):
            java_usage = self._uses_java(node)
            has_java_anno = self._has_java_required(node)
            if not has_java_anno and java_usage:
                text = self._missing_anno_tmpl.format('.'.join(context + [node.name]))
                yield java_usage.lineno, java_usage.col_offset, text, type(self)
            elif has_java_anno and not java_usage:
                text = self._extra_anno_tmpl.format('.'.join(context + [node.name]))
                yield node.lineno, node.col_offset, text, type(self)

        if isinstance(node, ast.FunctionDef):
            context = context + [node.name]

        if isinstance(node, ast.ClassDef):
            context = context + [node.name]

        # we probably don't need such an extensive list when searching for functions
        for attr in ['body', 'value', 'values', 'target', 'targets', 'left', 'right', 'elt', 'elts', 'keys', 'values', 'test', 'orelse', 'generators', 'func', 'args', 'keywords', 'format_spec']:
            if hasattr(node, attr):
                value = getattr(node, attr)
                if value is not None:
                    if hasattr(value, '__len__'):
                        for child in value:
                            yield from self._run(child, context)
                    else:
                        yield from self._run(value, context)

    def _has_java_required(self, function: ast.FunctionDef) -> bool:
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == 'required':
                    return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr == 'required':
                    return True
        return False

    def _uses_java(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Name):
            if node.id in ('java', 'J'):
                return node
        elif isinstance(node, ast.Attribute):
            if node.attr in ('java', 'J'):
                return node

        for attr in ['body', 'value', 'values', 'target', 'targets', 'left', 'right', 'elt', 'elts', 'keys', 'values', 'test', 'orelse', 'generators', 'func', 'args', 'keywords', 'format_spec']:
            if hasattr(node, attr):
                value = getattr(node, attr)
                if value is not None:
                    if hasattr(value, '__len__'):
                        for child in value:
                            res = self._uses_java(child)
                            if res:
                                return res
                    else:
                        res = self._uses_java(value)
                        if res:
                            return res
        return None
