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
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            has_java_anno, hard_require = self._has_java_required(node)
            java_usage = None
            if isinstance(node, ast.FunctionDef):
                for child in node.body:
                    java_usage = self._uses_java(child)
                    if java_usage: break
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        for child2 in child.body:
                            java_usage = self._uses_java(child2)
                            if java_usage: break
                    else:
                        java_usage = self._uses_java(child)
                    if java_usage: break
            if not has_java_anno and java_usage and isinstance(node, ast.FunctionDef):
                text = self._missing_anno_tmpl.format('.'.join(context + [node.name]))
                yield java_usage.lineno, java_usage.col_offset, text, type(self)
            elif has_java_anno and not java_usage and hard_require:
                text = self._extra_anno_tmpl.format('.'.join(context + [node.name]))
                yield node.lineno, node.col_offset, text, type(self)
            if has_java_anno and isinstance(node, ast.ClassDef):
                # simulate @required annotations on the methods defined by this class
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        child.decorator_list.append(ast.Name('__required__soft'))

        if isinstance(node, ast.FunctionDef):
            context = context + [node.name]

        if isinstance(node, ast.ClassDef):
            context = context + [node.name]

        # we probably don't need such an extensive list when searching for functions/classes
        for attr in ['body', 'value', 'values', 'target', 'targets', 'left', 'right', 'elt', 'elts', 'keys', 'values', 'test', 'orelse', 'generators', 'func', 'args', 'keywords', 'format_spec']:
            if hasattr(node, attr):
                value = getattr(node, attr)
                if value is not None:
                    if hasattr(value, '__len__'):
                        for child in value:
                            yield from self._run(child, context)
                    else:
                        yield from self._run(value, context)

    def _has_java_required(self, function: ast.FunctionDef) -> Tuple[bool, bool]:
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ('required', 'required_raise'):
                    return True, True
                if decorator.id == '__required__soft':
                    return True, False
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in ('required', 'required_raise'):
                    return True, True
                if decorator.attr == '__required__soft':
                    return True, False
        return False, False

    def _uses_java(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Name):
            if node.id in ('autoclass', 'cast', 'J'):
                return node
        elif isinstance(node, ast.Attribute):
            if node.attr in ('autoclass', 'cast', 'J'):
                return node
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name == 'jnius':
                    return node
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'jnius':
                return node
        elif isinstance(node, ast.ClassDef):
            return None # break search at class definintions
        elif isinstance(node, ast.FunctionDef):
            return None # break search at function definintions

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
