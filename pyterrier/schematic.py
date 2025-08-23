from copy import copy
import html
import uuid
from importlib import resources
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import numpy as np
import pyterrier as pt
import re

def _apply_default_schematic(schematic: Dict[str, Any], transformer: pt.Transformer, *, input_columns: Optional[List[str]] = None):
    assert schematic.setdefault('type', 'transformer') == 'transformer'

    if 'label' not in schematic:
        label = transformer.__class__.__name__
        if label.endswith('Transformer'):
            schematic['label'] = label[:-len('Transformer')]
        schematic['label'] = label

    if 'class_name' not in schematic:
        name = f'{transformer.__class__.__module__}.{transformer.__class__.__name__}'
        if name.startswith('pyterrier.'):
            name = 'pt.' + name[len('pyterrier.'):]
        schematic['name'] = name

    if 'input_columns' not in schematic or 'output_columns' not in schematic:
        if 'input_columns' not in schematic:
            schematic['input_columns'] = input_columns
        if 'output_columns' not in schematic:
            if input_columns is None:
                schematic['output_columns'] = None
            else:
                try:
                    schematic['output_columns'] = pt.inspect.transformer_outputs(transformer, input_columns)
                except pt.validate.InputValidationError as e:
                    schematic['output_columns'] = None
                    schematic['input_validation_error'] = e
                except pt.inspect.InspectError:
                    schematic['output_columns'] = None

    default_settings_applied = False
    if 'settings' not in schematic:
        default_settings_applied = True
        try:
            schematic['settings'] = {attr.name: attr.value for attr in pt.inspect.transformer_attributes(transformer)}
        except pt.inspect.InspectError:
            schematic['settings'] = {}

    if 'help_url' not in schematic:
        schematic['help_url'] = pt.documentation.url_for_class(transformer)

    if 'inner_pipelines' not in schematic:
        try:
            subtransformers = pt.inspect.subtransformers(transformer)
        except pt.inspect.InspectError:
            subtransformers = {}
        if subtransformers:
            subtransformer_inputs = input_columns or _INFER
            if schematic.get('inner_pipelines', 'unlinked') == 'unlinked':
                subtransformer_inputs = _INFER
            pipelines = []
            pipeline_labels = []
            for key, value in subtransformers.items():
                if default_settings_applied and key in schematic['settings']:
                    del schematic['settings'][key]
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        pipelines.append(transformer_schematic(v, input_columns=subtransformer_inputs))
                        pipeline_labels.append(f'{key}[{i}]')
                else:
                    pipelines.append(transformer_schematic(value, input_columns=subtransformer_inputs))
                    pipeline_labels.append(key)
            schematic['inner_pipelines'] = pipelines
            schematic['inner_pipelines_labels'] = pipeline_labels

    if schematic.get('inner_pipelines') and 'inner_pipelines_mode' not in schematic:
        schematic['inner_pipelines_mode'] = 'unlinked'


_INFER = object()
def transformer_schematic(
    transformer: pt.Transformer,
    *,
    input_columns: Optional[List[str]] = _INFER,
    default: bool = False,
) -> dict:
    """Builds a structured schematic of the transformer."""
    if input_columns is _INFER:
        all_input_column_configs = pt.inspect.transformer_inputs(transformer, strict=False)
        if all_input_column_configs is not None and len(all_input_column_configs) > 0:
            input_columns = all_input_column_configs[0] # pick the first one
        else:
            input_columns = None
    if not default and isinstance(transformer, HasSchematic):
        if callable(transformer.schematic):
            schematic = transformer.schematic(input_columns=input_columns)
        else:
            schematic = transformer.schematic
        schematic = copy(schematic) # we don't want to accidently modify the original schematic
        if 'type' in schematic:
            return schematic
    else:
        schematic = {
            'type': 'transformer',
        }
    _apply_default_schematic(schematic, transformer, input_columns=input_columns)
    return schematic

# Tools for converting the schematic diagrams to html
_css = None
_js = None
def _get_schematic_css_js(container_id):
    global _css, _js
    if _css is None or _js is None:
        _css = (resources.files('pyterrier') / 'data/schematic.css').read_text()
    if _js is None:
        _js = (resources.files('pyterrier') / 'data/schematic.js').read_text()
    css = _css.replace('#ID', f'#{container_id}')
    js = _js.replace('#ID', f'#{container_id}')
    return css, js


def draw(transformer: Union[pt.Transformer, dict], *, outer_class: Optional[str] = None) -> str:
    """Draws a transformer as an HTML schematic.

    If the transformer is already a ``SchematicDict``, it will be drawn directly.
    Otherwise, it will first convert the transformer to a structured schematic using :func:`transformer_schematic`,
    and draw that.

    Args:
        transformer: The transformer to draw, or a dict in ``SchematicDict`` format.
        outer_class: An optional CSS class to apply to the outer container of the schematic.

    Returns:
        An HTML string representing the schematic of the transformer.
    """
    if isinstance(transformer, dict):
        schematic = transformer
    else:
        schematic = transformer_schematic(transformer)
    return draw_html_schematic(schematic, outer_class=outer_class)


def draw_html_schematic(schematic: dict, *, outer_class: Optional[str] = None) -> str:
    """Draws a structured schematic as an HTML representation."""
    uid = str(uuid.uuid4())
    css, js = _get_schematic_css_js(f'id-{uid}')
    return f'''
    <div id="id-{uid}" class="{outer_class or ''}">
        <style>{css}</style>
        {_draw_html_schematic(schematic)}
        <div class="infobox">
            <div class="infobox-title"></div>
            <div class="infobox-body"></div>
        </div>
        <script>{js}</script>
    </div>
    '''


def _draw_html_schematic(schematic: dict, *, mode: str = 'outer') -> str:
    if schematic['type'] == 'transformer':
        return _draw_html_schematic({
            'type': 'pipeline',
            'input_columns': schematic.get('input_columns'),
            'output_columns': schematic.get('output_columns'),
            'title': None,
            'transformers': [schematic],
        }, mode=mode)
    if schematic['type'] == 'pipeline':
        result = '<div class="pipeline">'
        if mode == 'outer':
            clz = 'arr' if schematic['transformers'][0].get('inner_pipelines_mode') != 'combine' else ''
            result += '<div class="io-label">Input</div>'
            result += f'<div class="hline {clz} arr-input">{_draw_df_html(schematic["input_columns"])}</div>'
        elif mode == 'inner_linked':
            result += '<div class="hline arr arr-inner" style="width: 16px;"></div>'
        elif mode == 'inner_labeled':
            result += f'<div class="hline arr arr-input">{_draw_df_html(schematic["input_columns"])}</div>'
        columns = schematic["input_columns"]
        for i, record in enumerate(schematic['transformers']):
            assert record['type'] == 'transformer'
            assert record['input_columns'] == columns
            uid = str(uuid.uuid4())
            infobox = ''
            infobox_attr = ''
            error_cls = ''
            if record.get('settings') or record.get('name') or record.get('input_validation_error'):
                help_url = record.get('help_url') or ''
                name = record.get('name') or ''
                attrs = ''
                error_info = ''
                if record.get('input_validation_error'):
                    modes = record['input_validation_error'].modes
                    error_cls = 'input-validation-error'
                    if len(modes) == 1:
                        # Normal case: there's just one mode
                        error_info = '<div class="infobox-error">'
                        if len(modes[0].missing_columns) > 0:
                            error_info += f'Missing input columns: {", ".join(["<b>" + html.escape(c) + "</b>" for c in modes[0].missing_columns])}. '
                        if len(modes[0].extra_columns) > 0:
                            error_info += f'Unexpected input columns: {", ".join(["<b>" + html.escape(c) + "</b>" for c in modes[0].extra_columns])}. '
                        error_info += '</div>'
                    else:
                        error_info = '<div class="infobox-error"><div>None of the supported input modes matched:</div><ul>'
                        for i, error_mode in enumerate(modes):
                            error_info += f'<li>Mode {html.escape(error_mode.mode_name or str(i+1))}: '
                            if len(error_mode.missing_columns) > 0:
                                error_info += f'Missing input columns: {", ".join(["<b>" + html.escape(c) + "</b>" for c in error_mode.missing_columns])}. '
                            if len(error_mode.extra_columns) > 0:
                                error_info += f'Unexpected input columns: {", ".join(["<b>" + html.escape(c) + "</b>" for c in error_mode.extra_columns])}. '
                            error_info += '</li>'
                        error_info += '</ul></div>'
                if record['settings']:
                    attr_rows = []
                    for key, value in record['settings'].items():
                        attr_rows.append(f'<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>')
                    attrs = f'<table class="df-columns">{"".join(attr_rows)}</table>'
                infobox = f'''
                <div class="infobox-item" id="id-{uid}" data-title="Transformer">
                    <div style="font-family: monospace; padding: 3px 6px;">
                        {'<a href="' + html.escape(help_url) + '" target="_blank" onclick="window.event.stopPropagation();">' if help_url else ''}
                            {html.escape(name)}
                        {'</a>' if help_url else ''}
                    </div>
                    {error_info}
                    {attrs}
                </div>
                '''
                infobox_attr = f'data-infobox="id-{uid}"'
            if 'inner_pipelines' in record:
                if record['inner_pipelines_mode'] == 'linked':
                    pipelines = ''
                    for i, pipeline in enumerate(record['inner_pipelines']):
                        pipelines += '<div class="parallel-item"><div class="vline"></div>' + _draw_html_schematic(pipeline, mode='inner_linked') + '<div class="vline"></div></div>'
                    result += f'''
                    <div class="transformer inner parallel-scaffold {error_cls}" {infobox_attr}>
                        {infobox}
                        <div class="hline"></div>
                        <div class="transformer-title">{html.escape(record.get("label") or "")}</div>
                        <div class="inner-schematic inner-linked">{pipelines}</div>
                        <div class="hline arr"></div>
                    </div>
                    '''
                elif record['inner_pipelines_mode'] == 'combine':
                    pipelines = ''
                    for i, pipeline in enumerate(record['inner_pipelines']):
                        pipelines += '<div class="parallel-item"><div class="vline"></div>' + _draw_html_schematic(pipeline, mode='inner_linked') + '<div class="vline"></div></div>'
                    result += f'''
                    <div class="combine-box">
                        <div class="parallel-scaffold inner">
                            <div class="hline"></div>
                            <div class="inner-schematic inner-linked">{pipelines}</div>
                            <div class="hline arr"></div>
                        </div>
                        <div class="transformer {error_cls}" {infobox_attr}>
                            {infobox}
                            <div class="transformer-title">{html.escape(record.get("label") or "")}</div>
                        </div>
                    </div>
                    '''
                elif record['inner_pipelines_mode'] == 'unlinked':
                    pipelines = ''
                    if len(record['inner_pipelines_labels']) == len(record['inner_pipelines']):
                        for label, pipeline in zip(record['inner_pipelines_labels'], record['inner_pipelines']):
                            pipelines += f'''
                            <div class="transformer-title">{html.escape(label)}</div>
                            <div class="inner-schematic inner-labeled">{_draw_html_schematic(pipeline, mode='inner_labeled')}</div>
                            '''
                    else:
                        for pipeline in record['inner_pipelines']:
                            pipelines += _draw_html_schematic(pipeline, mode='inner_labeled')
                    result += f'''
                    <div class="transformer inner {error_cls}" {infobox_attr}>
                        {infobox}
                        <div class="transformer-title">{html.escape(record.get("label") or "")}</div>
                        <div class="inner-schematic inner-labeled">{pipelines}</div>
                    </div>
                    '''
            else:
                result += f'''
                <div class="transformer {error_cls}" {infobox_attr}>
                    {infobox}
                    <div class="transformer-title">{html.escape(record.get("label") or "")}</div>
                </div>
                '''
            if i != len(schematic['transformers']) - 1:
                result += f'<div class="hline arr arr-inner">{_draw_df_html(record["output_columns"], record["input_columns"])}</div>'
            columns = record['output_columns']
        if mode == 'outer':
            result += f'<div class="hline arr arr-output">{_draw_df_html(schematic["output_columns"], schematic["transformers"][-1]["input_columns"])}</div>'
            result += '<div class="io-label">Output</div>'
        elif mode == 'inner_linked':
            result += f'<div class="hline" style="flex-grow: 1;">{_draw_df_html(schematic["output_columns"], schematic["transformers"][-1]["input_columns"])}</div>'
        elif mode == 'inner_labeled':
            result += f'<div class="hline arr arr-output">{_draw_df_html(schematic["output_columns"], schematic["transformers"][-1]["input_columns"])}</div>'
        result += '</div>'
        return result


def _draw_df_html(columns, prev_columns = None) -> str:
    """Draws a DataFrame as an HTML table."""
    df_class = ''
    if columns is None:
        columns = []
        df_class = ' df-alert'
    frame_info = pt.model.frame_info(columns) or {'label': '?', 'title': 'Unknown Frame'}
    df_label = frame_info['label'] 
    df_label_long = frame_info['title'] 
    # change underscore subscript into HTML subscript
    df_label = re.sub(r'_(\w+)', r'<sub>\1</sub>', df_label)
    uid = str(uuid.uuid4())
    if columns:
        column_rows = []
        for col in columns:
            col_info = pt.model.column_info(col) or {}
            col_desc = ''
            type_name = ''
            if 'type' in col_info:
                type_name = str(col_info['type'])
                if col_info['type'] == np.array:
                    type_name = 'np.array'
                elif hasattr(col_info['type'], '__name__'):
                    type_name = col_info['type'].__name__
                type_name = f'<span style="font-family: monospace;">{html.escape(type_name)}</span>'
            if 'phrase' in col_info:
                col_desc += f'<i>({html.escape(col_info["phrase"])})</i> '
            if 'short_desc' in col_info:
                col_desc += f'{html.escape(col_info["short_desc"])} '
            is_added = prev_columns and col not in prev_columns
            column_rows.append(f'''
                <tr class="{"add" if is_added else ""}">
                    <th>{html.escape(col)}</th>
                    <td>{type_name}</td>
                    <td>{col_desc}</td>
                </tr>
            ''')
        col_table = f'''
        <div id="id-{uid}" class="infobox-item" data-title="{df_label_long}">
            <table class="df-columns">
                {''.join(column_rows)}
            </table>
        </div>'''
    else:
        col_table = f'''
        <div id="id-{uid}" class="infobox-item" data-title="{df_label_long}">
            <div class="infobox-error">Unknown/incompatible columns</div>
        </div>'''
    return f'<div class="df {df_class}" data-infobox="id-{uid}">{df_label}{col_table}</div>'


@runtime_checkable
class HasSchematic(Protocol):
    """Protocol for transformers override details about their schematic representation.

    This is an optional extension interface to :class:`pyterrier.Transformer` that allows
    transformers to provide customizations to their schematics.
    """
    def schematic(self, *, input_columns: Optional[List[str]]) -> Dict[str, Any]:
        """Returns a structured schematic representation of the transformer.

        The schematic should be a dictionary that follows the structure defined in :ref:`pt.schematic <pyterrier.schematic>`.

        For ease of use, the method can optionally return only some of the fields of the schematic; any missing fields
        will be filled in with default values.

        It can also be implemented as an instance or class member when the values do not need to be computed on-the-fly (e.g.,
        overriding the schematic label). When ``schematic`` is not ``callable``, it uses its dict value directly as the schematic.

        Args:
            input_columns: The input columns of the transformer, used to determine schematic fields such as the output columns.

        Returns:
            A dictionary representing the schematic of the transformer, which will be used to draw the schematic diagram.
        """
