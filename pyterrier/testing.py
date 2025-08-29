""" This module contains utilities for testing PyTerrier components and functionalities."""
import os
import unittest
import warnings

import pyterrier as pt

class TransformerTestCase(unittest.TestCase):
    """A base test case for testing PyTerrier transformers."""

    @classmethod
    def setUpClass(cls):
        cls.transformer = cls.get_transformer()
        cls.io_configs = []

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('PT_TEST_TRANSFORMER_REPORTS', '1') == '1':
            if cls.io_configs:
                io_configs = "\n".join([f"    - {' '.join(inp)} -> {' '.join(out)}" for inp, out in cls.io_configs])
            else:
                io_configs = '    [Unavailable (see test failures)]'
            warnings.warn('You can hide TransfomerTestCase reports by setting PT_TEST_TRANSFORMER_REPORTS=0')
            warnings.warn(f'''INFO: TransfomerTestCase Report for {str(cls.transformer)}
  Input/Output Configurations:
{io_configs}
            ''')
        cls.transformer = None

    @staticmethod
    def get_transformer():
        """Return the transformer instance to be tested."""
        raise NotImplementedError()

    def test_inspect_io(self):
        inputs = []
        with self.subTest("pt.inspect.transformer_inputs"):
            inputs = pt.inspect.transformer_inputs(self.transformer)
            self.assertIsInstance(inputs, list, "transformer_inputs not a list")
            if len(inputs) > 0:
                self.assertTrue(all(isinstance(i, list) for i in inputs), "transformer_inputs not a list of lists")
                self.assertTrue(all(isinstance(x, str) for i in inputs for x in i), "transformer_inputs not a list of list of strings")
            else:
                warnings.warn('Transformer does not support any input configurations. Are you sure this is right?')

        for input_config in inputs:
            with self.subTest(**{'pt.inspect.transformer_outputs': {'input_columns': input_config}}):
                outputs = pt.inspect.transformer_outputs(self.transformer, input_columns=input_config)
                self.assertIsInstance(outputs, list, "transformer_outputs not a list")
                self.assertTrue(all(isinstance(o, str) for o in outputs), "transformer_outputs not a list of strings")
                self.io_configs.append((input_config, outputs))

    # Disable for now:
    # def test_inspect_attr(self):
    #     with self.subTest('pt.inspect.transformer_attributes'):
    #         attrs = pt.inspect.transformer_attributes(self.transformer)
    #         self.assertIsInstance(attrs, list, "transformer_attributes not a list")
    #         self.assertTrue(all(isinstance(a, pt.inspect.TransformerAttribute) for a in attrs), "transformer_attributes not a list of TransformerAttribute")
    #         self.attrs.extend([a.name for a in attrs])

    #     with self.subTest('pt.inspect.transformer_apply_attributes'):
    #         new_transformer = pt.inspect.transformer_apply_attributes(self.transformer)
    #         self.assertFalse(self.transformer is new_transformer, "transformer_apply_attributes returned the same instance")

    #     with self.subTest('pt.inspect.subtransformers'):
    #         subtransformers = pt.inspect.subtransformers(self.transformer)
    #         self.assertIsInstance(subtransformers, dict, "subtransformers not a dict")

    def test_schematic(self):
        schematic = None
        with self.subTest('pt.schematic.transformer_schematic'):
            schematic = pt.schematic.transformer_schematic(self.transformer)
            if schematic['type'] == 'transformer' and len(schematic['label']) > 20:
                self.fail(f"Schematic label ({schematic['label']}) is long ({len(schematic['label'])} characters). This may affect rendering. Try to keep it below 20 characters.")

        if schematic is not None:
            with self.subTest('pt.schematic.draw'):
                schematic = pt.schematic.draw(schematic)


def transformer_test_class(fn):
    class _TransformerTestCase(TransformerTestCase):
        @staticmethod
        def get_transformer():
            return fn()
    return _TransformerTestCase
