import pyterrier as pt
import pandas as pd
import io
import contextlib
from .base_experiment import TestExperimentBase


class TestExperimentTree(TestExperimentBase):
    """Test suite for PyTerrier Experiment functionality specific to tree execution plans.
    
    This class is designed to hold tests that are specific to tree-based execution
    plans. Initially empty of tree-specific overrides, but ready for tree-specific 
    test cases to be added as the tree execution plan implementation develops.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_exp_kwargs = {'plan': 'tree'}

    def test_save_invalid_format(self):
        self.skipTest("Skipping test_save_invalid_format for tree execution plan; not yet implemented.")
    
    def test_save(self):
        self.skipTest("Skipping test_save for tree execution plan; not yet implemented.")

    def test_save_csv(self):
        self.skipTest("Skipping test_save_csv for tree execution plan; not yet implemented.")
    
    def test_verbose_pretty_print(self):
        topics = pd.DataFrame([
            ["q1", "hello world"],
        ], columns=["qid", "query"])
        qrels = pd.DataFrame([
            ["q1", "d1", 1],
        ], columns=["qid", "docno", "label"])

        res1 = pd.DataFrame([
            ["q1", "d1", 1.0, 0],
        ], columns=["qid", "docno", "score", "rank"])
        res2 = pd.DataFrame([
            ["q1", "d1", 0.8, 0],
        ], columns=["qid", "docno", "score", "rank"])

        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            pt.Experiment(
                [res1, res2],
                topics,
                qrels,
                eval_metrics=["map"],
                verbose=True,
                **self.pt_exp_kwargs)

        output = captured_stdout.getvalue()
        self.assertIn("Root", output)
