import unittest
import pandas as pd
import pyterrier as pt
from pyterrier._evaluation._trie import RadixTree, RadixNode, decompose_pipelines
from pyterrier._ops import Compose


class TestRadixTreeWithTransformers(unittest.TestCase):
    def test_tree1(self):
        radix_tree = RadixTree()
        pipeline = [["A"]]
        for sysid, p in enumerate(pipeline):
            radix_tree.insert(tuple(p), sysid)
        tree = radix_tree.describe_tree_structure()
        inner = tree[0]
        node, eval_index, children = inner
        assert eval_index == 0
        assert children == []  # No children for single node

    def test_tree2(self):
        radix_tree = RadixTree()
        pipeline = ["AB", "AC"]
        for sysid, p in enumerate(pipeline):
            radix_tree.insert(tuple(p), sysid)
        tree = radix_tree.describe_tree_structure()
        inner = tree[0]
        node, eval_index, children = inner
        self.assertIsNone(eval_index)
        self.assertEqual(len(children), 2)
        assert children[0][0] == ("B",)
        assert children[1][0] ==  ("C",)
        assert children[0][1] == 0
        assert children[1][1] ==  1

    def test_tree3(self):
        radix_tree = RadixTree()
        pipeline = ["ABC", "ABD"]
        for sysid,p in enumerate(pipeline):
            radix_tree.insert(tuple(p), sysid)
        tree = radix_tree.describe_tree_structure()
        d_node_1 = tree[0]
        print(d_node_1)
        node, eval_index, children = d_node_1
        print(children)
        self.assertEqual(node, ("A","B"))
        self.assertIsNone(eval_index)
        self.assertEqual(len(children), 2)
        assert children[0][0] == ("C",)
        assert children[1][0] == ("D",)
        assert children[0][1] == 0
        assert children[1][1] == 1

    def test_tree4(self):
        radix_tree = RadixTree()
        pipeline = ["AB", "ABC", "D"]
        for sysid, p in enumerate(pipeline):
            radix_tree.insert(tuple(p), sysid)
        tree = radix_tree.describe_tree_structure()
        d_node_1 = tree[0]
        d_node_2 = tree[1]
        node_1, eval_index_1, children_1 = d_node_1
        node_2, eval_index_2, children_2 = d_node_2
        self.assertEqual(node_1, ("A", "B"))
        self.assertEqual(node_2, ("D",))
        assert eval_index_1 == 0
        assert eval_index_2 == 2
        self.assertEqual(len(children_1), 1)
        self.assertEqual(len(children_2), 0)
        self.assertEqual(children_1[0][0], ("C",))
        assert children_1[0][1] == 1
        self.assertEqual(len(children_1[0][2]), 0)
 

    def test_empty_tree(self):
        tree = RadixTree()
        self.assertEqual(tree.describe_tree_structure(), [])

    def test_single_transformer_insertion(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        pipeline = [TF_IDF]
        tree.insert(tuple(pipeline), 0)
        structure = tree.describe_tree_structure()
        self.assertEqual(len(structure), 1)
        edge_label, value, children = structure[0]
        self.assertEqual(edge_label, tuple(pipeline))
        self.assertEqual(value, 0)
        self.assertEqual(children, [])
    
    def test_all_nodes_are_radix_node(self):
            radix_tree = RadixTree()
            vaswani = pt.datasets.get_dataset("vaswani")
            index = vaswani.get_index()
            TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
            BM25 = pt.terrier.Retriever(index, wmodel="BM25")
            PL2 = pt.terrier.Retriever(index, wmodel="PL2")
            pipelines = [TF_IDF>>BM25, TF_IDF>>PL2]
            for sysid, p in enumerate(decompose_pipelines(pipelines)):
                radix_tree.insert(tuple(p), sysid)

            def check_nodes(node):
                self.assertIsInstance(node, RadixNode)
                for child in node.children.values():
                    check_nodes(child)

            check_nodes(radix_tree.root)

    def test_two_pipelines_no_common_prefix(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        pipelines = [TF_IDF, BM25]
        for sysid, pipeline in enumerate(pipelines):
            tree.insert((pipeline,), sysid)
        structure = tree.describe_tree_structure()
        print(structure)
        self.assertEqual(len(structure), 2)
        edge_label_1, eval_1, children_1 = structure[0]
        edge_label_2, eval_2, children_2 = structure[1]
        self.assertEqual(edge_label_2, (TF_IDF,))
        self.assertEqual(edge_label_1, (BM25,))
        assert eval_2 == 0
        assert eval_1 == 1

    def test_pipelines_with_common_prefix(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        PL2 = pt.terrier.Retriever(index, wmodel="PL2")
        pipelines = [TF_IDF >> BM25, TF_IDF >> BM25%10 ]
        for sysid, pipeline in enumerate(pipelines):
            tree.insert(pipeline, sysid)
        structure = tree.describe_tree_structure()
        self.assertEqual(len(structure), 1)
        edge_label, value, children = structure[0]
        # The common prefix is the composed transformer TF_IDF >> BM25
        # self.assertEqual(edge_label, (TF_IDF, BM25))
        assert value == 0  # Not terminal
        self.assertEqual(len(children), 1)

    def test_prefix_pipeline_is_terminal(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        pipelines = [TF_IDF, TF_IDF >> BM25]
        for sysid, pipeline in enumerate(decompose_pipelines(pipelines)):
            tree.insert(tuple(pipeline), sysid)
        structure = tree.describe_tree_structure()
        edge_label, value, children = structure[0]
        self.assertEqual(edge_label, (TF_IDF,))
        assert value == 0
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0][0], (BM25,))
        assert children[0][1] == 1


    def test_traverse_single_transformer(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        pipeline = (TF_IDF,)
        tree.insert(pipeline, 0)
        test_df = pd.DataFrame({
            'qid': ['q1', 'q1'],
            'query': ['example query 1', 'example query 2'],

        })
        results = []
        def callback(res, sysid, time_ms):
            results.append((sysid, res))
        tree.root.traverse(test_df, callback)
        self.assertEqual(len(results), 1)
        sysid, result = results[0]
        self.assertEqual(sysid, 0)

    def test_traverse_shared_prefix(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        PL2 = pt.terrier.Retriever(index, wmodel="PL2")
        pipelines = [TF_IDF>>BM25, TF_IDF>>PL2]
        for sysid, pipeline in enumerate(decompose_pipelines(pipelines)):
            tree.insert(tuple(pipeline), sysid)
        test_df = pd.DataFrame({
            'qid': ['q1', 'q1'],
            'query': ['example query 1', 'example query 2'],
        })
        results = []
        def callback(res, sysid, time_ms):
            results.append((sysid, res))
        tree.root.traverse(test_df, callback)
        # Should have 2 results
        self.assertEqual(len(results), 2)
        sysids = [r[0] for r in results]
        assert sysids[0] ==0
        assert sysids[1] ==1

    def test_traverse_timing(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        pipeline = (TF_IDF, BM25)
        tree.insert(pipeline, 0)
        test_df = pd.DataFrame({
            'qid': ['q1', 'q1'],
            'query': ['example query 1', 'example query 2'],
        })
        results = []
        def callback(res, sysid, time_ms):
            results.append((sysid, time_ms))
        tree.root.traverse(test_df, callback)
        self.assertEqual(len(results), 1)
        sysid, time_ms = results[0]
        self.assertGreater(time_ms, 0)  

    def test_decompose_pipelines_single_transformer(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        pipelines = [TF_IDF]
        result = decompose_pipelines(pipelines)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [TF_IDF])

    def test_decompose_pipelines_multiple(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        PL2 = pt.terrier.Retriever(index, wmodel="PL2")
        pipelines = [TF_IDF >> BM25, TF_IDF >> PL2]
        result = decompose_pipelines(pipelines)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [TF_IDF, BM25])
        self.assertEqual(result[1], [TF_IDF, PL2])

    def test_decompose_pipelines_with_dataframe(self):
        test_df = pd.DataFrame({'qid': ['q1'], 'docno': ['d1'], 'score': [1.0]})
        pipelines = [test_df]
        result = decompose_pipelines(pipelines)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertIsInstance(result[0][0], pt.Transformer)

    def test_complex_pipeline_tree(self):
        tree = RadixTree()
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        PL2 = pt.terrier.Retriever(index, wmodel="PL2")
        pipelines = [
            TF_IDF>> BM25>> PL2, 
            TF_IDF>> PL2,       
            BM25               
        ]
        for sysid, pipeline in enumerate(decompose_pipelines(pipelines)):
            tree.insert(tuple(pipeline), sysid)
        structure = tree.describe_tree_structure()
        # Should have 2 top-level branches: TF_IDF and BM25
        self.assertEqual(len(structure), 2)


    def test_empty_pipeline_handling(self):
        tree = RadixTree()
        tree.insert([])
        self.assertEqual(tree.describe_tree_structure(), [])

    

    def test_ndcg_linear_equals_tree(self):
        vaswani = pt.datasets.get_dataset("vaswani")
        index = vaswani.get_index()
        TF_IDF = pt.terrier.Retriever(index, wmodel="TF_IDF")
        BM25 = pt.terrier.Retriever(index, wmodel="BM25")
        PL2 = pt.terrier.Retriever(index, wmodel="PL2")
        pipelines = [
            TF_IDF,
            BM25,
            PL2,
            TF_IDF >> BM25,
            BM25 >> PL2,
            TF_IDF >> BM25 >> PL2
        ]
        topics = vaswani.get_topics()
        qrels = vaswani.get_qrels()
        results_linear = pt.Experiment(
            pipelines, topics, qrels, ['ndcg'], plan="linear"
        )
        results_tree = pt.Experiment(
            pipelines, topics, qrels, ['ndcg'], plan="tree"
        )
        for ndcg_linear, ndcg_tree in zip(results_linear['ndcg'], results_tree['ndcg']):
            self.assertEqual(ndcg_linear, ndcg_tree)
            # self.assertAlmostEqual(ndcg_linear, ndcg_tree, places=6)
