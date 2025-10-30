from trie import RadixTree
def test_tree1():
    radix_tree = RadixTree()
    pipeline = ["A"]
    radix_tree.insert_all(pipeline)
    tree = radix_tree.describe_tree_structure()
    inner = tree[0]
    node, eval_index, children = inner
    assert eval_index == 0
    assert children == {}  # No children for single node

def test_tree2():
    radix_tree = RadixTree()
    pipeline = ["AB", "AC"]
    radix_tree.insert_all(pipeline)
    tree = radix_tree.describe_tree_structure()
    inner = tree[0]
    node, eval_index, children = inner
    assert eval_index is None
    assert len(children) == 2
    assert children[0][0] == 'B'
    assert children[1][0] == 'C'
    assert children[0][1] == 0
    assert children[1][1] == 1

def test_tree3():
    radix_tree = RadixTree()
    pipeline = ["ABC", "ABD"]
    radix_tree.insert_all(pipeline)
    tree = radix_tree.describe_tree_structure()
    d_node_1 = tree[0]
    node, eval_index, children = d_node_1 
    assert node == 'AB'
    assert eval_index is None
    assert len(children) == 2
    assert children[0][0] == 'C'
    assert children[1][0] == 'D'
    assert children[0][1] == 0
    assert children[1][1] == 1

def test_tree4():
    radix_tree = RadixTree()
    pipeline = ["AB", "ABC", "D"]
    radix_tree.insert_all(pipeline)
    tree = radix_tree.describe_tree_structure()
    d_node_1 = tree[0]
    d_node_2 = tree[1]
    node_1, eval_index_1, children_1 = d_node_1 
    node_2, eval_index_2, children_2 = d_node_2
    assert node_1 == 'AB'
    assert node_2 == 'D'
    assert eval_index_1 == 0
    assert eval_index_2 == 2
    assert len(children_1) == 1
    assert len(children_2) == 0
    assert children_1[0][0] == 'C'
    assert children_1[0][1] == 1
    assert len(children_1[0][2]) == 0
