from graph import DAG
from prefixes import compute_final_prefixes



def test_dag_structure_simple():
    """Test that DAG creates the correct node structure with edges."""
    dag = DAG()
    
    # Add edges
    dag.add_edge("AB", "C")
    dag.add_edge("AB", "D")
    dag.add_edge("C", "E")
    
    # Get root nodes
    root_nodes = dag.get_root_nodes()
    
    # Should have exactly one root node
    assert len(root_nodes) == 1
    
    # The root should be "AB"
    root = root_nodes[0]
    assert root.me == "AB"
    
    # AB should have 2 children: C and D
    ab_children = root.get_children()
    assert len(ab_children) == 2
    ab_children_names = sorted([child.me for child in ab_children])
    assert ab_children_names == ["C", "D"]
    
    # Find C node among AB's children
    c_node = next(child for child in ab_children if child.me == "C")
    
    # C should have 1 child: E
    c_children = c_node.get_children()
    assert len(c_children) == 1
    assert c_children[0].me == "E"
    
    # E should have no children
    e_children = c_children[0].get_children()
    assert len(e_children) == 0
    
    # Find D node among AB's children
    d_node = next(child for child in ab_children if child.me == "D")
    
    # D should have no children
    d_children = d_node.get_children()
    assert len(d_children) == 0
    
    # Verify the string representation matches expected structure
    # # Note: The exact order of C and D in the repr may vary, so we check the structure
    # root_repr = repr(root)
    # assert "Node(AB," in root_repr
    # assert "Node(C," in root_repr
    # assert "Node(D," in root_repr
    # assert "Node(E," in root_repr


def test_prefixes_case1():
    """Test prefix computation for pipelines: ['A', 'AB', 'D', 'ABC', 'BCDE', 'BCD', 'CDE']"""
    pipelines = ['A', 'AB', 'D', 'ABC', 'BCDE', 'BCD', 'CDE']
    result = compute_final_prefixes(pipelines)
    
    expected = {
        'A': ['A', 'AB', 'ABC'],
        'D': ['D'],
        'BCD': ['BCDE', 'BCD'],
        'CDE': ['CDE']
    }
    
    assert set(result.keys()) == set(expected.keys())
    for key in expected:
        assert set(result[key]) == set(expected[key])


def test_prefixes_case2():
    """Test prefix computation for pipelines: ['ABCDE', 'ABC', 'ABD']"""
    pipelines = ['ABCDE', 'ABC', 'ABD']
    result = compute_final_prefixes(pipelines)
    
    expected = {
        'AB': ['ABCDE', 'ABC', 'ABD']
    }
    
    assert set(result.keys()) == set(expected.keys())
    for key in expected:
        assert set(result[key]) == set(expected[key])


def test_prefixes_case3():
    """Test prefix computation for pipelines: ['A', 'AB', 'DE', 'ABC', 'BCDE', 'BCD', 'B']"""
    pipelines = ['A', 'AB', 'DE', 'ABC', 'BCDE', 'BCD', 'B']
    result = compute_final_prefixes(pipelines)
    
    expected = {
        'A': ['A', 'AB', 'ABC'],
        'B': ['B', 'BCDE', 'BCD'],
        'DE': ['DE']
    }
    
    assert set(result.keys()) == set(expected.keys())
    for key in expected:
        assert set(result[key]) == set(expected[key])


def test_prefixes_case4():
    """Test prefix computation for empty pipelines list"""
    pipelines = []
    result = compute_final_prefixes(pipelines)
    
    expected = {}
    
    assert result == expected