import pytest
import numpy as np
import os
import json
from data_generation.skeleton_converter import WindowSkeletonConverter
from pathlib import Path

def test_skeleton_structure():
    """
    Test the skeleton converter on a real sample from the dataset.
    This ensures it can at least parse and produce the expected JSON structure.
    """
    sample_path = "/home/davidm15/Projects/SkillTrace2/datasets/windows-v2/1_wr1fr_1/1_wr1fr_1.obj"
    if not os.path.exists(sample_path):
        pytest.skip("Sample OBJ not found.")
        
    converter = WindowSkeletonConverter()
    skeleton = converter.extract_skeleton(sample_path)
    
    assert "nodes" in skeleton
    assert "edges" in skeleton
    assert "metadata" in skeleton
    
    nodes = np.array(skeleton["nodes"])
    edges = np.array(skeleton["edges"])
    
    # Basic sanity checks
    assert len(nodes) >= 4
    assert len(edges) >= 4
    
    # Check that edges reference valid nodes
    for edge in edges:
        assert 0 <= edge[0] < len(nodes)
        assert 0 <= edge[1] < len(nodes)

def test_node_merging_logic():
    """
    Test the internal clustering logic of the converter.
    """
    converter = WindowSkeletonConverter()
    
    # Boundary levels for a single bar of thickness 20mm
    levels = [100.0, 120.0, 500.0, 520.0]
    centers = converter._cluster_to_centers(levels)
    
    # Should result in centers at (100+120)/2 and (500+520)/2
    assert len(centers) == 2
    assert pytest.approx(centers[0]) == 110.0
    assert pytest.approx(centers[1]) == 510.0

def test_deterministic_output():
    """
    Ensure that running the converter twice on the same file produces identical output.
    """
    sample_path = "/home/davidm15/Projects/SkillTrace2/datasets/windows-v2/1_wr1fr_1/1_wr1fr_1.obj"
    if not os.path.exists(sample_path):
        pytest.skip("Sample OBJ not found.")
        
    converter = WindowSkeletonConverter()
    s1 = converter.extract_skeleton(sample_path)
    s2 = converter.extract_skeleton(sample_path)
    
    assert s1 == s2

if __name__ == "__main__":
    pytest.main([__file__])
