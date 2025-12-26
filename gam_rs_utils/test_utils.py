"""
Test file for DatasetUtils and ModelUtils classes in gam_rs_utils/utils.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest
from collections import defaultdict
import re

from gam_rs_utils.utils import DatasetUtils, ModelUtils
from src.utils import get_X_y

MOCK_BINNED_HEADER = [
    'intercept', 
    'fone<=1.0', '1.0<fone<=2.0', '2.0<fone<=3.0',
    'ftwo<=3.0', '3.0<ftwo<=4.0', '4.0<ftwo<=5.0',
    'fthree<=5.0'
]
MOCK_CUMULATIVE_HEADER = [
    'intercept', 
    'fone<=1.0', 'fone<=2.0', 'fone<=3.0',
    'ftwo<=3.0', 'ftwo<=4.0', 'ftwo<=5.0',
    'fthree<=5.0'
]
MOCK_WEIGHTS = np.array([
    1.0, 
    0.5, 0.3, 0.2, 
    0.0, 0.0, 0.0,
    0.1,
])

class TestDatasetUtils:
    """Test suite for DatasetUtils class"""
    
    def test_get_feature_thresholds(self):
        """
        Test get_feature_thresholds method.
        """
        weights = MOCK_WEIGHTS
        columns = MOCK_BINNED_HEADER
        result = DatasetUtils.get_feature_thresholds(weights, columns)
        assert isinstance(result, defaultdict)
        assert result == defaultdict(list, {
            'intercept': [([], 1.0)],
            'fone': [([1.0], 0.5), ([1.0, 2.0], 0.3), ([2.0, 3.0], 0.2)], 
            'ftwo': [([3.0], 0.0), ([3.0, 4.0], 0.0), ([4.0, 5.0], 0.0)], 
            'fthree': [([5.0], 0.1)]
        })
    
    def test_get_feature_ranges(self):
        """
        Test get_feature_ranges method.
        """
        columns = MOCK_BINNED_HEADER
        result = DatasetUtils.get_feature_ranges(columns)
        assert isinstance(result, defaultdict)
        assert result == defaultdict(list, {
            'intercept': [[]],
            'fone': [[1.0], [1.0, 2.0], [2.0, 3.0]],
            'ftwo': [[3.0], [3.0, 4.0], [4.0, 5.0]],
            'fthree': [[5.0]]
        })
    
    def test_convert_cumulative_to_binned(self):
        """
        Test convert_cumulative_to_binned method.
        """
        X = np.array([
            [1, 0, 0, 1, 0, 0, 0, 0], 
            [1, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 1, 1, 1]
        ])
        header = MOCK_CUMULATIVE_HEADER
        X_new, header_new = DatasetUtils.convert_cumulative_to_binned(X, header)
        assert X_new.shape == X.shape
        assert len(header_new) == len(header)
        assert np.array_equal(X_new, np.array([
            [1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 1]
        ]))
        assert header_new == [
            'intercept',
            'fone<=1.0', '1.0<fone<=2.0', '2.0<fone<=3.0',
            'ftwo<=3.0', '3.0<ftwo<=4.0', '4.0<ftwo<=5.0',
            'fthree<=5.0',
        ]

class TestModelUtils:
    """Test suite for ModelUtils class"""
    
    def test_get_header_object(self):
        """
        Test get_header_object method.
        """
        header = MOCK_BINNED_HEADER
        result = ModelUtils.get_header_object(header)
        assert isinstance(result, dict)
        assert result == {
            'intercept': [], 
            'fone': [1.0, 2.0, 3.0], 
            'ftwo': [3.0, 4.0, 5.0], 
            'fthree': [5.0],
        }
    
        header = MOCK_CUMULATIVE_HEADER
        result = ModelUtils.get_header_object(header)
        assert isinstance(result, dict)
        assert result == {
            'intercept': [], 
            'fone': [1.0, 2.0, 3.0], 
            'ftwo': [3.0, 4.0, 5.0], 
            'fthree': [5.0],
        }
    
    def test_expand_w(self):
        """
        Test expand_w method.
        """
        w = np.array([
            1.0, 
            0.3, 0.1,
            0.2, 0.0, 0.3,
            0.4,
        ])
        sparse_header = {
            'intercept': [],
            'fone': [1.0, 3.0], 
            'ftwo': [3.0, 4.0, 5.0],
            'fthree': [6.0],
        }
        header = {
            'intercept': [], 
            'fone': [1.0, 2.0, 3.0], 
            'ftwo': [3.0, 4.0, 5.0], 
            'fthree': [5.0, 6.0],
            'ffour': [2.0, 3.0, 4.0],
        }
        result = ModelUtils.expand_w(w, sparse_header, header)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([
            1.0,
            0.3, 0.1, 0.1,
            0.2, 0.0, 0.3,
            0.4, 0.4,
            0.0, 0.0, 0.0,
        ]))

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
