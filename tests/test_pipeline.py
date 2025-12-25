"""
Unit tests for Face Mask Detection Pipeline
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Tests for configuration module."""
    
    def test_config_imports(self):
        """Test that config imports successfully."""
        from src.config import (
            IMG_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS,
            TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        assert IMG_SIZE == 224
        assert NUM_CLASSES == 3
        assert BATCH_SIZE > 0
        assert EPOCHS > 0
    
    def test_split_ratios(self):
        """Test that split ratios sum to 1."""
        from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
        assert abs(total - 1.0) < 0.001, f"Split ratios sum to {total}, expected 1.0"
    
    def test_class_mapping(self):
        """Test class mapping is correct."""
        from src.config import CLASSES, CLASS_NAMES
        assert len(CLASSES) == 3
        assert len(CLASS_NAMES) == 3
        assert "with_mask" in CLASSES
        assert "without_mask" in CLASSES


class TestModel:
    """Tests for model module."""
    
    def test_model_builds(self):
        """Test that model builds without errors."""
        from src.model import build_model
        model = build_model()
        assert model is not None
    
    def test_model_output_shapes(self):
        """Test model output shapes."""
        import numpy as np
        from src.model import build_model
        from src.config import IMG_SIZE, NUM_CLASSES
        
        model = build_model()
        
        # Create dummy input
        dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        
        # Get predictions
        bbox_pred, class_pred = model.predict(dummy_input, verbose=0)
        
        assert bbox_pred.shape == (1, 4), f"Expected bbox shape (1, 4), got {bbox_pred.shape}"
        assert class_pred.shape == (1, NUM_CLASSES), f"Expected class shape (1, {NUM_CLASSES}), got {class_pred.shape}"


class TestDataset:
    """Tests for dataset module."""
    
    def test_dataset_imports(self):
        """Test that dataset module imports."""
        from src.dataset import (
            parse_xml, load_and_preprocess_image, 
            get_file_pairs, split_data, create_dataset
        )
    
    def test_split_data_function(self):
        """Test data splitting logic."""
        from src.dataset import split_data
        
        # Create dummy pairs
        dummy_pairs = [(f"img_{i}.png", f"ann_{i}.xml") for i in range(100)]
        
        train, val, test = split_data(dummy_pairs)
        
        # Check split sizes approximately match ratios
        assert len(train) >= 60  # ~70%
        assert len(val) >= 10   # ~15%
        assert len(test) >= 10  # ~15%
        assert len(train) + len(val) + len(test) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
