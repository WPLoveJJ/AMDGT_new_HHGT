"""
Unit tests for utility functions.
Tests metric calculations, loss functions, and helper utilities.
"""

import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class TestMaskedLossFunctions:
    """Test suite for masked loss functions."""
    
    def test_masked_mae_cal_basic(self):
        """Test basic MAE calculation with mask."""
        from modeling.utils import masked_mae_cal
        
        inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        
        mae = masked_mae_cal(inputs, target, mask)
        
        # All values should contribute to the loss
        expected_mae = torch.mean(torch.abs(inputs - target))
        assert torch.isclose(mae, expected_mae, atol=1e-5)
    
    def test_masked_mae_cal_with_mask(self):
        """Test MAE calculation with partial mask."""
        from modeling.utils import masked_mae_cal
        
        inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        
        mae = masked_mae_cal(inputs, target, mask)
        
        # Only masked values should contribute
        diff = torch.abs(inputs - target) * mask
        expected_mae = torch.sum(diff) / torch.sum(mask)
        assert torch.isclose(mae, expected_mae, atol=1e-5)
    
    def test_masked_mae_cal_all_masked(self):
        """Test MAE calculation when all values are masked."""
        from modeling.utils import masked_mae_cal
        
        inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        mask = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        mae = masked_mae_cal(inputs, target, mask)
        
        # Should return 0 when all masked (due to division by small epsilon)
        assert mae >= 0
    
    def test_masked_mse_cal_basic(self):
        """Test basic MSE calculation with mask."""
        from modeling.utils import masked_mse_cal
        
        inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        
        mse = masked_mse_cal(inputs, target, mask)
        
        expected_mse = torch.mean((inputs - target) ** 2)
        assert torch.isclose(mse, expected_mse, atol=1e-5)
    
    def test_masked_rmse_cal_basic(self):
        """Test basic RMSE calculation with mask."""
        from modeling.utils import masked_rmse_cal
        
        inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        
        rmse = masked_rmse_cal(inputs, target, mask)
        
        expected_rmse = torch.sqrt(torch.mean((inputs - target) ** 2))
        assert torch.isclose(rmse, expected_rmse, atol=1e-5)
    
    def test_masked_mre_cal_basic(self):
        """Test basic MRE calculation with mask."""
        from modeling.utils import masked_mre_cal
        
        inputs = torch.tensor([[2.0, 4.0, 6.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        
        mre = masked_mre_cal(inputs, target, mask)
        
        # MRE should be (|2-1| + |4-2| + |6-3|) / (1 + 2 + 3) = 6/6 = 1
        expected_mre = 1.0
        assert torch.isclose(mre, torch.tensor(expected_mre), atol=1e-5)
    
    def test_masked_loss_non_negative(self):
        """Test that all masked loss functions return non-negative values."""
        from modeling.utils import masked_mae_cal, masked_mse_cal, masked_rmse_cal, masked_mre_cal
        
        inputs = torch.randn(5, 10)
        target = torch.randn(5, 10)
        mask = torch.ones(5, 10)
        
        mae = masked_mae_cal(inputs, target, mask)
        mse = masked_mse_cal(inputs, target, mask)
        rmse = masked_rmse_cal(inputs, target, mask)
        mre = masked_mre_cal(inputs, target, mask)
        
        assert mae >= 0
        assert mse >= 0
        assert rmse >= 0
        assert mre >= 0


class TestClassificationMetrics:
    """Test suite for classification metrics."""
    
    def test_get_metric_basic(self):
        """Test basic metric calculation."""
        from metric import get_metric
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.6, 0.8, 0.2])
        
        auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_true, y_pred, y_prob)
        
        # Check that metrics are in valid ranges
        assert 0 <= auc <= 1
        assert 0 <= aupr <= 1
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert -1 <= mcc <= 1
    
    def test_get_metric_perfect_classification(self):
        """Test metric calculation with perfect classification."""
        from metric import get_metric
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.1])
        
        auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_true, y_pred, y_prob)
        
        # Perfect classification should give high scores
        assert accuracy == 1.0
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
        assert mcc == 1.0
        assert auc == 1.0
    
    def test_get_metric_random_classification(self):
        """Test metric calculation with random classification."""
        from metric import get_metric
        
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        
        auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_true, y_pred, y_prob)
        
        # Check that metrics are in valid ranges
        assert 0 <= auc <= 1
        assert 0 <= aupr <= 1
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert -1 <= mcc <= 1
    
    def test_get_metric_single_class(self):
        """Test metric calculation with single class."""
        from metric import get_metric
        
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
        
        # This should handle edge case or raise appropriate error
        try:
            auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_true, y_pred, y_prob)
            assert accuracy == 1.0
        except (ValueError, UndefinedMetricWarning):
            # Expected for single class
            pass


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_str2bool_true(self):
        """Test str2bool with true values."""
        from modeling.utils import str2bool
        
        for val in ['yes', 'true', 't', 'y', '1']:
            assert str2bool(val) == True
    
    def test_str2bool_false(self):
        """Test str2bool with false values."""
        from modeling.utils import str2bool
        
        for val in ['no', 'false', 'f', 'n', '0']:
            assert str2bool(val) == False
    
    def test_str2bool_bool_input(self):
        """Test str2bool with boolean input."""
        from modeling.utils import str2bool
        
        assert str2bool(True) == True
        assert str2bool(False) == False
    
    def test_str2bool_invalid_input(self):
        """Test str2bool with invalid input."""
        from modeling.utils import str2bool
        
        with pytest.raises(TypeError):
            str2bool('invalid')
    
    def test_precision_recall(self):
        """Test precision-recall calculation."""
        from modeling.utils import precision_recall
        
        y_test = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.7])
        
        area, precisions, recalls, thresholds = precision_recall(y_pred, y_test)
        
        # Check output shapes
        assert len(precisions) == len(recalls)
        assert len(recalls) == len(thresholds) + 1
        assert 0 <= area <= 1
    
    def test_auc_roc(self):
        """Test AUC-ROC calculation."""
        from modeling.utils import auc_roc
        
        y_test = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.7])
        
        auc, fprs, tprs, thresholds = auc_roc(y_pred, y_test)
        
        # Check output shapes
        assert len(fprs) == len(tprs)
        assert len(tprs) == len(thresholds)
        assert 0 <= auc <= 1
        
        # Check that ROC curve starts at (0,0) and ends at (1,1)
        assert fprs[0] == 0.0
        assert tprs[0] == 0.0
        assert fprs[-1] == 1.0
        assert tprs[-1] == 1.0


class TestController:
    """Test suite for training controller."""
    
    def test_controller_init(self):
        """Test controller initialization."""
        from modeling.utils import Controller
        
        controller = Controller(early_stop_patience=10)
        
        assert controller.original_early_stop_patience_value == 10
        assert controller.early_stop_patience == 10
        assert controller.state_dict['train_step'] == 0
        assert controller.state_dict['val_step'] == 0
        assert controller.state_dict['epoch'] == 0
        assert controller.state_dict['best_imputation_MAE'] == 1e9
        assert controller.state_dict['should_stop'] == False
        assert controller.state_dict['save_model'] == False
    
    def test_controller_train_step(self):
        """Test controller training step."""
        from modeling.utils import Controller
        
        controller = Controller(early_stop_patience=10)
        
        for _ in range(5):
            state = controller(stage='train')
        
        assert state['train_step'] == 5
        assert state['val_step'] == 0
    
    def test_controller_val_step_improvement(self):
        """Test controller validation step with improvement."""
        from modeling.utils import Controller
        import logging
        
        logger = logging.getLogger('test')
        logger.setLevel(logging.CRITICAL)  # Suppress logs
        
        controller = Controller(early_stop_patience=10)
        
        info = {'imputation_MAE': 0.5}
        state = controller(stage='val', info=info, logger=logger)
        
        assert state['best_imputation_MAE'] == 0.5
        assert state['save_model'] == True
        assert state['should_stop'] == False
        assert controller.early_stop_patience == 10
    
    def test_controller_val_step_no_improvement(self):
        """Test controller validation step without improvement."""
        from modeling.utils import Controller
        import logging
        
        logger = logging.getLogger('test')
        logger.setLevel(logging.CRITICAL)
        
        controller = Controller(early_stop_patience=2)
        
        # First validation - set best
        info = {'imputation_MAE': 0.5}
        state = controller(stage='val', info=info, logger=logger)
        
        # Second validation - no improvement
        info = {'imputation_MAE': 0.6}
        state = controller(stage='val', info=info, logger=logger)
        
        assert state['save_model'] == False
        assert controller.early_stop_patience == 1
        assert state['should_stop'] == False
    
    def test_controller_early_stopping(self):
        """Test controller early stopping."""
        from modeling.utils import Controller
        import logging
        
        logger = logging.getLogger('test')
        logger.setLevel(logging.CRITICAL)
        
        controller = Controller(early_stop_patience=2)
        
        # First validation - set best
        info = {'imputation_MAE': 0.5}
        state = controller(stage='val', info=info, logger=logger)
        
        # Second and third validations - no improvement
        for _ in range(2):
            info = {'imputation_MAE': 0.6}
            state = controller(stage='val', info=info, logger=logger)
        
        assert state['should_stop'] == True
    
    def test_controller_epoch_increment(self):
        """Test controller epoch increment."""
        from modeling.utils import Controller
        
        controller = Controller(early_stop_patience=10)
        
        for _ in range(3):
            controller.epoch_num_plus_1()
        
        assert controller.state_dict['epoch'] == 3


class TestHypergraphUtils:
    """Test suite for hypergraph utility functions."""
    
    def test_extractv2e_basic(self):
        """Test ExtractV2E function."""
        from utils import ExtractV2E
        
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        num_nodes = 5
        num_hyperedges = 5
        
        V2E = ExtractV2E(edge_index, num_nodes, num_hyperedges)
        
        assert V2E.shape == (2, 3)
    
    def test_construct_h_basic(self):
        """Test ConstructH function."""
        from utils import ConstructH
        
        edge_index_0 = torch.tensor([[0, 1, 2], [0, 1, 2]])
        num_nodes = 3
        
        H = ConstructH(edge_index_0, num_nodes)
        
        assert H.shape == (3, 3)
        assert H.is_sparse
    
    def test_add_self_loops_basic(self):
        """Test add_self_loops function."""
        from utils import add_self_loops
        
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        num_nodes = 3
        
        edge_index_with_loops, edge_weight = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Check that self-loops were added
        assert edge_index_with_loops.shape[1] == edge_index.shape[1] + num_nodes
        assert edge_weight is not None
    
    def test_sparse_linear_basic(self):
        """Test SparseLinear layer."""
        from utils import SparseLinear
        
        in_features = 10
        out_features = 5
        
        layer = SparseLinear(in_features, out_features)
        
        # Create sparse input
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.randn(3)
        sparse_input = torch.sparse_coo_tensor(indices, values, (1, in_features))
        
        output = layer(sparse_input)
        
        assert output.shape == (1, out_features)
        assert torch.all(torch.isfinite(output))
    
    def test_sparse_linear_parameters(self):
        """Test SparseLinear parameter initialization."""
        from utils import SparseLinear
        
        in_features = 10
        out_features = 5
        
        layer = SparseLinear(in_features, out_features, bias=True)
        
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.weight.shape == (out_features, in_features)
        assert layer.bias is not None
        
        # Test without bias
        layer_no_bias = SparseLinear(in_features, out_features, bias=False)
        assert layer_no_bias.bias is None
    
    def test_sparse_linear_forward(self):
        """Test SparseLinear forward pass."""
        from utils import SparseLinear
        
        in_features = 10
        out_features = 5
        batch_size = 2
        
        layer = SparseLinear(in_features, out_features)
        
        # Create sparse inputs
        indices = torch.tensor([[0, 0, 1], [0, 1, 2]])
        values = torch.randn(3)
        sparse_input = torch.sparse_coo_tensor(indices, values, (batch_size, in_features))
        
        output = layer(sparse_input)
        
        assert output.shape == (batch_size, out_features)
        assert torch.all(torch.isfinite(output))
