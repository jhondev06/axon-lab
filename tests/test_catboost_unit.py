"""Unit tests for CatBoost model implementation."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from unittest.mock import patch, MagicMock

# Import CatBoost components
try:
    from src.models import CatBoostModel, CATBOOST_AVAILABLE, CB_GPU_AVAILABLE
    CB_AVAILABLE_IN_TESTS = CATBOOST_AVAILABLE
except ImportError:
    CB_AVAILABLE_IN_TESTS = False


@pytest.mark.skipif(not CB_AVAILABLE_IN_TESTS, reason="CatBoost not available")
class TestCatBoostModel:
    """Test CatBoost model functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def catboost_model(self):
        """Create CatBoost model instance."""
        return CatBoostModel(
            iterations=50,  # Small for testing
            depth=3,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )

    def test_model_initialization(self, catboost_model):
        """Test CatBoost model initialization."""
        assert catboost_model.model is None
        assert catboost_model.feature_names is None
        assert catboost_model.params['iterations'] == 50
        assert catboost_model.params['depth'] == 3
        assert catboost_model.params['learning_rate'] == 0.1

    def test_model_initialization_with_gpu(self, catboost_model):
        """Test GPU parameter setting."""
        if CB_GPU_AVAILABLE:
            assert catboost_model.params.get('task_type') == 'GPU'
            assert 'devices' in catboost_model.params
        else:
            assert catboost_model.params.get('task_type') == 'CPU'

    def test_fit_basic(self, sample_data, catboost_model):
        """Test basic model fitting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit model
        catboost_model.fit(X_train, y_train)

        # Check that model is fitted
        assert catboost_model.model is not None

        # Make predictions
        predictions = catboost_model.predict(X_test)
        probabilities = catboost_model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert set(predictions).issubset({0, 1})

    def test_fit_with_eval_set(self, sample_data, catboost_model):
        """Test fitting with validation set."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Fit with validation
        catboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert catboost_model.model is not None

    def test_predict_proba_format(self, sample_data, catboost_model):
        """Test predict_proba output format."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        catboost_model.fit(X_train, y_train)
        probabilities = catboost_model.predict_proba(X_test)

        # Check shape and properties
        assert probabilities.shape[1] == 2
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_with_dataframe(self, sample_data, catboost_model):
        """Test prediction with DataFrame input."""
        X, y = sample_data
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

        catboost_model.fit(X_train, y_train)
        predictions = catboost_model.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_feature_importance(self, sample_data, catboost_model):
        """Test feature importance extraction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        catboost_model.fit(X_train, y_train)
        importance = catboost_model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, (int, float, np.floating)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

    def test_save_load_model(self, sample_data, catboost_model, tmp_path):
        """Test model saving and loading."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        catboost_model.fit(X_train, y_train)

        # Save model
        model_path = tmp_path / "test_cb.model"
        catboost_model.save_model(str(model_path))

        # Load model
        loaded_model = CatBoostModel.load_model(str(model_path))

        # Test predictions are identical
        original_pred = catboost_model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_early_stopping(self, sample_data):
        """Test early stopping functionality."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        model = CatBoostModel(iterations=100, early_stopping_rounds=5, verbose=False)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=5)

        # Should have stopped early (less than 100 iterations)
        assert model.model.get_best_iteration() < 100

    def test_hyperparameter_validation(self):
        """Test hyperparameter validation."""
        # Valid parameters
        model = CatBoostModel(iterations=100, depth=6, learning_rate=0.1)
        assert model.params['iterations'] == 100

        # Test default parameters are set
        model = CatBoostModel()
        assert 'random_seed' in model.params
        assert 'verbose' in model.params
        assert 'early_stopping_rounds' in model.params

    def test_edge_cases_insufficient_data(self):
        """Test behavior with insufficient data."""
        X = np.array([[1, 2], [3, 4]])  # Very small dataset
        y = np.array([0, 1])

        model = CatBoostModel(iterations=1, verbose=False)  # Minimal model

        # Should handle gracefully
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)

    def test_predict_before_fit_raises_error(self, catboost_model):
        """Test that predict before fit raises error."""
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Model not fitted yet"):
            catboost_model.predict(X)

        with pytest.raises(ValueError, match="Model not fitted yet"):
            catboost_model.predict_proba(X)

        with pytest.raises(ValueError, match="Model not fitted yet"):
            catboost_model.get_feature_importance()

    def test_symmetric_tree_grow_policy(self):
        """Test SymmetricTree grow policy."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        model = CatBoostModel(
            iterations=20,
            depth=4,
            grow_policy='SymmetricTree',
            verbose=False
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_different_loss_functions(self):
        """Test different loss functions."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        # Logloss for binary classification
        model_logloss = CatBoostModel(loss_function='Logloss', verbose=False)
        model_logloss.fit(X, y)

        assert model_logloss.model is not None

    @pytest.mark.parametrize("iterations,depth", [
        (10, 3),
        (50, 5),
        (100, 6)
    ])
    def test_different_hyperparameters(self, iterations, depth):
        """Test model with different hyperparameters."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        model = CatBoostModel(iterations=iterations, depth=depth, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_memory_efficiency(self):
        """Test model handles memory constraints reasonably."""
        # Large dataset for memory test
        X, y = make_classification(n_samples=5000, n_features=50, random_state=42)

        model = CatBoostModel(iterations=10, depth=3, verbose=False)  # Small model
        model.fit(X, y)

        # Should complete without memory issues
        assert model.model is not None

    def test_feature_name_storage(self):
        """Test that feature names are stored correctly."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])

        model = CatBoostModel(verbose=False)
        model.fit(df, y)

        assert model.feature_names == ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']

    def test_prediction_consistency(self, sample_data, catboost_model):
        """Test prediction consistency across multiple calls."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        catboost_model.fit(X_train, y_train)

        # Multiple prediction calls should be identical
        pred1 = catboost_model.predict(X_test)
        pred2 = catboost_model.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

        prob1 = catboost_model.predict_proba(X_test)
        prob2 = catboost_model.predict_proba(X_test)

        np.testing.assert_array_equal(prob1, prob2)

    def test_bootstrap_types(self):
        """Test different bootstrap types."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        bootstrap_types = ['Bayesian', 'Bernoulli', 'MVS', 'No']

        for bootstrap_type in bootstrap_types:
            model = CatBoostModel(
                iterations=20,
                bootstrap_type=bootstrap_type,
                verbose=False
            )
            model.fit(X, y)

            assert model.model is not None

    def test_score_functions(self):
        """Test different score functions."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        score_functions = ['Cosine', 'L2', 'Newton', 'NewtonCosine']

        for score_func in score_functions:
            model = CatBoostModel(
                iterations=20,
                score_function=score_func,
                verbose=False
            )
            model.fit(X, y)

            assert model.model is not None


@pytest.mark.skipif(not CB_AVAILABLE_IN_TESTS, reason="CatBoost not available")
class TestCatBoostIntegration:
    """Test CatBoost integration with other components."""

    def test_sklearn_compatibility(self, sample_data):
        """Test sklearn-like interface compatibility."""
        X, y = sample_data
        model = CatBoostModel(verbose=False)

        # Should have sklearn-like methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

        # Should work in sklearn pipelines
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', CatBoostModel(verbose=False))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_cross_validation_compatibility(self, sample_data):
        """Test compatibility with sklearn cross-validation."""
        from sklearn.model_selection import cross_val_score

        X, y = sample_data
        model = CatBoostModel(iterations=10, verbose=False)  # Small for speed

        # Should work with cross_val_score
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

        assert len(scores) == 3
        assert all(isinstance(s, (int, float)) for s in scores)

    def test_feature_importance_types(self, sample_data):
        """Test different feature importance types."""
        X, y = sample_data
        model = CatBoostModel(iterations=20, verbose=False)
        model.fit(X, y)

        importance_types = ['FeatureImportance', 'PredictionDiff', 'LossFunctionChange']

        for imp_type in importance_types:
            importance = model.get_feature_importance(importance_type=imp_type)
            assert isinstance(importance, dict)
            assert len(importance) > 0


if __name__ == "__main__":
    pytest.main([__file__])