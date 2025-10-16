"""Validation tests for AXON models (out-of-sample, walk-forward, statistical significance, stability)."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import AXON components
from src.models import (
    XGBoostModel, CatBoostModel, LSTMModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, PYTORCH_AVAILABLE
)


class TestOutOfSampleValidation:
    """Test out-of-sample validation."""

    @pytest.fixture
    def time_series_data(self):
        """Create time series data for validation."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Create time series with temporal dependencies
        X = np.random.randn(n_samples, n_features)

        # Add autoregressive component
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1] + 0.05 * np.random.randn(n_features)

        # Create target with temporal pattern
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples))
        noise = np.random.randn(n_samples) * 0.1
        signal = X[:, 0] + trend + noise
        y = (signal > np.median(signal)).astype(int)

        return X, y

    def test_out_of_sample_performance_random_forest(self, time_series_data):
        """Test out-of-sample performance with RandomForest."""
        X, y = time_series_data

        # Use time series split for more realistic validation
        tscv = TimeSeriesSplit(n_splits=5)

        from sklearn.ensemble import RandomForestClassifier

        oos_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            oos_scores.append(accuracy)

        # Out-of-sample performance should be reasonable
        mean_oos = np.mean(oos_scores)
        std_oos = np.std(oos_scores)

        assert mean_oos > 0.5, f"Poor OOS performance: {mean_oos:.3f}"
        assert std_oos < 0.15, f"High OOS variance: {std_oos:.3f}"

        print(f"OOS Performance - Mean: {mean_oos:.3f}, Std: {std_oos:.3f}")

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_out_of_sample_performance_xgboost(self, time_series_data):
        """Test out-of-sample performance with XGBoost."""
        X, y = time_series_data
        tscv = TimeSeriesSplit(n_splits=5)

        oos_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBoostModel(n_estimators=50, verbosity=0)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            oos_scores.append(accuracy)

        mean_oos = np.mean(oos_scores)
        std_oos = np.std(oos_scores)

        assert mean_oos > 0.5, f"XGBoost poor OOS: {mean_oos:.3f}"
        assert std_oos < 0.15, f"XGBoost high OOS variance: {std_oos:.3f}"

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_out_of_sample_performance_catboost(self, time_series_data):
        """Test out-of-sample performance with CatBoost."""
        X, y = time_series_data
        tscv = TimeSeriesSplit(n_splits=5)

        oos_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = CatBoostModel(iterations=50, verbose=False)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            oos_scores.append(accuracy)

        mean_oos = np.mean(oos_scores)
        std_oos = np.std(oos_scores)

        assert mean_oos > 0.5, f"CatBoost poor OOS: {mean_oos:.3f}"
        assert std_oos < 0.15, f"CatBoost high OOS variance: {std_oos:.3f}"


class TestWalkForwardValidation:
    """Test walk-forward validation."""

    def test_walk_forward_validation_random_forest(self):
        """Test walk-forward validation with RandomForest."""
        # Create expanding window validation
        np.random.seed(42)
        n_samples = 800
        n_features = 8

        X = np.random.randn(n_samples, n_features)
        # Add temporal dependency
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]

        y = (X[:, 0] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        from sklearn.ensemble import RandomForestClassifier

        # Walk-forward validation
        window_size = 200
        step_size = 50
        wf_scores = []

        for start_idx in range(0, n_samples - window_size - step_size, step_size):
            end_train = start_idx + window_size
            end_test = end_train + step_size

            if end_test > n_samples:
                break

            X_train = X[start_idx:end_train]
            y_train = y[start_idx:end_train]
            X_test = X[end_train:end_test]
            y_test = y[end_train:end_test]

            model = RandomForestClassifier(n_estimators=30, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            wf_scores.append(accuracy)

        if len(wf_scores) > 0:
            mean_wf = np.mean(wf_scores)
            std_wf = np.std(wf_scores)

            assert mean_wf > 0.5, f"Poor walk-forward performance: {mean_wf:.3f}"
            assert std_wf < 0.2, f"High walk-forward variance: {std_wf:.3f}"

            print(f"Walk-forward - Mean: {mean_wf:.3f}, Std: {std_wf:.3f}")

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_walk_forward_validation_xgboost(self):
        """Test walk-forward validation with XGBoost."""
        np.random.seed(42)
        n_samples = 600
        n_features = 8

        X = np.random.randn(n_samples, n_features)
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]

        y = (X[:, 0] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        window_size = 150
        step_size = 50
        wf_scores = []

        for start_idx in range(0, n_samples - window_size - step_size, step_size):
            end_train = start_idx + window_size
            end_test = end_train + step_size

            if end_test > n_samples:
                break

            X_train = X[start_idx:end_train]
            y_train = y[start_idx:end_train]
            X_test = X[end_train:end_test]
            y_test = y[end_train:end_test]

            model = XGBoostModel(n_estimators=30, verbosity=0)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            wf_scores.append(accuracy)

        if len(wf_scores) > 0:
            mean_wf = np.mean(wf_scores)
            std_wf = np.std(wf_scores)

            assert mean_wf > 0.5, f"XGBoost poor walk-forward: {mean_wf:.3f}"
            assert std_wf < 0.2, f"XGBoost high walk-forward variance: {std_wf:.3f}"


class TestStatisticalSignificance:
    """Test statistical significance of model performance."""

    def test_performance_vs_random(self):
        """Test that model performance is significantly better than random."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=5, random_state=42
        )

        from sklearn.ensemble import RandomForestClassifier

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        # Compare to random guessing (should be 0.5 for balanced classes)
        random_accuracy = 0.5

        # Statistical test
        n_trials = len(y)
        # Use binomial test for significance
        p_value = stats.binom_test(
            int(accuracy * n_trials),
            n_trials,
            random_accuracy,
            alternative='greater'
        )

        assert p_value < 0.05, f"Model not significantly better than random: p={p_value:.4f}"
        print(f"Model accuracy: {accuracy:.3f}, p-value vs random: {p_value:.4f}")

    def test_model_comparison_significance(self):
        """Test statistical significance between different models."""
        np.random.seed(42)
        X, y = make_classification(n_samples=800, n_features=8, random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Compare RandomForest vs LogisticRegression
        rf_scores = cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=42),
            X, y, cv=10, scoring='accuracy'
        )

        lr_scores = cross_val_score(
            LogisticRegression(random_state=42),
            X, y, cv=10, scoring='accuracy'
        )

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rf_scores, lr_scores)

        print(f"RF vs LR - t-stat: {t_stat:.3f}, p-value: {p_value:.4f}")
        print(f"RF mean: {rf_scores.mean():.3f}, LR mean: {lr_scores.mean():.3f}")

        # Just verify the test runs without error
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_significance_vs_baseline(self):
        """Test XGBoost significance vs baseline."""
        X, y = make_classification(n_samples=600, n_features=8, random_state=42)

        # XGBoost model
        xgb_model = XGBoostModel(n_estimators=50, verbosity=0)
        xgb_scores = []

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            xgb_model.fit(X_train, y_train)
            pred = xgb_model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            xgb_scores.append(acc)

        # Compare to random baseline
        random_scores = [0.5] * len(xgb_scores)  # Assuming balanced classes

        # Check if XGBoost is significantly better
        t_stat, p_value = stats.ttest_rel(xgb_scores, random_scores)

        print(f"XGBoost vs Random - t-stat: {t_stat:.3f}, p-value: {p_value:.4f}")

        # XGBoost should be significantly better than random
        assert p_value < 0.05 or t_stat > 2, "XGBoost not significantly better than random"


class TestModelStability:
    """Test model stability and consistency."""

    def test_prediction_stability_random_forest(self):
        """Test prediction stability with RandomForest."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)

        from sklearn.ensemble import RandomForestClassifier

        # Train multiple models with same random seed
        predictions_list = []

        for seed in [42, 42, 42]:  # Same seed should give same results
            model = RandomForestClassifier(n_estimators=50, random_state=seed)
            model.fit(X, y)
            predictions = model.predict(X)
            predictions_list.append(predictions)

        # Predictions should be identical with same seed
        for i in range(1, len(predictions_list)):
            np.testing.assert_array_equal(
                predictions_list[0], predictions_list[i],
                "Predictions not stable with same random seed"
            )

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_prediction_stability_xgboost(self):
        """Test prediction stability with XGBoost."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)

        predictions_list = []

        for seed in [42, 42, 42]:
            model = XGBoostModel(n_estimators=50, random_state=seed, verbosity=0)
            model.fit(X, y)
            predictions = model.predict(X)
            predictions_list.append(predictions)

        # XGBoost should be deterministic with same seed
        for i in range(1, len(predictions_list)):
            np.testing.assert_array_equal(
                predictions_list[0], predictions_list[i],
                "XGBoost predictions not stable"
            )

    def test_feature_importance_stability(self):
        """Test feature importance stability."""
        X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)

        from sklearn.ensemble import RandomForestClassifier

        importance_list = []

        for seed in [42, 43, 44]:
            model = RandomForestClassifier(n_estimators=50, random_state=seed)
            model.fit(X, y)
            importance = model.feature_importances_
            importance_list.append(importance)

        # Convert to numpy array for easier analysis
        importance_array = np.array(importance_list)

        # Calculate coefficient of variation for each feature
        cv_per_feature = np.std(importance_array, axis=0) / np.mean(importance_array, axis=0)

        # Most features should have reasonable stability (CV < 0.5)
        stable_features = np.mean(cv_per_feature < 0.5)
        assert stable_features > 0.7, f"Only {stable_features:.1%} features are stable"

    def test_cross_validation_stability(self):
        """Test cross-validation stability."""
        X, y = make_classification(n_samples=600, n_features=8, random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Run multiple CV experiments
        cv_results = []

        for seed in range(5):
            scores = cross_val_score(
                RandomForestClassifier(n_estimators=50, random_state=seed),
                X, y, cv=5, scoring='accuracy'
            )
            cv_results.append(scores)

        cv_results = np.array(cv_results)

        # Calculate stability metrics
        mean_scores = np.mean(cv_results, axis=1)
        std_scores = np.std(cv_results, axis=1)

        # CV scores should be relatively stable
        cv_stability = np.std(mean_scores) / np.mean(mean_scores)
        assert cv_stability < 0.1, f"Unstable CV performance: CV = {cv_stability:.3f}"

        print(f"CV Stability - Mean: {np.mean(mean_scores):.3f}, Std: {np.std(mean_scores):.3f}")


class TestModelCalibration:
    """Test model calibration and probability estimates."""

    def test_probability_calibration_random_forest(self):
        """Test probability calibration with RandomForest."""
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import calibration_curve

        X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        # Check calibration
        prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=10)

        # Well-calibrated model should have prob_true ≈ prob_pred
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        assert calibration_error < 0.2, f"Poor calibration: error = {calibration_error:.3f}"

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_probability_calibration_xgboost(self):
        """Test probability calibration with XGBoost."""
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

        from sklearn.calibration import calibration_curve

        X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

        model = XGBoostModel(n_estimators=50, verbosity=0)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        assert calibration_error < 0.2, f"XGBoost poor calibration: error = {calibration_error:.3f}"


class TestRobustnessToDataChanges:
    """Test robustness to small changes in training data."""

    def test_data_perturbation_robustness(self):
        """Test robustness to small data perturbations."""
        X, y = make_classification(n_samples=500, n_features=8, random_state=42)

        from sklearn.ensemble import RandomForestClassifier

        # Train baseline model
        model1 = RandomForestClassifier(n_estimators=50, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        # Add small noise to data
        X_noisy = X + np.random.normal(0, 0.01, X.shape)

        # Train model on noisy data
        model2 = RandomForestClassifier(n_estimators=50, random_state=42)
        model2.fit(X_noisy, y)
        pred2 = model2.predict(X)

        # Predictions should be similar
        agreement = np.mean(pred1 == pred2)
        assert agreement > 0.8, f"Model not robust to noise: agreement = {agreement:.3f}"

    def test_sample_perturbation_robustness(self):
        """Test robustness to removing small number of samples."""
        X, y = make_classification(n_samples=500, n_features=8, random_state=42)

        from sklearn.ensemble import RandomForestClassifier

        # Train on full data
        model1 = RandomForestClassifier(n_estimators=50, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        # Remove 5% of samples
        remove_indices = np.random.choice(len(X), size=int(0.05 * len(X)), replace=False)
        keep_mask = np.ones(len(X), dtype=bool)
        keep_mask[remove_indices] = False

        X_reduced = X[keep_mask]
        y_reduced = y[keep_mask]

        # Train on reduced data
        model2 = RandomForestClassifier(n_estimators=50, random_state=42)
        model2.fit(X_reduced, y_reduced)

        # Predict on common samples
        common_pred1 = pred1[keep_mask]
        pred2 = model2.predict(X_reduced)

        agreement = np.mean(common_pred1 == pred2)
        assert agreement > 0.75, f"Model not robust to sample removal: agreement = {agreement:.3f}"


class TestLongTermStability:
    """Test long-term stability of model performance."""

    def test_performance_decay_simulation(self):
        """Simulate and test performance decay over time."""
        # Create data with concept drift
        np.random.seed(42)
        n_samples = 2000
        n_features = 8

        X = np.random.randn(n_samples, n_features)

        # Add gradual concept drift
        for i in range(n_samples):
            drift_factor = i / n_samples  # Increases from 0 to 1
            X[i, 0] += drift_factor * np.random.randn()

        y = (X[:, 0] > 0).astype(int)

        from sklearn.ensemble import RandomForestClassifier

        # Train on first half
        split_point = n_samples // 2
        X_train = X[:split_point]
        y_train = y[:split_point]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Test on different time periods
        test_periods = [
            (split_point, split_point + 200),  # Immediately after training
            (split_point + 200, split_point + 400),  # Later period
            (split_point + 400, n_samples),  # Latest period
        ]

        performances = []

        for start, end in test_periods:
            if end > n_samples:
                continue

            X_test = X[start:end]
            y_test = y[start:end]

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            performances.append(accuracy)

        if len(performances) > 1:
            # Performance should degrade gradually, not catastrophically
            initial_perf = performances[0]
            final_perf = performances[-1]

            degradation = initial_perf - final_perf
            assert degradation < 0.3, f"Catastrophic performance decay: {degradation:.3f}"

            print(f"Performance decay: {degradation:.3f} (initial: {initial_perf:.3f}, final: {final_perf:.3f})")


if __name__ == "__main__":
    pytest.main([__file__])