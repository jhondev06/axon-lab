"""Performance tests for AXON models and pipeline."""

import pytest
import time
import psutil
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from contextlib import contextmanager

# Import AXON components
from src.models import (
    XGBoostModel, CatBoostModel, LSTMModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, PYTORCH_AVAILABLE
)


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory

    print(".2f")


@contextmanager
def time_monitor():
    """Context manager to monitor execution time."""
    start_time = time.time()

    yield

    end_time = time.time()
    elapsed = end_time - start_time

    print(".4f")


class TestModelPerformanceBenchmarking:
    """Test model performance benchmarking."""

    @pytest.fixture
    def benchmark_data(self):
        """Create benchmark dataset."""
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        return X, y

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_training_performance(self, benchmark_data):
        """Test XGBoost training performance."""
        X, y = benchmark_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBoostModel(n_estimators=50, verbosity=0)

        with time_monitor():
            with memory_monitor():
                model.fit(X_train, y_train)

        # Verify model works
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        assert accuracy > 0.7  # Should have reasonable performance

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_training_performance(self, benchmark_data):
        """Test CatBoost training performance."""
        X, y = benchmark_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostModel(iterations=50, verbose=False)

        with time_monitor():
            with memory_monitor():
                model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        assert accuracy > 0.7

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_training_performance(self, benchmark_data):
        """Test LSTM training performance."""
        X, y = benchmark_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LSTMModel(
            sequence_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=5,
            batch_size=32
        )

        with time_monitor():
            with memory_monitor():
                model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        # LSTM predictions may be shorter due to sequences
        test_predictions = predictions[:min(len(predictions), len(y_test))]

        if len(test_predictions) > 0:
            accuracy = np.mean(test_predictions == y_test[:len(test_predictions)])
            assert accuracy > 0.5  # Lower threshold for LSTM

    def test_ensemble_training_performance(self, benchmark_data):
        """Test Ensemble training performance."""
        X, y = benchmark_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = EnsembleModel(
            ensemble_type='voting',
            base_models=['randomforest']
        )

        with time_monitor():
            with memory_monitor():
                model.fit(X_train, y_train, config={'models': {'randomforest': {}}})

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        assert accuracy > 0.7


class TestModelComparisonPerformance:
    """Test performance comparison between models."""

    @pytest.fixture
    def comparison_data(self):
        """Create dataset for model comparison."""
        X, y = make_classification(
            n_samples=500, n_features=15, n_informative=8,
            random_state=42
        )
        return X, y

    def test_model_speed_comparison(self, comparison_data):
        """Compare training speed between models."""
        X, y = comparison_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}

        # Test RandomForest (always available)
        from sklearn.ensemble import RandomForestClassifier

        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

        start_time = time.time()
        rf_model.fit(X_train, y_train)
        rf_time = time.time() - start_time

        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = np.mean(rf_predictions == y_test)

        results['randomforest'] = {
            'time': rf_time,
            'accuracy': rf_accuracy
        }

        # Test XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model = XGBoostModel(n_estimators=50, verbosity=0)

            start_time = time.time()
            xgb_model.fit(X_train, y_train)
            xgb_time = time.time() - start_time

            xgb_predictions = xgb_model.predict(X_test)
            xgb_accuracy = np.mean(xgb_predictions == y_test)

            results['xgboost'] = {
                'time': xgb_time,
                'accuracy': xgb_accuracy
            }

        # Test CatBoost if available
        if CATBOOST_AVAILABLE:
            cb_model = CatBoostModel(iterations=50, verbose=False)

            start_time = time.time()
            cb_model.fit(X_train, y_train)
            cb_time = time.time() - start_time

            cb_predictions = cb_model.predict(X_test)
            cb_accuracy = np.mean(cb_predictions == y_test)

            results['catboost'] = {
                'time': cb_time,
                'accuracy': cb_accuracy
            }

        # Verify we have at least one result
        assert len(results) > 0

        # Print comparison
        print("\nModel Performance Comparison:")
        for model_name, metrics in results.items():
            print(".4f")

        # All models should have reasonable accuracy
        for model_name, metrics in results.items():
            assert metrics['accuracy'] > 0.6

    def test_model_memory_comparison(self, comparison_data):
        """Compare memory usage between models."""
        X, y = comparison_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}

        # Test RandomForest
        from sklearn.ensemble import RandomForestClassifier

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        final_memory = process.memory_info().rss / 1024 / 1024
        rf_memory = final_memory - initial_memory

        results['randomforest'] = {'memory_mb': rf_memory}

        # Test XGBoost if available
        if XGBOOST_AVAILABLE:
            initial_memory = process.memory_info().rss / 1024 / 1024
            xgb_model = XGBoostModel(n_estimators=50, verbosity=0)
            xgb_model.fit(X_train, y_train)
            final_memory = process.memory_info().rss / 1024 / 1024
            xgb_memory = final_memory - initial_memory

            results['xgboost'] = {'memory_mb': xgb_memory}

        # Print memory comparison
        print("\nModel Memory Usage Comparison:")
        for model_name, metrics in results.items():
            print(".2f")

        # Memory usage should be reasonable (< 500 MB increase)
        for model_name, metrics in results.items():
            assert metrics['memory_mb'] < 500


class TestScalabilityPerformance:
    """Test performance scaling with data size."""

    def test_performance_scaling(self):
        """Test how performance scales with data size."""
        sizes = [100, 500, 1000]
        results = {}

        for n_samples in sizes:
            X, y = make_classification(
                n_samples=n_samples, n_features=10,
                n_informative=5, random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Test RandomForest scaling
            from sklearn.ensemble import RandomForestClassifier

            start_time = time.time()
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            predictions = model.predict(X_test)
            accuracy = np.mean(predictions == y_test)

            results[n_samples] = {
                'time': training_time,
                'accuracy': accuracy
            }

        # Verify scaling is reasonable (time shouldn't increase exponentially)
        print("\nScalability Results:")
        prev_time = None
        for size in sizes:
            metrics = results[size]
            print(f"Size {size}: Time={metrics['time']:.4f}s, Accuracy={metrics['accuracy']:.4f}")

            if prev_time is not None:
                # Time should scale roughly linearly or sub-linearly
                time_ratio = metrics['time'] / prev_time
                size_ratio = size / prev_size
                assert time_ratio < size_ratio * 2, f"Poor scaling: time ratio {time_ratio:.2f}, size ratio {size_ratio:.2f}"

            prev_time = metrics['time']
            prev_size = size

    def test_feature_scaling_performance(self):
        """Test performance with increasing number of features."""
        n_features_list = [5, 10, 20, 50]
        results = {}

        for n_features in n_features_list:
            X, y = make_classification(
                n_samples=500, n_features=n_features,
                n_informative=min(5, n_features//2), random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            from sklearn.ensemble import RandomForestClassifier

            start_time = time.time()
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            results[n_features] = {'time': training_time}

        print("\nFeature Scaling Results:")
        prev_time = None
        for n_feat in n_features_list:
            time_taken = results[n_feat]['time']
            print(f"Features {n_feat}: Time={time_taken:.4f}s")

            if prev_time is not None:
                time_ratio = time_taken / prev_time
                feat_ratio = n_feat / prev_feat
                # Allow some performance degradation but not exponential
                assert time_ratio < feat_ratio * 3, f"Poor feature scaling: {time_ratio:.2f} vs {feat_ratio:.2f}"

            prev_time = time_taken
            prev_feat = n_feat


class TestGPUPerformance:
    """Test GPU performance when available."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_gpu_performance(self):
        """Test XGBoost GPU performance if available."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Test CPU version
        cpu_model = XGBoostModel(n_estimators=50, verbosity=0)
        if hasattr(cpu_model, 'params'):
            cpu_model.params['tree_method'] = 'hist'  # Force CPU

        start_time = time.time()
        cpu_model.fit(X_train, y_train)
        cpu_time = time.time() - start_time

        cpu_predictions = cpu_model.predict(X_test)
        cpu_accuracy = np.mean(cpu_predictions == y_test)

        results = {'cpu': {'time': cpu_time, 'accuracy': cpu_accuracy}}

        # Test GPU version if available
        if hasattr(XGBoostModel, 'XGB_GPU_AVAILABLE') and XGBoostModel.XGB_GPU_AVAILABLE:
            gpu_model = XGBoostModel(n_estimators=50, verbosity=0)
            # GPU should be auto-detected

            start_time = time.time()
            gpu_model.fit(X_train, y_train)
            gpu_time = time.time() - start_time

            gpu_predictions = gpu_model.predict(X_test)
            gpu_accuracy = np.mean(gpu_predictions == y_test)

            results['gpu'] = {'time': gpu_time, 'accuracy': gpu_accuracy}

            print(".4f")
            print(".4f")

            # GPU should be at least as accurate as CPU
            assert gpu_accuracy >= cpu_accuracy - 0.05

        else:
            print("XGBoost GPU not available, skipping GPU test")

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_gpu_performance(self):
        """Test CatBoost GPU performance if available."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Test CPU version
        cpu_model = CatBoostModel(iterations=50, verbose=False, task_type='CPU')

        start_time = time.time()
        cpu_model.fit(X_train, y_train)
        cpu_time = time.time() - start_time

        cpu_predictions = cpu_model.predict(X_test)
        cpu_accuracy = np.mean(cpu_predictions == y_test)

        results = {'cpu': {'time': cpu_time, 'accuracy': cpu_accuracy}}

        # Test GPU version if available
        if hasattr(CatBoostModel, 'CB_GPU_AVAILABLE') and CatBoostModel.CB_GPU_AVAILABLE:
            gpu_model = CatBoostModel(iterations=50, verbose=False, task_type='GPU')

            start_time = time.time()
            gpu_model.fit(X_train, y_train)
            gpu_time = time.time() - start_time

            gpu_predictions = gpu_model.predict(X_test)
            gpu_accuracy = np.mean(gpu_predictions == y_test)

            results['gpu'] = {'time': gpu_time, 'accuracy': gpu_accuracy}

            print(".4f")
            print(".4f")

            # GPU should be at least as accurate as CPU
            assert gpu_accuracy >= cpu_accuracy - 0.05

        else:
            print("CatBoost GPU not available, skipping GPU test")


class TestInferencePerformance:
    """Test inference/prediction performance."""

    @pytest.fixture
    def inference_data(self):
        """Create data for inference testing."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        return X, y

    def test_batch_prediction_performance(self, inference_data):
        """Test prediction performance with different batch sizes."""
        X, y = inference_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        batch_sizes = [1, 10, 100, len(X_test)]

        print("\nBatch Prediction Performance:")
        for batch_size in batch_sizes:
            batch_X = X_test[:batch_size]

            start_time = time.time()
            predictions = model.predict(batch_X)
            inference_time = time.time() - start_time

            time_per_sample = inference_time / batch_size
            print(f"Batch size {batch_size}: {inference_time:.6f}s total, {time_per_sample:.6f}s per sample")

            assert len(predictions) == batch_size

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_inference_performance(self, inference_data):
        """Test XGBoost inference performance."""
        X, y = inference_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBoostModel(n_estimators=50, verbosity=0)
        model.fit(X_train, y_train)

        # Time predictions
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time

        time_per_sample = inference_time / len(X_test)

        print(".6f")

        assert len(predictions) == len(X_test)
        assert time_per_sample < 0.01  # Should be fast (< 10ms per sample)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_inference_performance(self, inference_data):
        """Test CatBoost inference performance."""
        X, y = inference_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostModel(iterations=50, verbose=False)
        model.fit(X_train, y_train)

        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time

        time_per_sample = inference_time / len(X_test)

        print(".6f")

        assert len(predictions) == len(X_test)
        assert time_per_sample < 0.01  # Should be fast


class TestMemoryEfficiency:
    """Test memory efficiency of models."""

    def test_memory_cleanup(self):
        """Test that models don't leak memory."""
        import gc

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create and train multiple models
        for i in range(5):
            X, y = make_classification(n_samples=500, n_features=10, random_state=i)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=20, random_state=i)
            model.fit(X_train, y_train)

            # Delete model
            del model
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        print(".2f")

        # Memory increase should be reasonable (< 100 MB)
        assert memory_increase < 100

    def test_large_dataset_memory_handling(self):
        """Test memory handling with large datasets."""
        # Create large dataset
        X, y = make_classification(
            n_samples=5000, n_features=50, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        print(".2f")

        # Should handle large dataset without excessive memory usage
        assert memory_used < 1000  # < 1GB increase


if __name__ == "__main__":
    pytest.main([__file__])