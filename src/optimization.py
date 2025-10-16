"""
AXON Hyperparameter Optimization Module

Sistema completo de otimização automática de hiperparâmetros com suporte a:
- Otimização multi-objetivo (Optuna)
- Grid Search sistemático
- Otimização Bayesiana
- Integração com backtest real
- Relatórios detalhados e visualizações
"""

import optuna
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import logging
import multiprocessing
from functools import partial
import time
import psutil

# Optional imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from skopt import gp_minimize, forest_minimize
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .utils import load_config, ensure_dir
from .backtest import BacktestEngine
from .metrics import sharpe_ratio, maximum_drawdown, calculate_returns
from axon.core.logging import get_logger

warnings.filterwarnings('ignore')

# Initialize logger
logger = get_logger(__name__)


class OptimizationEngine:
    """Main optimization engine with multiple methods support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.logger = logger

        # Initialize components
        self.optimizers = {
            'optuna': OptunaOptimizer(config),
            'grid_search': GridSearchOptimizer(config),
            'bayesian': BayesianOptimizer(config)
        }

        # Setup persistence
        self.persistence = StudyPersistence(config)

    def optimize_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                      y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model.

        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features (optional)
            y_test: Test labels (optional)

        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"🔧 Starting optimization for {model_name}")

        # Determine optimization method
        method = self._select_optimization_method(model_name)

        # Get optimizer
        optimizer = self.optimizers.get(method)
        if not optimizer:
            raise ValueError(f"Unsupported optimization method: {method}")

        # Run optimization
        results = optimizer.optimize(
            model_name, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Save results
        self.persistence.save_results(model_name, results)

        # Generate reports
        if self.opt_config.get('reporting', {}).get('enabled', True):
            self._generate_reports(model_name, results)

        return results

    def _select_optimization_method(self, model_name: str) -> str:
        """Select appropriate optimization method based on model and config."""
        method = self.opt_config.get('method', 'auto')

        if method != 'auto':
            return method

        # Auto-selection logic
        if model_name in ['lightgbm', 'lgb'] and LIGHTGBM_AVAILABLE:
            return 'optuna'
        elif model_name in ['xgboost', 'xgb'] and XGBOOST_AVAILABLE:
            return 'optuna'
        elif model_name in ['catboost', 'cb'] and CATBOOST_AVAILABLE:
            return 'optuna'
        elif model_name in ['randomforest', 'rf']:
            return 'grid_search'
        else:
            return 'optuna'  # default fallback

    def _generate_reports(self, model_name: str, results: Dict[str, Any]):
        """Generate optimization reports and visualizations."""
        reporter = OptimizationReporter(self.config)
        reporter.generate_report(model_name, results)


class ParallelOptimizationEngine:
    """Advanced parallel optimization engine with multi-model support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.parallel_config = self.opt_config.get('parallel', {})
        self.logger = logger

        # Resource management
        self.resource_manager = ResourceManager(config)

        # Initialize components
        self.model_optimizers = {}
        self.ensemble_optimizer = EnsembleOptimizer(config)

    def optimize_multiple_models(self, models: List[str], X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                               y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Optimize multiple models in parallel.

        Args:
            models: List of model names to optimize
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data (optional)

        Returns:
            Dictionary with optimization results for all models
        """
        self.logger.info(f"🚀 Starting parallel optimization for {len(models)} models")

        # Check resource availability
        if not self.resource_manager.check_resources():
            self.logger.warning("⚠️  Resource constraints detected, falling back to sequential optimization")
            return self._sequential_optimization(models, X_train, y_train, X_val, y_val, X_test, y_test)

        # Prepare optimization tasks
        optimization_tasks = []
        for model_name in models:
            task = {
                'model_name': model_name,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'config': self.config
            }
            optimization_tasks.append(task)

        # Execute parallel optimization
        max_parallel = self.parallel_config.get('max_parallel_jobs', min(len(models), multiprocessing.cpu_count()))
        timeout = self.parallel_config.get('timeout_per_trial', 300)

        self.logger.info(f"📊 Using {max_parallel} parallel jobs with {timeout}s timeout per trial")

        if JOBLIB_AVAILABLE:
            # Use joblib for parallel execution
            results = Parallel(
                n_jobs=max_parallel,
                backend='multiprocessing',
                timeout=timeout,
                verbose=10
            )(
                delayed(self._optimize_single_model_joblib)(task)
                for task in optimization_tasks
            )
        else:
            # Fallback to ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                futures = [
                    executor.submit(self._optimize_single_model_process, task)
                    for task in optimization_tasks
                ]

                results = []
                for future in as_completed(futures, timeout=timeout * len(models)):
                    try:
                        result = future.result(timeout=timeout)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"❌ Optimization task failed: {e}")
                        results.append({'error': str(e)})

        # Process results
        optimization_results = {}
        for i, result in enumerate(results):
            model_name = models[i]
            if 'error' not in result:
                optimization_results[model_name] = result
                self.logger.info(f"✅ {model_name} optimization completed")
            else:
                self.logger.error(f"❌ {model_name} optimization failed: {result['error']}")

        # Perform ensemble optimization if multiple models succeeded
        if len(optimization_results) > 1:
            self.logger.info("🎯 Optimizing ensemble with multiple models")
            ensemble_result = self.ensemble_optimizer.optimize_ensemble(
                optimization_results, X_train, y_train, X_val, y_val, X_test, y_test
            )
            optimization_results['ensemble'] = ensemble_result

        # Generate comprehensive reports
        if self.opt_config.get('reporting', {}).get('enabled', True):
            self._generate_parallel_reports(optimization_results)

        return optimization_results

    def _optimize_single_model_joblib(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Joblib-compatible single model optimization."""
        try:
            model_name = task['model_name']

            # Create optimizer instance
            if model_name in ['lightgbm', 'lgb']:
                optimizer = OptunaOptimizer(task['config'])
            elif model_name in ['xgboost', 'xgb']:
                optimizer = OptunaOptimizer(task['config'])
            elif model_name in ['catboost', 'cb']:
                optimizer = OptunaOptimizer(task['config'])
            elif model_name in ['randomforest', 'rf']:
                optimizer = GridSearchOptimizer(task['config'])
            else:
                raise ValueError(f"Unsupported model for parallel optimization: {model_name}")

            # Run optimization
            result = optimizer.optimize(
                model_name,
                task['X_train'], task['y_train'],
                task['X_val'], task['y_val'],
                task['X_test'], task['y_test']
            )

            return result

        except Exception as e:
            return {'error': str(e)}

    def _optimize_single_model_process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process pool compatible single model optimization."""
        return self._optimize_single_model_joblib(task)

    def _sequential_optimization(self, models: List[str], X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                               y_test: pd.Series = None) -> Dict[str, Any]:
        """Fallback sequential optimization."""
        self.logger.info("🔄 Running sequential optimization")

        results = {}
        for model_name in models:
            try:
                optimizer = OptimizationEngine(self.config)
                result = optimizer.optimize_model(
                    model_name, X_train, y_train, X_val, y_val, X_test, y_test
                )
                results[model_name] = result
            except Exception as e:
                self.logger.error(f"❌ Sequential optimization failed for {model_name}: {e}")

        return results

    def _generate_parallel_reports(self, results: Dict[str, Any]):
        """Generate comprehensive parallel optimization reports."""
        reporter = ParallelOptimizationReporter(self.config)
        reporter.generate_parallel_report(results)


class ResourceManager:
    """Resource management for optimization processes."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.resource_config = self.opt_config.get('resource_management', {})

    def check_resources(self) -> bool:
        """Check if sufficient resources are available for parallel optimization."""
        try:
            # Check CPU cores
            available_cores = multiprocessing.cpu_count()
            requested_cores = self.resource_config.get('max_parallel_jobs', available_cores)

            if requested_cores > available_cores:
                logger.warning(f"⚠️  Requested {requested_cores} cores but only {available_cores} available")
                return False

            # Check memory
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            min_memory_gb = self.resource_config.get('min_memory_gb', 2.0)

            if available_gb < min_memory_gb:
                logger.warning(f"⚠️  Only {available_gb:.1f}GB memory available, need {min_memory_gb}GB")
                return False

            # Check GPU if enabled
            gpu_config = self.config.get('gpu', {})
            if gpu_config.get('enable', False):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("⚠️  GPU enabled but not available")
                        return False

                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    min_gpu_memory_gb = self.resource_config.get('min_gpu_memory_gb', 4.0)

                    if gpu_memory < min_gpu_memory_gb:
                        logger.warning(f"⚠️  Only {gpu_memory:.1f}GB GPU memory available, need {min_gpu_memory_gb}GB")
                        return False

                except ImportError:
                    logger.warning("⚠️  PyTorch not available for GPU check")
                    return False

            return True

        except Exception as e:
            logger.warning(f"⚠️  Resource check failed: {e}")
            return False

    def allocate_resources(self, n_jobs: int) -> Dict[str, Any]:
        """Allocate resources for optimization."""
        allocation = {
            'cpu_cores': min(n_jobs, multiprocessing.cpu_count()),
            'memory_limit': self.resource_config.get('memory_limit_gb', 8.0),
            'timeout': self.resource_config.get('timeout_seconds', 3600)
        }

        # GPU allocation
        if self.config.get('gpu', {}).get('enable', False):
            try:
                import torch
                allocation['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
                allocation['gpu_device'] = 0
            except:
                pass

        return allocation

    def monitor_resources(self) -> Dict[str, float]:
        """Monitor current resource usage."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            monitoring = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info.percent,
                'memory_used_gb': memory_info.used / (1024**3),
                'memory_available_gb': memory_info.available / (1024**3)
            }

            # GPU monitoring
            if self.config.get('gpu', {}).get('enable', False):
                try:
                    import torch
                    gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                    monitoring['gpu_memory_used_gb'] = gpu_memory
                    monitoring['gpu_memory_cached_gb'] = gpu_memory_cached
                except:
                    pass

            return monitoring

        except Exception as e:
            logger.warning(f"⚠️  Resource monitoring failed: {e}")
            return {}


class OptunaOptimizer:
    """Optuna-based optimizer with multi-objective support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.optuna_config = self.opt_config.get('optuna', {})
        self.logger = logger

    def optimize(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                y_test: pd.Series = None) -> Dict[str, Any]:
        """Run Optuna optimization."""

        self.logger.info(f"🚀 Starting Optuna optimization for {model_name}")

        # Create study
        study = self._create_study(model_name)

        # Define objective function
        objective_func = self._get_objective_function(
            model_name, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Run optimization
        n_trials = self.opt_config.get('n_trials', 100)
        study.optimize(objective_func, n_trials=n_trials, timeout=self.opt_config.get('constraints', {}).get('max_optimization_time'))

        # Extract results
        results = self._extract_study_results(study)

        self.logger.info(f"✅ Optuna optimization completed for {model_name}")
        self.logger.info(f"Best value: {results['best_value']:.4f}")

        return results

    def _create_study(self, model_name: str) -> optuna.Study:
        """Create Optuna study with appropriate settings."""
        study_name = f"{model_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Study direction
        if self.opt_config.get('multi_objective', False):
            direction = 'minimize'  # For multi-objective, we minimize a composite score
        else:
            direction = self.optuna_config.get('direction', 'maximize')

        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )

        return study

    def _get_sampler(self):
        """Get Optuna sampler based on config."""
        sampler_type = self.optuna_config.get('sampler', 'tpe')

        if sampler_type == 'tpe':
            return optuna.samplers.TPESampler()
        elif sampler_type == 'random':
            return optuna.samplers.RandomSampler()
        elif sampler_type == 'cmaes':
            return optuna.samplers.CmaEsSampler()
        elif sampler_type == 'nsga2':
            return optuna.samplers.NSGAIISampler()
        else:
            return optuna.samplers.TPESampler()

    def _get_pruner(self):
        """Get Optuna pruner based on config."""
        pruner_type = self.optuna_config.get('pruner', 'median')

        if pruner_type == 'median':
            return optuna.pruners.MedianPruner()
        elif pruner_type == 'hyperband':
            return optuna.pruners.HyperbandPruner()
        elif pruner_type == 'patient':
            return optuna.pruners.PatientPruner()
        else:
            return optuna.pruners.MedianPruner()

    def _get_objective_function(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                               y_test: pd.Series = None):
        """Get appropriate objective function."""

        if self.opt_config.get('multi_objective', False):
            return lambda trial: self._multi_objective_function(
                trial, model_name, X_train, y_train, X_val, y_val, X_test, y_test
            )
        else:
            return lambda trial: self._single_objective_function(
                trial, model_name, X_train, y_train, X_val, y_val, X_test, y_test
            )

    def _single_objective_function(self, trial: optuna.Trial, model_name: str,
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  X_test: pd.DataFrame = None, y_test: pd.Series = None) -> float:
        """Single objective optimization function."""

        # Get hyperparameters
        params = self._suggest_parameters(trial, model_name)

        try:
            # Train and evaluate model
            if model_name in ['lightgbm', 'lgb']:
                score = self._evaluate_lightgbm(params, X_train, y_train, X_val, y_val, X_test, y_test)
            elif model_name in ['xgboost', 'xgb']:
                score = self._evaluate_xgboost(params, X_train, y_train, X_val, y_val, X_test, y_test)
            elif model_name in ['catboost', 'cb']:
                score = self._evaluate_catboost(params, X_train, y_train, X_val, y_val, X_test, y_test)
            elif model_name in ['randomforest', 'rf']:
                score = self._evaluate_randomforest(params, X_train, y_train, X_val, y_val, X_test, y_test)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            return score

        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -float('inf')

    def _multi_objective_function(self, trial: optuna.Trial, model_name: str,
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 X_test: pd.DataFrame = None, y_test: pd.Series = None) -> float:
        """Multi-objective optimization function."""

        # Get hyperparameters
        params = self._suggest_parameters(trial, model_name)

        try:
            # Train model and get predictions
            if model_name in ['lightgbm', 'lgb']:
                model = self._train_lightgbm(params, X_train, y_train, X_val, y_val)
                y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_name in ['xgboost', 'xgb']:
                model = self._train_xgboost(params, X_train, y_train, X_val, y_val)
                y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_name in ['catboost', 'cb']:
                model = self._train_catboost(params, X_train, y_train, X_val, y_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            elif model_name in ['randomforest', 'rf']:
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            # Calculate objectives
            objectives = self.opt_config.get('objectives', [])
            composite_score = 0.0

            for obj in objectives:
                metric_name = obj['metric']
                weight = obj.get('weight', 1.0)
                direction = obj['direction']
                target = obj.get('target', None)

                # Calculate metric value
                if metric_name == 'sharpe_ratio':
                    # Use predictions as proxy for returns
                    returns = calculate_returns(pd.Series(y_pred_proba))
                    if len(returns) > 0:
                        metric_value = sharpe_ratio(returns)
                    else:
                        metric_value = 0.0
                elif metric_name == 'max_drawdown':
                    # Calculate drawdown from prediction confidence
                    equity_curve = pd.Series(y_pred_proba.cumsum())
                    dd_metrics = maximum_drawdown(equity_curve)
                    metric_value = dd_metrics['max_drawdown_pct']
                else:
                    continue

                # Apply direction and target constraints
                if direction == 'maximize':
                    if target and metric_value < target:
                        metric_value = -abs(target - metric_value)  # Penalty
                    score_component = metric_value * weight
                else:  # minimize
                    if target and metric_value > target:
                        metric_value = abs(target - metric_value)  # Penalty
                    score_component = -metric_value * weight

                composite_score += score_component

            return composite_score

        except Exception as e:
            self.logger.warning(f"Multi-objective trial failed: {e}")
            return -float('inf')

    def _suggest_parameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Suggest hyperparameters based on model type."""

        if model_name in ['lightgbm', 'lgb']:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            }
        elif model_name in ['xgboost', 'xgb']:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0,
                'random_state': 42
            }
        elif model_name in ['catboost', 'cb']:
            return {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0.1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 128, 255),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50,
                'use_best_model': True
            }
        elif model_name in ['randomforest', 'rf']:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            raise ValueError(f"No parameter suggestions for model: {model_name}")

    def _evaluate_lightgbm(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                          y_test: pd.Series = None) -> float:
        """Evaluate LightGBM model with backtest integration."""

        # Train model
        model = self._train_lightgbm(params, X_train, y_train, X_val, y_val)

        # Get predictions
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

        # Use Sharpe ratio as objective
        returns = calculate_returns(pd.Series(y_pred_proba))
        if len(returns) > 0:
            return sharpe_ratio(returns)
        else:
            return 0.0

    def _evaluate_randomforest(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                              y_test: pd.Series = None) -> float:
        """Evaluate RandomForest model."""

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Use Sharpe ratio as objective
        returns = calculate_returns(pd.Series(y_pred_proba))
        if len(returns) > 0:
            return sharpe_ratio(returns)
        else:
            return 0.0

    def _evaluate_xgboost(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                          y_test: pd.Series = None) -> float:
        """Evaluate XGBoost model with backtest integration."""

        # Train model
        model = self._train_xgboost(params, X_train, y_train, X_val, y_val)

        # Get predictions
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

        # Use Sharpe ratio as objective
        returns = calculate_returns(pd.Series(y_pred_proba))
        if len(returns) > 0:
            return sharpe_ratio(returns)
        else:
            return 0.0

    def _train_lightgbm(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series):
        """Train LightGBM model with early stopping."""

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        train_params = {
            'num_boost_round': params['n_estimators'],
            'early_stopping_rounds': self.optuna_config.get('early_stopping_rounds', 20),
            'verbose_eval': False
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            **train_params
        )

        return model

    def _train_xgboost(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series):
        """Train XGBoost model with early stopping."""

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        train_params = params.copy()
        num_boost_round = train_params.pop('n_estimators', 100)

        train_params.update({
            'early_stopping_rounds': self.optuna_config.get('early_stopping_rounds', 20)
        })

        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            verbose_eval=False
        )

        return model

    def _evaluate_catboost(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                           y_test: pd.Series = None) -> float:
        """Evaluate CatBoost model with backtest integration."""

        # Train model
        model = self._train_catboost(params, X_train, y_train, X_val, y_val)

        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Use Sharpe ratio as objective
        returns = calculate_returns(pd.Series(y_pred_proba))
        if len(returns) > 0:
            return sharpe_ratio(returns)
        else:
            return 0.0

    def _train_catboost(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series):
        """Train CatBoost model with early stopping."""

        # Create model
        model = cb.CatBoostClassifier(**params)

        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            use_best_model=params.get('use_best_model', True),
            verbose=False
        )

        return model

    def _extract_study_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Extract comprehensive results from Optuna study."""

        # Get best parameters and value
        best_params = study.best_params
        best_value = study.best_value

        # Get trials dataframe
        trials_df = study.trials_dataframe()

        # Calculate optimization statistics
        stats = {
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.TrialState.FAIL]),
            'best_trial_number': study.best_trial.number,
            'optimization_time': sum([t.duration.total_seconds() for t in study.trials if t.duration]),
            'average_trial_time': np.mean([t.duration.total_seconds() for t in study.trials if t.duration])
        }

        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study,
            'trials_df': trials_df,
            'statistics': stats,
            'method': 'optuna'
        }


class GridSearchOptimizer:
    """Grid Search optimizer for systematic parameter exploration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.grid_config = self.opt_config.get('grid_search', {})
        self.logger = logger

    def optimize(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                y_test: pd.Series = None) -> Dict[str, Any]:
        """Run Grid Search optimization."""

        self.logger.info(f"🔍 Starting Grid Search optimization for {model_name}")

        # Define parameter grid
        param_grid = self._get_parameter_grid(model_name)

        # Create model
        if model_name in ['randomforest', 'rf']:
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Grid Search not supported for model: {model_name}")

        # Setup cross-validation
        cv = self.grid_config.get('cv_folds', 5)
        scoring = self.grid_config.get('scoring', 'neg_mean_squared_error')
        n_jobs = self.grid_config.get('n_jobs', -1)

        # Run grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        # Fit on training data
        grid_search.fit(X_train, y_train)

        # Evaluate on validation set
        best_model = grid_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        returns = calculate_returns(pd.Series(y_pred_proba))
        sharpe = sharpe_ratio(returns) if len(returns) > 0 else 0.0

        # Extract results
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'sharpe_ratio': sharpe,
            'cv_results': grid_search.cv_results_,
            'method': 'grid_search'
        }

        self.logger.info(f"✅ Grid Search completed for {model_name}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")

        return results

    def _get_parameter_grid(self, model_name: str) -> Dict[str, List]:
        """Get parameter grid for model."""

        if model_name in ['randomforest', 'rf']:
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        else:
            return {}


class StudyPersistence:
    """Handle persistence of optimization studies and results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.persistence_config = self.opt_config.get('persistence', {})
        self.logger = logger

    def save_results(self, model_name: str, results: Dict[str, Any]):
        """Save optimization results to disk."""

        if not self.persistence_config.get('enabled', True):
            return

        # Create directory
        study_dir = self.persistence_config.get('study_dir', 'outputs/optuna_studies')
        ensure_dir(study_dir)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{model_name}_{timestamp}"

        # Save best parameters
        params_file = f"{study_dir}/{base_filename}_best_params.json"
        with open(params_file, 'w') as f:
            json.dump(results['best_params'], f, indent=2)

        # Save full results
        results_file = f"{study_dir}/{base_filename}_results.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)

        # Save Optuna study if available
        if 'study' in results and results['study']:
            study_file = f"{study_dir}/{base_filename}_study.pkl"
            with open(study_file, 'wb') as f:
                pickle.dump(results['study'], f)

        self.logger.info(f"💾 Optimization results saved to {study_dir}")

    def load_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load previous optimization results."""

        if not self.persistence_config.get('load_existing', True):
            return None

        study_dir = self.persistence_config.get('study_dir', 'outputs/optuna_studies')

        if not Path(study_dir).exists():
            return None

        # Find most recent results for model
        result_files = list(Path(study_dir).glob(f"{model_name}_*_results.json"))

        if not result_files:
            return None

        # Load most recent
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)

            self.logger.info(f"📂 Loaded previous results from {latest_file}")
            return results

        except Exception as e:
            self.logger.warning(f"Failed to load previous results: {e}")
            return None

    def _make_serializable(self, obj):
        """Make object serializable by removing non-serializable components."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                   if k not in ['study', 'optimizer_result']}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


class OptimizationReporter:
    """Generate comprehensive optimization reports and visualizations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.reporting_config = self.opt_config.get('reporting', {})
        self.logger = logger

    def generate_report(self, model_name: str, results: Dict[str, Any]):
        """Generate comprehensive optimization report."""

        if not self.reporting_config.get('enabled', True):
            return

        self.logger.info(f"📊 Generating optimization report for {model_name}")

        # Create reports directory
        ensure_dir("outputs/reports")

        # Generate different report formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Markdown report
        if self.reporting_config.get('report_format', 'markdown') == 'markdown':
            self._generate_markdown_report(model_name, results, timestamp)

        # JSON report
        self._generate_json_report(model_name, results, timestamp)

        # Visualizations
        if self.reporting_config.get('include_visualizations', True) and PLOTLY_AVAILABLE:
            self._generate_visualizations(model_name, results, timestamp)

    def _generate_markdown_report(self, model_name: str, results: Dict[str, Any], timestamp: str):
        """Generate Markdown optimization report."""

        report_content = f"# AXON Hyperparameter Optimization Report\n\n"
        report_content += f"**Model:** {model_name}\n"
        report_content += f"**Method:** {results.get('method', 'unknown')}\n"
        report_content += f"**Timestamp:** {timestamp}\n"
        report_content += f"**Best Value:** {results.get('best_value', 'N/A'):.4f}\n\n"

        # Best parameters
        report_content += "## Best Parameters\n\n"
        best_params = results.get('best_params', {})
        for param, value in best_params.items():
            report_content += f"- **{param}:** {value}\n"

        # Statistics
        if 'statistics' in results:
            stats = results['statistics']
            report_content += "\n## Optimization Statistics\n\n"
            report_content += f"- **Total Trials:** {stats.get('total_trials', 'N/A')}\n"
            report_content += f"- **Completed Trials:** {stats.get('completed_trials', 'N/A')}\n"
            report_content += f"- **Pruned Trials:** {stats.get('pruned_trials', 'N/A')}\n"
            report_content += f"- **Failed Trials:** {stats.get('failed_trials', 'N/A')}\n"
            report_content += f"- **Average Trial Time:** {stats.get('average_trial_time', 'N/A'):.2f}s\n"

        # Top configurations
        if 'trials_df' in results and not results['trials_df'].empty:
            report_content += "\n## Top 5 Configurations\n\n"
            top_trials = results['trials_df'].nlargest(5, 'value')
            report_content += top_trials.to_markdown(index=False)

        # Parameter importance
        if 'trials_df' in results and not results['trials_df'].empty:
            report_content += "\n## Parameter Importance\n\n"
            trials_df = results['trials_df']
            correlations = trials_df.corr()['value'].drop('value', errors='ignore')

            for param, corr in correlations.items():
                if not pd.isna(corr):
                    impact = "Positive" if corr > 0 else "Negative"
                    report_content += f"- **{param}:** {corr:.3f} ({impact} impact)\n"

        # Save report
        report_path = f"outputs/reports/{model_name}_optimization_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"📄 Markdown report saved: {report_path}")

    def _generate_json_report(self, model_name: str, results: Dict[str, Any], timestamp: str):
        """Generate JSON optimization report."""

        # Make results serializable
        serializable_results = StudyPersistence(self.config)._make_serializable(results)

        report_path = f"outputs/reports/{model_name}_optimization_results_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        self.logger.info(f"📋 JSON report saved: {report_path}")

    def _generate_visualizations(self, model_name: str, results: Dict[str, Any], timestamp: str):
        """Generate optimization visualizations."""

        if not PLOTLY_AVAILABLE:
            return

        # Create visualizations directory
        ensure_dir("outputs/figures")

        # Optimization history
        if 'trials_df' in results and not results['trials_df'].empty:
            self._plot_optimization_history(model_name, results['trials_df'], timestamp)

        # Parameter importance
        if 'trials_df' in results and not results['trials_df'].empty:
            self._plot_parameter_importance(model_name, results['trials_df'], timestamp)

    def _plot_optimization_history(self, model_name: str, trials_df: pd.DataFrame, timestamp: str):
        """Plot optimization history."""

        fig = go.Figure()

        # Best value over time
        best_values = trials_df['value'].expanding().max()

        fig.add_trace(go.Scatter(
            x=trials_df.index,
            y=best_values,
            mode='lines+markers',
            name='Best Value',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=trials_df.index,
            y=trials_df['value'],
            mode='markers',
            name='Trial Values',
            marker=dict(color='lightblue', size=6)
        ))

        fig.update_layout(
            title=f"Optimization History - {model_name}",
            xaxis_title="Trial Number",
            yaxis_title="Objective Value",
            template="plotly_white"
        )

        # Save plot
        plot_path = f"outputs/figures/{model_name}_optimization_history_{timestamp}.html"
        fig.write_html(plot_path)

        self.logger.info(f"📈 Optimization history plot saved: {plot_path}")

    def _plot_parameter_importance(self, model_name: str, trials_df: pd.DataFrame, timestamp: str):
        """Plot parameter importance."""

        # Calculate correlations
        correlations = trials_df.corr()['value'].drop('value', errors='ignore')
        correlations = correlations.abs().sort_values(ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker_color='lightgreen'
        ))

        fig.update_layout(
            title=f"Parameter Importance - {model_name}",
            xaxis_title="Absolute Correlation",
            yaxis_title="Parameter",
            template="plotly_white"
        )

        # Save plot
        plot_path = f"outputs/figures/{model_name}_parameter_importance_{timestamp}.html"
        fig.write_html(plot_path)

        self.logger.info(f"📊 Parameter importance plot saved: {plot_path}")


class ParallelOptimizationReporter:
    """Advanced reporter for parallel optimization results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.reporting_config = self.opt_config.get('reporting', {})
        self.logger = logger

    def generate_parallel_report(self, results: Dict[str, Any]):
        """Generate comprehensive parallel optimization report."""
        if not self.reporting_config.get('enabled', True):
            return

        self.logger.info("📊 Generating parallel optimization report")

        # Create reports directory
        ensure_dir("outputs/reports")

        # Generate different report formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Parallel optimization summary
        self._generate_parallel_summary_report(results, timestamp)

        # Model comparison matrix
        if len(results) > 1:
            self._generate_model_comparison_matrix(results, timestamp)

        # Performance stability analysis
        self._generate_stability_analysis(results, timestamp)

        # Convergence plots
        if PLOTLY_AVAILABLE:
            self._generate_convergence_plots(results, timestamp)

    def _generate_parallel_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Generate parallel optimization summary report."""
        report_content = "# AXON Parallel Multi-Model Optimization Report\n\n"
        report_content += f"**Timestamp:** {timestamp}\n"
        report_content += f"**Models Optimized:** {len([k for k in results.keys() if k != 'ensemble'])}\n"
        report_content += f"**Ensemble Included:** {'Yes' if 'ensemble' in results else 'No'}\n\n"

        # Individual model results
        report_content += "## Individual Model Results\n\n"

        for model_name, result in results.items():
            if model_name == 'ensemble':
                continue

            report_content += f"### {model_name.upper()}\n\n"
            report_content += f"- **Best Value:** {result.get('best_value', 'N/A'):.4f}\n"
            report_content += f"- **Method:** {result.get('method', 'N/A')}\n"

            if 'statistics' in result:
                stats = result['statistics']
                report_content += f"- **Trials:** {stats.get('total_trials', 'N/A')}\n"
                report_content += f"- **Optimization Time:** {stats.get('optimization_time', 0):.1f}s\n"

            report_content += "\n**Best Parameters:**\n"
            best_params = result.get('best_params', {})
            for param, value in best_params.items():
                report_content += f"  - {param}: {value}\n"
            report_content += "\n"

        # Ensemble results
        if 'ensemble' in results:
            ensemble_result = results['ensemble']
            report_content += "## Ensemble Results\n\n"
            report_content += f"- **Type:** {ensemble_result.get('ensemble_type', 'N/A')}\n"
            report_content += f"- **Best Sharpe:** {ensemble_result.get('best_sharpe', 'N/A'):.4f}\n"
            report_content += f"- **Weights:** {ensemble_result.get('weights', {})}\n\n"

        # Save report
        report_path = f"outputs/reports/parallel_optimization_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"📄 Parallel optimization report saved: {report_path}")

    def _generate_model_comparison_matrix(self, results: Dict[str, Any], timestamp: str):
        """Generate model comparison matrix."""
        # Extract metrics for comparison
        comparison_data = {}

        for model_name, result in results.items():
            if model_name == 'ensemble':
                continue

            metrics = {
                'Best_Value': result.get('best_value', 0),
                'Method': result.get('method', 'N/A'),
                'Trials': result.get('statistics', {}).get('total_trials', 0),
                'Opt_Time_s': result.get('statistics', {}).get('optimization_time', 0)
            }
            comparison_data[model_name] = metrics

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T

        # Save CSV
        csv_path = f"outputs/reports/model_comparison_matrix_{timestamp}.csv"
        comparison_df.to_csv(csv_path)

        self.logger.info(f"📊 Model comparison matrix saved: {csv_path}")

    def _generate_stability_analysis(self, results: Dict[str, Any], timestamp: str):
        """Generate performance stability analysis."""
        stability_report = "# Performance Stability Analysis\n\n"

        for model_name, result in results.items():
            if model_name == 'ensemble' or 'trials_df' not in result:
                continue

            trials_df = result['trials_df']
            if trials_df.empty:
                continue

            stability_report += f"## {model_name.upper()} Stability\n\n"

            # Calculate stability metrics
            values = trials_df['value'].dropna()
            if len(values) > 1:
                stability_report += f"- **Mean:** {values.mean():.4f}\n"
                stability_report += f"- **Std:** {values.std():.4f}\n"
                stability_report += f"- **CV:** {values.std()/values.mean():.4f}\n"
                stability_report += f"- **Best:** {values.max():.4f}\n"
                stability_report += f"- **Worst:** {values.min():.4f}\n"
                stability_report += f"- **Range:** {values.max() - values.min():.4f}\n\n"

        # Save stability report
        stability_path = f"outputs/reports/stability_analysis_{timestamp}.md"
        with open(stability_path, 'w', encoding='utf-8') as f:
            f.write(stability_report)

        self.logger.info(f"📈 Stability analysis saved: {stability_path}")

    def _generate_convergence_plots(self, results: Dict[str, Any], timestamp: str):
        """Generate convergence plots for all models."""
        if not PLOTLY_AVAILABLE:
            return

        ensure_dir("outputs/figures")

        # Create subplot figure
        fig = go.Figure()

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        for i, (model_name, result) in enumerate(results.items()):
            if model_name == 'ensemble' or 'trials_df' not in result:
                continue

            trials_df = result['trials_df']
            if trials_df.empty:
                continue

            color = colors[i % len(colors)]

            # Best value over time
            best_values = trials_df['value'].expanding().max()

            fig.add_trace(go.Scatter(
                x=trials_df.index,
                y=best_values,
                mode='lines+markers',
                name=f'{model_name} Best',
                line=dict(color=color, width=2),
                showlegend=True
            ))

        fig.update_layout(
            title="Parallel Optimization Convergence",
            xaxis_title="Trial Number",
            yaxis_title="Best Objective Value",
            template="plotly_white"
        )

        # Save plot
        plot_path = f"outputs/figures/parallel_convergence_{timestamp}.html"
        fig.write_html(plot_path)

        self.logger.info(f"📈 Parallel convergence plot saved: {plot_path}")


class EnsembleOptimizer:
    """Advanced ensemble optimization with multiple strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.ensemble_config = self.opt_config.get('ensemble', {})
        self.logger = logger

    def optimize_ensemble(self, model_results: Dict[str, Any], X_train: pd.DataFrame,
                         y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                         X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Optimize ensemble combining multiple models.

        Args:
            model_results: Dictionary with individual model optimization results
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data (optional)

        Returns:
            Dictionary with ensemble optimization results
        """
        self.logger.info("🎯 Starting ensemble optimization")

        # Extract successful models
        successful_models = {}
        for model_name, result in model_results.items():
            if 'error' not in result and 'best_params' in result:
                successful_models[model_name] = result

        if len(successful_models) < 2:
            self.logger.warning("⚠️  Need at least 2 successful models for ensemble")
            return {'error': 'Insufficient models for ensemble'}

        # Train models with optimized parameters
        trained_models = {}
        for model_name, result in successful_models.items():
            try:
                model = self._train_model_with_params(model_name, result['best_params'],
                                                    X_train, y_train, X_val, y_val)
                trained_models[model_name] = model
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to train {model_name}: {e}")

        if len(trained_models) < 2:
            return {'error': 'Failed to train enough models for ensemble'}

        # Optimize ensemble weights
        ensemble_result = self._optimize_ensemble_weights(
            trained_models, X_val, y_val, X_test, y_test
        )

        # Add ensemble metadata
        ensemble_result.update({
            'ensemble_type': self.ensemble_config.get('ensemble_type', 'weighted'),
            'base_models': list(trained_models.keys()),
            'optimization_method': 'performance_based_weighting'
        })

        self.logger.info("✅ Ensemble optimization completed")
        return ensemble_result

    def _train_model_with_params(self, model_name: str, best_params: Dict[str, Any],
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series):
        """Train a model with optimized parameters."""
        from .models import ModelRegistry

        registry = ModelRegistry(self.config)
        model = registry.get_model(model_name)

        # Merge best params with model config
        if isinstance(model, dict) and model.get('model_class') == 'lightgbm':
            # LightGBM special handling
            train_model_func = registry.train_model
            trained_model, _ = train_model_func(
                model, X_train, y_train, X_val, y_val, model_name, self.config
            )
            return trained_model
        else:
            # Standard scikit-learn interface
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            return model

    def _optimize_ensemble_weights(self, trained_models: Dict[str, Any],
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """Optimize ensemble weights using various strategies."""

        # Get predictions from all models
        predictions = {}
        for model_name, model in trained_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_val)[:, 1]
                else:
                    pred = model.predict(X_val).astype(float)
                    pred_proba = pred

                predictions[model_name] = pred_proba
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to get predictions from {model_name}: {e}")
                predictions[model_name] = np.zeros(len(X_val))

        # Calculate performance-based weights
        weights = self._calculate_performance_weights(predictions, y_val)

        # Evaluate ensemble performance
        ensemble_pred = self._combine_predictions(predictions, weights)
        ensemble_sharpe = self._calculate_sharpe_from_predictions(ensemble_pred)

        return {
            'weights': weights,
            'best_sharpe': ensemble_sharpe,
            'predictions': ensemble_pred,
            'individual_performances': {
                name: self._calculate_sharpe_from_predictions(pred)
                for name, pred in predictions.items()
            }
        }

    def _calculate_performance_weights(self, predictions: Dict[str, np.ndarray],
                                     y_true: pd.Series) -> Dict[str, float]:
        """Calculate weights based on individual model performance."""
        performances = {}

        for model_name, pred_proba in predictions.items():
            try:
                sharpe = self._calculate_sharpe_from_predictions(pred_proba)
                performances[model_name] = max(sharpe, 0.1)  # Minimum weight
            except:
                performances[model_name] = 0.1

        # Normalize weights
        total_perf = sum(performances.values())
        if total_perf > 0:
            weights = {name: perf / total_perf for name, perf in performances.items()}
        else:
            # Equal weights fallback
            n_models = len(performances)
            weights = {name: 1.0 / n_models for name in performances.keys()}

        return weights

    def _combine_predictions(self, predictions: Dict[str, np.ndarray],
                           weights: Dict[str, float]) -> np.ndarray:
        """Combine predictions using weighted average."""
        weighted_sum = np.zeros(len(next(iter(predictions.values()))))

        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            weighted_sum += weight * pred

        return weighted_sum

    def _calculate_sharpe_from_predictions(self, predictions: np.ndarray) -> float:
        """Calculate Sharpe ratio from prediction array."""
        try:
            from .metrics import sharpe_ratio, calculate_returns
            returns = calculate_returns(pd.Series(predictions))
            if len(returns) > 0:
                return sharpe_ratio(returns)
            else:
                return 0.0
        except:
            return 0.0


class MultiObjectiveOptimizer:
    """Multi-objective optimization using PyMOO with Pareto front analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.multi_config = self.opt_config.get('multi_objective', {})
        self.logger = logger

    def optimize_pareto_front(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                            y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Perform multi-objective optimization to find Pareto front.

        Args:
            model_name: Name of the model to optimize
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data (optional)

        Returns:
            Dictionary with Pareto front results
        """
        if not PYMOO_AVAILABLE:
            self.logger.warning("⚠️  PyMOO not available, falling back to single-objective optimization")
            optimizer = OptunaOptimizer(self.config)
            return optimizer.optimize(model_name, X_train, y_train, X_val, y_val, X_test, y_test)

        self.logger.info(f"🎯 Starting Pareto front optimization for {model_name}")

        # Define the multi-objective problem
        problem = TradingOptimizationProblem(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=self.config
        )

        # Select algorithm
        algorithm_name = self.multi_config.get('algorithm', 'nsga2')

        if algorithm_name == 'nsga2':
            algorithm = NSGA2(pop_size=self.multi_config.get('pop_size', 50))
        else:
            algorithm = NSGA2(pop_size=self.multi_config.get('pop_size', 50))

        # Run optimization
        n_generations = self.multi_config.get('n_generations', 100)

        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', n_generations),
            seed=42,
            verbose=True
        )

        # Extract Pareto front results
        pareto_results = self._extract_pareto_results(res, model_name)

        # Generate Pareto front visualization
        if PLOTLY_AVAILABLE and len(res.F) > 1:
            self._plot_pareto_front(res, model_name)

        self.logger.info(f"✅ Pareto front optimization completed for {model_name}")
        return pareto_results

    def _extract_pareto_results(self, res, model_name: str) -> Dict[str, Any]:
        """Extract comprehensive results from Pareto front optimization."""
        # Get Pareto optimal solutions
        pareto_front = res.F
        pareto_solutions = res.X

        # Find best compromise solution (closest to ideal point)
        ideal_point = np.array([1.0, 0.0])  # Max Sharpe, Min Drawdown
        distances = np.sqrt(np.sum((pareto_front - ideal_point)**2, axis=1))
        best_compromise_idx = np.argmin(distances)

        best_solution = pareto_solutions[best_compromise_idx]
        best_objectives = pareto_front[best_compromise_idx]

        # Convert solution back to parameter dictionary
        param_names = self._get_parameter_names(model_name)
        best_params = dict(zip(param_names, best_solution))

        return {
            'pareto_front': pareto_front,
            'pareto_solutions': pareto_solutions,
            'best_params': best_params,
            'best_objectives': {
                'sharpe_ratio': best_objectives[0],
                'max_drawdown': best_objectives[1]
            },
            'method': 'pareto_front',
            'n_solutions': len(pareto_front),
            'hypervolume': self._calculate_hypervolume(pareto_front) if len(pareto_front) > 1 else 0
        }

    def _get_parameter_names(self, model_name: str) -> List[str]:
        """Get parameter names for the model."""
        if model_name in ['lightgbm', 'lgb']:
            return ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples',
                   'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
        elif model_name in ['xgboost', 'xgb']:
            return ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                   'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']
        elif model_name in ['catboost', 'cb']:
            return ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
                   'random_strength', 'bagging_temperature', 'border_count']
        else:
            return ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

    def _calculate_hypervolume(self, pareto_front: np.ndarray) -> float:
        """Calculate hypervolume of Pareto front."""
        try:
            from pymoo.indicators.hv import HV
            ref_point = np.array([0.0, 1.0])  # Reference point (worse than any solution)
            ind = HV(ref_point=ref_point)
            return ind(pareto_front)
        except:
            return 0.0

    def _plot_pareto_front(self, res, model_name: str):
        """Plot Pareto front visualization."""
        if not PLOTLY_AVAILABLE:
            return

        ensure_dir("outputs/figures")

        fig = go.Figure()

        # Pareto front points
        fig.add_trace(go.Scatter(
            x=res.F[:, 0],
            y=res.F[:, 1],
            mode='markers',
            name='Pareto Front',
            marker=dict(color='blue', size=8)
        ))

        # Ideal point
        fig.add_trace(go.Scatter(
            x=[1.0],
            y=[0.0],
            mode='markers',
            name='Ideal Point',
            marker=dict(color='red', size=12, symbol='star')
        ))

        fig.update_layout(
            title=f"Pareto Front - {model_name}",
            xaxis_title="Sharpe Ratio (maximize)",
            yaxis_title="Max Drawdown (minimize)",
            template="plotly_white"
        )

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"outputs/figures/pareto_front_{model_name}_{timestamp}.html"
        fig.write_html(plot_path)

        self.logger.info(f"📈 Pareto front plot saved: {plot_path}")


class TradingOptimizationProblem:
    """Multi-objective optimization problem for trading models."""

    def __init__(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame = None,
                 y_test: pd.Series = None, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.config = config or {}

        # Define parameter bounds
        self._setup_parameter_bounds()

    def _setup_parameter_bounds(self):
        """Setup parameter bounds for optimization."""
        if self.model_name in ['lightgbm', 'lgb']:
            self.param_bounds = {
                'n_estimators': (50, 500),
                'max_depth': (3, 15),
                'num_leaves': (20, 150),
                'min_child_samples': (5, 100),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0.0, 1.0),
                'reg_lambda': (0.0, 1.0)
            }
        elif self.model_name in ['xgboost', 'xgb']:
            self.param_bounds = {
                'n_estimators': (50, 500),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_weight': (1, 10),
                'gamma': (0.0, 1.0),
                'reg_alpha': (0.0, 1.0),
                'reg_lambda': (0.0, 1.0)
            }
        else:
            self.param_bounds = {
                'n_estimators': (50, 300),
                'max_depth': (5, 30),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }

        # Create bounds arrays for PyMOO
        self.param_names = list(self.param_bounds.keys())
        self.n_var = len(self.param_names)
        self.xl = np.array([bounds[0] for bounds in self.param_bounds.values()])
        self.xu = np.array([bounds[1] for bounds in self.param_bounds.values()])

    def _evaluate(self, X: np.ndarray, out: Dict[str, np.ndarray], *args, **kwargs):
        """Evaluate objective functions for a population of solutions."""
        n_solutions = X.shape[0]
        F = np.zeros((n_solutions, 2))  # 2 objectives: Sharpe (max), Drawdown (min)

        for i in range(n_solutions):
            params = dict(zip(self.param_names, X[i]))

            try:
                # Evaluate model with parameters
                objectives = self._evaluate_model(params)
                F[i, 0] = -objectives['sharpe_ratio']  # Minimize negative Sharpe
                F[i, 1] = objectives['max_drawdown']   # Minimize drawdown

            except Exception as e:
                # Penalize failed evaluations
                F[i, 0] = 10.0  # Large negative Sharpe
                F[i, 1] = 1.0   # Large drawdown

        out["F"] = F

    def _evaluate_model(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model with given parameters."""
        try:
            # Create and train model
            if self.model_name in ['lightgbm', 'lgb']:
                model = self._train_lightgbm(params)
                y_pred_proba = model.predict(self.X_val, num_iteration=model.best_iteration)
            elif self.model_name in ['xgboost', 'xgb']:
                model = self._train_xgboost(params)
                y_pred_proba = model.predict(self.X_val, num_iteration=model.best_iteration)
            elif self.model_name in ['catboost', 'cb']:
                model = self._train_catboost(params)
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            else:
                # RandomForest fallback
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(self.X_train, self.y_train)
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]

            # Calculate objectives
            from .metrics import sharpe_ratio, maximum_drawdown, calculate_returns

            returns = calculate_returns(pd.Series(y_pred_proba))
            sharpe = sharpe_ratio(returns) if len(returns) > 0 else 0.0

            equity_curve = pd.Series(y_pred_proba.cumsum())
            dd_metrics = maximum_drawdown(equity_curve)
            max_dd = dd_metrics['max_drawdown_pct']

            return {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            }

        except Exception as e:
            return {
                'sharpe_ratio': -10.0,  # Very bad Sharpe
                'max_drawdown': 1.0     # Very bad drawdown
            }

    def _train_lightgbm(self, params: Dict[str, Any]):
        """Train LightGBM model."""
        import lightgbm as lgb

        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)

        train_params = {
            'num_boost_round': params['n_estimators'],
            'early_stopping_rounds': 20,
            'verbose_eval': False
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            **train_params
        )

        return model

    def _train_xgboost(self, params: Dict[str, Any]):
        """Train XGBoost model."""
        import xgboost as xgb

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)

        train_params = params.copy()
        num_boost_round = train_params.pop('n_estimators', 100)

        train_params.update({
            'early_stopping_rounds': 20
        })

        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            verbose_eval=False
        )

        return model

    def _train_catboost(self, params: Dict[str, Any]):
        """Train CatBoost model."""
        import catboost as cb

        model = cb.CatBoostClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        return model


class TimeSeriesCrossValidator:
    """Rigorous time series cross-validation for financial models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_config = config.get('optimization', {}).get('cross_validation', {})
        self.logger = logger

    def validate_model_stability(self, model_name: str, best_params: Dict[str, Any],
                               X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform rigorous time series cross-validation to assess model stability.

        Args:
            model_name: Name of the model
            best_params: Optimized parameters
            X, y: Full dataset
            n_splits: Number of CV splits

        Returns:
            Dictionary with stability analysis results
        """
        self.logger.info(f"🔍 Starting time series cross-validation for {model_name}")

        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_results = []
        predictions_all = []
        labels_all = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"📊 CV Fold {fold + 1}/{n_splits}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            try:
                # Train model on fold
                fold_result = self._train_and_evaluate_fold(
                    model_name, best_params, X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold, fold
                )

                cv_results.append(fold_result)

                # Collect predictions for overall analysis
                predictions_all.extend(fold_result['predictions'])
                labels_all.extend(y_val_fold.tolist())

            except Exception as e:
                self.logger.error(f"❌ Failed CV fold {fold}: {e}")
                cv_results.append({'fold': fold, 'error': str(e)})

        # Analyze stability
        stability_analysis = self._analyze_stability(cv_results, predictions_all, labels_all)

        # Generate stability report
        self._generate_stability_report(model_name, stability_analysis)

        return {
            'cv_results': cv_results,
            'stability_analysis': stability_analysis,
            'n_splits': n_splits,
            'method': 'time_series_cv'
        }

    def _train_and_evaluate_fold(self, model_name: str, params: Dict[str, Any],
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series, fold: int) -> Dict[str, Any]:
        """Train and evaluate model on a single CV fold."""
        try:
            # Create and train model
            if model_name in ['lightgbm', 'lgb']:
                model = self._train_lightgbm(params, X_train, y_train, X_val, y_val)
                predictions = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_name in ['xgboost', 'xgb']:
                model = self._train_xgboost(params, X_train, y_train, X_val, y_val)
                predictions = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_name in ['catboost', 'cb']:
                model = self._train_catboost(params, X_train, y_train, X_val, y_val)
                predictions = model.predict_proba(X_val)[:, 1]
            else:
                # RandomForest
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict_proba(X_val)[:, 1]

            # Calculate fold metrics
            from .metrics import sharpe_ratio, maximum_drawdown, calculate_returns

            returns = calculate_returns(pd.Series(predictions))
            sharpe = sharpe_ratio(returns) if len(returns) > 0 else 0.0

            equity_curve = pd.Series(predictions.cumsum())
            dd_metrics = maximum_drawdown(equity_curve)
            max_dd = dd_metrics['max_drawdown_pct']

            # Classification metrics
            pred_binary = (predictions > 0.5).astype(int)
            accuracy = (pred_binary == y_val.values).mean()

            return {
                'fold': fold,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'accuracy': accuracy,
                'predictions': predictions.tolist(),
                'true_labels': y_val.tolist()
            }

        except Exception as e:
            return {
                'fold': fold,
                'error': str(e),
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'accuracy': 0.0
            }

    def _analyze_stability(self, cv_results: List[Dict], predictions_all: List[float],
                         labels_all: List[int]) -> Dict[str, Any]:
        """Analyze stability across CV folds."""
        # Extract metrics from successful folds
        successful_folds = [r for r in cv_results if 'error' not in r]

        if not successful_folds:
            return {'error': 'No successful CV folds'}

        sharpe_ratios = [r['sharpe_ratio'] for r in successful_folds]
        max_drawdowns = [r['max_drawdown'] for r in successful_folds]
        accuracies = [r['accuracy'] for r in successful_folds]

        # Calculate stability metrics
        stability = {
            'sharpe_mean': np.mean(sharpe_ratios),
            'sharpe_std': np.std(sharpe_ratios),
            'sharpe_cv': np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else float('inf'),
            'sharpe_min': np.min(sharpe_ratios),
            'sharpe_max': np.max(sharpe_ratios),

            'drawdown_mean': np.mean(max_drawdowns),
            'drawdown_std': np.std(max_drawdowns),
            'drawdown_cv': np.std(max_drawdowns) / np.mean(max_drawdowns) if np.mean(max_drawdowns) != 0 else float('inf'),

            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),

            'n_successful_folds': len(successful_folds),
            'total_folds': len(cv_results)
        }

        # Stability classification
        if stability['sharpe_cv'] < 0.2 and stability['drawdown_cv'] < 0.3:
            stability['stability_rating'] = 'High'
        elif stability['sharpe_cv'] < 0.5 and stability['drawdown_cv'] < 0.7:
            stability['stability_rating'] = 'Medium'
        else:
            stability['stability_rating'] = 'Low'

        return stability

    def _generate_stability_report(self, model_name: str, stability: Dict[str, Any]):
        """Generate stability analysis report."""
        ensure_dir("outputs/reports")

        report_content = f"# Model Stability Analysis - {model_name}\n\n"
        report_content += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if 'error' in stability:
            report_content += f"❌ Error: {stability['error']}\n"
        else:
            report_content += "## Stability Metrics\n\n"
            report_content += f"- **Stability Rating:** {stability['stability_rating']}\n"
            report_content += f"- **Successful Folds:** {stability['n_successful_folds']}/{stability['total_folds']}\n\n"

            report_content += "### Sharpe Ratio Stability\n"
            report_content += f"- Mean: {stability['sharpe_mean']:.4f}\n"
            report_content += f"- Std: {stability['sharpe_std']:.4f}\n"
            report_content += f"- CV: {stability['sharpe_cv']:.4f}\n"
            report_content += f"- Range: [{stability['sharpe_min']:.4f}, {stability['sharpe_max']:.4f}]\n\n"

            report_content += "### Drawdown Stability\n"
            report_content += f"- Mean: {stability['drawdown_mean']:.4f}\n"
            report_content += f"- Std: {stability['drawdown_std']:.4f}\n"
            report_content += f"- CV: {stability['drawdown_cv']:.4f}\n\n"

            report_content += "### Accuracy Stability\n"
            report_content += f"- Mean: {stability['accuracy_mean']:.4f}\n"
            report_content += f"- Std: {stability['accuracy_std']:.4f}\n"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"outputs/reports/stability_analysis_{model_name}_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"📊 Stability report saved: {report_path}")

    def _train_lightgbm(self, params, X_train, y_train, X_val, y_val):
        """Train LightGBM model for CV."""
        import lightgbm as lgb

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        train_params = {
            'num_boost_round': params.get('n_estimators', 100),
            'early_stopping_rounds': 20,
            'verbose_eval': False
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            **train_params
        )

        return model

    def _train_xgboost(self, params, X_train, y_train, X_val, y_val):
        """Train XGBoost model for CV."""
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        train_params = params.copy()
        num_boost_round = train_params.pop('n_estimators', 100)

        train_params.update({
            'early_stopping_rounds': 20
        })

        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            verbose_eval=False
        )

        return model

    def _train_catboost(self, params, X_train, y_train, X_val, y_val):
        """Train CatBoost model for CV."""
        import catboost as cb

        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        return model


class RegimeBasedSelector:
    """Regime-specific model selection and weighting."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_config = config.get('optimization', {}).get('regime_detection', {})
        self.logger = logger

    def detect_market_regime(self, X: pd.DataFrame, y: pd.Series = None) -> str:
        """
        Detect current market regime for adaptive model selection.

        Args:
            X: Feature data
            y: Target data (optional)

        Returns:
            Regime classification string
        """
        try:
            # Calculate volatility (rolling standard deviation)
            if hasattr(X, 'rolling'):
                volatility = X.rolling(window=self.regime_config.get('volatility_window', 20)).std().mean(axis=1).iloc[-1]
            else:
                volatility = np.std(X, axis=0).mean()

            # Calculate trend (SMA difference)
            if hasattr(X, 'rolling'):
                sma_short = X.rolling(window=10).mean().iloc[-1].mean()
                sma_long = X.rolling(window=30).mean().iloc[-1].mean()
                trend = 'bullish' if sma_short > sma_long else 'bearish'
            else:
                trend = 'neutral'

            # Classify regime
            vol_threshold = self.regime_config.get('volatility_threshold', 0.02)

            if volatility > vol_threshold * 1.5:
                regime = 'high_volatility'
            elif trend == 'bullish':
                regime = 'bull_trend'
            elif trend == 'bearish':
                regime = 'bear_trend'
            else:
                regime = 'neutral'

            self.logger.info(f"🎯 Detected market regime: {regime} (volatility: {volatility:.4f})")
            return regime

        except Exception as e:
            self.logger.warning(f"⚠️  Regime detection failed: {e}")
            return 'neutral'

    def get_regime_weights(self, regime: str, model_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Get regime-specific model weights.

        Args:
            regime: Current market regime
            model_results: Individual model optimization results

        Returns:
            Dictionary of model weights for the regime
        """
        # Regime-specific weight adjustments
        regime_multipliers = {
            'high_volatility': {
                'lightgbm': 1.2, 'lgb': 1.2,
                'xgboost': 1.1, 'xgb': 1.1,
                'catboost': 1.0, 'cb': 1.0,
                'lstm': 0.8,
                'randomforest': 0.9, 'rf': 0.9
            },
            'bull_trend': {
                'lightgbm': 1.1, 'lgb': 1.1,
                'xgboost': 1.0, 'xgb': 1.0,
                'catboost': 1.2, 'cb': 1.2,
                'lstm': 0.9,
                'randomforest': 0.8, 'rf': 0.8
            },
            'bear_trend': {
                'lightgbm': 1.0, 'lgb': 1.0,
                'xgboost': 1.2, 'xgb': 1.2,
                'catboost': 1.1, 'cb': 1.1,
                'lstm': 0.8,
                'randomforest': 0.9, 'rf': 0.9
            },
            'neutral': {
                'lightgbm': 1.0, 'lgb': 1.0,
                'xgboost': 1.0, 'xgb': 1.0,
                'catboost': 1.0, 'cb': 1.0,
                'lstm': 1.0,
                'randomforest': 1.0, 'rf': 1.0
            }
        }

        multipliers = regime_multipliers.get(regime, regime_multipliers['neutral'])

        # Calculate base weights from performance
        base_weights = {}
        total_performance = 0

        for model_name, result in model_results.items():
            if 'best_value' in result:
                performance = max(result['best_value'], 0.1)  # Minimum performance
                base_weights[model_name] = performance
                total_performance += performance

        # Apply regime multipliers and normalize
        regime_weights = {}
        total_weight = 0

        for model_name in base_weights.keys():
            multiplier = multipliers.get(model_name, 1.0)
            regime_weights[model_name] = base_weights[model_name] * multiplier
            total_weight += regime_weights[model_name]

        # Normalize to sum to 1
        if total_weight > 0:
            regime_weights = {name: w / total_weight for name, w in regime_weights.items()}

        self.logger.info(f"⚖️  Regime '{regime}' weights: {regime_weights}")
        return regime_weights


def main():
    """Advanced multi-model optimization pipeline with parallel processing."""
    print("=== AXON Advanced Multi-Model Optimization ===")

    # Load configuration
    config = load_config()

    # Check if optimization is enabled
    opt_config = config.get('optimization', {})
    if not opt_config.get('enabled', False):
        print("❌ Optimization disabled in configuration")
        return

    # Load processed features
    print("📂 Loading processed features...")

    train_path = "data/processed/train_features.parquet"
    val_path = "data/processed/validation_features.parquet"

    if not Path(train_path).exists() or not Path(val_path).exists():
        raise FileNotFoundError("❌ Processed features not found. Run feature engineering first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    print(f"✅ Train data: {train_df.shape}")
    print(f"✅ Validation data: {val_df.shape}")

    # Prepare features and labels
    target_col = config.get('target', 'y')
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Target distribution (train): {y_train.value_counts().to_dict()}")

    # Get models to optimize
    models_to_optimize = opt_config.get('models', ['lightgbm'])
    optimization_method = opt_config.get('method', 'parallel')

    print(f"🎯 Optimization method: {optimization_method}")
    print(f"🏃 Models to optimize: {models_to_optimize}")

    all_results = {}

    if optimization_method == 'parallel':
        # Use parallel optimization
        print(f"\n{'='*60}")
        print("🚀 STARTING PARALLEL MULTI-MODEL OPTIMIZATION")
        print(f"{'='*60}")

        parallel_engine = ParallelOptimizationEngine(config)

        try:
            results = parallel_engine.optimize_multiple_models(
                models_to_optimize, X_train, y_train, X_val, y_val
            )
            all_results.update(results)

            print("✅ Parallel optimization completed!")

        except Exception as e:
            print(f"❌ Parallel optimization failed: {e}")
            # Fallback to sequential
            optimization_method = 'sequential'

    if optimization_method in ['pareto', 'auto'] or optimization_method != 'parallel':
        # Use Pareto front optimization for individual models
        print(f"\n{'='*60}")
        print("🎯 PARETO FRONT MULTI-OBJECTIVE OPTIMIZATION")
        print(f"{'='*60}")

        pareto_optimizer = MultiObjectiveOptimizer(config)

        for model_name in models_to_optimize:
            print(f"\n🎯 Optimizing {model_name.upper()} with Pareto front")

            try:
                results = pareto_optimizer.optimize_pareto_front(
                    model_name, X_train, y_train, X_val, y_val
                )
                all_results[model_name] = results

                print(f"✅ {model_name} Pareto optimization completed!")
                print(f"   Solutions found: {results.get('n_solutions', 0)}")
                print(f"   Hypervolume: {results.get('hypervolume', 0):.4f}")

            except Exception as e:
                print(f"❌ Failed Pareto optimization for {model_name}: {e}")
                continue

    # Perform stability analysis if enabled
    if opt_config.get('validation', {}).get('enabled', True):
        print(f"\n{'='*60}")
        print("🔍 MODEL STABILITY ANALYSIS")
        print(f"{'='*60}")

        stability_validator = TimeSeriesCrossValidator(config)

        # Combine train and validation for full CV
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])

        for model_name, results in all_results.items():
            if model_name == 'ensemble' or 'best_params' not in results:
                continue

            print(f"\n🔍 Analyzing stability for {model_name}")

            try:
                stability_results = stability_validator.validate_model_stability(
                    model_name, results['best_params'], X_full, y_full,
                    n_splits=opt_config.get('cross_validation', {}).get('n_splits', 5)
                )

                results['stability_analysis'] = stability_results
                print(f"✅ Stability analysis completed for {model_name}")

            except Exception as e:
                print(f"❌ Stability analysis failed for {model_name}: {e}")

    # Regime-based analysis
    if opt_config.get('regime_detection', {}).get('enabled', True):
        print(f"\n{'='*60}")
        print("🎭 REGIME-BASED MODEL ANALYSIS")
        print(f"{'='*60}")

        regime_selector = RegimeBasedSelector(config)
        current_regime = regime_selector.detect_market_regime(X_val, y_val)

        print(f"🎯 Current market regime: {current_regime}")

        # Get regime-specific weights
        regime_weights = regime_selector.get_regime_weights(current_regime, all_results)

        # Store regime analysis
        all_results['regime_analysis'] = {
            'current_regime': current_regime,
            'regime_weights': regime_weights
        }

    # Final model comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("🏆 FINAL OPTIMIZATION COMPARISON")
        print(f"{'='*60}")

        comparison_data = {}
        for model_name, results in all_results.items():
            if model_name in ['ensemble', 'regime_analysis']:
                continue

            best_value = results.get('best_value', results.get('best_objectives', {}).get('sharpe_ratio', 0))
            method = results.get('method', 'unknown')
            trials = results.get('statistics', {}).get('total_trials', 'N/A')
            stability = results.get('stability_analysis', {}).get('stability_rating', 'N/A')

            comparison_data[model_name] = {
                'Best_Value': f"{best_value:.4f}",
                'Method': method,
                'Trials': trials,
                'Stability': stability
            }

        comparison_df = pd.DataFrame(comparison_data).T
        print(comparison_df)

        # Save comparison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = f"outputs/reports/final_optimization_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_path)
        print(f"\n💾 Final comparison saved: {comparison_path}")

        # Find best model
        best_model_name = max(
            [name for name in all_results.keys() if name not in ['ensemble', 'regime_analysis']],
            key=lambda x: all_results[x].get('best_value',
                all_results[x].get('best_objectives', {}).get('sharpe_ratio', 0))
        )

        best_performance = all_results[best_model_name].get('best_value',
            all_results[best_model_name].get('best_objectives', {}).get('sharpe_ratio', 0))

        print(f"\n🏆 Best performing model: {best_model_name} (Performance: {best_performance:.4f})")

    print(f"\n✅ Advanced optimization pipeline completed successfully!")
    print(f"📊 Results saved in outputs/optuna_studies/")
    print(f"📋 Reports saved in outputs/reports/")
    print(f"📈 Visualizations saved in outputs/figures/")


if __name__ == "__main__":
    main()