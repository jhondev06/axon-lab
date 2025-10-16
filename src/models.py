"""AXON Models Module

Model registry and training utilities.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports for LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    PYTORCH_AVAILABLE = True

    # Check for GPU support
    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device('cuda')
        print("PyTorch GPU support detected")
    else:
        TORCH_DEVICE = torch.device('cpu')
        print("PyTorch using CPU")

except ImportError:
    PYTORCH_AVAILABLE = False
    TORCH_DEVICE = torch.device('cpu')
    print("PyTorch not installed - LSTM model unavailable")

from .utils import load_config, ensure_dir


class XGBoostModel:
    """XGBoost model wrapper with GPU support and scikit-learn interface."""

    def __init__(self, **params):
        self.params = params.copy()
        self.model = None
        self.feature_names = None

        # Set default parameters
        defaults = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
        }

        # Add GPU support if available
        if XGB_GPU_AVAILABLE:
            defaults['tree_method'] = 'gpu_hist'
            defaults['gpu_id'] = 0
        else:
            defaults['tree_method'] = 'hist'

        # Update with provided params
        defaults.update(self.params)
        self.params = defaults

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """Fit the XGBoost model."""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        dtrain = xgb.DMatrix(X, label=y)

        eval_list = []
        if eval_set is not None:
            if isinstance(eval_set, list) and len(eval_set) == 2:
                X_val, y_val = eval_set
                if hasattr(X_val, 'values'):
                    X_val = X_val.values
                if hasattr(y_val, 'values'):
                    y_val = y_val.values
                dval = xgb.DMatrix(X_val, label=y_val)
                eval_list = [(dtrain, 'train'), (dval, 'validation')]
            else:
                eval_list = [(dtrain, 'train')]

        # Training parameters
        train_params = self.params.copy()
        num_boost_round = train_params.pop('n_estimators', 100)

        if early_stopping_rounds and eval_list:
            train_params['early_stopping_rounds'] = early_stopping_rounds

        callbacks = []
        if not verbose:
            callbacks.append(xgb.callback.EvaluationMonitor(show_stdv=False))

        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=eval_list if eval_list else None,
            callbacks=callbacks,
            verbose_eval=verbose
        )

        return self

    def predict(self, X, num_iteration=None):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(X, 'values'):
            X = X.values

        dtest = xgb.DMatrix(X)

        # Use best_iteration if available (when early stopping was used), otherwise use all trees
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            iteration_range = (0, num_iteration or self.model.best_iteration)
        else:
            iteration_range = (0, num_iteration or self.model.num_boosted_rounds())

        pred_proba = self.model.predict(dtest, iteration_range=iteration_range)
        return (pred_proba > 0.5).astype(int)

    def predict_proba(self, X, num_iteration=None):
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(X, 'values'):
            X = X.values

        dtest = xgb.DMatrix(X)

        # Use best_iteration if available (when early stopping was used), otherwise use all trees
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            iteration_range = (0, num_iteration or self.model.best_iteration)
        else:
            iteration_range = (0, num_iteration or self.model.num_boosted_rounds())

        pred_proba = self.model.predict(dtest, iteration_range=iteration_range)

        # Return probabilities for both classes
        return np.column_stack([1 - pred_proba, pred_proba])

    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        importance = self.model.get_score(importance_type=importance_type)
        return importance

    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        self.model.save_model(filepath)

    @classmethod
    def load_model(cls, filepath):
        """Load model from file."""
        instance = cls()
        instance.model = xgb.Booster()
        instance.model.load_model(filepath)
        return instance


class CatBoostModel:
    """CatBoost model wrapper with GPU support and scikit-learn interface."""

    def __init__(self, **params):
        self.params = params.copy()
        self.model = None
        self.feature_names = None

        # Set default parameters optimized for time series
        defaults = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'bagging_temperature': 1,
            'border_count': 254,
            'grow_policy': 'SymmetricTree',  # Symmetric trees for better generalization
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50,
            'use_best_model': True,
            'task_type': 'CPU',
            'devices': '0',
            'bootstrap_type': 'Bayesian',
            'boosting_type': 'Plain',
            'score_function': 'Cosine',
            'leaf_estimation_method': 'Newton',
            'leaf_estimation_iterations': 1,
        }

        # Add GPU support if available
        if CB_GPU_AVAILABLE:
            defaults['task_type'] = 'GPU'
            defaults['devices'] = '0'
        else:
            defaults['task_type'] = 'CPU'

        # Update with provided params
        defaults.update(self.params)
        self.params = defaults

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """Fit the CatBoost model."""
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Prepare evaluation set
        eval_data = None
        if eval_set is not None:
            if isinstance(eval_set, list) and len(eval_set) == 2:
                X_val, y_val = eval_set
                if hasattr(X_val, 'values'):
                    X_val = X_val.values
                if hasattr(y_val, 'values'):
                    y_val = y_val.values
                eval_data = [(X_val, y_val)]

        # Training parameters
        train_params = self.params.copy()

        # Override early stopping if specified
        if early_stopping_rounds is not None:
            train_params['early_stopping_rounds'] = early_stopping_rounds
        elif 'early_stopping_rounds' not in train_params:
            train_params['early_stopping_rounds'] = 50

        # Create model
        self.model = cb.CatBoostClassifier(**train_params)

        if verbose:
            self.model.set_params(verbose=True)

        # Fit model with appropriate parameters
        fit_params = {
            'X': X,
            'y': y,
            'verbose': verbose
        }

        # Only use eval_set and use_best_model if eval_set is provided
        if eval_data:
            fit_params['eval_set'] = eval_data
            fit_params['early_stopping_rounds'] = train_params.get('early_stopping_rounds', 50)
            fit_params['use_best_model'] = train_params.get('use_best_model', True)
        else:
            # Remove use_best_model and early_stopping_rounds from model params if no eval_set
            clean_params = train_params.copy()
            clean_params.pop('use_best_model', None)
            clean_params.pop('early_stopping_rounds', None)
            # Recreate model without these params
            self.model = cb.CatBoostClassifier(**clean_params)

        self.model.fit(**fit_params)

        return self

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(X, 'values'):
            X = X.values

        return self.model.predict(X).astype(int).flatten()

    def predict_proba(self, X):
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(X, 'values'):
            X = X.values

        proba = self.model.predict_proba(X)
        return proba

    def get_feature_importance(self, importance_type='FeatureImportance'):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        importance = self.model.get_feature_importance(type=importance_type)
        return dict(zip(self.feature_names or [], importance))

    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        self.model.save_model(filepath)

    @classmethod
    def load_model(cls, filepath):
        """Load model from file."""
        instance = cls()
        instance.model = cb.CatBoostClassifier()
        instance.model.load_model(filepath)
        return instance


# Try to import LightGBM with GPU support detection
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

    # Check for GPU support
    try:
        # Try to create a GPU dataset to test GPU availability
        test_data = np.random.random((100, 10))
        test_labels = np.random.randint(0, 2, 100)
        gpu_dataset = lgb.Dataset(test_data, label=test_labels)
        gpu_params = {'device': 'gpu', 'objective': 'binary', 'verbose': -1}
        lgb.train(gpu_params, gpu_dataset, num_boost_round=1, valid_sets=[gpu_dataset], callbacks=[lgb.log_evaluation(0)])
        GPU_AVAILABLE = True
        print("LightGBM GPU support detected")
    except Exception:
        GPU_AVAILABLE = False
        print("LightGBM GPU support not available, using CPU")

except ImportError:
    LIGHTGBM_AVAILABLE = False
    GPU_AVAILABLE = False
    print("LightGBM not installed, only RandomForest available")

# Try to import CatBoost with GPU support detection
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True

    # Check for GPU support
    try:
        # Try to create a GPU model to test GPU availability
        test_data = np.random.random((100, 10))
        test_labels = np.random.randint(0, 2, 100)
        gpu_model = cb.CatBoostClassifier(task_type='GPU', devices='0', verbose=False)
        gpu_model.fit(test_data, test_labels)
        CB_GPU_AVAILABLE = True
        print("CatBoost GPU support detected")
    except Exception:
        CB_GPU_AVAILABLE = False
        print("CatBoost GPU support not available, using CPU")

except ImportError:
    CATBOOST_AVAILABLE = False
    CB_GPU_AVAILABLE = False
    print("CatBoost not installed")

# Try to import XGBoost with GPU support detection
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

    # Check for GPU support
    try:
        # Try to create a GPU model to test GPU availability
        test_data = np.random.random((100, 10))
        test_labels = np.random.randint(0, 2, 100)
        dtrain = xgb.DMatrix(test_data, label=test_labels)
        gpu_params = {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'verbosity': 0}
        xgb.train(gpu_params, dtrain, num_boost_round=1)
        XGB_GPU_AVAILABLE = True
        print("XGBoost GPU support detected")
    except Exception:
        XGB_GPU_AVAILABLE = False
        print("XGBoost GPU support not available, using CPU")

except ImportError:
    XGBOOST_AVAILABLE = False
    XGB_GPU_AVAILABLE = False
    print("XGBoost not installed")


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""

    def __init__(self, X, y, sequence_length=20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_seq = self.X[idx]  # Already a sequence: (seq_len, n_features)
        if self.y is not None:
            y_seq = self.y[idx]  # Target for this sequence
            return X_seq, y_seq
        return X_seq


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM."""

    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, 1)  # Bidirectional LSTM has 2*hidden_size

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size * 2)
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with attention for classification."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention layer
        self.attention = AttentionLayer(hidden_size)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 32)
        self.fc2 = nn.Linear(32, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_len, input_size) or (batch_size, 1, seq_len, input_size)
        # Handle potential extra dimension from DataLoader
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove extra dimension

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        attended_out = self.attention(lstm_out)

        # Fully connected layers
        out = self.dropout(attended_out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        # Sigmoid for binary classification
        out = self.sigmoid(out)
        return out


class LSTMModel:
    """LSTM model wrapper with scikit-learn interface for time series classification."""

    def __init__(self, **params):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch torchvision")

        self.params = params.copy()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.sequence_length = None

        # Set default parameters
        defaults = {
            'input_size': None,  # Will be set during fit
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True,
            'sequence_length': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,  # L2 regularization
            'epochs': 100,
            'patience': 10,  # Early stopping patience
            'device': TORCH_DEVICE,
        }

        # Update with provided params
        defaults.update(self.params)
        self.params = defaults

        # Set sequence_length attribute
        self.sequence_length = self.params['sequence_length']

    def _create_sequences(self, X, y=None, sequence_length=None):
        """Create sliding window sequences from data with edge case handling."""
        if sequence_length is None:
            sequence_length = self.sequence_length

        # Edge case: not enough data for sequences
        if len(X) < sequence_length:
            print(f"Warning: Data length ({len(X)}) < sequence_length ({sequence_length})")
            print("Using available data with padding or truncation")

            # Option 1: Pad with zeros at the beginning
            padding_needed = sequence_length - len(X)
            if padding_needed > 0:
                padding = np.zeros((padding_needed, X.shape[1]))
                X_padded = np.vstack([padding, X])
                X = X_padded

                # Pad y if provided
                if y is not None:
                    y_padding = np.full(padding_needed, np.nan)  # Use NaN for padding
                    y = np.concatenate([y_padding, y])

        sequences = []
        targets = []

        for i in range(len(X) - sequence_length + 1):
            seq = X[i:i + sequence_length]
            sequences.append(seq)

            if y is not None:
                target = y[i + sequence_length - 1]  # Target is the last value in sequence
                # Skip if target is NaN (padded data)
                if not np.isnan(target):
                    targets.append(target)
                else:
                    continue  # Skip this sequence

        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences)

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """Fit the LSTM model."""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        # Normalize features
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Set input size and sequence length
        self.params['input_size'] = X_scaled.shape[1]
        self.sequence_length = self.params['sequence_length']

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y, self.sequence_length)

        # Create datasets
        train_dataset = TimeSeriesDataset(X_seq, y_seq, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)

        # Validation data
        val_loader = None
        if eval_set is not None:
            if isinstance(eval_set, list) and len(eval_set) == 2:
                X_val, y_val = eval_set
                if hasattr(X_val, 'values'):
                    X_val = X_val.values
                if hasattr(y_val, 'values'):
                    y_val = y_val.values

                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val, self.sequence_length)
                val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq, self.sequence_length)
                val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=False)

        # Initialize model
        self.model = LSTMClassifier(
            input_size=self.params['input_size'],
            hidden_size=self.params['hidden_size'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            bidirectional=self.params['bidirectional']
        ).to(self.params['device'])

        # Loss and optimizer with stability improvements
        criterion = nn.BCELoss()

        # Use AdamW for better weight decay regularization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay'],
            betas=(0.9, 0.999),  # Default Adam betas
            eps=1e-8  # Numerical stability
        )

        # Learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # Mixed precision training if available (for GPU memory efficiency)
        scaler = None
        if TORCH_DEVICE.type == 'cuda':
            try:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                print("Using mixed precision training")
            except ImportError:
                print("Mixed precision not available, using regular precision")

        # Training loop with early stopping
        best_loss = float('inf')
        patience = early_stopping_rounds or self.params['patience']
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.params['epochs']):
            self.model.train()
            train_loss = 0.0
            nan_count = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.params['device'])
                batch_y = batch_y.to(self.params['device'])

                # Check for NaN/inf in inputs
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    print(f"Warning: NaN/inf detected in batch inputs at epoch {epoch}")
                    continue

                optimizer.zero_grad()

                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/inf loss detected at epoch {epoch}")
                        nan_count += 1
                        if nan_count > 5:  # Too many NaN losses, break
                            print("Too many NaN losses, stopping training")
                            break
                        continue

                    # Mixed precision backward pass with gradient scaling
                    scaler.scale(loss).backward()

                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular precision training
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/inf loss detected at epoch {epoch}")
                        nan_count += 1
                        if nan_count > 5:
                            print("Too many NaN losses, stopping training")
                            break
                        continue

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Update learning rate scheduler
            if val_loader is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step(train_loss)

            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.params['device'])
                        batch_y = batch_y.to(self.params['device'])

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.params['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.params['epochs']}, Train Loss: {train_loss:.4f}")

        # Load best model if early stopping was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(X, 'values'):
            X = X.values

        # Normalize
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq = self._create_sequences(X_scaled, sequence_length=self.sequence_length)

        # Create dataset
        dataset = TimeSeriesDataset(X_seq, None, self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.params['device'])
                outputs = self.model(batch_X)
                preds = (outputs > 0.5).float().cpu().numpy().flatten()
                predictions.extend(preds)

        return np.array(predictions).astype(int)

    def predict_proba(self, X):
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(X, 'values'):
            X = X.values

        # Normalize
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq = self._create_sequences(X_scaled, sequence_length=self.sequence_length)

        # Create dataset
        dataset = TimeSeriesDataset(X_seq, None, self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.params['device'])
                outputs = self.model(batch_X)
                probs = outputs.cpu().numpy().flatten()
                probabilities.extend(probs)

        # Return probabilities for both classes
        probabilities = np.array(probabilities)
        return np.column_stack([1 - probabilities, probabilities])

    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        model_state = {
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
        }

        torch.save(model_state, filepath)

    @classmethod
    def load_model(cls, filepath):
        """Load model from file."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        model_state = torch.load(filepath, map_location=TORCH_DEVICE, weights_only=False)

        instance = cls(**model_state['params'])
        instance.scaler = model_state['scaler']
        instance.feature_names = model_state['feature_names']
        instance.sequence_length = model_state['sequence_length']

        # Recreate model
        instance.model = LSTMClassifier(
            input_size=instance.params['input_size'],
            hidden_size=instance.params['hidden_size'],
            num_layers=instance.params['num_layers'],
            dropout=instance.params['dropout'],
            bidirectional=instance.params['bidirectional']
        ).to(instance.params['device'])

        instance.model.load_state_dict(model_state['model_state_dict'])
        instance.model.eval()

        return instance


class ModelRegistry:
    """Central registry for all ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.model_configs = {
            'lightgbm': self._get_lightgbm_config,
            'randomforest': self._get_randomforest_config,
            'xgboost': self._get_xgboost_config,
            'catboost': self._get_catboost_config,
            'lstm': self._get_lstm_config,
            'ensemble': self._get_ensemble_config,
            'lgb': self._get_lightgbm_config,  # alias
            'rf': self._get_randomforest_config,  # alias
            'xgb': self._get_xgboost_config,  # alias
            'cb': self._get_catboost_config,  # alias
        }
    
    def _get_lightgbm_config(self) -> Dict[str, Any]:
        """Get LightGBM configuration with GPU support if available."""
        base_config = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
        }
        
        # Add GPU support if available
        if GPU_AVAILABLE:
            base_config['device'] = 'gpu'
            base_config['gpu_platform_id'] = 0
            base_config['gpu_device_id'] = 0
        
        # Override with config file settings
        model_config = self.config.get('models', {}).get('lightgbm', {})
        base_config.update(model_config)
        
        return base_config
    
    def _get_randomforest_config(self) -> Dict[str, Any]:
        """Get RandomForest configuration."""
        base_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            'class_weight': 'balanced',
        }

        # Override with config file settings
        model_config = self.config.get('models', {}).get('randomforest', {})
        base_config.update(model_config)

        return base_config

    def _get_xgboost_config(self) -> Dict[str, Any]:
        """Get XGBoost configuration with GPU support if available."""
        base_config = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
        }

        # Add GPU support if available
        if XGB_GPU_AVAILABLE:
            base_config['tree_method'] = 'gpu_hist'
            base_config['gpu_id'] = 0
        else:
            base_config['tree_method'] = 'hist'

        # Override with config file settings
        model_config = self.config.get('models', {}).get('xgboost', {})
        base_config.update(model_config)

        return base_config

    def _get_catboost_config(self) -> Dict[str, Any]:
        """Get CatBoost configuration with GPU support if available."""
        base_config = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'bagging_temperature': 1,
            'border_count': 254,
            'grow_policy': 'SymmetricTree',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50,
            'use_best_model': True,
            'bootstrap_type': 'Bayesian',
            'boosting_type': 'Plain',
            'score_function': 'Cosine',
            'leaf_estimation_method': 'Newton',
            'leaf_estimation_iterations': 1,
        }

        # Add GPU support if available
        if CB_GPU_AVAILABLE:
            base_config['task_type'] = 'GPU'
            base_config['devices'] = '0'
        else:
            base_config['task_type'] = 'CPU'

        # Override with config file settings
        model_config = self.config.get('models', {}).get('catboost', {})
        base_config.update(model_config)

        return base_config

    def _get_lstm_config(self) -> Dict[str, Any]:
        """Get LSTM configuration."""
        base_config = {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True,
            'sequence_length': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'epochs': 100,
            'patience': 10,
        }

        # Override with config file settings
        model_config = self.config.get('models', {}).get('lstm', {})
        base_config.update(model_config)

        return base_config

    def _get_ensemble_config(self) -> Dict[str, Any]:
        """Get Ensemble configuration."""
        base_config = {
            'ensemble_type': 'weighted',  # voting, stacking, blending, weighted
            'combination_strategy': 'performance_based',  # performance_based, diversity_based, adaptive, equal
            'base_models': ['lightgbm', 'xgboost', 'catboost'],  # list of model names
            'voting_type': 'soft',  # hard, soft (for voting ensemble)
            'cv_folds': 5,  # for stacking/blending
            'holdout_size': 0.2,  # for blending
            'meta_model': 'xgboost',  # meta-model for stacking
            'weights_update_freq': 'epoch',  # epoch, batch (for adaptive weighting)
            'diversity_metric': 'correlation',  # correlation, disagreement (for diversity weighting)
            'performance_metric': 'sharpe_ratio',  # metric for performance weighting
            'regime_detection': True,  # enable regime-based weighting
            'regime_window': 50,  # window for regime detection
            'random_state': 42,
        }

        # Override with config file settings
        model_config = self.config.get('models', {}).get('ensemble', {})
        base_config.update(model_config)

        return base_config

    def get_model(self, model_name: str, **kwargs) -> Any:
        """Get model instance by name."""
        model_name = model_name.lower()
        
        if model_name not in self.model_configs:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available_models}")
        
        # Get base configuration
        config = self.model_configs[model_name]()
        config.update(kwargs)  # Override with any provided kwargs
        
        # Create model instance
        if model_name in ['lightgbm', 'lgb']:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")

            # Extract training parameters
            train_params = {
                'num_boost_round': config.pop('n_estimators', 100),
                'valid_sets': None,  # Will be set during training
                'callbacks': [lgb.log_evaluation(0)],  # Silent training
            }

            return {
                'model_class': 'lightgbm',
                'params': config,
                'train_params': train_params,
            }

        elif model_name in ['xgboost', 'xgb']:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")

            return XGBoostModel(**config)

        elif model_name in ['catboost', 'cb']:
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available. Install with: pip install catboost")

            return CatBoostModel(**config)

        elif model_name == 'lstm':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Install with: pip install torch torchvision")
            return LSTMModel(**config)

        elif model_name in ['randomforest', 'rf']:
            return RandomForestClassifier(**config)

        elif model_name == 'ensemble':
            return EnsembleModel(**config)

        else:
            raise ValueError(f"Model implementation not found: {model_name}")
    
    def list_available_models(self) -> list:
        """List all available models."""
        available = []

        if LIGHTGBM_AVAILABLE:
            available.extend(['lightgbm', 'lgb'])

        if XGBOOST_AVAILABLE:
            available.extend(['xgboost', 'xgb'])

        if CATBOOST_AVAILABLE:
            available.extend(['catboost', 'cb'])

        if PYTORCH_AVAILABLE:
            available.append('lstm')

        available.extend(['randomforest', 'rf', 'ensemble'])

        return available

def get_model(name, cfg_entry, gpu_enable=False):
    """Get model instance from configuration."""
    # Implementation will be added by business logic agent
    pass

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series,
                model_name: str, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model and return the trained model with validation metrics.

    Args:
        model: Model instance or config dict
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model
        config: Configuration dictionary

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    metrics = {}
    
    if isinstance(model, dict) and model.get('model_class') == 'lightgbm':
        # LightGBM training
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Update training parameters
        train_params = model['train_params'].copy()
        train_params['valid_sets'] = [train_data, val_data]
        train_params['valid_names'] = ['train', 'valid']

        # Train model
        trained_model = lgb.train(
            model['params'],
            train_data,
            **train_params
        )

        # Get predictions
        y_pred_proba = trained_model.predict(X_val, num_iteration=trained_model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

    elif isinstance(model, XGBoostModel):
        # XGBoost training
        trained_model = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

        # Get predictions
        y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
        y_pred = trained_model.predict(X_val)

    elif isinstance(model, CatBoostModel):
        # CatBoost training
        trained_model = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

        # Get predictions
        y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
        y_pred = trained_model.predict(X_val)

    elif isinstance(model, LSTMModel):
        # LSTM training
        trained_model = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

        # Get predictions
        y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
        y_pred = trained_model.predict(X_val)

    elif isinstance(model, EnsembleModel):
        # Ensemble training
        trained_model = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], config=config)

        # Get predictions
        y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
        y_pred = trained_model.predict(X_val)

    else:
        # Scikit-learn style training (RandomForest, etc.)
        trained_model = model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
        y_pred = trained_model.predict(X_val)
    
    # Calculate validation metrics
    try:
        metrics['accuracy'] = accuracy_score(y_val, y_pred)
        metrics['precision'] = precision_score(y_val, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_val, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_val, y_pred, zero_division=0)

        # AUC only if we have both classes
        if len(np.unique(y_val)) > 1:
            metrics['auc'] = roc_auc_score(y_val, y_pred_proba)
        else:
            metrics['auc'] = 0.5

    except Exception as e:
        metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.5}

    return trained_model, metrics

def save_model(model, model_name: str, metrics: Dict[str, float], 
               feature_names: list, config: Dict[str, Any]) -> str:
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Validation metrics
        feature_names: List of feature names
        config: Configuration dictionary
    
    Returns:
        Path to saved model
    """
    ensure_dir("outputs/artifacts")
    
    # Create model filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}"
    
    # Save model
    model_path = f"outputs/artifacts/{model_filename}.pkl"
    
    if hasattr(model, 'save_model') and hasattr(model, 'get_score'):  # XGBoost
        # Save XGBoost model in native format
        xgb_path = f"outputs/artifacts/{model_filename}.xgb"
        model.save_model(xgb_path)

        # Also save as pickle for consistency
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    elif hasattr(model, 'save_model') and hasattr(model, 'get_feature_importance'):  # CatBoost
        # Save CatBoost model in native format
        cb_path = f"outputs/artifacts/{model_filename}.cb"
        model.save_model(cb_path)

        # Also save as pickle for consistency
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    elif hasattr(model, 'save_model') and hasattr(model, 'model'):  # LSTM (has model attribute)
        # Save LSTM model in PyTorch format
        lstm_path = f"outputs/artifacts/{model_filename}.pth"
        model.save_model(lstm_path)

        # Also save as pickle for consistency
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    elif hasattr(model, 'save_model'):  # LightGBM
        # Save LightGBM model in native format
        lgb_path = f"outputs/artifacts/{model_filename}.lgb"
        model.save_model(lgb_path)

        # Also save as pickle for consistency
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        # Save scikit-learn model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_names': feature_names,
        'config': config,
        'model_path': model_path,
    }
    
    metadata_path = f"outputs/artifacts/{model_filename}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    
    return model_path

def load_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a saved model and its metadata.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, metadata

def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model instance
        X: Features
        y: Labels
        cv: Number of cross-validation folds
    
    Returns:
        Dictionary with cross-validation metrics
    """
    print(f"Performing {cv}-fold cross-validation...")
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    metrics = {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_precision_mean': cv_precision.mean(),
        'cv_precision_std': cv_precision.std(),
        'cv_recall_mean': cv_recall.mean(),
        'cv_recall_std': cv_recall.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
    }
    
    print(f"Cross-validation results:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return metrics

def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """
    Get feature importance from trained model.

    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model

    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        # Scikit-learn style (RandomForest)
        importance = model.feature_importances_
    elif hasattr(model, 'get_score'):
        # XGBoost
        importance_dict = model.get_score(importance_type='gain')
        importance = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            importance[i] = importance_dict.get(f'f{i}', 0)
    elif hasattr(model, 'feature_importance'):
        # LightGBM
        importance = model.feature_importance(importance_type='gain')
    elif hasattr(model, 'get_feature_importance'):
        # CatBoost
        importance = model.get_feature_importance('FeatureImportance')
        if isinstance(importance, dict):
            importance = np.array([importance.get(f'f{i}', 0) for i in range(len(feature_names))])
    else:
        return pd.DataFrame()

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df

def train_model_with_optimized_params(model_name: str, best_params: Dict[str, Any],
                                    X_train: pd.DataFrame, y_train: pd.Series,
                                    X_val: pd.DataFrame, y_val: pd.Series,
                                    config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model using optimized hyperparameters.

    Args:
        model_name: Name of the model
        best_params: Optimized hyperparameters
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f"🏆 Training {model_name} with optimized parameters...")

    try:
        if model_name in ['lightgbm', 'lgb']:
            # LightGBM training with optimized params
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Merge optimized params with base config
            model_params = best_params.copy()
            model_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            })

            # Add GPU support if available
            if GPU_AVAILABLE:
                model_params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })

            # Training parameters
            train_params = {
                'num_boost_round': model_params.pop('n_estimators', 100),
                'early_stopping_rounds': 20,
                'verbose_eval': False
            }

            # Train model
            trained_model = lgb.train(
                model_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                **train_params
            )

            # Get predictions
            y_pred_proba = trained_model.predict(X_val, num_iteration=trained_model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)

        elif model_name in ['xgboost', 'xgb']:
            # XGBoost training with optimized params
            model_params = best_params.copy()
            model_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0,
                'random_state': 42
            })

            # Add GPU support if available
            if XGB_GPU_AVAILABLE:
                model_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0
                })
            else:
                model_params['tree_method'] = 'hist'

            trained_model = XGBoostModel(**model_params)
            trained_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

            # Get predictions
            y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
            y_pred = trained_model.predict(X_val)

        elif model_name in ['catboost', 'cb']:
            # CatBoost training with optimized params
            model_params = best_params.copy()
            model_params.update({
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50,
                'use_best_model': True
            })

            # Add GPU support if available
            if CB_GPU_AVAILABLE:
                model_params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })
            else:
                model_params['task_type'] = 'CPU'

            trained_model = CatBoostModel(**model_params)
            trained_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

            # Get predictions
            y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
            y_pred = trained_model.predict(X_val)

        elif model_name == 'lstm':
            # LSTM training with optimized params
            model_params = best_params.copy()
            model_params.update({
                'device': TORCH_DEVICE,
            })

            trained_model = LSTMModel(**model_params)
            trained_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

            # Get predictions
            y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
            y_pred = trained_model.predict(X_val)

        elif model_name in ['randomforest', 'rf']:
            # RandomForest training with optimized params
            model_params = best_params.copy()
            model_params.update({
                'random_state': 42,
                'n_jobs': -1
            })

            trained_model = RandomForestClassifier(**model_params)
            trained_model.fit(X_train, y_train)

            # Get predictions
            y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
            y_pred = trained_model.predict(X_val)

        elif model_name == 'ensemble':
            # Ensemble training with optimized params
            model_params = best_params.copy()
            model_params.update({
                'random_state': 42
            })

            trained_model = EnsembleModel(**model_params)
            trained_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], config=config)

            # Get predictions
            y_pred_proba = trained_model.predict_proba(X_val)[:, 1]
            y_pred = trained_model.predict(X_val)

        else:
            raise ValueError(f"Unsupported model for optimized training: {model_name}")

        # Calculate validation metrics
        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(y_val, y_pred)
            metrics['precision'] = precision_score(y_val, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_val, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_val, y_pred, zero_division=0)

            # AUC only if we have both classes
            if len(np.unique(y_val)) > 1:
                metrics['auc'] = roc_auc_score(y_val, y_pred_proba)
            else:
                metrics['auc'] = 0.5

            # Add Sharpe ratio for comparison
            from .metrics import sharpe_ratio, calculate_returns
            returns = calculate_returns(pd.Series(y_pred_proba))
            if len(returns) > 0:
                metrics['sharpe_ratio'] = sharpe_ratio(returns)
            else:
                metrics['sharpe_ratio'] = 0.0

        except Exception as e:
            print(f"⚠️  Error calculating metrics: {e}")
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.5, 'sharpe_ratio': 0.0}

        print(f"✅ Optimized {model_name} trained successfully!")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"   F1 Score: {metrics.get('f1', 0):.4f}")

        return trained_model, metrics

    except Exception as e:
        print(f"❌ Failed to train optimized {model_name}: {e}")
        raise


class EnsembleModel:
    """
    Ensemble model wrapper that combines multiple base models using various strategies.

    Supports:
    - Voting (hard/soft)
    - Stacking with meta-model
    - Blending with holdout set
    - Weighted averaging (performance-based, diversity-based, adaptive)
    """

    def __init__(self, **params):
        self.params = params.copy()
        self.base_models = {}
        self.meta_model = None
        self.weights = None
        self.feature_names = None
        self.ensemble_type = None
        self.combination_strategy = None

        # Set default parameters
        defaults = {
            'ensemble_type': 'voting',  # voting, stacking, blending, weighted
            'combination_strategy': 'performance_based',  # performance_based, diversity_based, adaptive, equal
            'base_models': ['lightgbm', 'xgboost', 'catboost'],  # list of model names
            'voting_type': 'soft',  # hard, soft (for voting ensemble)
            'cv_folds': 5,  # for stacking/blending
            'holdout_size': 0.2,  # for blending
            'meta_model': 'xgboost',  # meta-model for stacking
            'weights_update_freq': 'epoch',  # epoch, batch (for adaptive weighting)
            'diversity_metric': 'correlation',  # correlation, disagreement (for diversity weighting)
            'performance_metric': 'sharpe_ratio',  # metric for performance weighting
            'regime_detection': True,  # enable regime-based weighting
            'regime_window': 50,  # window for regime detection
            'random_state': 42,
        }

        # Update with provided params
        defaults.update(self.params)
        self.params = defaults

        # Validate ensemble type
        valid_types = ['voting', 'stacking', 'blending', 'weighted']
        if self.params['ensemble_type'] not in valid_types:
            raise ValueError(f"Invalid ensemble_type. Must be one of: {valid_types}")

        # Validate combination strategy
        valid_strategies = ['performance_based', 'diversity_based', 'adaptive', 'equal']
        if self.params['combination_strategy'] not in valid_strategies:
            raise ValueError(f"Invalid combination_strategy. Must be one of: {valid_strategies}")

    def _initialize_base_models(self, config):
        """Initialize base models from configuration."""
        registry = ModelRegistry(config)
        self.base_models = {}

        for model_name in self.params['base_models']:
            try:
                model = registry.get_model(model_name)
                self.base_models[model_name] = model
                print(f"✅ Initialized {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to initialize {model_name}: {e}")
                continue

        if not self.base_models:
            raise ValueError("No base models could be initialized")

    def _detect_market_regime(self, X, y=None):
        """Detect current market regime for adaptive weighting."""
        if not self.params['regime_detection'] or len(X) < self.params['regime_window']:
            return 'neutral'

        # Simple regime detection based on volatility and trend
        window = self.params['regime_window']
        recent_data = X.iloc[-window:] if hasattr(X, 'iloc') else X[-window:]

        # Calculate volatility (standard deviation of returns)
        if hasattr(recent_data, 'pct_change'):
            returns = recent_data.pct_change().dropna()
            volatility = returns.std().mean() if hasattr(returns, 'std') else np.std(returns)
        else:
            volatility = np.std(recent_data, axis=0).mean()

        # Simple trend detection (SMA slope)
        if hasattr(recent_data, 'rolling'):
            sma_short = recent_data.rolling(10).mean().iloc[-1]
            sma_long = recent_data.rolling(30).mean().iloc[-1]
            trend = 'bullish' if sma_short > sma_long else 'bearish'
        else:
            # Fallback for numpy arrays
            trend = 'neutral'

        # Classify regime
        if volatility > np.percentile(volatility, 75) if isinstance(volatility, (int, float)) else volatility > 0.02:
            regime = 'high_volatility'
        elif trend == 'bullish':
            regime = 'bull_trend'
        elif trend == 'bearish':
            regime = 'bear_trend'
        else:
            regime = 'neutral'

        return regime

    def _calculate_model_weights(self, X_val, y_val, regime='neutral'):
        """Calculate weights for base models based on combination strategy."""
        if self.params['combination_strategy'] == 'equal':
            n_models = len(self.base_models)
            return {model_name: 1.0 / n_models for model_name in self.base_models.keys()}

        elif self.params['combination_strategy'] == 'performance_based':
            return self._performance_based_weights(X_val, y_val)

        elif self.params['combination_strategy'] == 'diversity_based':
            return self._diversity_based_weights(X_val, y_val)

        elif self.params['combination_strategy'] == 'adaptive':
            return self._adaptive_weights(X_val, y_val, regime)

        else:
            raise ValueError(f"Unknown combination strategy: {self.params['combination_strategy']}")

    def _performance_based_weights(self, X_val, y_val):
        """Calculate weights based on historical performance."""
        performances = {}
        total_performance = 0

        for model_name, model in self.base_models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)
                    y_pred_proba = y_pred.astype(float)

                # Calculate Sharpe ratio as performance metric
                from .metrics import sharpe_ratio, calculate_returns
                returns = calculate_returns(pd.Series(y_pred_proba))
                if len(returns) > 0:
                    perf = sharpe_ratio(returns)
                else:
                    perf = 0.0

                performances[model_name] = max(perf, 0.1)  # Minimum weight
                total_performance += performances[model_name]

            except Exception as e:
                print(f"⚠️  Error calculating performance for {model_name}: {e}")
                performances[model_name] = 0.1
                total_performance += 0.1

        # Normalize weights
        weights = {}
        for model_name, perf in performances.items():
            weights[model_name] = perf / total_performance if total_performance > 0 else 1.0 / len(performances)

        return weights

    def _diversity_based_weights(self, X_val, y_val):
        """Calculate weights based on model diversity."""
        predictions = {}

        # Get predictions from all models
        for model_name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]
                else:
                    pred = model.predict(X_val).astype(float)
                predictions[model_name] = pred
            except Exception as e:
                print(f"⚠️  Error getting predictions for {model_name}: {e}")
                predictions[model_name] = np.zeros(len(X_val))

        # Calculate diversity matrix
        n_models = len(predictions)
        diversity_matrix = np.zeros((n_models, n_models))

        model_names = list(predictions.keys())
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    if self.params['diversity_metric'] == 'correlation':
                        corr = np.corrcoef(predictions[name1], predictions[name2])[0, 1]
                        diversity_matrix[i, j] = 1 - abs(corr)  # Higher diversity = lower correlation
                    else:  # disagreement
                        disagreement = np.mean(predictions[name1] != predictions[name2])
                        diversity_matrix[i, j] = disagreement

        # Calculate weights as average diversity
        weights = {}
        for i, name in enumerate(model_names):
            avg_diversity = np.mean(diversity_matrix[i])
            weights[name] = avg_diversity

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            weights = {name: 1.0 / n_models for name in model_names}

        return weights

    def _adaptive_weights(self, X_val, y_val, regime):
        """Calculate adaptive weights based on market regime."""
        # Base weights from performance
        base_weights = self._performance_based_weights(X_val, y_val)

        # Regime-specific adjustments
        regime_multipliers = {
            'high_volatility': {'lightgbm': 1.2, 'xgboost': 1.1, 'catboost': 1.0, 'lstm': 0.8},
            'bull_trend': {'lightgbm': 1.1, 'xgboost': 1.0, 'catboost': 1.2, 'lstm': 0.9},
            'bear_trend': {'lightgbm': 1.0, 'xgboost': 1.2, 'catboost': 1.1, 'lstm': 0.8},
            'neutral': {'lightgbm': 1.0, 'xgboost': 1.0, 'catboost': 1.0, 'lstm': 1.0}
        }

        multipliers = regime_multipliers.get(regime, regime_multipliers['neutral'])

        # Apply multipliers
        adaptive_weights = {}
        for model_name, base_weight in base_weights.items():
            multiplier = multipliers.get(model_name, 1.0)
            adaptive_weights[model_name] = base_weight * multiplier

        # Re-normalize
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {name: w / total_weight for name, w in adaptive_weights.items()}

        return adaptive_weights

    def _create_meta_features(self, X, predictions_dict):
        """Create meta-features for stacking."""
        meta_features = []

        for model_name, preds in predictions_dict.items():
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                # Probability predictions
                meta_features.append(preds[:, 1])  # Positive class probability
            else:
                # Binary predictions
                meta_features.append(preds.flatten())

        # Add original features (subset)
        if hasattr(X, 'values'):
            X_vals = X.values
        else:
            X_vals = X

        # Select top features by variance
        variances = np.var(X_vals, axis=0)
        top_feature_indices = np.argsort(variances)[-10:]  # Top 10 features
        selected_features = X_vals[:, top_feature_indices]

        meta_features.extend([selected_features[:, i] for i in range(selected_features.shape[1])])

        return np.column_stack(meta_features)

    def fit(self, X, y, eval_set=None, config=None, **kwargs):
        """Fit the ensemble model."""
        if config is None:
            from .utils import load_config
            config = load_config()

        # Initialize base models
        self._initialize_base_models(config)

        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        # Split data for different ensemble types
        if self.params['ensemble_type'] in ['stacking', 'blending']:
            from sklearn.model_selection import train_test_split
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.params['holdout_size'],
                random_state=self.params['random_state'], stratify=y
            )
        else:
            X_train, y_train = X, y
            X_holdout, y_holdout = None, None

        # Train base models
        trained_models = {}
        base_predictions = {}

        for model_name, model in self.base_models.items():
            print(f"🏃 Training base model: {model_name}")

            try:
                if self.params['ensemble_type'] in ['stacking', 'blending']:
                    # Train on training set only
                    trained_model, _ = train_model(
                        model, X_train, y_train, X_train, y_train, model_name, config
                    )
                else:
                    # Train on full data
                    trained_model, _ = train_model(
                        model, X_train, y_train, eval_set[0] if eval_set else X_train,
                        eval_set[1] if eval_set else y_train, model_name, config
                    )

                trained_models[model_name] = trained_model

                # Get predictions for meta-model training
                if self.params['ensemble_type'] in ['stacking', 'blending'] and X_holdout is not None:
                    if hasattr(trained_model, 'predict_proba'):
                        preds = trained_model.predict_proba(X_holdout)
                    else:
                        preds = trained_model.predict(X_holdout)
                        preds = np.column_stack([1 - preds, preds])
                    base_predictions[model_name] = preds

            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
                continue

        # Update base models
        self.base_models = trained_models

        # Train meta-model for stacking
        if self.params['ensemble_type'] == 'stacking' and base_predictions:
            print("🏗️  Training meta-model for stacking...")

            # Create meta-features
            meta_X = self._create_meta_features(X_holdout, base_predictions)
            meta_y = y_holdout

            # Train meta-model
            registry = ModelRegistry(config)
            self.meta_model = registry.get_model(self.params['meta_model'])

            try:
                self.meta_model, _ = train_model(
                    self.meta_model, meta_X, meta_y, meta_X, meta_y,
                    f"{self.params['meta_model']}_meta", config
                )
                print("✅ Meta-model trained successfully")
            except Exception as e:
                print(f"❌ Failed to train meta-model: {e}")
                self.meta_model = None

        # Calculate initial weights
        if eval_set and len(eval_set) == 2:
            X_val, y_val = eval_set
            regime = self._detect_market_regime(X_val, y_val)
            self.weights = self._calculate_model_weights(X_val, y_val, regime)
        else:
            # Default equal weights
            n_models = len(self.base_models)
            self.weights = {name: 1.0 / n_models for name in self.base_models.keys()}

        print(f"✅ Ensemble model fitted with {len(self.base_models)} base models")
        print(f"   Weights: {self.weights}")

        return self

    def predict(self, X):
        """Make predictions."""
        if not self.base_models:
            raise ValueError("Ensemble not fitted yet")

        predictions = {}

        # Get predictions from all base models
        for model_name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                    pred_proba = pred.astype(float)

                predictions[model_name] = pred_proba

            except Exception as e:
                print(f"⚠️  Error predicting with {model_name}: {e}")
                predictions[model_name] = np.zeros(len(X))

        # Combine predictions based on ensemble type
        if self.params['ensemble_type'] == 'voting':
            return self._voting_predict(predictions)
        elif self.params['ensemble_type'] == 'stacking':
            return self._stacking_predict(X, predictions)
        elif self.params['ensemble_type'] == 'blending':
            return self._blending_predict(predictions)
        elif self.params['ensemble_type'] == 'weighted':
            return self._weighted_predict(predictions)
        else:
            raise ValueError(f"Unknown ensemble type: {self.params['ensemble_type']}")

    def _voting_predict(self, predictions):
        """Voting ensemble prediction."""
        if self.params['voting_type'] == 'hard':
            # Hard voting: majority vote
            binary_preds = {}
            for name, probs in predictions.items():
                binary_preds[name] = (probs > 0.5).astype(int)

            # Stack predictions and take majority vote
            pred_matrix = np.column_stack(list(binary_preds.values()))
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=pred_matrix
            )

        else:  # soft voting
            # Soft voting: average probabilities
            prob_matrix = np.column_stack(list(predictions.values()))
            avg_probs = np.mean(prob_matrix, axis=1)
            final_pred = (avg_probs > 0.5).astype(int)

        return final_pred

    def _stacking_predict(self, X, predictions):
        """Stacking ensemble prediction."""
        if self.meta_model is None:
            # Fallback to voting
            return self._voting_predict(predictions)

        # Create meta-features
        meta_X = self._create_meta_features(X, predictions)

        # Get meta-model predictions
        if hasattr(self.meta_model, 'predict_proba'):
            meta_pred_proba = self.meta_model.predict_proba(meta_X)[:, 1]
        else:
            meta_pred = self.meta_model.predict(meta_X)
            meta_pred_proba = meta_pred.astype(float)

        return (meta_pred_proba > 0.5).astype(int)

    def _blending_predict(self, predictions):
        """Blending ensemble prediction."""
        # Simple average of predictions (could be improved with holdout-based weights)
        prob_matrix = np.column_stack(list(predictions.values()))
        avg_probs = np.mean(prob_matrix, axis=1)
        return (avg_probs > 0.5).astype(int)

    def _weighted_predict(self, predictions):
        """Weighted ensemble prediction."""
        if self.weights is None:
            # Fallback to equal weights
            n_models = len(predictions)
            weights = np.ones(n_models) / n_models
        else:
            weights = np.array([self.weights.get(name, 0) for name in predictions.keys()])

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average of probabilities
        prob_matrix = np.column_stack(list(predictions.values()))
        weighted_probs = np.average(prob_matrix, axis=1, weights=weights)

        return (weighted_probs > 0.5).astype(int)

    def predict_proba(self, X):
        """Make probability predictions."""
        if not self.base_models:
            raise ValueError("Ensemble not fitted yet")

        predictions = {}

        # Get predictions from all base models
        for model_name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    predictions[model_name] = model.predict_proba(X)
                else:
                    pred = model.predict(X).astype(float)
                    predictions[model_name] = np.column_stack([1 - pred, pred])

            except Exception as e:
                print(f"⚠️  Error predicting with {model_name}: {e}")
                n_samples = len(X)
                predictions[model_name] = np.full((n_samples, 2), 0.5)

        # Combine predictions based on ensemble type
        if self.params['ensemble_type'] == 'voting':
            return self._voting_predict_proba(predictions)
        elif self.params['ensemble_type'] == 'stacking':
            return self._stacking_predict_proba(X, predictions)
        elif self.params['ensemble_type'] == 'blending':
            return self._blending_predict_proba(predictions)
        elif self.params['ensemble_type'] == 'weighted':
            return self._weighted_predict_proba(predictions)
        else:
            raise ValueError(f"Unknown ensemble type: {self.params['ensemble_type']}")

    def _voting_predict_proba(self, predictions):
        """Voting ensemble probability prediction."""
        if self.params['voting_type'] == 'hard':
            # Convert to binary and average
            binary_preds = {}
            for name, probs in predictions.items():
                binary_preds[name] = (probs[:, 1] > 0.5).astype(int)

            pred_matrix = np.column_stack(list(binary_preds.values()))
            avg_binary = np.mean(pred_matrix, axis=1)
            return np.column_stack([1 - avg_binary, avg_binary])

        else:  # soft voting
            # Average probabilities
            prob_arrays = [probs[:, 1] for probs in predictions.values()]
            prob_matrix = np.column_stack(prob_arrays)
            avg_probs = np.mean(prob_matrix, axis=1)
            return np.column_stack([1 - avg_probs, avg_probs])

    def _stacking_predict_proba(self, X, predictions):
        """Stacking ensemble probability prediction."""
        if self.meta_model is None:
            return self._voting_predict_proba(predictions)

        # Create meta-features
        meta_X = self._create_meta_features(X, predictions)

        # Get meta-model predictions
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_X)
        else:
            pred = self.meta_model.predict(meta_X).astype(float)
            return np.column_stack([1 - pred, pred])

    def _blending_predict_proba(self, predictions):
        """Blending ensemble probability prediction."""
        prob_arrays = [probs[:, 1] for probs in predictions.values()]
        prob_matrix = np.column_stack(prob_arrays)
        avg_probs = np.mean(prob_matrix, axis=1)
        return np.column_stack([1 - avg_probs, avg_probs])

    def _weighted_predict_proba(self, predictions):
        """Weighted ensemble probability prediction."""
        if self.weights is None:
            n_models = len(predictions)
            weights = np.ones(n_models) / n_models
        else:
            weights = np.array([self.weights.get(name, 0) for name in predictions.keys()])

        weights = weights / np.sum(weights)

        prob_arrays = [probs[:, 1] for probs in predictions.values()]
        prob_matrix = np.column_stack(prob_arrays)
        weighted_probs = np.average(prob_matrix, axis=1, weights=weights)

        return np.column_stack([1 - weighted_probs, weighted_probs])

    def save_model(self, filepath):
        """Save ensemble model."""
        if not self.base_models:
            raise ValueError("Ensemble not fitted yet")

        model_state = {
            'params': self.params,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'base_models': {},
            'meta_model': None,
        }

        # Save base models
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'save_model'):
                    model_path = f"{filepath}_{name}.pkl"
                    model.save_model(model_path)
                    model_state['base_models'][name] = model_path
                else:
                    model_state['base_models'][name] = model
            except Exception as e:
                print(f"⚠️  Could not save {name}: {e}")

        # Save meta-model
        if self.meta_model is not None:
            try:
                if hasattr(self.meta_model, 'save_model'):
                    meta_path = f"{filepath}_meta.pkl"
                    self.meta_model.save_model(meta_path)
                    model_state['meta_model'] = meta_path
                else:
                    model_state['meta_model'] = self.meta_model
            except Exception as e:
                print(f"⚠️  Could not save meta-model: {e}")

        # Save ensemble state
        torch.save(model_state, filepath) if PYTORCH_AVAILABLE else pickle.dump(model_state, open(filepath, 'wb'))
        print(f"✅ Ensemble model saved: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load ensemble model."""
        if PYTORCH_AVAILABLE:
            model_state = torch.load(filepath, map_location=TORCH_DEVICE, weights_only=False)
        else:
            model_state = pickle.load(open(filepath, 'rb'))

        instance = cls(**model_state['params'])
        instance.weights = model_state['weights']
        instance.feature_names = model_state['feature_names']

        # Load base models
        instance.base_models = {}
        for name, model_path in model_state['base_models'].items():
            try:
                if isinstance(model_path, str):
                    # Load from file
                    if name in ['lightgbm', 'lgb']:
                        instance.base_models[name] = lgb.Booster(model_file=model_path)
                    elif name in ['xgboost', 'xgb']:
                        instance.base_models[name] = xgb.Booster()
                        instance.base_models[name].load_model(model_path)
                    elif name in ['catboost', 'cb']:
                        instance.base_models[name] = cb.CatBoostClassifier()
                        instance.base_models[name].load_model(model_path)
                    elif name == 'lstm':
                        instance.base_models[name] = LSTMModel.load_model(model_path)
                    else:
                        instance.base_models[name] = pickle.load(open(model_path, 'rb'))
                else:
                    instance.base_models[name] = model_path
            except Exception as e:
                print(f"⚠️  Could not load {name}: {e}")

        # Load meta-model
        if model_state['meta_model'] is not None:
            try:
                if isinstance(model_state['meta_model'], str):
                    instance.meta_model = pickle.load(open(model_state['meta_model'], 'rb'))
                else:
                    instance.meta_model = model_state['meta_model']
            except Exception as e:
                print(f"⚠️  Could not load meta-model: {e}")

        return instance


def main():
    """Main model training pipeline with integrated optimization."""
    print("=== AXON Model Training Pipeline ===")

    # Load configuration
    config = load_config()

    try:
        # Check if optimization is enabled
        optimize_config = config.get('optimization', {})
        optimization_enabled = optimize_config.get('enabled', False)

        if optimization_enabled:
            print("🔧 Optimization enabled - running hyperparameter optimization...")

            # Import optimization engine
            from .optimization import OptimizationEngine

            # Initialize optimization engine
            opt_engine = OptimizationEngine(config)

            # Load processed features
            print("\n📂 Loading processed features...")
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
            models_to_optimize = optimize_config.get('models', ['lightgbm'])

            optimized_models = {}
            optimization_results = {}

            # Run optimization for each model
            for model_name in models_to_optimize:
                print(f"\n{'='*60}")
                print(f"🎯 OPTIMIZING {model_name.upper()}")
                print(f"{'='*60}")

                try:
                    # Run optimization
                    results = opt_engine.optimize_model(
                        model_name, X_train, y_train, X_val, y_val
                    )

                    optimization_results[model_name] = results

                    # Train final model with optimized parameters
                    print(f"\n🏆 Training final {model_name} with optimized parameters...")
                    best_params = results['best_params']

                    final_model, final_metrics = train_model_with_optimized_params(
                        model_name, best_params, X_train, y_train, X_val, y_val, config
                    )

                    # Save optimized model
                    model_path = save_model(
                        final_model, f"{model_name}_optimized", final_metrics, feature_cols, config
                    )

                    # Get feature importance
                    importance_df = get_feature_importance(final_model, feature_cols, model_name)
                    if not importance_df.empty:
                        importance_path = f"outputs/artifacts/{model_name}_optimized_feature_importance.csv"
                        importance_df.to_csv(importance_path, index=False)
                        print(f"📊 Feature importance saved: {importance_path}")

                    optimized_models[model_name] = {
                        'model': final_model,
                        'path': model_path,
                        'metrics': final_metrics,
                        'best_params': best_params,
                        'optimization_results': results
                    }

                    print(f"✅ {model_name} optimization and training completed!")

                except Exception as e:
                    print(f"❌ Failed to optimize {model_name}: {e}")
                    continue

            # Optimization comparison
            if len(optimized_models) > 1:
                print(f"\n{'='*60}")
                print("🏆 OPTIMIZATION COMPARISON")
                print(f"{'='*60}")

                comparison_data = {}
                for model_name, model_info in optimized_models.items():
                    metrics = model_info['metrics']
                    opt_results = model_info['optimization_results']

                    comparison_data[model_name] = {
                        'Best Value': f"{opt_results['best_value']:.4f}",
                        'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.4f}",
                        'F1 Score': f"{metrics.get('f1', 0):.4f}",
                        'AUC': f"{metrics.get('auc', 0):.4f}",
                        'Method': opt_results['method']
                    }

                comparison_df = pd.DataFrame(comparison_data).T
                print(comparison_df)

                # Save comparison
                comparison_path = "outputs/artifacts/optimization_comparison.csv"
                comparison_df.to_csv(comparison_path)
                print(f"\n💾 Optimization comparison saved: {comparison_path}")

                # Find best optimized model
                best_model_name = max(optimized_models.keys(),
                                    key=lambda k: optimized_models[k]['metrics'].get('sharpe_ratio', 0))
                best_sharpe = optimized_models[best_model_name]['metrics'].get('sharpe_ratio', 0)
                print(f"\n🏆 Best optimized model: {best_model_name} (Sharpe: {best_sharpe:.4f})")

            print(f"\n✅ Optimization pipeline completed successfully!")
            print(f"📊 Optimized models: {list(optimized_models.keys())}")
            print(f"📋 Reports saved in outputs/reports/")
            print(f"💾 Models saved in outputs/artifacts/")

            return

        # Standard training pipeline (when optimization is disabled)
        print("🔧 Optimization disabled - running standard training...")

        # Initialize model registry
        registry = ModelRegistry(config)

        print(f"📋 Available models: {registry.list_available_models()}")

        # Load processed features
        print("\n📂 Loading processed features...")

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

        # Get models to train from config
        models_to_train = config.get('models', {}).get('train', ['lightgbm', 'randomforest'])

        trained_models = {}
        all_metrics = {}

        # Train each model
        for model_name in models_to_train:
            print(f"\n{'='*50}")
            print(f"🏃 Training {model_name.upper()}")
            print(f"{'='*50}")

            try:
                # Get model
                model = registry.get_model(model_name)

                # Train model
                trained_model, metrics = train_model(
                    model, X_train, y_train, X_val, y_val, model_name, config
                )

                # Save model
                model_path = save_model(
                    trained_model, model_name, metrics, feature_cols, config
                )

                # Get feature importance
                importance_df = get_feature_importance(trained_model, feature_cols, model_name)
                if not importance_df.empty:
                    importance_path = f"outputs/artifacts/{model_name}_feature_importance.csv"
                    importance_df.to_csv(importance_path, index=False)
                    print(f"📊 Feature importance saved: {importance_path}")

                    # Print top 10 features
                    print(f"\n🔍 Top 10 features for {model_name}:")
                    for idx, row in importance_df.head(10).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.4f}")

                trained_models[model_name] = {
                    'model': trained_model,
                    'path': model_path,
                    'metrics': metrics
                }
                all_metrics[model_name] = metrics

            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
                continue

        # Model comparison
        if len(trained_models) > 1:
            print(f"\n{'='*50}")
            print("📊 MODEL COMPARISON")
            print(f"{'='*50}")

            comparison_df = pd.DataFrame(all_metrics).T
            print(comparison_df.round(4))

            # Save comparison
            comparison_path = "outputs/artifacts/model_comparison.csv"
            comparison_df.to_csv(comparison_path)
            print(f"\n💾 Model comparison saved: {comparison_path}")

            # Find best model
            best_model_name = comparison_df['f1'].idxmax()
            best_f1 = comparison_df.loc[best_model_name, 'f1']
            print(f"\n🏆 Best model: {best_model_name} (F1: {best_f1:.4f})")

        print(f"\n✅ Standard training completed successfully!")
        print(f"🏃 Trained models: {list(trained_models.keys())}")

    except Exception as e:
        print(f"❌ Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()