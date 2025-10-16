"""AXON Features Module

Feature engineering and label generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .utils import load_config, ensure_dir
from .dataset import load_processed_data


class FeatureEngineer:
    """Feature engineering class for AXON system."""
    
    def __init__(self, config=None):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def create_features(self, df):
        """Create features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        return create_feature_matrix(df, self.config)
    
    def prepare_features(self, df, target_col='y'):
        """Prepare features for model training.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            
        Returns:
            Prepared features DataFrame
        """
        return prepare_features(df, self.config, target_col)


def calculate_ema(series, span):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        series: Price series (typically close prices)
        window: RSI calculation window (default 14)

    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_imbalance_features(df, windows=[5, 10, 20]):
    """
    Calculate order book imbalance proxy features using OHLCV data.
    
    Since we don't have real order book data, we approximate imbalance using:
    - Price momentum vs volume
    - Intraday price action patterns
    - Volume-weighted price movements
    """
    features = {}
    
    for window in windows:
        # Volume-weighted average price (VWAP) approximation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        features[f'vwap_{window}'] = vwap
        
        # Price vs VWAP deviation (imbalance proxy)
        features[f'price_vwap_ratio_{window}'] = df['close'] / vwap
        
        # Volume momentum (buying/selling pressure proxy)
        price_change = df['close'].pct_change()
        volume_momentum = (price_change * df['volume']).rolling(window).sum()
        features[f'volume_momentum_{window}'] = volume_momentum
        
        # Intraday strength (where price closes within the range)
        high_low_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / high_low_range.replace(0, np.nan)
        features[f'intraday_strength_{window}'] = close_position.rolling(window).mean()
        
        # Volume-adjusted returns (imbalance impact on price)
        vol_adj_returns = df['returns'] * np.log1p(df['volume'] / df['volume'].rolling(window).mean())
        features[f'vol_adj_returns_{window}'] = vol_adj_returns
    
    return pd.DataFrame(features, index=df.index)

def generate_triple_barrier_labels(df, horizon_minutes=15, pt_pct=0.02, sl_pct=0.02):
    """
    Generate triple barrier labels for classification.

    Args:
        df: DataFrame with OHLCV data
        horizon_minutes: Maximum holding period in minutes
        pt_pct: Profit target as percentage (e.g., 0.02 = 2%)
        sl_pct: Stop loss as percentage (e.g., 0.02 = 2%)

    Returns:
        Series with labels: 1 (profit target hit), -1 (stop loss hit), 0 (timeout)
    """
    labels = pd.Series(index=df.index, dtype=int)

    for i in range(len(df) - horizon_minutes):
        entry_price = df['close'].iloc[i]

        # Define barriers
        pt_price = entry_price * (1 + pt_pct)  # Profit target
        sl_price = entry_price * (1 - sl_pct)  # Stop loss

        # Look ahead for barrier hits (this is OK for labeling, not for features)
        future_highs = df['high'].iloc[i+1:i+1+horizon_minutes]
        future_lows = df['low'].iloc[i+1:i+1+horizon_minutes]

        # Check which barrier is hit first
        pt_hit_times = future_highs[future_highs >= pt_price].index
        sl_hit_times = future_lows[future_lows <= sl_price].index

        if len(pt_hit_times) > 0 and len(sl_hit_times) > 0:
            # Both barriers hit, check which came first
            if pt_hit_times[0] <= sl_hit_times[0]:
                labels.iloc[i] = 1  # Profit target hit first
            else:
                labels.iloc[i] = -1  # Stop loss hit first
        elif len(pt_hit_times) > 0:
            labels.iloc[i] = 1  # Only profit target hit
        elif len(sl_hit_times) > 0:
            labels.iloc[i] = -1  # Only stop loss hit
        else:
            labels.iloc[i] = 0  # Timeout (no barrier hit)

    # Fill remaining values with 0 (can't look ahead enough)
    labels = labels.fillna(0)

    return labels

def generate_sign_return_labels(df, horizon_minutes=15):
    """
    Generate simple sign of future return labels.
    
    Args:
        df: DataFrame with OHLCV data
        horizon_minutes: Forward looking period in minutes
    
    Returns:
        Series with labels: 1 (positive return), 0 (negative return)
    """
    # Calculate forward returns (this is OK for labeling)
    future_returns = df['close'].shift(-horizon_minutes) / df['close'] - 1
    
    # Convert to binary labels
    labels = (future_returns > 0).astype(int)
    
    # Remove last horizon_minutes rows (can't calculate forward return)
    labels.iloc[-horizon_minutes:] = np.nan
    
    return labels

def create_feature_matrix(df, config):
    """
    Create the complete feature matrix based on configuration.
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary
    
    Returns:
        DataFrame with all features
    """
    print("Creating feature matrix...")
    
    # Start with the base dataframe
    features_df = df.copy()
    
    # Get feature configuration
    feature_config = config.get('features', {})
    use_features = feature_config.get('use', ['ret_1m', 'ema_5', 'ema_20', 'rsi_14', 'imb_10'])
    
    print(f"Configured features: {use_features}")
    
    # Calculate returns features FIRST (needed for other features)
    features_df['returns'] = df['close'].pct_change()

    if any('ret' in f for f in use_features):
        features_df['ret_1m'] = features_df['returns']
        features_df['ret_5m'] = df['close'].pct_change(5)
        features_df['ret_15m'] = df['close'].pct_change(15)
        features_df['ret_60m'] = df['close'].pct_change(60)

    # Calculate EMA features
    if any('ema' in f for f in use_features):
        features_df['ema_5'] = calculate_ema(df['close'], 5)
        features_df['ema_20'] = calculate_ema(df['close'], 20)
        features_df['ema_50'] = calculate_ema(df['close'], 50)

        # EMA ratios (price relative to EMA)
        features_df['price_ema5_ratio'] = df['close'] / features_df['ema_5']
        features_df['price_ema20_ratio'] = df['close'] / features_df['ema_20']
        features_df['ema5_ema20_ratio'] = features_df['ema_5'] / features_df['ema_20']

    # Calculate RSI features
    if any('rsi' in f for f in use_features):
        features_df['rsi_14'] = calculate_rsi(df['close'], 14)
        features_df['rsi_7'] = calculate_rsi(df['close'], 7)
        features_df['rsi_21'] = calculate_rsi(df['close'], 21)

    # Calculate imbalance features
    if any('imb' in f for f in use_features):
        imbalance_features = calculate_imbalance_features(features_df)
        features_df = pd.concat([features_df, imbalance_features], axis=1)

    # Calculate additional technical features
    # Bollinger Bands
    bb_window = 20
    bb_std = 2
    bb_ma = df['close'].rolling(bb_window).mean()
    bb_std_val = df['close'].rolling(bb_window).std()
    features_df['bb_upper'] = bb_ma + (bb_std_val * bb_std)
    features_df['bb_lower'] = bb_ma - (bb_std_val * bb_std)
    features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])

    # Volatility features
    features_df['volatility_5m'] = features_df['returns'].rolling(5).std()
    features_df['volatility_20m'] = features_df['returns'].rolling(20).std()
    features_df['volatility_ratio'] = features_df['volatility_5m'] / features_df['volatility_20m']

    # Volume features
    features_df['volume_sma_20'] = df['volume'].rolling(20).mean()
    features_df['volume_ratio_20'] = df['volume'] / features_df['volume_sma_20']

    # Price action features
    features_df['high_low_ratio'] = df['high'] / df['low']
    features_df['open_close_ratio'] = df['open'] / df['close']
    
    print(f"Created {len(features_df.columns)} total features")
    
    return features_df

def generate_labels(df, config):
    """
    Generate labels based on configuration.
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary
    
    Returns:
        Series with labels
    """
    print("Generating labels...")
    
    label_config = config.get('labels', {})
    method = label_config.get('method', 'triple_barrier')
    horizon = label_config.get('horizon', '15m')
    
    # Convert horizon to minutes
    if isinstance(horizon, str):
        if horizon.endswith('m'):
            horizon_minutes = int(horizon[:-1])
        elif horizon.endswith('h'):
            horizon_minutes = int(horizon[:-1]) * 60
        else:
            horizon_minutes = 15  # default
    else:
        horizon_minutes = int(horizon)
    
    print(f"Label method: {method}, horizon: {horizon_minutes} minutes")
    
    if method == 'triple_barrier':
        labels = generate_triple_barrier_labels(df, horizon_minutes)
        # Convert to binary for simplicity (combine -1 and 0 into 0, keep 1 as 1)
        labels = (labels == 1).astype(int)
    elif method == 'sign_return_h':
        labels = generate_sign_return_labels(df, horizon_minutes)
    else:
        raise ValueError(f"Unknown label method: {method}")
    
    print(f"Generated {len(labels)} labels, distribution: {labels.value_counts().to_dict()}")
    
    return labels

def filter_features(features_df, config):
    """
    Filter features based on configuration.
    
    Args:
        features_df: DataFrame with all features
        config: Configuration dictionary
    
    Returns:
        DataFrame with selected features only
    """
    feature_config = config.get('features', {})
    use_features = feature_config.get('use', [])
    
    # Always keep basic columns
    keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Add configured features if they exist
    available_features = [col for col in features_df.columns if col not in keep_cols]
    
    if use_features:
        # Map feature names to actual column names (flexible matching)
        selected_features = []
        for feature_name in use_features:
            matching_cols = [col for col in available_features if feature_name in col]
            selected_features.extend(matching_cols)
        
        # Remove duplicates and add to keep_cols
        selected_features = list(set(selected_features))
        keep_cols.extend(selected_features)
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
    else:
        # If no specific features configured, use all
        keep_cols.extend(available_features)
        print(f"Using all {len(available_features)} available features")
    
    # Filter dataframe
    filtered_df = features_df[keep_cols].copy()
    
    return filtered_df

def save_features_and_labels(features_df, labels, config):
    """
    Save features and labels to processed data directory.
    
    Args:
        features_df: DataFrame with features
        labels: Series with labels
        config: Configuration dictionary
    """
    ensure_dir("data/processed")
    
    # Add labels to features dataframe
    target_col = config.get('target', 'y')
    features_df[target_col] = labels
    
    # Save to parquet for efficient loading
    output_path = "data/processed/features_labels.parquet"
    features_df.to_parquet(output_path, index=False)
    
    print(f"Saved features and labels to {output_path}")
    print(f"Shape: {features_df.shape}")
    print(f"Features: {[col for col in features_df.columns if col != target_col]}")
    print(f"Target: {target_col}")
    
    return output_path

def main():
    """Main feature engineering pipeline."""
    print("=== AXON Feature Engineering ===")
    
    # Load configuration
    config = load_config()
    
    try:
        # Load processed data splits
        print("Loading processed data...")
        splits = load_processed_data()
        
        if not splits:
            raise FileNotFoundError("No processed data found. Run dataset pipeline first.")
        
        # Process each split
        processed_splits = {}
        all_labels = {}

        # Compute features and labels on the full timeline to avoid cold-start NaNs in val/test
        print("\nBuilding full timeline for feature computation...")
        # Capture time ranges per split
        split_ranges = {
            name: (df['timestamp'].min(), df['timestamp'].max())
            for name, df in splits.items()
        }

        # Concatenate all splits and sort by time
        df_all = pd.concat(list(splits.values()), axis=0, ignore_index=True)
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)

        print(f"Full timeline rows: {len(df_all):,}")

        # Compute features on full data
        features_all = create_feature_matrix(df_all, config)
        features_all = filter_features(features_all, config)

        # Clean features globally (using only past info)
        features_all = features_all.replace([np.inf, -np.inf], np.nan)
        features_all = features_all.ffill()
        features_all = features_all.dropna()

        # Generate labels on full data (used for train/validation only)
        labels_all = generate_labels(df_all, config)

        # Slice back into splits by time ranges
        for split_name, (start_ts, end_ts) in split_ranges.items():
            print(f"\nMaterializing {split_name} split from full features timeline...")
            mask = (features_all['timestamp'] >= start_ts) & (features_all['timestamp'] <= end_ts)
            split_features = features_all.loc[mask].copy()

            # Align labels for train/validation; keep NaN for test
            if split_name in ['train', 'validation']:
                split_labels = labels_all.loc[split_features.index]
                # Drop rows where labels are NaN (e.g., last horizon minutes)
                if split_labels.isna().any():
                    valid_mask = split_labels.notna()
                    split_features = split_features.loc[valid_mask]
                    split_labels = split_labels.loc[valid_mask]
            else:
                split_labels = pd.Series(np.nan, index=split_features.index)

            removed = mask.sum() - len(split_features)
            if removed > 0:
                print(f"Removed {removed} rows with NaNs when materializing {split_name}")

            processed_splits[split_name] = split_features
            all_labels[split_name] = split_labels

        # Save processed features and labels
        for split_name in processed_splits:
            features_df = processed_splits[split_name]
            labels = all_labels[split_name]
            
            # Add labels to features
            target_col = config.get('target', 'y')
            features_df[target_col] = labels
            
            # Save split
            output_path = f"data/processed/{split_name}_features.parquet"
            features_df.to_parquet(output_path, index=False)
            print(f"Saved {split_name} features: {features_df.shape} -> {output_path}")
        
        print(f"\n[SUCCESS] Feature engineering completed successfully!")
        
        # Summary
        train_df = processed_splits['train']
        feature_cols = [col for col in train_df.columns if col not in ['timestamp', target_col]]
        print(f"\nFeature Engineering Summary:")
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Target column: {target_col}")
        print(f"  Train samples: {len(processed_splits['train']):,}")
        print(f"  Validation samples: {len(processed_splits['validation']):,}")
        print(f"  Test samples: {len(processed_splits['test']):,}")
        
        # Label distribution for train set
        train_labels = all_labels['train'].dropna()
        if len(train_labels) > 0:
            label_dist = train_labels.value_counts(normalize=True)
            print(f"  Label distribution (train): {label_dist.to_dict()}")
        
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        raise

def prepare_sequences_for_nn(features_df, target_col='y', sequence_length=20, stride=1):
    """
    Prepare sliding window sequences for neural network training.

    Args:
        features_df: DataFrame with features and target
        target_col: Name of target column
        sequence_length: Length of each sequence
        stride: Step size for sliding window

    Returns:
        Tuple of (X_sequences, y_targets) as numpy arrays
    """
    print(f"Preparing sequences for NN: sequence_length={sequence_length}, stride={stride}")

    # Separate features and target
    feature_cols = [col for col in features_df.columns if col not in ['timestamp', target_col]]
    X = features_df[feature_cols].values
    y = features_df[target_col].values if target_col in features_df.columns else None

    sequences = []
    targets = []

    # Create sliding windows
    for i in range(0, len(X) - sequence_length + 1, stride):
        seq = X[i:i + sequence_length]
        sequences.append(seq)

        if y is not None:
            # Target is the value at the end of the sequence
            target = y[i + sequence_length - 1]
            targets.append(target)

    X_seq = np.array(sequences)
    y_seq = np.array(targets) if targets else None

    print(f"Created {len(X_seq)} sequences of shape {X_seq.shape}")

    if y_seq is not None:
        print(f"Target distribution: {np.bincount(y_seq.astype(int))}")

    return X_seq, y_seq


def create_nn_scaler():
    """
    Create MinMaxScaler for neural network feature normalization.

    Returns:
        MinMaxScaler instance
    """
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler(feature_range=(-1, 1))


def scale_features_for_nn(X_train, X_val=None, X_test=None, scaler=None):
    """
    Scale features using MinMaxScaler for neural networks.

    Args:
        X_train: Training features (2D array)
        X_val: Validation features (2D array, optional)
        X_test: Test features (2D array, optional)
        scaler: Pre-fitted scaler (optional)

    Returns:
        Tuple of (scaled_X_train, scaled_X_val, scaled_X_test, scaler)
    """
    if scaler is None:
        scaler = create_nn_scaler()

    # Fit on training data
    if X_train.ndim == 2:
        # Regular features
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    else:
        # Sequence data (3D: samples, sequence_length, features)
        # Reshape for scaling
        n_samples, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        scaler.fit(X_train_reshaped)

        X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, seq_len, n_features)

        if X_val is not None:
            n_val_samples = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_val_samples, seq_len, n_features)
        else:
            X_val_scaled = None

        if X_test is not None:
            n_test_samples = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_test_samples, seq_len, n_features)
        else:
            X_test_scaled = None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_nn_data(features_df, target_col='y', sequence_length=20, val_split=0.2, test_split=0.1):
    """
    Prepare complete dataset for neural network training.

    Args:
        features_df: DataFrame with features and target
        target_col: Name of target column
        sequence_length: Length of sequences
        val_split: Validation split ratio
        test_split: Test split ratio

    Returns:
        Dictionary with train/val/test splits and scaler
    """
    print("Preparing NN data splits...")

    # Create sequences
    X_seq, y_seq = prepare_sequences_for_nn(features_df, target_col, sequence_length)

    # Split into train/val/test
    n_total = len(X_seq)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    # Sequential split (time-ordered)
    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]

    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]

    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features_for_nn(
        X_train, X_val, X_test
    )

    # Convert targets to appropriate format for NN
    if y_train is not None:
        # For binary classification, keep as is (0/1)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)

    data = {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'sequence_length': sequence_length
    }

    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    return data


def save_nn_data(nn_data, output_dir="data/processed"):
    """
    Save prepared NN data to disk.

    Args:
        nn_data: Dictionary from prepare_nn_data
        output_dir: Output directory
    """
    ensure_dir(output_dir)

    # Save numpy arrays
    np.save(f"{output_dir}/nn_X_train.npy", nn_data['X_train'])
    np.save(f"{output_dir}/nn_y_train.npy", nn_data['y_train'])
    np.save(f"{output_dir}/nn_X_val.npy", nn_data['X_val'])
    np.save(f"{output_dir}/nn_y_val.npy", nn_data['y_val'])
    np.save(f"{output_dir}/nn_X_test.npy", nn_data['X_test'])
    np.save(f"{output_dir}/nn_y_test.npy", nn_data['y_test'])

    # Save scaler
    import joblib
    joblib.dump(nn_data['scaler'], f"{output_dir}/nn_scaler.pkl")

    # Save metadata
    metadata = {
        'sequence_length': nn_data['sequence_length'],
        'train_shape': nn_data['X_train'].shape,
        'val_shape': nn_data['X_val'].shape,
        'test_shape': nn_data['X_test'].shape,
    }

    with open(f"{output_dir}/nn_metadata.json", 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"Saved NN data to {output_dir}")


if __name__ == "__main__":
    main()