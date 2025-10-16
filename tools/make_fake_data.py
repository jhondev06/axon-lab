"""AXON Synthetic Data Generator

Generates BTC-like 1m OHLCV data (2019-2024) with noise and regimes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_btc_like_data(start_date='2019-01-01', end_date='2024-12-31', seed=42):
    """
    Generate synthetic BTC-like 1-minute OHLCV data with realistic patterns.
    
    Features:
    - Multiple market regimes (bull, bear, sideways)
    - Realistic volatility clustering
    - Weekend gaps and lower volume
    - Intraday patterns
    - Proper OHLC relationships
    """
    np.random.seed(seed)
    
    # Create minute-by-minute timestamp range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    timestamps = pd.date_range(start=start, end=end, freq='1min')
    
    n_periods = len(timestamps)
    print(f"Generating {n_periods:,} 1-minute candles from {start_date} to {end_date}")
    
    # Base price parameters (BTC-like)
    initial_price = 7000.0  # Starting price around 2019 levels
    
    # Market regime parameters
    regime_length = int(n_periods * 0.15)  # ~15% of data per regime
    n_regimes = max(3, n_periods // regime_length)
    
    # Generate market regimes
    regimes = np.random.choice(['bull', 'bear', 'sideways'], size=n_regimes, p=[0.3, 0.2, 0.5])
    regime_changes = np.linspace(0, n_periods, n_regimes + 1, dtype=int)
    
    # Initialize arrays
    log_returns = np.zeros(n_periods)
    volumes = np.zeros(n_periods)
    
    # Generate returns by regime
    for i in range(n_regimes):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i + 1]
        regime_length_actual = end_idx - start_idx
        
        if regimes[i] == 'bull':
            # Bull market: positive drift, moderate volatility
            drift = 0.00008  # ~0.008% per minute
            base_vol = 0.002  # ~0.2% per minute
        elif regimes[i] == 'bear':
            # Bear market: negative drift, high volatility
            drift = -0.00006  # ~-0.006% per minute
            base_vol = 0.0035  # ~0.35% per minute
        else:  # sideways
            # Sideways: no drift, low volatility
            drift = 0.00001
            base_vol = 0.0015  # ~0.15% per minute
        
        # Generate volatility clustering using GARCH-like process
        vol_persistence = 0.95
        vol_innovation = 0.05
        volatility = np.zeros(regime_length_actual)
        volatility[0] = base_vol
        
        for j in range(1, regime_length_actual):
            vol_shock = np.random.normal(0, vol_innovation * base_vol)
            volatility[j] = vol_persistence * volatility[j-1] + (1 - vol_persistence) * base_vol + abs(vol_shock)
        
        # Generate returns with time-varying volatility
        regime_returns = np.random.normal(drift, volatility)
        
        # Add intraday patterns (higher volatility during certain hours)
        timestamps_regime = timestamps[start_idx:end_idx]
        hour_multiplier = 1 + 0.3 * np.sin(2 * np.pi * timestamps_regime.hour / 24)
        regime_returns *= hour_multiplier
        
        # Add weekend effects (lower activity)
        weekend_mask = timestamps_regime.weekday >= 5
        regime_returns[weekend_mask] *= 0.6
        
        log_returns[start_idx:end_idx] = regime_returns
    
    # Generate volume patterns
    base_volume = 1000000  # Base volume
    for i in range(n_periods):
        ts = timestamps[i]
        
        # Intraday volume pattern (higher during business hours)
        hour_factor = 0.5 + 0.8 * (1 + np.sin(2 * np.pi * (ts.hour - 6) / 24))
        
        # Weekend lower volume
        weekend_factor = 0.3 if ts.weekday() >= 5 else 1.0
        
        # Volume related to volatility
        vol_factor = 1 + 2 * abs(log_returns[i]) / 0.01
        
        # Random component
        noise_factor = np.random.lognormal(0, 0.5)
        
        volumes[i] = base_volume * hour_factor * weekend_factor * vol_factor * noise_factor
    
    # Convert log returns to prices
    prices = initial_price * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC from prices
    opens = np.zeros(n_periods)
    highs = np.zeros(n_periods)
    lows = np.zeros(n_periods)
    closes = prices.copy()
    
    opens[0] = initial_price
    for i in range(1, n_periods):
        opens[i] = closes[i-1]
    
    # Generate realistic high/low spreads
    for i in range(n_periods):
        # Spread based on volatility and volume
        vol_proxy = abs(log_returns[i])
        spread_factor = max(0.0001, vol_proxy * 2)  # Minimum 0.01% spread
        
        # High and low around open/close
        mid_price = (opens[i] + closes[i]) / 2
        spread = mid_price * spread_factor
        
        # Ensure proper OHLC relationships
        highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, spread)
        lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, spread)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes.astype(int)
    })
    
    # Add some derived columns for convenience
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    print(f"Generated data summary:")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Total return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.1f}%")
    print(f"  Avg daily volume: {df['volume'].mean():,.0f}")
    print(f"  Volatility (annualized): {df['returns'].std() * np.sqrt(365 * 24 * 60) * 100:.1f}%")
    
    return df

def main():
    """Generate and save synthetic data."""
    print("AXON Synthetic Data Generator")
    
    # Generate data
    df = generate_btc_like_data()
    
    # Save to data/raw/
    output_path = Path("data/raw/btc_synthetic_1m.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset (without derived columns to keep it clean)
    main_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    main_df.to_csv(output_path, index=False)
    
    print(f"\nSynthetic data saved to {output_path}")
    print(f"Shape: {main_df.shape}")
    print(f"Columns: {list(main_df.columns)}")
    print(f"Date range: {main_df['timestamp'].min()} to {main_df['timestamp'].max()}")


if __name__ == "__main__":
    main()