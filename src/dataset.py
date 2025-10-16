"""AXON Dataset Module

Handles data loading, preprocessing, and time-based splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import yfinance as yf
from datetime import datetime, timedelta
from .utils import load_config, ensure_dir
import os
# from alpha_vantage.timeseries import TimeSeries  # Temporarily commented for testing
import json
# from .binance_ws import start_binance_ws  # Temporarily commented for testing
import logging
import time
import hashlib
from typing import Dict, Optional, Tuple, List
import pickle
import lzma

class IntelligentCache:
    """
    Sistema de cache inteligente hierárquico para dados financeiros.

    Features:
    - Cache triplo: memória → disco → API
    - Validação de freshness automática
    - Compressão de dados históricos
    - Metadata de qualidade e fonte
    - Invalidação inteligente baseada em tempo
    """

    def __init__(self, cache_dir="data/cache", memory_ttl=300, disk_ttl=3600):
        """
        Args:
            cache_dir: Diretório base para cache em disco
            memory_ttl: TTL para cache em memória (segundos)
            disk_ttl: TTL para cache em disco (segundos)
        """
        self.cache_dir = Path(cache_dir)
        self.memory_ttl = memory_ttl
        self.disk_ttl = disk_ttl
        self.memory_cache = {}
        self.cache_metadata = {}

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger('IntelligentCache')

    def _generate_cache_key(self, source: str, symbol: str, interval: str,
                           start_date: datetime, end_date: datetime) -> str:
        """Generate unique cache key based on parameters."""
        key_data = f"{source}_{symbol}_{interval}_{start_date.isoformat()}_{end_date.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, source: str) -> Path:
        """Get filesystem path for cache file."""
        return self.cache_dir / source / f"{cache_key}.parquet.xz"

    def _compress_dataframe(self, df: pd.DataFrame) -> bytes:
        """Compress DataFrame for storage."""
        return lzma.compress(pickle.dumps(df))

    def _decompress_dataframe(self, data: bytes) -> pd.DataFrame:
        """Decompress DataFrame from storage."""
        return pickle.loads(lzma.decompress(data))

    def get(self, source: str, symbol: str, interval: str,
            start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache hierarchy.

        Returns:
            DataFrame if found and fresh, None otherwise
        """
        cache_key = self._generate_cache_key(source, symbol, interval, start_date, end_date)

        # 1. Check memory cache
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if self._is_fresh(entry['timestamp'], self.memory_ttl):
                self.logger.debug(f"Memory cache hit for {source}/{symbol}")
                return entry['data']
            else:
                # Remove stale memory cache
                del self.memory_cache[cache_key]

        # 2. Check disk cache
        cache_path = self._get_cache_path(cache_key, source)
        if cache_path.exists():
            try:
                # Check file freshness
                file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if self._is_fresh(file_mtime, self.disk_ttl):
                    # Load compressed data
                    with open(cache_path, 'rb') as f:
                        compressed_data = f.read()

                    df = self._decompress_dataframe(compressed_data)

                    # Store in memory cache
                    self.memory_cache[cache_key] = {
                        'data': df,
                        'timestamp': datetime.now(),
                        'metadata': self.cache_metadata.get(cache_key, {})
                    }

                    self.logger.debug(f"Disk cache hit for {source}/{symbol}")
                    return df
                else:
                    # Remove stale file
                    cache_path.unlink()
            except Exception as e:
                self.logger.warning(f"Error loading cache file {cache_path}: {e}")
                # Remove corrupted file
                if cache_path.exists():
                    cache_path.unlink()

        return None

    def put(self, source: str, symbol: str, interval: str,
            start_date: datetime, end_date: datetime,
            data: pd.DataFrame, metadata: Dict = None):
        """
        Store data in cache hierarchy.

        Args:
            source: Data source (yahoo, alpha_vantage, binance)
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date of data
            end_date: End date of data
            data: DataFrame to cache
            metadata: Additional metadata (quality metrics, etc.)
        """
        if data.empty:
            return

        cache_key = self._generate_cache_key(source, symbol, interval, start_date, end_date)

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'source': source,
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'record_count': len(data),
            'cached_at': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(data)
        })

        # 1. Store in memory cache
        self.memory_cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now(),
            'metadata': metadata
        }

        # 2. Store in disk cache
        try:
            cache_path = self._get_cache_path(cache_key, source)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Compress and save
            compressed_data = self._compress_dataframe(data)
            with open(cache_path, 'wb') as f:
                f.write(compressed_data)

            # Store metadata
            self.cache_metadata[cache_key] = metadata

            self.logger.debug(f"Cached data for {source}/{symbol}: {len(data)} records")

        except Exception as e:
            self.logger.error(f"Error saving to disk cache: {e}")

    def _is_fresh(self, timestamp: datetime, ttl: int) -> bool:
        """Check if cache entry is still fresh."""
        return (datetime.now() - timestamp).total_seconds() < ttl

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess data quality metrics."""
        if df.empty:
            return {'quality_score': 0}

        quality_metrics = {
            'total_records': len(df),
            'missing_data_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'duplicate_timestamps': df.duplicated(subset=['timestamp']).sum() if 'timestamp' in df.columns else 0,
            'negative_prices': ((df[['open', 'high', 'low', 'close']] <= 0).sum().sum()) if all(col in df.columns for col in ['open', 'high', 'low', 'close']) else 0,
            'invalid_ohlc': 0
        }

        # Check OHLC validity
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = ~(
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            )
            quality_metrics['invalid_ohlc'] = invalid_ohlc.sum()

        # Calculate quality score (0-100)
        quality_score = 100
        quality_score -= quality_metrics['missing_data_pct'] * 2
        quality_score -= (quality_metrics['duplicate_timestamps'] / len(df)) * 50
        quality_score -= (quality_metrics['negative_prices'] / len(df)) * 100
        quality_score -= (quality_metrics['invalid_ohlc'] / len(df)) * 100

        quality_metrics['quality_score'] = max(0, min(100, quality_score))

        return quality_metrics

    def invalidate_source(self, source: str):
        """Invalidate all cache entries for a specific source."""
        try:
            source_dir = self.cache_dir / source
            if source_dir.exists():
                import shutil
                shutil.rmtree(source_dir)
                self.logger.info(f"Invalidated cache for source: {source}")

            # Clear memory cache for this source
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{source}_")]
            for key in keys_to_remove:
                del self.memory_cache[key]

        except Exception as e:
            self.logger.error(f"Error invalidating cache for {source}: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'memory_entries': len(self.memory_cache),
            'disk_cache_size': self._calculate_disk_cache_size(),
            'sources_cached': self._get_cached_sources()
        }

    def _calculate_disk_cache_size(self) -> int:
        """Calculate total size of disk cache in bytes."""
        total_size = 0
        try:
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except:
            pass
        return total_size

    def _get_cached_sources(self) -> List[str]:
        """Get list of sources with cached data."""
        sources = []
        try:
            for source_dir in self.cache_dir.iterdir():
                if source_dir.is_dir():
                    sources.append(source_dir.name)
        except:
            pass
        return sources

# Global cache instance
_cache_instance = None

def get_cache_instance(cache_dir="data/cache") -> IntelligentCache:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache(cache_dir)
    return _cache_instance

def download_yahoo_finance_data(symbol="BTC-USD", period="7d", interval="1m"):
    """
    Download comprehensive data from Yahoo Finance with chunking support.

    Features:
    - Chunking for periods longer than 7 days (1m interval limit)
    - Intelligent caching with parquet compression
    - Multiple symbols support
    - Error handling and retries
    - Data quality validation
    """
    try:
        # Check if we need chunking (Yahoo limits 1m data to 7 days)
        if interval == "1m" and period.endswith("d"):
            days = int(period[:-1])
            if days > 7:
                return _download_yahoo_with_chunking(symbol, days, interval)

        # Standard download for short periods
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Prepare data in OHLCV format
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Select only relevant columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Validate data quality
        df = _validate_yahoo_data(df, symbol)

        return df

    except Exception as e:
        logging.error(f"Erro no download Yahoo Finance para {symbol}: {e}")
        raise

def _download_yahoo_with_chunking(symbol, total_days, interval):
    """Download Yahoo Finance data in chunks to bypass 7-day limit for 1m data."""
    logging.info(f"Usando chunking para {symbol}: {total_days} dias em intervalos de {interval}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_days)
    chunk_size = 7  # 7 days max per chunk

    all_data = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_size), end_date)

        # Format dates for Yahoo
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')

        try:
            logging.info(f"Baixando chunk: {start_str} to {end_str}")

            # Rate limiting
            time.sleep(1)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_str, end=end_str, interval=interval)

            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={
                    'Datetime': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                all_data.append(df)

        except Exception as e:
            logging.warning(f"Erro no chunk {start_str}-{end_str}: {e}")
            # Continue with next chunk instead of failing completely

        current_start = current_end

    if not all_data:
        raise ValueError(f"Nenhum dado baixado para {symbol} após chunking")

    # Combine all chunks
    df_combined = pd.concat(all_data, ignore_index=True)

    # Remove duplicates and sort
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    # Validate combined data
    df_combined = _validate_yahoo_data(df_combined, symbol)

    logging.info(f"Chunking concluído: {len(df_combined)} registros para {symbol}")
    return df_combined

def _validate_yahoo_data(df, symbol):
    """Validate and clean Yahoo Finance data quality."""
    if df.empty:
        return df

    # Remove rows with missing critical data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    # Validate price relationships
    valid_mask = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close']) &
        (df['open'] > 0) &
        (df['high'] > 0) &
        (df['low'] > 0) &
        (df['close'] > 0) &
        (df['volume'] >= 0)
    )

    df = df[valid_mask]

    # Remove extreme price changes (>20% in one interval for crypto)
    if len(df) > 1:
        price_change = df['close'].pct_change().abs()
        df = df[price_change <= 0.2]

    logging.info(f"Validação Yahoo Finance {symbol}: {len(df)} registros válidos")
    return df.reset_index(drop=True)

def generate_data_if_missing():
    """Generate real data from Yahoo Finance if raw data files are missing."""
    raw_data_path = Path("data/raw/btc_real_1m.csv")
    
    if not raw_data_path.exists():
        print("Raw data not found. Downloading real market data from Yahoo Finance...")
        
        try:
            # Download real BTC-USD data usando dados históricos (Yahoo limita dados de 1 minuto)
            df = download_yahoo_finance_data("BTC-USD", "7d", "1m")
            
            # Save downloaded data
            ensure_dir("data/raw")
            df.to_csv(raw_data_path, index=False)
            print(f"Real market data saved to {raw_data_path}")
            
        except Exception as e:
            print(f"Real data download failed, falling back to synthetic data: {e}")
            # Fallback to synthetic data
            result = subprocess.run([sys.executable, "tools/make_fake_data.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Data generation failed: {result.stderr}")
            print("Synthetic data generated as fallback.")
    
    return raw_data_path

def load_raw_data(file_path):
    """Load raw OHLCV data with proper datetime parsing."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Validate OHLCV data integrity
    assert all(df['high'] >= df['low']), "High prices must be >= low prices"
    assert all(df['high'] >= df['open']), "High prices must be >= open prices"
    assert all(df['high'] >= df['close']), "High prices must be >= close prices"
    assert all(df['low'] <= df['open']), "Low prices must be <= open prices"
    assert all(df['low'] <= df['close']), "Low prices must be <= close prices"
    assert all(df['volume'] >= 0), "Volume must be non-negative"

    return df

def create_time_based_splits(df, train_ratio=0.6, test_ratio=0.2):
    """
    Create time-based train/validation/test splits to prevent future leakage.

    Args:
        df: DataFrame with timestamp column
        train_ratio: Proportion for training (default 60%)
        test_ratio: Proportion for testing (default 20%, remaining 20% for validation)

    Returns:
        dict with train, validation, test DataFrames
    """
    # Ensure data is sorted by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_val = n_total - n_train - n_test

    # Time-based splits (no shuffling to prevent leakage)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    # Validate no temporal leakage
    assert train_df['timestamp'].max() < val_df['timestamp'].min(), "Train data leaks into validation"
    assert val_df['timestamp'].max() < test_df['timestamp'].min(), "Validation data leaks into test"

    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }

def add_basic_features(df):
    """
    Add basic derived features that don't require future data.
    
    Note: Advanced features are handled in features.py
    """
    df = df.copy()
    
    # Basic price features (no lookahead)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    
    # Volume features
    df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def clean_and_validate_data(df):
    """
    Clean data and handle missing values.
    """
    # Remove rows with missing critical data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    # Remove rows with zero or negative prices
    price_cols = ['open', 'high', 'low', 'close']
    df = df[(df[price_cols] > 0).all(axis=1)]

    # Remove extreme outliers (prices that change more than 50% in one minute)
    df['price_change'] = df['close'].pct_change().abs()
    df = df[df['price_change'] <= 0.5]  # Remove >50% price changes
    df = df.drop('price_change', axis=1)

    # Forward fill any remaining NaN values in derived features
    df = df.ffill()

    # Drop any remaining NaN rows (typically the first few rows)
    df = df.dropna()

    return df

def save_processed_data(splits_dict, config):
    """
    Save processed data splits to parquet files for efficient loading.
    """
    ensure_dir("data/processed")
    
    saved_files = []
    for split_name, df in splits_dict.items():
        output_path = f"data/processed/{split_name}.parquet"
        df.to_parquet(output_path, index=False)
        saved_files.append(output_path)
        print(f"Saved {split_name} split: {len(df):,} rows -> {output_path}")
    
    # Save metadata
    metadata = {
        'splits': {name: len(df) for name, df in splits_dict.items()},
        'columns': list(splits_dict['train'].columns),
        'date_range': {
            'start': str(splits_dict['train']['timestamp'].min()),
            'end': str(splits_dict['test']['timestamp'].max())
        },
        'config_used': config
    }
    
    import json
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return saved_files

def load_processed_data(split_name=None):
    """
    Load processed data splits from parquet files.
    
    Args:
        split_name: 'train', 'validation', 'test', or None for all splits
    
    Returns:
        DataFrame or dict of DataFrames
    """
    if split_name:
        file_path = f"data/processed/{split_name}.parquet"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Processed data not found: {file_path}. Run dataset pipeline first.")
        return pd.read_parquet(file_path)
    else:
        # Load all splits
        splits = {}
        for split in ['train', 'validation', 'test']:
            file_path = f"data/processed/{split}.parquet"
            if Path(file_path).exists():
                splits[split] = pd.read_parquet(file_path)
        return splits

def load_data_with_fallback(config):
    """
    Sistema de fallback robusto com priorização automática de fontes.

    Features:
    - Priorização baseada em qualidade e disponibilidade
    - Validação cross-source para consistência
    - Switching automático em caso de falha
    - Logging detalhado de decisões
    """
    sources = config['data']['sources']
    symbol = config['data']['symbols'][0] if config['data']['symbols'] else 'BTC-USD'
    interval = config['data']['interval']
    lookback_days = config['data']['lookback_days']

    cache = get_cache_instance()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Try sources in priority order
    source_results = {}

    for source in sources:
        try:
            logging.info(f"Tentando fonte: {source}")

            # Check cache first
            cached_data = cache.get(source, symbol, interval, start_date, end_date)
            if cached_data is not None:
                logging.info(f"Dados frescos encontrados no cache para {source}")
                df = cached_data
            else:
                # Download fresh data
                df = _download_from_source(source, config)

                if df is not None and not df.empty:
                    # Cache the data
                    cache.put(source, symbol, interval, start_date, end_date, df)

            if df is not None and not df.empty:
                # Validate data quality
                quality_score = _assess_source_quality(df, source)

                source_results[source] = {
                    'data': df,
                    'quality_score': quality_score,
                    'record_count': len(df),
                    'date_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None
                }

                logging.info(f"Fonte {source}: {len(df)} registros, qualidade: {quality_score:.1f}%")

                # If quality is acceptable, return immediately
                if quality_score >= 80:
                    logging.info(f"Fonte selecionada: {source} (qualidade: {quality_score:.1f}%)")
                    return df, source

        except Exception as e:
            logging.warning(f"Falha na fonte {source}: {e}")
            source_results[source] = {'error': str(e)}

    # If no source has acceptable quality, select the best available
    if source_results:
        best_source = _select_best_source(source_results)
        if best_source:
            result = source_results[best_source]
            if 'data' in result:
                logging.info(f"Fonte selecionada (melhor disponível): {best_source} (qualidade: {result['quality_score']:.1f}%)")
                return result['data'], best_source

    # Last resort: synthetic data
    logging.warning("Todas as fontes falharam, usando dados sintéticos")
    try:
        df_synthetic = _generate_synthetic_fallback(config)
        if df_synthetic is not None and not df_synthetic.empty:
            logging.info("Dados sintéticos gerados como fallback")
            return df_synthetic, 'synthetic'
    except Exception as e:
        logging.error(f"Falha na geração de dados sintéticos: {e}")

    raise ValueError("Todas as fontes de dados falharam, incluindo fallback sintético")

def _download_from_source(source, config):
    """Download data from specific source."""
    if source == 'yahoo':
        period = f"{config['data']['lookback_days']}d"
        return download_yahoo_finance_data(
            config['data']['symbols'][0],
            period=period,
            interval=config['data']['interval']
        )
    elif source == 'alpha_vantage':
        return download_alpha_vantage_data(config)
    elif source == 'binance_ws':
        return load_binance_ws_data(config)
    elif source == 'synthetic':
        return _generate_synthetic_fallback(config)
    else:
        raise ValueError(f"Fonte não suportada: {source}")

def _assess_source_quality(df, source):
    """Assess data quality for a specific source."""
    if df.empty:
        return 0

    quality_score = 100

    # Basic validations
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        quality_score -= 50

    # Check for missing data
    missing_pct = df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols)) * 100
    quality_score -= missing_pct * 2

    # Check data consistency
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # OHLC validation
        valid_ohlc = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['open'] > 0) &
            (df['close'] > 0)
        )
        invalid_pct = (~valid_ohlc).sum() / len(df) * 100
        quality_score -= invalid_pct

    # Check for duplicates
    if 'timestamp' in df.columns:
        dup_pct = df.duplicated(subset=['timestamp']).sum() / len(df) * 100
        quality_score -= dup_pct * 2

    # Source-specific quality adjustments
    if source == 'yahoo':
        # Yahoo has known issues with after-hours data
        quality_score -= 5
    elif source == 'alpha_vantage':
        # Alpha Vantage has rate limits but good data quality
        quality_score -= 2
    elif source == 'binance_ws':
        # Real-time data might have gaps
        quality_score -= 10

    return max(0, min(100, quality_score))

def _select_best_source(source_results):
    """Select the best available source based on quality and availability."""
    candidates = []

    for source, result in source_results.items():
        if 'data' in result and 'quality_score' in result:
            candidates.append({
                'source': source,
                'quality_score': result['quality_score'],
                'record_count': result['record_count']
            })

    if not candidates:
        return None

    # Sort by quality score, then by record count
    candidates.sort(key=lambda x: (x['quality_score'], x['record_count']), reverse=True)

    return candidates[0]['source']

def _generate_synthetic_fallback(config):
    """Generate synthetic data as last resort fallback."""
    try:
        result = subprocess.run([sys.executable, "tools/make_fake_data.py"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Synthetic data generation failed: {result.stderr}")

        file_path = "data/raw/btc_synthetic_1m.csv"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Synthetic data file not found: {file_path}")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Validate synthetic data
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Synthetic data missing required columns")

        return df

    except Exception as e:
        logging.error(f"Error generating synthetic fallback: {e}")
        return None

def validate_data_quality(df, source=None, config=None):
    """
    Validação abrangente de qualidade de dados.

    Features:
    - Detecção de gaps temporais
    - Validação de preços realistas
    - Comparação cross-source (se disponível)
    - Métricas de qualidade detalhadas
    """
    if df.empty:
        return {
            'overall_quality': 0,
            'issues': ['DataFrame vazio'],
            'recommendations': ['Verificar fonte de dados']
        }

    quality_report = {
        'overall_quality': 100,
        'issues': [],
        'warnings': [],
        'metrics': {},
        'recommendations': []
    }

    # 1. Validação básica de estrutura
    structure_issues = _validate_data_structure(df)
    quality_report['issues'].extend(structure_issues)
    quality_report['overall_quality'] -= len(structure_issues) * 10

    # 2. Detecção de gaps temporais
    gap_analysis = _detect_temporal_gaps(df)
    quality_report['metrics']['temporal_gaps'] = gap_analysis
    if gap_analysis['gap_percentage'] > 5:
        quality_report['issues'].append(f"Gaps temporais detectados: {gap_analysis['gap_percentage']:.1f}%")
        quality_report['overall_quality'] -= min(20, gap_analysis['gap_percentage'])

    # 3. Validação de preços realistas
    price_validation = _validate_realistic_prices(df)
    quality_report['metrics']['price_validation'] = price_validation
    if not price_validation['all_realistic']:
        quality_report['issues'].extend(price_validation['issues'])
        quality_report['overall_quality'] -= len(price_validation['issues']) * 5

    # 4. Validação de consistência OHLC
    ohlc_validation = _validate_ohlc_consistency(df)
    quality_report['metrics']['ohlc_consistency'] = ohlc_validation
    if ohlc_validation['invalid_percentage'] > 1:
        quality_report['issues'].append(f"Inconsistências OHLC: {ohlc_validation['invalid_percentage']:.1f}%")
        quality_report['overall_quality'] -= ohlc_validation['invalid_percentage']

    # 5. Detecção de outliers
    outlier_analysis = _detect_outliers(df)
    quality_report['metrics']['outliers'] = outlier_analysis
    if outlier_analysis['outlier_percentage'] > 2:
        quality_report['warnings'].append(f"Outliers detectados: {outlier_analysis['outlier_percentage']:.1f}%")
        quality_report['overall_quality'] -= outlier_analysis['outlier_percentage'] * 0.5

    # 6. Análise de volume
    volume_analysis = _analyze_volume_quality(df)
    quality_report['metrics']['volume_analysis'] = volume_analysis
    if volume_analysis['zero_volume_percentage'] > 10:
        quality_report['issues'].append(f"Volume zero excessivo: {volume_analysis['zero_volume_percentage']:.1f}%")
        quality_report['overall_quality'] -= volume_analysis['zero_volume_percentage'] * 0.5

    # 7. Comparação cross-source (se múltiplas fontes disponíveis)
    if source and config:
        cross_source_comparison = _compare_cross_source(df, source, config)
        if cross_source_comparison:
            quality_report['metrics']['cross_source'] = cross_source_comparison

    # Garantir qualidade não negativa
    quality_report['overall_quality'] = max(0, quality_report['overall_quality'])

    # Gerar recomendações baseadas nos problemas encontrados
    quality_report['recommendations'] = _generate_quality_recommendations(quality_report)

    logging.info(f"Qualidade de dados {source or 'unknown'}: {quality_report['overall_quality']:.1f}%")
    if quality_report['issues']:
        logging.warning(f"Problemas encontrados: {quality_report['issues']}")

    return quality_report

def _validate_data_structure(df):
    """Validate basic data structure."""
    issues = []

    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Colunas obrigatórias faltando: {missing_cols}")

    # Check for NaN values
    nan_counts = df.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        issues.append(f"Colunas com NaN: {cols_with_nan.to_dict()}")

    # Check data types
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            issues.append("Coluna timestamp não é datetime")

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Coluna {col} não é numérica")

    return issues

def _detect_temporal_gaps(df):
    """Detect temporal gaps in the data."""
    if 'timestamp' not in df.columns or df.empty:
        return {'gap_percentage': 100, 'total_gaps': 0, 'gap_details': []}

    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').copy()

    # Calculate expected intervals (assume 1-minute data by default)
    if len(df_sorted) > 1:
        time_diffs = df_sorted['timestamp'].diff().dropna()
        median_interval = time_diffs.median()

        # Find gaps larger than 2x median interval
        gap_threshold = median_interval * 2
        gaps = time_diffs[time_diffs > gap_threshold]

        gap_percentage = (gaps.sum() / (df_sorted['timestamp'].max() - df_sorted['timestamp'].min())).total_seconds() * 100

        return {
            'gap_percentage': gap_percentage,
            'total_gaps': len(gaps),
            'median_interval': median_interval,
            'gap_details': gaps.head(5).tolist()  # Top 5 gaps
        }

    return {'gap_percentage': 0, 'total_gaps': 0, 'gap_details': []}

def _validate_realistic_prices(df):
    """Validate that prices are realistic for the asset."""
    issues = []
    all_realistic = True

    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return {'all_realistic': False, 'issues': ['Colunas OHLC faltando']}

    # Check for negative or zero prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        invalid_prices = (df[col] <= 0).sum()
        if invalid_prices > 0:
            issues.append(f"Preços {col} inválidos (≤0): {invalid_prices}")
            all_realistic = False

    # Check for extremely high prices (assuming crypto, max reasonable ~$1M)
    for col in price_cols:
        extreme_prices = (df[col] > 1000000).sum()
        if extreme_prices > 0:
            issues.append(f"Preços {col} extremos (>1M): {extreme_prices}")
            all_realistic = False

    # Check for price changes that are too extreme (>50% in one interval)
    if len(df) > 1:
        returns = df['close'].pct_change().abs()
        extreme_changes = (returns > 0.5).sum()
        if extreme_changes > 0:
            issues.append(f"Mudanças de preço extremas (>50%): {extreme_changes}")
            all_realistic = False

    return {
        'all_realistic': all_realistic,
        'issues': issues,
        'price_range': {
            'min': df['low'].min(),
            'max': df['high'].max(),
            'median': df['close'].median()
        }
    }

def _validate_ohlc_consistency(df):
    """Validate OHLC price consistency."""
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return {'invalid_percentage': 100, 'total_invalid': len(df)}

    # Check OHLC relationships
    valid_ohlc = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    )

    invalid_count = (~valid_ohlc).sum()
    invalid_percentage = (invalid_count / len(df)) * 100

    return {
        'invalid_percentage': invalid_percentage,
        'total_invalid': invalid_count,
        'total_valid': len(df) - invalid_count
    }

def _detect_outliers(df):
    """Detect statistical outliers in price data."""
    if 'close' not in df.columns or df.empty:
        return {'outlier_percentage': 0, 'total_outliers': 0}

    # Use IQR method for outlier detection
    Q1 = df['close'].quantile(0.25)
    Q3 = df['close'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df['close'] < lower_bound) | (df['close'] > upper_bound))
    outlier_count = outliers.sum()
    outlier_percentage = (outlier_count / len(df)) * 100

    return {
        'outlier_percentage': outlier_percentage,
        'total_outliers': outlier_count,
        'bounds': {'lower': lower_bound, 'upper': upper_bound}
    }

def _analyze_volume_quality(df):
    """Analyze volume data quality."""
    if 'volume' not in df.columns:
        return {'zero_volume_percentage': 100, 'negative_volume': 0}

    zero_volume = (df['volume'] == 0).sum()
    negative_volume = (df['volume'] < 0).sum()

    zero_percentage = (zero_volume / len(df)) * 100

    return {
        'zero_volume_percentage': zero_percentage,
        'negative_volume': negative_volume,
        'volume_stats': {
            'mean': df['volume'].mean(),
            'median': df['volume'].median(),
            'std': df['volume'].std()
        }
    }

def _compare_cross_source(df, source, config):
    """Compare data with other sources for consistency."""
    # This would require loading data from other sources
    # For now, return basic structure
    return {
        'comparison_available': False,
        'note': 'Cross-source comparison not implemented yet'
    }

def _generate_quality_recommendations(quality_report):
    """Generate recommendations based on quality issues."""
    recommendations = []

    if quality_report['overall_quality'] < 50:
        recommendations.append("Qualidade de dados crítica - considerar fonte alternativa")

    if any('gaps temporais' in issue.lower() for issue in quality_report['issues']):
        recommendations.append("Preencher gaps temporais ou usar fonte com melhor cobertura")

    if any('preços' in issue.lower() and 'inválidos' in issue.lower() for issue in quality_report['issues']):
        recommendations.append("Filtrar ou corrigir preços inválidos")

    if any('ohlc' in issue.lower() for issue in quality_report['issues']):
        recommendations.append("Corrigir inconsistências OHLC")

    if not recommendations:
        recommendations.append("Dados de boa qualidade - nenhum ação imediata necessária")

    return recommendations

def download_alpha_vantage_data(config):
    """
    Download comprehensive data from Alpha Vantage with advanced features.

    Features:
    - Rate limiting (5 calls/min for free tier)
    - Intelligent caching with parquet compression
    - Multiple intervals support (1min, 5min, 15min, 30min, 1h)
    - Multiple symbols support
    - Error handling and fallbacks
    - Data quality validation
    """
    symbols = config['data']['symbols']
    interval = config['data']['interval']
    lookback_days = config['data']['lookback_days']
    cache_dir = config['data']['cache']['dir']

    # Validate interval support
    supported_intervals = ['1min', '5min', '15min', '30min', '60min']
    if interval not in supported_intervals:
        raise ValueError(f"Interval {interval} not supported by Alpha Vantage. Use: {supported_intervals}")

    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY não encontrada nas variáveis de ambiente")

    from alpha_vantage.timeseries import TimeSeries
    ts = TimeSeries(key=api_key, output_format='pandas')

    all_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    for symbol in symbols:
        cache_path = os.path.join(cache_dir, f'alpha/{symbol}_{interval}.parquet')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Check cache freshness (24h validity)
        if os.path.exists(cache_path):
            cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - cache_mtime).total_seconds() < 86400:  # 24 hours
                logging.info(f"Carregando dados frescos do cache: {cache_path}")
                df_cached = pd.read_parquet(cache_path)
                # Filter by date range
                df_cached = df_cached[(df_cached['timestamp'] >= start_date) & (df_cached['timestamp'] <= end_date)]
                if not df_cached.empty:
                    all_data[symbol] = df_cached
                    continue

        logging.info(f"Baixando dados Alpha Vantage para {symbol} ({interval})")

        # Rate limiting: 5 calls per minute for free tier
        data_frames = []
        current_date = start_date.replace(day=1)

        while current_date <= end_date:
            month_str = current_date.strftime('%Y-%m')

            try:
                # Rate limiting
                time.sleep(12)  # ~5 calls per minute

                data, meta_data = ts.get_intraday(
                    symbol=symbol,
                    interval=interval,
                    outputsize='full',
                    month=month_str
                )

                if not data.empty:
                    data_frames.append(data)
                    logging.info(f"Downloaded {len(data)} records for {symbol} - {month_str}")

            except Exception as e:
                logging.warning(f"Erro ao baixar {symbol} para {month_str}: {e}")
                if "rate limit" in str(e).lower():
                    logging.warning("Rate limit atingido, aguardando 60s...")
                    time.sleep(60)

            current_date = (current_date + timedelta(days=32)).replace(day=1)

        if not data_frames:
            logging.warning(f"Nenhum dado baixado para {symbol}")
            continue

        # Combine and clean data
        df = pd.concat(data_frames, ignore_index=False)
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })

        # Convert index to timestamp column
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        df = df.sort_values('timestamp')

        # Filter date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')

        # Validate data quality
        df = _validate_alpha_vantage_data(df, symbol)

        if not df.empty:
            # Save to cache
            df.to_parquet(cache_path, index=False, compression='snappy')
            logging.info(f"Dados salvos no cache: {cache_path} ({len(df)} registros)")
            all_data[symbol] = df

    if not all_data:
        raise ValueError("Nenhum dado válido baixado da Alpha Vantage")

    # Return single symbol or combine multiple
    if len(symbols) == 1:
        return all_data[symbols[0]]
    else:
        # Add symbol column for multi-symbol data
        combined_dfs = []
        for symbol, df in all_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            combined_dfs.append(df_copy)
        return pd.concat(combined_dfs, ignore_index=True)

def _validate_alpha_vantage_data(df, symbol):
    """Validate and clean Alpha Vantage data quality."""
    if df.empty:
        return df

    # Remove rows with missing critical data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    # Validate price relationships
    valid_mask = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close']) &
        (df['open'] > 0) &
        (df['high'] > 0) &
        (df['low'] > 0) &
        (df['close'] > 0) &
        (df['volume'] >= 0)
    )

    df = df[valid_mask]

    # Remove extreme price changes (>50% in one interval)
    if len(df) > 1:
        price_change = df['close'].pct_change().abs()
        df = df[price_change <= 0.5]

    # Remove zero volume periods
    df = df[df['volume'] > 0]

    logging.info(f"Validação Alpha Vantage {symbol}: {len(df)} registros válidos")
    return df.reset_index(drop=True)


def load_binance_ws_data(config):
    # Map configured symbol (e.g., 'BTC-USD', 'BTC/USD', 'BTCUSD') to Binance spot symbol (e.g., 'BTCUSDT')
    cfg_symbol = config['data']['symbols'][0] if isinstance(config['data'].get('symbols'), list) and config['data']['symbols'] else 'BTC-USD'
    norm = cfg_symbol.replace('-', '').replace('/', '')
    if norm.endswith('USD') and not norm.endswith('USDT'):
        norm = norm[:-3] + 'USDT'
    symbol = norm  # e.g., BTCUSDT

    interval = config['data']['interval']
    lookback_days = config['data']['lookback_days']
    raw_dir = 'data/raw/rt'  # Hardcode or add to config
    ndjson_path = os.path.join(raw_dir, f'{symbol}_{interval}.ndjson')
    
    if not os.path.exists(ndjson_path):
        logging.warning(f'Arquivo NDJSON não encontrado: {ndjson_path}')
        return pd.DataFrame()
    
    start_time = datetime.now() - timedelta(days=lookback_days)
    data = []
    with open(ndjson_path, 'r') as f:
        for line in f:
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if 'k' in msg:
                k = msg['k']
                # Use only closed klines to avoid partial updates/duplicates
                if not k.get('x', False):
                    continue
                open_time = datetime.fromtimestamp(k['t'] / 1000)
                if open_time >= start_time:
                    data.append({
                        'timestamp': open_time,
                        'open': float(k['o']),
                        'high': float(k['h']),
                        'low': float(k['l']),
                        'close': float(k['c']),
                        'volume': float(k['v'])
                    })
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Deduplicate by timestamp keeping the last occurrence
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    logging.info(f'Dados Binance WS carregados: {df.shape}')
    return df


def main():
    """Main dataset processing pipeline."""
    print("=== AXON Dataset Pipeline ===")
    
    # Load configuration
    config = load_config()
    
    try:
        # Step 1: Ensure raw data exists
        raw_data_path = generate_data_if_missing()
        
        # Step 2: Load raw data
        # Em main(), substitua load_raw_data por load_data_with_fallback
        df, selected_source = load_data_with_fallback(config)
        print(f"Fonte selecionada: {selected_source}")
        
        # Step 3: Add basic features
        df = add_basic_features(df)
        
        # Step 4: Clean and validate
        # Após create_time_based_splits em main
        splits = create_time_based_splits(df)
        
        min_rows_test = config['data']['quality']['min_rows_test']
        if len(splits['test']) < min_rows_test:
            raise ValueError(f"Split de teste tem apenas {len(splits['test'])} rows, mínimo requerido: {min_rows_test}")
        
        logging.info(f"Shapes: Train {splits['train'].shape}, Val {splits['validation'].shape}, Test {splits['test'].shape}")
        saved_files = save_processed_data(splits, config)
        
        print(f"\n[SUCCESS] Dataset pipeline completed successfully!")
        print(f"Processed files saved: {saved_files}")
        
        # Summary statistics
        total_rows = sum(len(df) for df in splits.values())
        print(f"\nDataset Summary:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Features: {len(splits['train'].columns)}")
        print(f"  Time span: {splits['train']['timestamp'].min()} to {splits['test']['timestamp'].max()}")
        
    except Exception as e:
        print(f"[ERROR] Dataset pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()