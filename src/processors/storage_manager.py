"""
Storage Manager - Handles data persistence
Supports Parquet format with compression
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages data storage in various formats"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['storage']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        
    def save(self, tweets, format='parquet', stage='processed'):
        """Save tweets to storage."""
        logger.info(f"Saving {len(tweets)} tweets in {format} format (stage={stage})...")
        
        # Convert to DataFrame
        df = pd.DataFrame(tweets)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        target_dir = self.output_dir / stage
        target_dir.mkdir(exist_ok=True)
        
        if format == 'parquet':
            filepath = target_dir / f'tweets_{timestamp}.parquet'
            df.to_parquet(
                filepath,
                compression=self.config['storage']['compression'],
                index=False
            )
        elif format == 'csv':
            filepath = target_dir / f'tweets_{timestamp}.csv'
            df.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = target_dir / f'tweets_{timestamp}.json'
            df.to_json(filepath, orient='records', indent=2)
        
        logger.info(f"[ok] Saved to: {filepath}")
        
        # Save metadata
        self._save_metadata(df, filepath)
        
        return filepath
    
    def save_signals(self, signals):
        """Save trading signals"""
        logger.info(f"Saving {len(signals)} signals...")
        
        df = pd.DataFrame(signals)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f'signals_{timestamp}.csv'
        
        df.to_csv(filepath, index=False)
        
        logger.info(f"[ok] Signals saved to: {filepath}")
        return filepath
    
    def _save_metadata(self, df, data_filepath):
        """Save dataset metadata"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'file_size_mb': data_filepath.stat().st_size / (1024 * 1024),
            'data_file': str(data_filepath)
        }
        
        meta_path = data_filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Metadata saved to {meta_path}")
    
    def load(self, filepath):
        """Load tweets from storage"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
    
    def get_latest_file(self, pattern='tweets_*.parquet'):
        """Get most recent data file"""
        files = list(self.output_dir.glob(f'**/{pattern}'))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
