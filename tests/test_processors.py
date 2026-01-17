# tests/test_processors.py
"""
Unit tests for data processors
"""

import pytest
from src.processors.data_cleaner import DataCleaner
from src.processors.deduplicator import Deduplicator
import yaml


@pytest.fixture
def config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_tweets():
    """Sample tweet data for testing"""
    return [
        {
            'id': '1',
            'username': 'trader1',
            'content': 'NIFTY looking bullish! #nifty50 #stockmarket',
            'timestamp': '2025-01-17T10:00:00',
            'likes': 10,
            'retweets': 5,
            'replies': 2,
            'hashtags': ['#nifty50', '#stockmarket'],
            'mentions': []
        },
        {
            'id': '2',
            'username': 'trader2',
            'content': 'NIFTY looking bullish! #nifty50 #stockmarket',  # Duplicate
            'timestamp': '2025-01-17T10:05:00',
            'likes': 8,
            'retweets': 3,
            'replies': 1,
            'hashtags': ['#nifty50', '#stockmarket'],
            'mentions': []
        },
        {
            'id': '3',
            'username': 'trader3',
            'content': 'Bank Nifty breakout coming! 📈',
            'timestamp': '2025-01-17T10:10:00',
            'likes': 15,
            'retweets': 7,
            'replies': 3,
            'hashtags': ['#banknifty'],
            'mentions': ['@trader1']
        }
    ]


def test_data_cleaner(config, sample_tweets):
    """Test data cleaning"""
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(sample_tweets)
    
    assert len(cleaned) > 0
    assert all('content_clean' in t for t in cleaned)
    assert all('engagement_score' in t for t in cleaned)


def test_deduplicator(config, sample_tweets):
    """Test deduplication"""
    deduplicator = Deduplicator(config)
    unique = deduplicator.deduplicate(sample_tweets)
    
    # Should remove 1 duplicate
    assert len(unique) == 2


def test_hash_content(config):
    """Test content hashing"""
    deduplicator = Deduplicator(config)
    
    hash1 = deduplicator._hash_content("Test content")
    hash2 = deduplicator._hash_content("Test content")
    hash3 = deduplicator._hash_content("Different content")
    
    assert hash1 == hash2
    assert hash1 != hash3
