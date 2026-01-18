import pytest
from src.processors.data_cleaner import DataCleaner
from src.processors.deduplicator import Deduplicator
import yaml


@pytest.fixture
def config():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['scraping']['mode'] = 'demo'
    cfg['scraping']['target_tweets'] = 25
    cfg['scraping']['time_window_hours'] = 24 * 365 * 5  # keep fixtures valid
    return cfg


@pytest.fixture
def sample_tweets():
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
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(sample_tweets)
    
    assert len(cleaned) > 0
    assert all('content_clean' in t for t in cleaned)
    assert all('engagement_score' in t for t in cleaned)
    assert all('language' in t for t in cleaned)


def test_deduplicator(config, sample_tweets):
    deduplicator = Deduplicator(config)
    unique = deduplicator.deduplicate(sample_tweets)
    
    # Should remove 1 duplicate
    assert len(unique) == 2


def test_hash_content(config):
    deduplicator = Deduplicator(config)
    
    hash1 = deduplicator._hash_content("Test content")
    hash2 = deduplicator._hash_content("Test content")
    hash3 = deduplicator._hash_content("Different content")
    
    assert hash1 == hash2
    assert hash1 != hash3
