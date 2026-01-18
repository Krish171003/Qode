import pytest
from src.collectors.twitter_scraper import TwitterScraper
import yaml


@pytest.fixture
def config():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    # Force demo mode for tests to avoid browser/network dependencies
    cfg['scraping']['mode'] = 'demo'
    cfg['scraping']['target_tweets'] = 25
    return cfg


@pytest.fixture
def scraper(config):
    return TwitterScraper(config)


def test_scraper_initialization(scraper):
    assert scraper is not None
    assert scraper.tweets_collected == []
    # In demo mode driver should not be created
    assert scraper.driver is None


def test_parse_tweet_element():
    # This would need a mock HTML element
    pass


def test_human_scroll(scraper):
    scraper._human_scroll()  # Should no-op safely without driver
    assert True


def test_demo_data_generation(scraper):
    tweets = scraper._fallback_demo_data(scraper.config['scraping']['target_tweets'])
    assert len(tweets) == scraper.config['scraping']['target_tweets']
    assert all('content' in t for t in tweets)
