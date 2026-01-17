"""
Twitter/X Scraper using Selenium
Implements anti-detection and rate limiting strategies
"""

import time
import random
import json
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class TwitterScraper:
    """Scrapes Twitter/X for market intelligence without using paid APIs"""
    
    def __init__(self, config):
        self.config = config
        self.driver = None
        self.tweets_collected = []
        self.seen_ids = set()  # For quick duplicate checking
        
        # Initialize browser
        self._setup_driver()
    
    def _setup_driver(self):
        """Initialize Selenium WebDriver with anti-detection measures"""
        browser_type = self.config['scraping']['browser']
        
        logger.info(f"Setting up {browser_type} driver...")
        
        if browser_type == 'chrome':
            options = ChromeOptions()
            
            if self.config['scraping']['headless']:
                options.add_argument('--headless=new')
            
            # Anti-detection flags
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Performance & stealth
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            
            # Random user agent
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
            options.add_argument(f'user-agent={random.choice(user_agents)}')
            
            self.driver = webdriver.Chrome(options=options)
            
        else:  # firefox
            options = FirefoxOptions()
            
            if self.config['scraping']['headless']:
                options.add_argument('--headless')
            
            options.set_preference("general.useragent.override", 
                                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0")
            
            self.driver = webdriver.Firefox(options=options)
        
        # Execute stealth script
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            '''
        })
        
        logger.info("✓ Driver initialized successfully")
    
    def scrape_tweets(self):
        """Main scraping orchestration method"""
        target_count = self.config['scraping']['target_tweets']
        hashtags = self.config['scraping']['hashtags']
        
        logger.info(f"Starting scrape for {target_count} tweets...")
        
        for hashtag in hashtags:
            if len(self.tweets_collected) >= target_count:
                break
            
            logger.info(f"\nScraping hashtag: {hashtag}")
            self._scrape_hashtag(hashtag, target_count)
            
            # Random delay between hashtags
            if len(self.tweets_collected) < target_count:
                delay = random.uniform(5, 10)
                logger.debug(f"Waiting {delay:.1f}s before next hashtag...")
                time.sleep(delay)
        
        logger.info(f"\n✓ Scraping complete: {len(self.tweets_collected)} tweets")
        return self.tweets_collected
    
    def _scrape_hashtag(self, hashtag, target_count):
        """Scrape tweets for a specific hashtag"""
        # Clean hashtag
        clean_tag = hashtag.replace('#', '')
        
        # Navigate to hashtag search
        url = f"https://x.com/search?q=%23{clean_tag}&src=typed_query&f=live"
        logger.debug(f"Navigating to: {url}")
        
        try:
            self.driver.get(url)
            time.sleep(self.config['scraping']['delays']['page_load'])
            
            # Scroll and collect tweets
            scroll_attempts = 0
            no_new_tweets_count = 0
            max_no_new = 5
            
            while len(self.tweets_collected) < target_count and no_new_tweets_count < max_no_new:
                before_count = len(self.tweets_collected)
                
                # Extract tweets from current view
                self._extract_tweets_from_page()
                
                after_count = len(self.tweets_collected)
                
                if after_count == before_count:
                    no_new_tweets_count += 1
                    logger.debug(f"No new tweets found (attempt {no_new_tweets_count}/{max_no_new})")
                else:
                    no_new_tweets_count = 0
                    logger.info(f"Progress: {len(self.tweets_collected)}/{target_count} tweets")
                
                # Scroll down with human-like behavior
                self._human_scroll()
                
                scroll_attempts += 1
                
                # Safety limit
                if scroll_attempts > 50:
                    logger.warning("Max scroll attempts reached")
                    break
            
        except Exception as e:
            logger.error(f"Error scraping {hashtag}: {str(e)}")
    
    def _extract_tweets_from_page(self):
        """Extract tweet data from current page view"""
        try:
            # Wait for tweets to load
            time.sleep(2)
            
            # Get page source and parse
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            
            # Find tweet articles
            articles = soup.find_all('article', {'data-testid': 'tweet'})
            
            for article in articles:
                try:
                    tweet_data = self._parse_tweet_element(article)
                    if tweet_data and tweet_data['id'] not in self.seen_ids:
                        self.tweets_collected.append(tweet_data)
                        self.seen_ids.add(tweet_data['id'])
                except Exception as e:
                    logger.debug(f"Error parsing tweet: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Error extracting tweets: {str(e)}")
    
    def _parse_tweet_element(self, article):
        """Parse individual tweet element"""
        try:
            # Extract text content
            text_elem = article.find('div', {'data-testid': 'tweetText'})
            if not text_elem:
                return None
            
            content = text_elem.get_text(strip=True)
            
            # Generate a simple ID from content hash
            tweet_id = str(hash(content))
            
            # Extract username
            username = "Unknown"
            user_elem = article.find('div', {'data-testid': 'User-Name'})
            if user_elem:
                user_links = user_elem.find_all('a')
                if user_links:
                    username = user_links[0].get('href', '/unknown').split('/')[-1]
            
            # Extract engagement metrics
            likes = self._extract_metric(article, 'like')
            retweets = self._extract_metric(article, 'retweet')
            replies = self._extract_metric(article, 'reply')
            
            # Extract hashtags
            hashtags = [tag.get_text() for tag in text_elem.find_all('a') 
                       if tag.get_text().startswith('#')]
            
            # Extract mentions
            mentions = [tag.get_text() for tag in text_elem.find_all('a') 
                       if tag.get_text().startswith('@')]
            
            # Timestamp (current time as approximation)
            timestamp = datetime.now().isoformat()
            
            return {
                'id': tweet_id,
                'username': username,
                'content': content,
                'timestamp': timestamp,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'hashtags': hashtags,
                'mentions': mentions,
                'url': f"https://x.com/{username}/status/{tweet_id}"
            }
            
        except Exception as e:
            logger.debug(f"Parse error: {str(e)}")
            return None
    
    def _extract_metric(self, article, metric_type):
        """Extract engagement metric (likes, retweets, replies)"""
        try:
            metric_elem = article.find('button', {'data-testid': f'{metric_type}'})
            if metric_elem:
                aria_label = metric_elem.get('aria-label', '')
                # Extract number from aria-label
                numbers = ''.join(filter(str.isdigit, aria_label))
                return int(numbers) if numbers else 0
            return 0
        except:
            return 0
    
    def _human_scroll(self):
        """Simulate human-like scrolling behavior"""
        # Random scroll amount
        scroll_amount = random.randint(300, 700)
        
        self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        
        # Random delay
        delay = random.uniform(
            self.config['scraping']['delays']['min_scroll'],
            self.config['scraping']['delays']['max_scroll']
        )
        time.sleep(delay)
        
        # Occasionally scroll back up (human behavior)
        if random.random() < 0.1:
            back_scroll = random.randint(50, 150)
            self.driver.execute_script(f"window.scrollBy(0, -{back_scroll});")
            time.sleep(0.5)
    
    def close(self):
        """Cleanup resources"""
        if self.driver:
            logger.info("Closing browser...")
            self.driver.quit()