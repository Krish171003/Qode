# src/processors/data_cleaner.py
"""
Data Cleaning and Normalization Module
Handles text preprocessing, Unicode normalization, and data validation
"""

import re
import logging
from datetime import datetime, timedelta, timezone
import emoji
import unicodedata

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and normalizes tweet data for analysis"""
    
    def __init__(self, config):
        self.config = config
        self.min_length = config['processing']['min_text_length']
        self.languages = config['processing']['languages']
        self.allowed_languages = set(config['processing']['languages'])
        self.time_window_hours = config['scraping']['time_window_hours']
        
    def clean(self, tweets):
        """Main cleaning pipeline"""
        logger.info("Starting data cleaning pipeline...")
        
        cleaned = []
        stats = {
            'removed_short': 0,
            'removed_invalid': 0,
            'removed_out_of_window': 0,
            'removed_invalid_timestamp': 0,
            'cleaned': 0,
        }
        cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=self.time_window_hours)
        
        for tweet in tweets:
            try:
                tweet_timestamp = self._parse_timestamp(tweet.get('timestamp'))
                if not tweet_timestamp:
                    stats['removed_invalid_timestamp'] += 1
                    continue
                if tweet_timestamp < cutoff_time:
                    stats['removed_out_of_window'] += 1
                    continue

                # Skip if too short
                if len(tweet.get('content', '')) < self.min_length:
                    stats['removed_short'] += 1
                    continue

                # Language gate (keeps English/Hindi focus)
                language = self._detect_language(tweet.get('content', ''))
                if self.allowed_languages and language not in self.allowed_languages:
                    stats['removed_invalid'] += 1
                    continue
                
                # Clean the tweet
                cleaned_tweet = self._clean_tweet(tweet, tweet_timestamp, language)
                
                if cleaned_tweet:
                    cleaned.append(cleaned_tweet)
                    stats['cleaned'] += 1
                else:
                    stats['removed_invalid'] += 1
                    
            except Exception as e:
                logger.debug(f"Error cleaning tweet: {str(e)}")
                stats['removed_invalid'] += 1
        
        logger.info(f"Cleaning stats: {stats}")
        return cleaned
    
    def _clean_tweet(self, tweet, tweet_timestamp, language):
        """Clean individual tweet"""
        # Deep copy to avoid modifying original
        cleaned = tweet.copy()
        
        # Clean content
        content = tweet.get('content', '')
        
        # Normalize Unicode
        if self.config['processing']['normalize_unicode']:
            content = self._normalize_unicode(content)
        
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Convert emojis to text descriptions (preserves sentiment)
        content = emoji.demojize(content, delimiters=(" :", ": "))
        
        # Clean URLs but keep them marked
        content = re.sub(r'http\S+|www.\S+', '[URL]', content)
        
        # Normalize hashtags (remove # but keep text)
        # We keep original hashtags in separate field
        
        # Remove @mentions from main text (keep in mentions field)
        content = re.sub(r'@\w+', '', content)
        
        # Remove extra spaces again
        content = ' '.join(content.split())
        
        # Update content
        cleaned['content'] = content
        cleaned['content_clean'] = content.lower()  # Lowercase version
        cleaned['language'] = language
        
        # Ensure all required fields exist
        cleaned.setdefault('likes', 0)
        cleaned.setdefault('retweets', 0)
        cleaned.setdefault('replies', 0)
        cleaned.setdefault('hashtags', [])
        cleaned.setdefault('mentions', [])
        
        # Calculate engagement score
        cleaned['engagement_score'] = (
            cleaned['likes'] + 
            cleaned['retweets'] * 2 + 
            cleaned['replies'] * 1.5
        )
        
        # Normalize timestamp to ISO format for downstream processing
        cleaned['timestamp'] = tweet_timestamp.isoformat()
        cleaned['processed_at'] = datetime.now().isoformat()
        
        return cleaned if len(content) >= self.min_length else None
    
    def _normalize_unicode(self, text):
        """Normalize Unicode characters (handles Hindi, emojis, etc.)"""
        text = unicodedata.normalize('NFKC', text)
        # Strip zero-width and control characters
        text = re.sub(r'[\u200b-\u200d\uFEFF]', '', text)
        text = text.replace('\r', ' ')
        text = ' '.join(text.split())
        return text.strip()
    
    def _detect_language(self, text):
        """Simple language detection (English vs Hindi)"""
        # Count Devanagari characters
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'unknown'
        
        hindi_ratio = hindi_chars / total_chars
        
        if hindi_ratio > 0.3:
            return 'hi'
        return 'en'

    def _parse_timestamp(self, timestamp):
        """Parse tweet timestamps from Nitter or ISO formats."""
        if not timestamp:
            return None

        if isinstance(timestamp, datetime):
            return timestamp.astimezone(timezone.utc).replace(tzinfo=None) if timestamp.tzinfo else timestamp

        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp)
            except (OSError, ValueError):
                return None

        if isinstance(timestamp, str):
            cleaned = re.sub(r'[·\u2022]', '', timestamp).strip()
            cleaned = ' '.join(cleaned.split())
            formats = [
                '%b %d, %Y %I:%M %p %Z',
                '%b %d, %Y %I:%M %p',
                '%b %d, %Y',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S',
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(cleaned, fmt)
                except ValueError:
                    continue

            try:
                parsed = datetime.fromisoformat(cleaned.replace('Z', '+00:00'))
                if parsed.tzinfo:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                return parsed
            except ValueError:
                logger.debug("Unable to parse timestamp: %s", timestamp)
                return None

        return None
