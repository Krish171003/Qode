# src/processors/data_cleaner.py
"""
Data Cleaning and Normalization Module
Handles text preprocessing, Unicode normalization, and data validation
"""

import re
import logging
from datetime import datetime
import emoji
import unicodedata

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and normalizes tweet data for analysis"""
    
    def __init__(self, config):
        self.config = config
        self.min_length = config['processing']['min_text_length']
        self.languages = config['processing']['languages']
        
    def clean(self, tweets):
        """Main cleaning pipeline"""
        logger.info("Starting data cleaning pipeline...")
        
        cleaned = []
        stats = {'removed_short': 0, 'removed_invalid': 0, 'cleaned': 0}
        
        for tweet in tweets:
            try:
                # Skip if too short
                if len(tweet.get('content', '')) < self.min_length:
                    stats['removed_short'] += 1
                    continue
                
                # Clean the tweet
                cleaned_tweet = self._clean_tweet(tweet)
                
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
    
    def _clean_tweet(self, tweet):
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
        
        # Add processing timestamp
        cleaned['processed_at'] = datetime.now().isoformat()
        
        return cleaned if len(content) >= self.min_length else None
    
    def _normalize_unicode(self, text):
        """Normalize Unicode characters (handles Hindi, emojis, etc.)"""
        # NFD normalization then remove combining characters
        text = unicodedata.normalize('NFD', text)
        # Keep only allowed characters (preserve Hindi/Devanagari)
        return text
    
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
