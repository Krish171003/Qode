import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class DataValidator:
    
    @staticmethod
    def validate_tweet(tweet):
        required_fields = ['content', 'timestamp', 'username']
        
        # Check required fields
        for field in required_fields:
            if field not in tweet:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate content
        if not isinstance(tweet['content'], str) or len(tweet['content']) < 5:
            logger.warning("Invalid content")
            return False
        
        # Validate timestamp
        try:
            if isinstance(tweet['timestamp'], str):
                datetime.fromisoformat(tweet['timestamp'].replace('Z', '+00:00'))
        except:
            logger.warning("Invalid timestamp format")
            return False
        
        # Validate engagement metrics
        for metric in ['likes', 'retweets', 'replies']:
            if metric in tweet:
                if not isinstance(tweet[metric], (int, float)) or tweet[metric] < 0:
                    logger.warning(f"Invalid {metric} value")
                    tweet[metric] = 0  # Fix it
        
        return True
    
    @staticmethod
    def validate_batch(tweets):
        valid_tweets = []
        invalid_count = 0
        
        for tweet in tweets:
            if DataValidator.validate_tweet(tweet):
                valid_tweets.append(tweet)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid tweets")
        
        return valid_tweets
    
    @staticmethod
    def sanitize_text(text):
        if not isinstance(text, str):
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters except newlines
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        return text













