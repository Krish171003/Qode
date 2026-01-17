# src/processors/deduplicator.py
"""
Efficient Deduplication Module
Uses hash-based approach for O(1) duplicate detection
"""

import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class Deduplicator:
    """Removes duplicate tweets efficiently"""
    
    def __init__(self, config):
        self.config = config
        self.hash_cache = set()
        
    def deduplicate(self, tweets):
        """Remove duplicates using content hash"""
        logger.info("Deduplicating tweets...")
        
        unique_tweets = []
        seen_hashes = set()
        duplicate_count = 0
        
        for tweet in tweets:
            # Generate content hash
            content_hash = self._hash_content(tweet['content'])
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_tweets.append(tweet)
            else:
                duplicate_count += 1
        
        logger.info(f"Removed {duplicate_count} duplicates")
        logger.info(f"Unique tweets: {len(unique_tweets)}")
        
        return unique_tweets
    
    def _hash_content(self, content):
        """Generate SHA256 hash of content"""
        # Normalize content before hashing
        normalized = content.lower().strip()
        normalized = ' '.join(normalized.split())  # Remove extra spaces
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def find_near_duplicates(self, tweets, threshold=0.8):
        """Find near-duplicates using Jaccard similarity (optional)"""
        # This is more computationally expensive but catches similar tweets
        logger.info("Finding near-duplicates...")
        
        unique = []
        
        for i, tweet in enumerate(tweets):
            is_duplicate = False
            tokens_i = set(tweet['content'].lower().split())
            
            for j in range(len(unique)):
                tokens_j = set(unique[j]['content'].lower().split())
                
                # Jaccard similarity
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                
                if union > 0:
                    similarity = intersection / union
                    
                    if similarity >= threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(tweet)
        
        logger.info(f"Found {len(tweets) - len(unique)} near-duplicates")
        return unique
