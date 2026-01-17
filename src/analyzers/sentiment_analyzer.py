# src/analyzers/sentiment_analyzer.py
"""
Sentiment Analysis Module
Provides enhanced sentiment scoring
"""

import logging
import re
import numpy as np

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Advanced sentiment analysis for market tweets"""
    
    def __init__(self, config):
        self.config = config
        
        # Market-specific lexicons
        self.bullish_terms = [
            'buy', 'long', 'bullish', 'call', 'up', 'gain', 'profit', 'rise',
            'surge', 'rally', 'breakout', 'support', 'bounce', 'uptrend',
            'accumulate', 'target', 'green'
        ]
        
        self.bearish_terms = [
            'sell', 'short', 'bearish', 'put', 'down', 'loss', 'fall', 'drop',
            'crash', 'decline', 'breakdown', 'resistance', 'dump', 'downtrend',
            'exit', 'stoploss', 'red'
        ]
        
    def analyze(self, text):
        """Analyze sentiment of text"""
        text_lower = text.lower()
        
        # Count bullish/bearish terms
        bullish_score = sum(1 for term in self.bullish_terms if term in text_lower)
        bearish_score = sum(1 for term in self.bearish_terms if term in text_lower)
        
        # Normalize
        total = bullish_score + bearish_score
        if total == 0:
            return 0.0
        
        sentiment = (bullish_score - bearish_score) / total
        
        return sentiment
    
    def batch_analyze(self, texts):
        """Analyze sentiment for multiple texts"""
        return [self.analyze(text) for text in texts]