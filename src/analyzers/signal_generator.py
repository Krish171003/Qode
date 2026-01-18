
# src/analyzers/signal_generator.py
"""
Trading Signal Generation Module
Converts text features into actionable trading signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals from tweet features"""
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config['analysis']['signals']['confidence_threshold']
        self.lookback_minutes = config['analysis']['signals']['lookback_minutes']
        
    def generate_signals(self, tweets, features):
        """Main signal generation pipeline"""
        logger.info("Generating trading signals...")
        
        signals = []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(tweets)
        df['features'] = list(features)
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Generate signals by time window
        signals = self._generate_windowed_signals(df)
        
        logger.info(f"[ok] Generated {len(signals)} signals")
        
        return signals
    
    def _generate_windowed_signals(self, df):
        """Generate signals using sliding time windows"""
        signals = []
        
        # Group by time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{self.lookback_minutes}min')
        
        for window, group in df.groupby('time_window'):
            if len(group) < 3:  # Need minimum tweets for signal
                continue
            
            signal = self._analyze_window(group, window)
            
            if signal and signal['confidence'] >= self.confidence_threshold:
                signals.append(signal)
        
        return signals
    
    def _analyze_window(self, group, window_time):
        """Analyze a time window to generate signal"""
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment(group)
        
        # Calculate momentum score
        momentum_score = self._calculate_momentum(group)
        
        # Detect dominant index
        dominant_index = self._detect_dominant_index(group)
        
        # Calculate signal strength
        signal_strength = (sentiment_score + momentum_score) / 2
        
        # Determine direction
        if signal_strength > 0.3:
            direction = 'BULLISH'
        elif signal_strength < -0.3:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Calculate confidence
        confidence = min(abs(signal_strength), 1.0)
        
        # Average engagement
        avg_engagement = group['engagement_score'].mean()
        
        return {
            'timestamp': window_time,
            'index': dominant_index,
            'direction': direction,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'momentum_score': momentum_score,
            'tweet_count': len(group),
            'avg_engagement': avg_engagement,
            'top_hashtags': self._get_top_hashtags(group, n=3)
        }
    
    def _calculate_sentiment(self, group):
        """Calculate aggregate sentiment from tweets"""
        scores = []
        
        for _, row in group.iterrows():
            content = row.get('content', '').lower()
            
            # Simple sentiment scoring
            bullish_words = ['buy', 'long', 'bullish', 'up', 'gain', 'profit', 'rise', 'surge']
            bearish_words = ['sell', 'short', 'bearish', 'down', 'loss', 'fall', 'drop', 'crash']
            
            bullish_count = sum(1 for word in bullish_words if word in content)
            bearish_count = sum(1 for word in bearish_words if word in content)
            
            # Weight by engagement
            weight = np.log1p(row.get('engagement_score', 1))
            
            tweet_sentiment = (bullish_count - bearish_count) * weight
            scores.append(tweet_sentiment)
        
        # Aggregate
        if len(scores) == 0:
            return 0.0
        
        return np.tanh(np.mean(scores))  # Normalize to [-1, 1]
    
    def _calculate_momentum(self, group):
        """Calculate momentum from engagement patterns"""
        
        # Sort by time
        group = group.sort_values('timestamp')
        
        # Split into first and second half
        mid = len(group) // 2
        first_half = group.iloc[:mid]
        second_half = group.iloc[mid:]
        
        # Compare engagement
        if len(first_half) == 0 or len(second_half) == 0:
            return 0.0
        
        first_engagement = first_half['engagement_score'].mean()
        second_engagement = second_half['engagement_score'].mean()
        
        # Momentum as rate of change
        if first_engagement == 0:
            return 0.0
        
        momentum = (second_engagement - first_engagement) / (first_engagement + 1)
        
        return np.tanh(momentum)  # Normalize
    
    def _detect_dominant_index(self, group):
        """Detect which index is most mentioned"""
        index_counts = defaultdict(int)
        
        for _, row in group.iterrows():
            content = row.get('content', '').lower()
            hashtags = ' '.join(row.get('hashtags', [])).lower()
            combined = content + ' ' + hashtags
            
            if 'nifty' in combined and 'bank' in combined:
                index_counts['BANKNIFTY'] += 1
            elif 'nifty' in combined:
                index_counts['NIFTY50'] += 1
            elif 'sensex' in combined:
                index_counts['SENSEX'] += 1
            elif 'bank' in combined:
                index_counts['BANKNIFTY'] += 1
        
        if not index_counts:
            return 'GENERAL'
        
        return max(index_counts, key=index_counts.get)
    
    def _get_top_hashtags(self, group, n=3):
        """Get most common hashtags in window"""
        all_hashtags = []
        for hashtags in group['hashtags']:
            all_hashtags.extend(hashtags)
        
        # Count frequencies
        from collections import Counter
        hashtag_counts = Counter(all_hashtags)
        
        # Get top N
        top = [tag for tag, _ in hashtag_counts.most_common(n)]
        
        return ', '.join(top) if top else ''
