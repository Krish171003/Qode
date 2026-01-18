import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
import re
from src.analyzers.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class TextVectorizer:
    
    def __init__(self, config):
        self.config = config
        
        # Initialize TF-IDF vectorizer
        ngram_range = tuple(config['analysis']['vectorization']['ngram_range'])
        max_features = config['analysis']['vectorization']['max_features']
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.sentiment = SentimentAnalyzer(config) if config['analysis']['sentiment'].get('enabled') else None
        
    def transform(self, tweets):
        logger.info("Generating text features...")
        
        # Extract text content
        texts = [t.get('content_clean', t.get('content', '')) for t in tweets]
        
        # TF-IDF features
        logger.debug("Computing TF-IDF vectors...")
        tfidf_features = self.tfidf.fit_transform(texts)
        
        # Custom features
        logger.debug("Extracting custom features...")
        custom_features = self._extract_custom_features(tweets)
        
        # Combine features
        # Convert sparse to dense for TF-IDF
        tfidf_dense = tfidf_features.toarray()
        
        # Combine
        combined = np.hstack([tfidf_dense, custom_features])
        
        # Store feature names
        self.feature_names = (
            list(self.tfidf.get_feature_names_out()) + 
            self._get_custom_feature_names()
        )
        
        logger.info(f"[ok] Generated {combined.shape[1]} features")
        
        return combined
    
    def _extract_custom_features(self, tweets):
        features = []
        
        for tweet in tweets:
            content = tweet.get('content', '')
            
            # Feature engineering
            feat = {
                # Engagement features
                'log_likes': np.log1p(tweet.get('likes', 0)),
                'log_retweets': np.log1p(tweet.get('retweets', 0)),
                'log_replies': np.log1p(tweet.get('replies', 0)),
                'engagement_score': tweet.get('engagement_score', 0),
                
                # Content features
                'text_length': len(content),
                'word_count': len(content.split()),
                'hashtag_count': len(tweet.get('hashtags', [])),
                'mention_count': len(tweet.get('mentions', [])),
                'url_count': content.count('[URL]'),
                
                # Market-specific features
                'has_nifty': int('nifty' in content.lower()),
                'has_sensex': int('sensex' in content.lower()),
                'has_banknifty': int('banknifty' in content.lower() or 'bank nifty' in content.lower()),
                
                # Action keywords
                'has_buy': int(bool(re.search(r'\b(buy|long|bullish)\b', content.lower()))),
                'has_sell': int(bool(re.search(r'\b(sell|short|bearish)\b', content.lower()))),
                
                # Numbers (targets/levels)
                'number_count': len(re.findall(r'\d+', content)),
                
                # Urgency indicators
                'has_urgency': int(bool(re.search(r'\b(now|urgent|quick|fast)\b', content.lower()))),
                'exclamation_count': content.count('!'),
                
                # Question vs statement
                'is_question': int('?' in content),
                
                # Time references
                'has_time_ref': int(bool(re.search(r'\b(today|tomorrow|intraday)\b', content.lower()))),
                'lex_sentiment': self.sentiment.analyze(content) if self.sentiment else 0.0,
            }
            
            features.append(list(feat.values()))
        
        # Convert to numpy array
        feature_array = np.array(features, dtype=np.float32)
        
        # Scale features
        feature_array = self.scaler.fit_transform(feature_array)
        
        return feature_array
    
    def _get_custom_feature_names(self):
        return [
            'log_likes', 'log_retweets', 'log_replies', 'engagement_score',
            'text_length', 'word_count', 'hashtag_count', 'mention_count', 'url_count',
            'has_nifty', 'has_sensex', 'has_banknifty',
            'has_buy', 'has_sell', 'number_count',
            'has_urgency', 'exclamation_count', 'is_question', 'has_time_ref', 'lex_sentiment'
        ]
    
    def get_top_features(self, n=20):
        # Get TF-IDF feature importances
        tfidf_scores = self.tfidf.idf_
        tfidf_names = self.tfidf.get_feature_names_out()
        
        # Sort by importance
        top_indices = np.argsort(tfidf_scores)[-n:]
        top_features = [(tfidf_names[i], tfidf_scores[i]) for i in top_indices]
        
        return sorted(top_features, key=lambda x: x[1], reverse=True)