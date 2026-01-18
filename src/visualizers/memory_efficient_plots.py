import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


class MemoryEfficientPlotter:
    """Creates visualizations with memory constraints"""
    
    def __init__(self, config):
        self.config = config
        self.sample_size = config['visualization']['sample_size']
        self.output_dir = Path('output/plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_sentiment_timeline(self, df, output_name='sentiment_timeline.png'):
        """Plot sentiment over time"""
        logger.info("Creating sentiment timeline plot...")
        
        # Sample if needed
        df_plot = self._sample_data(df)
        
        # Ensure timestamp column
        if 'timestamp' in df_plot.columns:
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
            df_plot = df_plot.sort_values('timestamp')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create time-based aggregation
        df_plot.set_index('timestamp', inplace=True)
        
        # Resample to hourly if we have enough data
        if len(df_plot) > 100:
            hourly = df_plot.resample('1H').agg({
                'engagement_score': 'mean',
                'content': 'count'
            }).rename(columns={'content': 'tweet_count'})
            
            ax.plot(hourly.index, hourly['engagement_score'], 
                   marker='o', linewidth=2, markersize=4, label='Avg Engagement')
            ax.set_ylabel('Average Engagement', fontsize=12)
            
            # Twin axis for tweet count
            ax2 = ax.twinx()
            ax2.bar(hourly.index, hourly['tweet_count'], 
                   alpha=0.3, color='green', label='Tweet Volume')
            ax2.set_ylabel('Tweet Count', fontsize=12)
            
        else:
            ax.scatter(df_plot.index, df_plot['engagement_score'], alpha=0.6)
            ax.set_ylabel('Engagement Score', fontsize=12)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title('Market Sentiment Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        if 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[ok] Saved to {output_path}")
        return output_path
    
    def plot_hashtag_frequency(self, df, output_name='hashtag_frequency.png', top_n=15):
        """Plot most common hashtags"""
        logger.info("Creating hashtag frequency plot...")
        
        # Collect all hashtags
        all_hashtags = []
        for hashtags in df['hashtags']:
            if isinstance(hashtags, list):
                all_hashtags.extend(hashtags)
        
        # Count frequencies
        from collections import Counter
        hashtag_counts = Counter(all_hashtags)
        top_hashtags = hashtag_counts.most_common(top_n)
        
        if not top_hashtags:
            logger.warning("No hashtags found")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        tags, counts = zip(*top_hashtags)
        
        # Horizontal bar chart
        y_pos = np.arange(len(tags))
        bars = ax.barh(y_pos, counts, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(tags))))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tags, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title('Top Hashtags in Market Discussions', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[ok] Saved to {output_path}")
        return output_path
    
    def plot_engagement_heatmap(self, df, output_name='engagement_heatmap.png'):
        """Plot engagement patterns by hour and day"""
        logger.info("Creating engagement heatmap...")
        
        df = self._sample_data(df)
        
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        
        # Create pivot table
        pivot = df.pivot_table(
            values='engagement_score',
            index='day',
            columns='hour',
            aggfunc='mean'
        )
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(pivot, cmap='YlOrRd', annot=False, fmt='.0f', 
                   cbar_kws={'label': 'Avg Engagement'}, ax=ax)
        
        ax.set_title('Engagement Patterns by Day and Hour', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[ok] Saved to {output_path}")
        return output_path
    
    def plot_signal_strength(self, signals_df, output_name='signal_strength.png'):
        """Plot trading signal strengths"""
        logger.info("Creating signal strength plot...")
        
        if signals_df.empty:
            logger.warning("No signals to plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Signal strength over time
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        signals_df = signals_df.sort_values('timestamp')
        
        colors = {'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'gray'}
        
        for direction, color in colors.items():
            mask = signals_df['direction'] == direction
            subset = signals_df[mask]
            if len(subset) > 0:
                ax1.scatter(subset['timestamp'], subset['signal_strength'],
                          c=color, label=direction, alpha=0.6, s=100)
        
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Signal Strength', fontsize=12)
        ax1.set_title('Trading Signals Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Signal distribution by index
        index_counts = signals_df['index'].value_counts()
        
        ax2.bar(range(len(index_counts)), index_counts.values,
               color=plt.cm.Set3(np.linspace(0, 1, len(index_counts))))
        ax2.set_xticks(range(len(index_counts)))
        ax2.set_xticklabels(index_counts.index, rotation=45, ha='right')
        ax2.set_ylabel('Signal Count', fontsize=12)
        ax2.set_title('Signals by Index', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[ok] Saved to {output_path}")
        return output_path
    
    def _sample_data(self, df):
        """Sample data if it exceeds memory limit"""
        if len(df) <= self.sample_size:
            return df.copy()
        
        logger.debug(f"Sampling {self.sample_size} from {len(df)} records")
        return df.sample(n=self.sample_size, random_state=42)
