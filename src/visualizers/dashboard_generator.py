import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DashboardGenerator:
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
    def create_dashboard(self, tweets, signals):
        logger.info("Generating dashboard...")
        
        # Convert to DataFrames
        tweets_df = pd.DataFrame(tweets)
        signals_df = pd.DataFrame(signals)
        
        # Generate plots
        from src.visualizers.memory_efficient_plots import MemoryEfficientPlotter
        plotter = MemoryEfficientPlotter(self.config)
        
        plot1 = plotter.plot_sentiment_timeline(tweets_df)
        plot2 = plotter.plot_hashtag_frequency(tweets_df)
        plot3 = plotter.plot_engagement_heatmap(tweets_df)
        plot4 = plotter.plot_signal_strength(signals_df)
        
        # Generate statistics
        stats = self._calculate_statistics(tweets_df, signals_df)
        
        # Create HTML
        html = self._generate_html(stats, plot1, plot2, plot3, plot4)
        
        # Save dashboard
        dashboard_path = self.output_dir / 'dashboard.html'
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"[ok] Dashboard saved to {dashboard_path}")
        return dashboard_path
    
    def _calculate_statistics(self, tweets_df, signals_df):
        stats = {
            'total_tweets': len(tweets_df),
            'unique_users': tweets_df['username'].nunique() if 'username' in tweets_df.columns else 0,
            'total_engagement': tweets_df['engagement_score'].sum() if 'engagement_score' in tweets_df.columns else 0,
            'avg_engagement': tweets_df['engagement_score'].mean() if 'engagement_score' in tweets_df.columns else 0,
            'total_signals': len(signals_df),
            'bullish_signals': len(signals_df[signals_df['direction'] == 'BULLISH']) if 'direction' in signals_df.columns else 0,
            'bearish_signals': len(signals_df[signals_df['direction'] == 'BEARISH']) if 'direction' in signals_df.columns else 0,
            'neutral_signals': len(signals_df[signals_df['direction'] == 'NEUTRAL']) if 'direction' in signals_df.columns else 0,
        }
        
        return stats
    
    def _generate_html(self, stats, plot1, plot2, plot3, plot4):
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qode Market Intelligence Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .plots {{
            padding: 30px;
        }}
        .plot-section {{
            margin-bottom: 40px;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        .plot-section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        .plot-section img {{
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .footer {{
            background: #2d3748;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .signal-indicator {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 0 5px;
        }}
        .bullish {{ background: #48bb78; color: white; }}
        .bearish {{ background: #f56565; color: white; }}
        .neutral {{ background: #cbd5e0; color: #2d3748; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Qode Market Intelligence</h1>
            <p>Real-time Social Media Market Analysis</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {timestamp}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Tweets</div>
                <div class="stat-value">{stats['total_tweets']:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unique Users</div>
                <div class="stat-value">{stats['unique_users']:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Engagement</div>
                <div class="stat-value">{stats['total_engagement']:,.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Engagement</div>
                <div class="stat-value">{stats['avg_engagement']:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trading Signals</div>
                <div class="stat-value">{stats['total_signals']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Market Sentiment</div>
                <div class="stat-value">
                    <span class="signal-indicator bullish">{stats['bullish_signals']}</span>
                    <span class="signal-indicator bearish">{stats['bearish_signals']}</span>
                </div>
            </div>
        </div>
        
        <div class="plots">
            <div class="plot-section">
                <h2>📈 Sentiment Timeline</h2>
                <img src="plots/sentiment_timeline.png" alt="Sentiment Timeline">
            </div>
            
            <div class="plot-section">
                <h2>🏷️ Popular Hashtags</h2>
                <img src="plots/hashtag_frequency.png" alt="Hashtag Frequency">
            </div>
            
            <div class="plot-section">
                <h2>🔥 Engagement Heatmap</h2>
                <img src="plots/engagement_heatmap.png" alt="Engagement Heatmap">
            </div>
            
            <div class="plot-section">
                <h2>📊 Trading Signals</h2>
                <img src="plots/signal_strength.png" alt="Signal Strength">
            </div>
        </div>
        
        <div class="footer">
            <p>Built with Qode Market Intelligence System</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                Production-ready data collection and analysis for algorithmic trading
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html
