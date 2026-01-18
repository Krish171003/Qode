#!/usr/bin/env python3
"""
Qode Market Intelligence - Main Orchestration Script
Author: Built for Qode Technical Assignment
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.collectors.twitter_scraper import TwitterScraper
from src.processors.data_cleaner import DataCleaner
from src.processors.deduplicator import Deduplicator
from src.processors.storage_manager import StorageManager
from src.analyzers.text_vectorizer import TextVectorizer
from src.analyzers.signal_generator import SignalGenerator
from src.visualizers.dashboard_generator import DashboardGenerator
from src.utils.performance import PerformanceMonitor
import yaml


def load_config():
    """Load configuration from yaml file"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qode Market Intelligence System"
    )
    parser.add_argument(
        '--target',
        type=int,
        default=None,
        help='Target number of tweets (default: from config)'
    )
    parser.add_argument(
        '--browser',
        choices=['chrome', 'firefox'],
        default=None,
        help='Browser to use'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=None,
        help='Time window in hours'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    parser.add_argument(
        '--mode',
        choices=['snscrape', 'nitter', 'demo'],
        default=None,
        help='Scraping mode: snscrape (default), nitter (selenium), or demo (synthetic)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with synthetic data (no scraping required)'
    )
    
    return parser.parse_args()


def main():
    """Main execution flow"""
    
    args = parse_arguments()
    config = load_config()
    
    # Override config with command line args
    if args.target:
        config['scraping']['target_tweets'] = args.target
    if args.browser:
        config['scraping']['browser'] = args.browser
    if args.hours:
        config['scraping']['time_window_hours'] = args.hours
    if args.headless:
        config['scraping']['headless'] = True
    if args.mode:
        config['scraping']['mode'] = args.mode
    if args.demo:
        config['scraping']['mode'] = 'demo'
    if args.debug:
        config['logging']['level'] = 'DEBUG'
    
    # Setup logging
    logger = setup_logger(config)
    logger.info("=" * 60)
    logger.info("Qode Market Intelligence System Starting...")
    logger.info("=" * 60)
    
    if args.demo:
        logger.warning("\n[WARN] RUNNING IN DEMO MODE - Using synthetic data only\n")
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()
    
    try:
        # Step 1: Data Collection
        logger.info("\n[STEP 1/5] Initializing Twitter Scraper...")
        scraper = TwitterScraper(config)
        
        logger.info(f"Target: {config['scraping']['target_tweets']} tweets")
        logger.info(f"Hashtags: {', '.join(config['scraping']['hashtags'])}")
        logger.info(f"Scrape mode: {config['scraping'].get('mode', 'snscrape')}")
        if config['scraping'].get('mode') == 'nitter':
            logger.info(f"Browser: {config['scraping']['browser']}")
        
        raw_tweets = scraper.scrape_tweets()
        logger.info(f"[ok] Collected {len(raw_tweets)} raw tweets")

        if len(raw_tweets) == 0:
            logger.warning("No tweets collected. Exiting...")
            logger.warning("\nTip: Try running with --demo flag to see the full pipeline:")
            logger.warning("  python main.py --demo")
            return
        
        storage = StorageManager(config)
        raw_format = 'json' if args.demo else config['storage']['format']
        raw_snapshot = storage.save(raw_tweets, raw_format, stage='raw')
        logger.info(f"[ok] Raw snapshot saved to: {raw_snapshot}")
        
        # Step 2: Data Cleaning
        logger.info("\n[STEP 2/5] Cleaning and Processing Data...")
        cleaner = DataCleaner(config)
        cleaned_tweets = cleaner.clean(raw_tweets)
        logger.info(f"[ok] Cleaned {len(cleaned_tweets)} tweets")
        
        # Step 3: Deduplication
        logger.info("\n[STEP 3/5] Removing Duplicates...")
        deduplicator = Deduplicator(config)
        unique_tweets = deduplicator.deduplicate(cleaned_tweets)
        logger.info(f"[ok] {len(unique_tweets)} unique tweets remaining")
        
        # Step 4: Storage
        logger.info("\n[STEP 4/5] Saving to Storage...")
        # Save as CSV for compatibility
        format_type = 'csv' if args.demo else config['storage']['format']
        data_path = storage.save(unique_tweets, format_type)
        logger.info(f"[ok] Data saved to: {data_path}")
        
        # Step 5: Analysis & Signals
        logger.info("\n[STEP 5/5] Generating Trading Signals...")
        
        # Vectorization
        vectorizer = TextVectorizer(config)
        features = vectorizer.transform(unique_tweets)
        logger.info(f"[ok] Generated {features.shape[1]} features")
        
        # Signal Generation
        signal_gen = SignalGenerator(config)
        signals = signal_gen.generate_signals(unique_tweets, features)
        logger.info(f"[ok] Generated {len(signals)} trading signals")
        
        # Save signals
        signals_path = storage.save_signals(signals)
        logger.info(f"[ok] Signals saved to: {signals_path}")
        
        # Step 6: Visualization
        logger.info("\n[BONUS] Generating Dashboard...")
        dashboard_gen = DashboardGenerator(config)
        dashboard_path = dashboard_gen.create_dashboard(
            unique_tweets, 
            signals
        )
        logger.info(f"[ok] Dashboard created: {dashboard_path}")
        
        # Performance Summary
        perf_monitor.stop()
        stats = perf_monitor.get_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Runtime: {stats['elapsed_time']:.2f} seconds")
        logger.info(f"Peak Memory: {stats['peak_memory_mb']:.2f} MB")
        logger.info(f"Tweets Processed: {len(unique_tweets)}")
        logger.info(f"Processing Rate: {len(unique_tweets)/stats['elapsed_time']:.2f} tweets/sec")
        logger.info(f"Trading Signals: {len(signals)}")
        logger.info("=" * 60)
        logger.info("[ok] All tasks completed successfully!")
        logger.info(f"[ok] Check outputs in: {config['storage']['output_dir']}/")
        logger.info(f"[ok] Open dashboard: {dashboard_path}")
        logger.info("=" * 60)
        
        if args.demo:
            logger.warning("\n[WARN] Demo mode used synthetic data. Switch mode to snscrape or nitter for real tweets.")
        
    except KeyboardInterrupt:
        logger.warning("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if 'scraper' in locals():
            scraper.close()


if __name__ == "__main__":
    main()
