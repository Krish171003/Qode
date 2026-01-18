"""
Twitter scraper for market intelligence without paid APIs.

Primary path: `snscrape` (no auth needed, fast, resilient).
Fallback path: Nitter via Selenium when public mirrors are available.
Demo path: synthetic data for offline testing and CI.
"""

import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

try:
    import snscrape.modules.twitter as sntwitter
except Exception:  # pragma: no cover - optional dependency
    sntwitter = None

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions

logger = logging.getLogger(__name__)


class TwitterScraper:
    """Scrapes real Twitter data without using paid APIs."""

    def __init__(self, config):
        self.config = config
        self.mode = config["scraping"].get("mode", "snscrape").lower()
        self.driver = None
        self.tweets_collected = []
        self.seen_ids = set()

        self.time_window_hours = self.config["scraping"]["time_window_hours"]
        self.cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.time_window_hours)

        # Working Nitter instances (as of Jan 2025)
        self.nitter_instances = [
            "https://nitter.privacydev.net",
            "https://nitter.poast.org",
            "https://nitter.bird.trom.tf",
            "https://nitter.woodland.cafe",
            "https://nitter.1d4.us",
        ]
        self.current_instance = None

        # Only spin up Selenium if needed
        if self.mode == "nitter":
            self._setup_driver()

    def _setup_driver(self):
        """Initialize Selenium WebDriver."""
        browser_type = self.config["scraping"]["browser"]

        logger.info("Setting up %s driver for Nitter scraping...", browser_type)

        options = ChromeOptions()
        if self.config["scraping"]["headless"]:
            options.add_argument("--headless=new")

        # Performance optimizations
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-logging")
        options.add_argument("--log-level=3")

        # Anti-detection (helps on public mirrors)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        logger.info("[init] Selenium driver ready")

    def scrape_tweets(self):
        """Main scraping orchestration."""
        target_count = self.config["scraping"]["target_tweets"]
        hashtags = self.config["scraping"]["hashtags"]

        logger.info("Starting scrape for %s tweets (mode=%s)", target_count, self.mode)

        if self.mode == "demo":
            logger.warning("Demo mode selected; generating synthetic tweets only.")
            return self._fallback_demo_data(target_count)

        # Primary: snscrape (fast, no browser required)
        if self.mode == "snscrape":
            tweets = self._scrape_with_snscrape(target_count, hashtags)
            if tweets:
                logger.info("Collected %s tweets via snscrape", len(tweets))
                return tweets
            logger.warning("snscrape returned no tweets; falling back to Nitter.")

        # Fallback: Nitter via Selenium
        tweets = self._scrape_with_nitter(target_count, hashtags)
        if tweets:
            return tweets

        # Final fallback: synthetic data so pipeline remains testable
        logger.warning("All scraping strategies failed; generating demo data.")
        return self._fallback_demo_data(target_count)

    # --- snscrape path ----------------------------------------------------
    def _scrape_with_snscrape(self, target_count, hashtags):
        """Scrape using snscrape with concurrent hashtag workers."""
        if not sntwitter:
            logger.error("snscrape is not installed; install it or switch mode to nitter.")
            return []

        per_tag_quota = max(target_count // max(len(hashtags), 1), 150)
        workers = min(
            len(hashtags), self.config["performance"].get("concurrent_workers", 4)
        )
        collected = []

        def _scrape_tag(tag):
            results = []
            since_date = (datetime.now(timezone.utc) - timedelta(hours=self.time_window_hours)).date()
            until_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()
            query = f"{tag} since:{since_date} until:{until_date}"
            logger.debug("snscrape query for %s: %s", tag, query)

            try:
                for item in sntwitter.TwitterSearchScraper(query).get_items():
                    item_date = item.date
                    if item_date.tzinfo is None:
                        item_date = item_date.replace(tzinfo=timezone.utc)
                    if item_date < self.cutoff_time:
                        break

                    results.append(
                        {
                            "id": str(item.id),
                            "username": item.user.username if item.user else "unknown",
                            "content": item.rawContent or item.content,
                            "timestamp": item_date.isoformat(),
                            "likes": item.likeCount or 0,
                            "retweets": item.retweetCount or 0,
                            "replies": item.replyCount or 0,
                            "hashtags": [f"#{h}" for h in (item.hashtags or [])],
                            "mentions": [
                                f"@{m.username}" for m in (item.mentionedUsers or [])
                            ],
                            "url": item.url,
                        }
                    )

                    if len(results) >= per_tag_quota:
                        break

            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("snscrape failed for %s: %s", tag, exc)

            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scrape_tag, tag): tag for tag in hashtags}
            for future in as_completed(futures):
                for tweet in future.result():
                    if tweet["id"] in self.seen_ids:
                        continue
                    self.seen_ids.add(tweet["id"])
                    collected.append(tweet)
                    if len(collected) >= target_count:
                        logger.debug("Reached target via snscrape.")
                        return collected[:target_count]

        return collected[:target_count]

    # --- Nitter (Selenium) path ------------------------------------------
    def _scrape_with_nitter(self, target_count, hashtags):
        """Scrape using public Nitter mirrors with Selenium."""
        if self.mode != "nitter" and not self.driver:
            try:
                self._setup_driver()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.error("Failed to start Selenium: %s", exc)
                return []

        if not self.driver:
            return []

        logger.info("Using Nitter fallback (headless=%s)", self.config["scraping"]["headless"])
        self.current_instance = self._find_working_instance()

        if not self.current_instance:
            logger.error("No working Nitter instances found.")
            return []

        for hashtag in hashtags:
            if len(self.tweets_collected) >= target_count:
                break

            logger.info("Scraping hashtag via Nitter: %s", hashtag)
            self._scrape_nitter_hashtag(hashtag, target_count)

            if len(self.tweets_collected) < target_count:
                time.sleep(random.uniform(2, 4))

        logger.info("Nitter scrape complete: %s tweets", len(self.tweets_collected))
        return self.tweets_collected

    def _find_working_instance(self):
        """Test Nitter instances to find a working one."""
        logger.info("Testing Nitter instances (mirrors can be unstable).")

        for instance in self.nitter_instances:
            try:
                logger.debug("Testing %s...", instance)
                self.driver.get(f"{instance}/twitter")
                time.sleep(2)

                if "Nitter" in self.driver.title or "Twitter" in self.driver.page_source:
                    logger.info("Using Nitter instance: %s", instance)
                    return instance

            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug("%s failed: %s", instance, exc)
                continue

        return None

    def _scrape_nitter_hashtag(self, hashtag, target_count):
        """Scrape tweets for a specific hashtag from Nitter."""
        clean_tag = hashtag.replace("#", "").strip()
        since_date = (datetime.now(timezone.utc) - timedelta(hours=self.time_window_hours)).strftime(
            "%Y-%m-%d"
        )
        query = f"%23{clean_tag}%20since%3A{since_date}"
        url = f"{self.current_instance}/search?f=tweets&q={query}"

        try:
            logger.debug("Navigating to %s", url)
            self.driver.get(url)
            time.sleep(3)

            scroll_attempts = 0
            max_scrolls = self.config["scraping"].get("max_scrolls", 20)
            no_new_count = 0

            while len(self.tweets_collected) < target_count and scroll_attempts < max_scrolls:
                initial_count = len(self.tweets_collected)

                self._extract_nitter_tweets()
                new_tweets = len(self.tweets_collected) - initial_count

                if new_tweets > 0:
                    logger.info("Progress: %s/%s tweets", len(self.tweets_collected), target_count)
                    no_new_count = 0
                else:
                    no_new_count += 1
                    if no_new_count >= 3:
                        logger.debug("No new tweets after multiple scrolls; moving on.")
                        break

                self._human_scroll()
                scroll_attempts += 1
                time.sleep(random.uniform(2, 3))

        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Error scraping %s: %s", hashtag, exc)

    def _extract_nitter_tweets(self):
        """Extract tweets from the current Nitter page."""
        try:
            tweet_containers = self.driver.find_elements(By.CSS_SELECTOR, ".timeline-item")
            if not tweet_containers:
                tweet_containers = self.driver.find_elements(
                    By.XPATH, '//div[contains(@class, "tweet")]'
                )

            logger.info("Found %s tweet containers on page", len(tweet_containers))

            for container in tweet_containers:
                try:
                    tweet_data = self._parse_nitter_tweet(container)

                    if tweet_data and tweet_data["id"] not in self.seen_ids:
                        self.tweets_collected.append(tweet_data)
                        self.seen_ids.add(tweet_data["id"])
                        logger.debug("Captured tweet: %s", tweet_data["content"][:80])

                except Exception as exc:
                    logger.debug("Error parsing tweet: %s", exc)
                    continue

        except Exception as exc:  # pragma: no cover - browser dependent
            logger.error("Error extracting tweets: %s", exc, exc_info=True)

    def _parse_nitter_tweet(self, container):
        """Parse individual Nitter tweet element."""
        try:
            content_elem = container.find_element(By.CLASS_NAME, "tweet-content")
            content = content_elem.text.strip()
            if not content or len(content) < 5:
                return None

            tweet_id = str(abs(hash(content)))

            username = "unknown"
            try:
                username_elem = container.find_element(By.CLASS_NAME, "username")
                username = username_elem.text.strip().replace("@", "")
            except Exception:
                pass

            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                time_elem = container.find_element(By.CLASS_NAME, "tweet-date")
                time_text = time_elem.get_attribute("title")
                if time_text:
                    timestamp = time_text
            except Exception:
                pass

            likes = self._extract_nitter_metric(container, "icon-heart")
            retweets = self._extract_nitter_metric(container, "icon-retweet")
            replies = self._extract_nitter_metric(container, "icon-comment")

            hashtags = re.findall(r"#\w+", content)
            mentions = re.findall(r"@\w+", content)

            tweet_url = ""
            try:
                link_elem = container.find_element(By.CLASS_NAME, "tweet-link")
                tweet_url = link_elem.get_attribute("href")
                if tweet_url and not tweet_url.startswith("http"):
                    tweet_url = self.current_instance + tweet_url
            except Exception:
                pass

            return {
                "id": tweet_id,
                "username": username,
                "content": content,
                "timestamp": timestamp,
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "hashtags": hashtags,
                "mentions": mentions,
                "url": tweet_url,
            }

        except Exception as exc:
            logger.debug("Parse error: %s", exc)
            return None

    def _extract_nitter_metric(self, container, icon_class):
        """Extract engagement metric from Nitter."""
        try:
            stats = container.find_elements(By.CLASS_NAME, "tweet-stat")

            for stat in stats:
                if icon_class in stat.get_attribute("class"):
                    num_elem = stat.find_element(By.CLASS_NAME, "icon-text")
                    num_text = num_elem.text.strip()

                    if not num_text:
                        return 0

                    if "K" in num_text.upper():
                        return int(float(num_text.upper().replace("K", "")) * 1000)
                    if "M" in num_text.upper():
                        return int(float(num_text.upper().replace("M", "")) * 1000000)
                    return int(re.sub(r"[^\d]", "", num_text))

            return 0

        except Exception:
            return 0

    def _human_scroll(self):
        """Human-like scrolling to reduce bot detection on mirrors."""
        if not self.driver:
            return
        scroll_height = random.uniform(0.6, 1.0)
        self.driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight * arguments[0]);", scroll_height
        )
        delay = random.uniform(
            self.config["scraping"]["delays"]["min_scroll"],
            self.config["scraping"]["delays"]["max_scroll"],
        )
        time.sleep(delay)

    # --- Demo data --------------------------------------------------------
    def _fallback_demo_data(self, target_count):
        """Fallback to demo data if scraping fails or demo mode selected."""
        logger.warning("=" * 60)
        logger.warning("DEMO MODE: Generating synthetic data")
        logger.warning("=" * 60)
        logger.warning("Note: As of Jan 2025, free Twitter scraping is highly restricted.")
        logger.warning("This demo mode showcases the complete data pipeline using")
        logger.warning("realistic synthetic data for reliable demonstration.")
        logger.warning("For production, integrate Twitter API or paid data providers.")
        logger.warning("=" * 60)
        logger.warning("Generating %s synthetic tweets for demo/testing.", target_count)

        tweets = []
        current_time = datetime.now(timezone.utc)

        templates = [
            "NIFTY showing bullish momentum at {price}. Target {target}. #nifty50 #stockmarket",
            "Bank Nifty strong support at {support}. Good entry point. #banknifty #intraday",
            "Sensex consolidating. Watch {price} level carefully. #sensex #trading",
        ]

        for i in range(target_count):
            hours_ago = random.uniform(0, 24)
            tweet_time = current_time - timedelta(hours=hours_ago)

            template = random.choice(templates)
            price = 24000 + random.randint(-300, 300)
            target = price + random.randint(50, 200)
            support = price - random.randint(50, 150)

            content = template.format(price=price, target=target, support=support)

            tweets.append(
                {
                    "id": str(i + 1000),
                    "username": f"trader_{random.randint(1, 100)}",
                    "content": content,
                    "timestamp": tweet_time.isoformat(),
                    "likes": random.randint(1, 100),
                    "retweets": random.randint(1, 50),
                    "replies": random.randint(1, 30),
                    "hashtags": re.findall(r"#\w+", content),
                    "mentions": [],
                    "url": f"https://twitter.com/demo/status/{i}",
                }
            )

            if (i + 1) % 500 == 0:
                logger.info("Generated %s/%s demo tweets", i + 1, target_count)

        return tweets

    def close(self):
        """Cleanup resources."""
        if self.driver:
            logger.info("Closing browser...")
            self.driver.quit()
