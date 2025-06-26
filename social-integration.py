#!/usr/bin/env python3
"""
Social Media API Integration for RL Meta-Agent
Practical implementation for Instagram, Facebook, and Twitter posting.
"""

import os
import datetime as dt
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import requests
from abc import ABC, abstractmethod

# Import the meta-agent
from astro_rl_meta_agent import AstrologyMetaAgent, PostMetrics, AstrologyPost

# Social Media API Clients
try:
    from instagram_private_api import Client as InstagramAPI
    from facebook import GraphAPI
    import tweepy
    APIS_AVAILABLE = True
except ImportError:
    APIS_AVAILABLE = False
    print("⚠️  Install social media APIs: pip install instagram-private-api facebook-sdk tweepy")


# ============== Base Social Media Client ==============
class SocialMediaClient(ABC):
    """Abstract base class for social media clients."""
    
    @abstractmethod
    def post_content(self, caption: str, image_path: Optional[str] = None, 
                    video_path: Optional[str] = None) -> Dict:
        """Post content to the platform."""
        pass
    
    @abstractmethod
    def get_post_metrics(self, post_id: str) -> PostMetrics:
        """Retrieve metrics for a specific post."""
        pass
    
    @abstractmethod
    def get_account_insights(self, days: int = 7) -> Dict:
        """Get account-level insights."""
        pass


# ============== Instagram Client ==============
class InstagramClient(SocialMediaClient):
    """Instagram API client for posting and analytics."""
    
    def __init__(self, username: str, password: str):
        """Initialize Instagram client."""
        if not APIS_AVAILABLE:
            raise ImportError("instagram-private-api not installed")
        
        self.api = InstagramAPI(username, password)
        self.user_id = self.api.authenticated_user_id
    
    def post_content(self, caption: str, image_path: Optional[str] = None,
                    video_path: Optional[str] = None) -> Dict:
        """Post to Instagram."""
        try:
            if video_path and os.path.exists(video_path):
                # Post video
                with open(video_path, 'rb') as video:
                    result = self.api.post_video(video, caption=caption)
            elif image_path and os.path.exists(image_path):
                # Post image
                with open(image_path, 'rb') as photo:
                    result = self.api.post_photo(photo, caption=caption)
            else:
                # Text-only posts not supported on Instagram
                print("Instagram requires media. Skipping text-only post.")
                return {}
            
            return {
                'post_id': result.get('media', {}).get('pk'),
                'status': 'success',
                'timestamp': dt.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Instagram posting error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_post_metrics(self, post_id: str) -> PostMetrics:
        """Get metrics for an Instagram post."""
        try:
            media_info = self.api.media_info(post_id)
            media = media_info.get('items', [{}])[0]
            
            metrics = PostMetrics(
                likes=media.get('like_count', 0),
                comments=media.get('comment_count', 0),
                views=media.get('view_count', 0),  # For videos
                saves=media.get('saved_collection_ids', []),  # Approximate
                timestamp=dt.datetime.now()
            )
            
            # Calculate engagement rate
            if metrics.views > 0:
                metrics.engagement_rate = (metrics.likes + metrics.comments) / metrics.views
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching Instagram metrics: {e}")
            return PostMetrics()
    
    def get_account_insights(self, days: int = 7) -> Dict:
        """Get Instagram account insights."""
        try:
            # Get recent media
            recent_media = self.api.user_feed(self.user_id)
            posts = recent_media.get('items', [])[:20]  # Last 20 posts
            
            total_likes = sum(p.get('like_count', 0) for p in posts)
            total_comments = sum(p.get('comment_count', 0) for p in posts)
            
            return {
                'follower_count': self.api.user_info(self.user_id).get('user', {}).get('follower_count', 0),
                'average_likes': total_likes / len(posts) if posts else 0,
                'average_comments': total_comments / len(posts) if posts else 0,
                'post_count': len(posts)
            }
            
        except Exception as e:
            print(f"Error fetching Instagram insights: {e}")
            return {}


# ============== Facebook Client ==============
class FacebookClient(SocialMediaClient):
    """Facebook API client for posting and analytics."""
    
    def __init__(self, access_token: str, page_id: Optional[str] = None):
        """Initialize Facebook client."""
        if not APIS_AVAILABLE:
            raise ImportError("facebook-sdk not installed")
        
        self.graph = GraphAPI(access_token)
        self.page_id = page_id or "me"
    
    def post_content(self, caption: str, image_path: Optional[str] = None,
                    video_path: Optional[str] = None) -> Dict:
        """Post to Facebook."""
        try:
            if video_path and os.path.exists(video_path):
                # Post video
                with open(video_path, 'rb') as video:
                    result = self.graph.put_video(
                        video, 
                        description=caption,
                        title="Astrology Update"
                    )
            elif image_path and os.path.exists(image_path):
                # Post image
                with open(image_path, 'rb') as photo:
                    result = self.graph.put_photo(
                        photo,
                        message=caption
                    )
            else:
                # Text post
                result = self.graph.put_object(
                    self.page_id,
                    "feed",
                    message=caption
                )
            
            return {
                'post_id': result.get('id'),
                'status': 'success',
                'timestamp': dt.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Facebook posting error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_post_metrics(self, post_id: str) -> PostMetrics:
        """Get metrics for a Facebook post."""
        try:
            # Get post insights
            insights = self.graph.get_connections(
                post_id,
                'insights',
                metric='post_impressions,post_engaged_users,post_reactions_by_type_total'
            )
            
            reactions = self.graph.get_connections(post_id, 'reactions')
            comments = self.graph.get_connections(post_id, 'comments')
            
            metrics = PostMetrics(
                likes=len(reactions.get('data', [])),
                comments=len(comments.get('data', [])),
                views=self._extract_metric(insights, 'post_impressions'),
                timestamp=dt.datetime.now()
            )
            
            engaged_users = self._extract_metric(insights, 'post_engaged_users')
            if metrics.views > 0:
                metrics.engagement_rate = engaged_users / metrics.views
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching Facebook metrics: {e}")
            return PostMetrics()
    
    def _extract_metric(self, insights: Dict, metric_name: str) -> int:
        """Extract specific metric from Facebook insights."""
        for metric in insights.get('data', []):
            if metric.get('name') == metric_name:
                values = metric.get('values', [{}])
                if values:
                    return values[0].get('value', 0)
        return 0
    
    def get_account_insights(self, days: int = 7) -> Dict:
        """Get Facebook page insights."""
        try:
            page_info = self.graph.get_object(self.page_id)
            insights = self.graph.get_connections(
                self.page_id,
                'insights',
                metric='page_fans,page_impressions,page_engaged_users',
                period='day',
                since=(dt.datetime.now() - dt.timedelta(days=days)).isoformat()
            )
            
            return {
                'follower_count': page_info.get('fan_count', 0),
                'page_impressions': self._extract_metric(insights, 'page_impressions'),
                'engaged_users': self._extract_metric(insights, 'page_engaged_users')
            }
            
        except Exception as e:
            print(f"Error fetching Facebook insights: {e}")
            return {}


# ============== Twitter Client ==============
class TwitterClient(SocialMediaClient):
    """Twitter API client for posting and analytics."""
    
    def __init__(self, consumer_key: str, consumer_secret: str,
                 access_token: str, access_token_secret: str):
        """Initialize Twitter client."""
        if not APIS_AVAILABLE:
            raise ImportError("tweepy not installed")
        
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        self.client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
    
    def post_content(self, caption: str, image_path: Optional[str] = None,
                    video_path: Optional[str] = None) -> Dict:
        """Post to Twitter."""
        try:
            media_ids = []
            
            if video_path and os.path.exists(video_path):
                # Upload video
                media = self.api.media_upload(video_path)
                media_ids.append(media.media_id)
            elif image_path and os.path.exists(image_path):
                # Upload image
                media = self.api.media_upload(image_path)
                media_ids.append(media.media_id)
            
            # Post tweet
            result = self.client.create_tweet(
                text=caption[:280],  # Twitter character limit
                media_ids=media_ids if media_ids else None
            )
            
            return {
                'post_id': result.data['id'],
                'status': 'success',
                'timestamp': dt.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Twitter posting error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_post_metrics(self, post_id: str) -> PostMetrics:
        """Get metrics for a Twitter post."""
        try:
            # Get tweet with metrics
            tweet = self.client.get_tweet(
                post_id,
                tweet_fields=['public_metrics', 'created_at']
            )
            
            if tweet.data:
                metrics_data = tweet.data.get('public_metrics', {})
                
                metrics = PostMetrics(
                    likes=metrics_data.get('like_count', 0),
                    comments=metrics_data.get('reply_count', 0),
                    shares=metrics_data.get('retweet_count', 0),
                    views=metrics_data.get('impression_count', 0),
                    timestamp=dt.datetime.now()
                )
                
                if metrics.views > 0:
                    total_engagement = metrics.likes + metrics.comments + metrics.shares
                    metrics.engagement_rate = total_engagement / metrics.views
                
                return metrics
            
            return PostMetrics()
            
        except Exception as e:
            print(f"Error fetching Twitter metrics: {e}")
            return PostMetrics()
    
    def get_account_insights(self, days: int = 7) -> Dict:
        """Get Twitter account insights."""
        try:
            # Get user info
            user = self.api.verify_credentials()
            
            # Get recent tweets
            tweets = self.client.get_users_tweets(
                user.id,
                max_results=100,
                tweet_fields=['public_metrics']
            )
            
            total_likes = 0
            total_retweets = 0
            
            if tweets.data:
                for tweet in tweets.data:
                    metrics = tweet.get('public_metrics', {})
                    total_likes += metrics.get('like_count', 0)
                    total_retweets += metrics.get('retweet_count', 0)
            
            return {
                'follower_count': user.followers_count,
                'average_likes': total_likes / len(tweets.data) if tweets.data else 0,
                'average_retweets': total_retweets / len(tweets.data) if tweets.data else 0
            }
            
        except Exception as e:
            print(f"Error fetching Twitter insights: {e}")
            return {}


# ============== Multi-Platform Manager ==============
class MultiPlatformManager:
    """Manages posting across multiple social media platforms."""
    
    def __init__(self, config_file: str = "social_config.json"):
        """Initialize with configuration file."""
        self.clients = {}
        self.config = self._load_config(config_file)
        self._initialize_clients()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load social media configuration."""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create template config
            template = {
                "instagram": {
                    "enabled": False,
                    "username": "your_username",
                    "password": "your_password"
                },
                "facebook": {
                    "enabled": False,
                    "access_token": "your_access_token",
                    "page_id": "your_page_id"
                },
                "twitter": {
                    "enabled": False,
                    "consumer_key": "your_consumer_key",
                    "consumer_secret": "your_consumer_secret",
                    "access_token": "your_access_token",
                    "access_token_secret": "your_access_token_secret"
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(template, f, indent=2)
            
            print(f"Created config template: {config_file}")
            print("Please fill in your API credentials.")
            return template
    
    def _initialize_clients(self):
        """Initialize enabled social media clients."""
        if not APIS_AVAILABLE:
            print("Social media APIs not available. Install with:")
            print("pip install instagram-private-api facebook-sdk tweepy")
            return
        
        # Instagram
        if self.config.get('instagram', {}).get('enabled'):
            try:
                self.clients['instagram'] = InstagramClient(
                    self.config['instagram']['username'],
                    self.config['instagram']['password']
                )
                print("✅ Instagram client initialized")
            except Exception as e:
                print(f"❌ Instagram initialization failed: {e}")
        
        # Facebook
        if self.config.get('facebook', {}).get('enabled'):
            try:
                self.clients['facebook'] = FacebookClient(
                    self.config['facebook']['access_token'],
                    self.config['facebook'].get('page_id')
                )
                print("✅ Facebook client initialized")
            except Exception as e:
                print(f"❌ Facebook initialization failed: {e}")
        
        # Twitter
        if self.config.get('twitter', {}).get('enabled'):
            try:
                self.clients['twitter'] = TwitterClient(
                    self.config['twitter']['consumer_key'],
                    self.config['twitter']['consumer_secret'],
                    self.config['twitter']['access_token'],
                    self.config['twitter']['access_token_secret']
                )
                print("✅ Twitter client initialized")
            except Exception as e:
                print(f"❌ Twitter initialization failed: {e}")
    
    def post_to_all(self, caption: str, image_path: Optional[str] = None,
                    video_path: Optional[str] = None) -> Dict[str, Dict]:
        """Post content to all enabled platforms."""
        results = {}
        
        for platform, client in self.clients.items():
            print(f"📱 Posting to {platform}...")
            result = client.post_content(caption, image_path, video_path)
            results[platform] = result
            
            # Rate limiting
            time.sleep(2)
        
        return results
    
    def collect_metrics(self, post_ids: Dict[str, str]) -> Dict[str, PostMetrics]:
        """Collect metrics from all platforms."""
        metrics = {}
        
        for platform, post_id in post_ids.items():
            if platform in self.clients and post_id:
                print(f"📊 Collecting metrics from {platform}...")
                metrics[platform] = self.clients[platform].get_post_metrics(post_id)
        
        return metrics
    
    def get_aggregated_insights(self, days: int = 7) -> Dict:
        """Get aggregated insights across all platforms."""
        insights = {
            'total_followers': 0,
            'average_engagement': 0,
            'platform_breakdown': {}
        }
        
        engagement_rates = []
        
        for platform, client in self.clients.items():
            platform_insights = client.get_account_insights(days)
            insights['platform_breakdown'][platform] = platform_insights
            
            # Aggregate followers
            insights['total_followers'] += platform_insights.get('follower_count', 0)
            
            # Calculate engagement (platform-specific)
            if platform == 'instagram':
                if platform_insights.get('average_likes'):
                    avg_engagement = platform_insights['average_likes'] / max(1, platform_insights.get('follower_count', 1))
                    engagement_rates.append(avg_engagement)
        
        if engagement_rates:
            insights['average_engagement'] = sum(engagement_rates) / len(engagement_rates)
        
        return insights


# ============== RL Integration ==============
class RLOptimizedPoster:
    """Integrates RL meta-agent with real social media posting."""
    
    def __init__(self, meta_agent: AstrologyMetaAgent, 
                 platform_manager: MultiPlatformManager):
        self.meta_agent = meta_agent
        self.platform_manager = platform_manager
        self.posting_history = []
    
    def execute_learned_strategy(self, days_ahead: int = 7):
        """Execute the learned posting strategy."""
        print(f"\n🚀 Executing learned strategy for next {days_ahead} days")
        
        # Get optimal schedule from meta-agent
        schedule = self.meta_agent.deploy_policy(real_posting=False)
        
        # Filter to next N days
        end_date = dt.date.today() + dt.timedelta(days=days_ahead)
        upcoming_posts = [
            post for post in schedule 
            if dt.datetime.strptime(post['date'], '%Y-%m-%d').date() <= end_date
        ]
        
        print(f"📅 Scheduling {len(upcoming_posts)} posts")
        
        for post in upcoming_posts:
            scheduled_time = dt.datetime.strptime(
                f"{post['date']} {post['time']}", 
                "%Y-%m-%d %H:%M"
            )
            
            # Wait until scheduled time
            wait_time = (scheduled_time - dt.datetime.now()).total_seconds()
            if wait_time > 0:
                print(f"⏰ Waiting until {scheduled_time} for {post['sun_sign']} post...")
                time.sleep(min(wait_time, 3600))  # Max 1 hour wait in demo
            
            # Post content
            results = self.platform_manager.post_to_all(
                caption=post['caption'],
                video_path=post.get('video_file')
            )
            
            # Store results
            self.posting_history.append({
                'post': post,
                'results': results,
                'timestamp': dt.datetime.now().isoformat()
            })
            
            print(f"✅ Posted: {post['sun_sign']} | {post['moon_phase']}")
    
    def collect_feedback_and_retrain(self, wait_hours: int = 24):
        """Collect real metrics and retrain the RL agent."""
        print(f"\n🔄 Collecting feedback after {wait_hours} hours...")
        
        # Wait for engagement to accumulate
        time.sleep(wait_hours * 3600)
        
        # Collect metrics for all recent posts
        real_metrics = []
        
        for entry in self.posting_history[-30:]:  # Last 30 posts
            post_ids = {
                platform: result.get('post_id')
                for platform, result in entry['results'].items()
                if result.get('status') == 'success'
            }
            
            if post_ids:
                metrics = self.platform_manager.collect_metrics(post_ids)
                
                # Aggregate metrics across platforms
                aggregated = PostMetrics()
                for platform_metrics in metrics.values():
                    aggregated.likes += platform_metrics.likes
                    aggregated.comments += platform_metrics.comments
                    aggregated.shares += platform_metrics.shares
                    aggregated.views += platform_metrics.views
                
                aggregated.calculate_engagement_rate()
                real_metrics.append(aggregated)
        
        print(f"📊 Collected metrics for {len(real_metrics)} posts")
        
        # Update environment with real metrics
        self._update_environment_with_real_data(real_metrics)
        
        # Retrain agent
        print("🎯 Retraining with real-world feedback...")
        self.meta_agent.train(num_episodes=50)  # Shorter retraining
        
        return real_metrics
    
    def _update_environment_with_real_data(self, real_metrics: List[PostMetrics]):
        """Update the RL environment with real-world data."""
        # This would update the environment's reward function
        # to better reflect real-world engagement patterns
        if real_metrics:
            avg_engagement = sum(m.engagement_rate for m in real_metrics) / len(real_metrics)
            print(f"📈 Average real engagement rate: {avg_engagement:.2%}")


# ============== Usage Examples ==============
def example_basic_usage():
    """Basic usage example."""
    print("🌟 Basic RL Social Media Optimization Example")
    
    # 1. Create and train meta-agent
    agent = AstrologyMetaAgent(algorithm="PPO", calendar_file="astro_calendar.csv")
    agent.train(num_episodes=100)
    
    # 2. Deploy learned policy
    schedule = agent.deploy_policy()
    print(f"\n📅 Generated optimal schedule for {len(schedule)} posts")


def example_full_integration():
    """Full integration with real social media platforms."""
    print("🚀 Full Social Media Integration Example")
    
    # 1. Initialize platform manager
    platform_manager = MultiPlatformManager("social_config.json")
    
    # 2. Create and train meta-agent
    agent = AstrologyMetaAgent(algorithm="PPO")
    agent.train(num_episodes=200)
    
    # 3. Create integrated poster
    poster = RLOptimizedPoster(agent, platform_manager)
    
    # 4. Execute strategy
    poster.execute_learned_strategy(days_ahead=7)
    
    # 5. Collect feedback and retrain
    real_metrics = poster.collect_feedback_and_retrain(wait_hours=24)
    
    # 6. Generate insights report
    insights = platform_manager.get_aggregated_insights()
    print(f"\n📊 Total followers across platforms: {insights['total_followers']:,}")
    print(f"📈 Average engagement rate: {insights['average_engagement']:.2%}")


def example_continuous_optimization():
    """Continuous optimization example."""
    print("♾️  Continuous Optimization Example")
    
    # Initialize components
    platform_manager = MultiPlatformManager()
    agent = AstrologyMetaAgent(algorithm="PPO")
    poster = RLOptimizedPoster(agent, platform_manager)
    
    # Initial training
    agent.train(num_episodes=100)
    
    # Continuous optimization loop
    for cycle in range(10):  # 10 optimization cycles
        print(f"\n🔄 Optimization Cycle {cycle + 1}")
        
        # Execute current policy
        poster.execute_learned_strategy(days_ahead=3)
        
        # Collect feedback and retrain
        metrics = poster.collect_feedback_and_retrain(wait_hours=72)
        
        # Analyze improvement
        if metrics:
            avg_engagement = sum(m.engagement_rate for m in metrics) / len(metrics)
            print(f"📊 Cycle {cycle + 1} avg engagement: {avg_engagement:.2%}")
        
        # Save checkpoint
        agent.save_checkpoint(f"optimization_cycle_{cycle + 1}.pt")


# ============== Main ==============
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            example_basic_usage()
        elif sys.argv[1] == "full":
            example_full_integration()
        elif sys.argv[1] == "continuous":
            example_continuous_optimization()
        else:
            print("Usage: python social_media_integration.py [basic|full|continuous]")
    else:
        # Default: show basic usage
        example_basic_usage()
