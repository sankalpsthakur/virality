#!/usr/bin/env python3
"""
Astrology Social Media RL Meta-Agent
Uses PPO/DQN to learn optimal posting policies for astrology content.
Integrates with the astrology calendar generator to optimize engagement.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
from collections import deque
import datetime as dt
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import os
import time
import random
from enum import Enum

# Import social media APIs (mock implementations for now)
try:
    import facebook
    import tweepy
    import instagram_private_api
    SOCIAL_APIS_AVAILABLE = True
except ImportError:
    SOCIAL_APIS_AVAILABLE = False
    print("⚠️  Social media APIs not available, using mock implementations")


# ============== Data Classes ==============
@dataclass
class PostMetrics:
    """Metrics for a single post."""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0
    clicks: int = 0
    saves: int = 0
    engagement_rate: float = 0.0
    timestamp: dt.datetime = None
    
    def calculate_engagement_rate(self) -> float:
        """Calculate engagement rate from metrics."""
        total_engagement = self.likes + self.comments + self.shares + self.saves
        if self.views > 0:
            self.engagement_rate = total_engagement / self.views
        return self.engagement_rate


@dataclass
class AstrologyPost:
    """Represents an astrology post with all metadata."""
    date: dt.date
    sun_sign: str
    moon_phase: str
    caption: str
    video_file: Optional[str] = None
    scheduled_time: Optional[dt.datetime] = None
    posted: bool = False
    metrics: Optional[PostMetrics] = None
    platform: str = "instagram"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = self._generate_tags()
    
    def _generate_tags(self) -> List[str]:
        """Generate relevant hashtags based on content."""
        base_tags = ["#astrology", "#zodiac", "#horoscope", "#spiritual", "#cosmic"]
        sign_tags = [f"#{self.sun_sign.lower()}", f"#{self.sun_sign.lower()}season"]
        phase_tags = [f"#{self.moon_phase.lower().replace(' ', '')}", "#moonphase"]
        return base_tags + sign_tags + phase_tags


class PostingStrategy(Enum):
    """Different posting strategies to explore."""
    MORNING_PEAK = "morning_peak"      # 6-9 AM
    LUNCH_BREAK = "lunch_break"        # 11 AM - 1 PM
    EVENING_PRIME = "evening_prime"    # 5-8 PM
    NIGHT_OWL = "night_owl"            # 9-11 PM
    ADAPTIVE = "adaptive"              # Learn from data


# ============== RL Environment ==============
class AstrologyPostingEnv(gym.Env):
    """Gym environment for social media posting optimization."""
    
    def __init__(self, posts_queue: List[AstrologyPost], 
                 platform: str = "instagram",
                 history_window: int = 30):
        super().__init__()
        
        self.posts_queue = posts_queue
        self.platform = platform
        self.history_window = history_window
        self.current_idx = 0
        self.posted_history = deque(maxlen=history_window)
        self.time_step = 0
        
        # Define action space: [hour (0-23), use_video (0/1), caption_style (0-4)]
        self.action_space = spaces.MultiDiscrete([24, 2, 5])
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._get_state_dim(),), 
            dtype=np.float32
        )
        
        # Analytics tracking
        self.episode_rewards = []
        self.metrics_history = []
        
    def _get_state_dim(self) -> int:
        """Calculate state dimension."""
        # Features: day_of_week(7), hour(24), sun_sign(12), moon_phase(8), 
        # past_performance(10), content_features(5), total=66
        return 66
    
    def _get_state(self) -> np.ndarray:
        """Extract current state features."""
        if self.current_idx >= len(self.posts_queue):
            return np.zeros(self._get_state_dim())
        
        post = self.posts_queue[self.current_idx]
        state = []
        
        # Temporal features
        current_time = dt.datetime.now()
        day_of_week = np.zeros(7)
        day_of_week[current_time.weekday()] = 1
        state.extend(day_of_week)
        
        hour_encoding = np.zeros(24)
        hour_encoding[current_time.hour] = 1
        state.extend(hour_encoding)
        
        # Astrological features
        sun_signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                     "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
        sun_sign_encoding = np.zeros(12)
        if post.sun_sign in sun_signs:
            sun_sign_encoding[sun_signs.index(post.sun_sign)] = 1
        state.extend(sun_sign_encoding)
        
        moon_phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                       "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
        moon_phase_encoding = np.zeros(8)
        if post.moon_phase in moon_phases:
            moon_phase_encoding[moon_phases.index(post.moon_phase)] = 1
        state.extend(moon_phase_encoding)
        
        # Past performance features (10 features)
        if len(self.posted_history) > 0:
            recent_metrics = [p.metrics for p in self.posted_history if p.metrics]
            if recent_metrics:
                avg_likes = np.mean([m.likes for m in recent_metrics])
                avg_comments = np.mean([m.comments for m in recent_metrics])
                avg_shares = np.mean([m.shares for m in recent_metrics])
                avg_engagement = np.mean([m.engagement_rate for m in recent_metrics])
                trend_likes = self._calculate_trend([m.likes for m in recent_metrics])
                trend_engagement = self._calculate_trend([m.engagement_rate for m in recent_metrics])
                
                state.extend([
                    avg_likes / 1000,  # Normalize
                    avg_comments / 100,
                    avg_shares / 100,
                    avg_engagement * 100,
                    trend_likes,
                    trend_engagement,
                    len(recent_metrics) / self.history_window,  # Fill rate
                    0, 0, 0  # Padding
                ])
            else:
                state.extend([0] * 10)
        else:
            state.extend([0] * 10)
        
        # Content features (5 features)
        state.extend([
            len(post.caption) / 280,  # Caption length ratio
            1 if post.video_file else 0,  # Has video
            len(post.tags) / 30,  # Tag density
            self._get_caption_sentiment(post.caption),  # Sentiment score
            self._get_caption_complexity(post.caption)  # Complexity score
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from recent values."""
        if len(values) < 2:
            return 0.0
        # Simple linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return np.tanh(slope)  # Normalize to [-1, 1]
    
    def _get_caption_sentiment(self, caption: str) -> float:
        """Simple sentiment scoring (would use proper NLP in production)."""
        positive_words = ["love", "amazing", "blessed", "powerful", "magic", "beautiful", "wonderful"]
        negative_words = ["difficult", "challenge", "struggle", "hard", "tough"]
        
        caption_lower = caption.lower()
        pos_score = sum(word in caption_lower for word in positive_words)
        neg_score = sum(word in caption_lower for word in negative_words)
        
        return (pos_score - neg_score) / max(1, pos_score + neg_score)
    
    def _get_caption_complexity(self, caption: str) -> float:
        """Measure caption complexity."""
        words = caption.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        return min(avg_word_length / 10, 1.0)  # Normalize
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state, reward, done, info."""
        if self.current_idx >= len(self.posts_queue):
            return self._get_state(), 0, True, {}
        
        # Parse action
        post_hour = int(action[0])
        use_video = bool(action[1])
        caption_style = int(action[2])
        
        # Get current post
        post = self.posts_queue[self.current_idx]
        
        # Schedule and "post" content
        scheduled_time = dt.datetime.combine(post.date, dt.time(post_hour, 0))
        post.scheduled_time = scheduled_time
        
        # Simulate posting and getting metrics
        metrics = self._simulate_posting(post, use_video, caption_style)
        post.metrics = metrics
        post.posted = True
        
        # Calculate reward
        reward = self._calculate_reward(metrics)
        
        # Update history
        self.posted_history.append(post)
        self.metrics_history.append(metrics)
        self.episode_rewards.append(reward)
        
        # Move to next post
        self.current_idx += 1
        self.time_step += 1
        
        # Check if done
        done = self.current_idx >= len(self.posts_queue)
        
        # Get new state
        new_state = self._get_state()
        
        info = {
            "metrics": asdict(metrics),
            "post_time": scheduled_time.isoformat(),
            "sun_sign": post.sun_sign,
            "moon_phase": post.moon_phase
        }
        
        return new_state, reward, done, info
    
    def _simulate_posting(self, post: AstrologyPost, use_video: bool, 
                         caption_style: int) -> PostMetrics:
        """Simulate posting and generate realistic metrics."""
        # Base engagement rates by time of day
        hour = post.scheduled_time.hour
        time_multipliers = {
            range(6, 9): 1.2,    # Morning peak
            range(11, 14): 1.1,  # Lunch
            range(17, 21): 1.3,  # Evening peak
            range(21, 24): 0.9,  # Late night
        }
        
        time_mult = 0.7  # Default
        for time_range, mult in time_multipliers.items():
            if hour in time_range:
                time_mult = mult
                break
        
        # Content multipliers
        video_mult = 1.5 if use_video else 1.0
        
        # Sign popularity (some signs naturally get more engagement)
        sign_popularity = {
            "Scorpio": 1.3, "Leo": 1.2, "Pisces": 1.15,
            "Gemini": 1.1, "Libra": 1.1, "Aquarius": 1.05
        }
        sign_mult = sign_popularity.get(post.sun_sign, 1.0)
        
        # Moon phase impact
        phase_impact = {
            "Full Moon": 1.25, "New Moon": 1.2,
            "First Quarter": 1.1, "Last Quarter": 1.1
        }
        phase_mult = phase_impact.get(post.moon_phase, 1.0)
        
        # Caption style impact
        style_impacts = [1.0, 0.9, 1.1, 1.2, 0.95]  # Different styles
        style_mult = style_impacts[caption_style]
        
        # Calculate base metrics with randomness
        base_views = np.random.poisson(5000)
        total_mult = time_mult * video_mult * sign_mult * phase_mult * style_mult
        
        views = int(base_views * total_mult * (0.8 + 0.4 * np.random.random()))
        engagement_rate = 0.03 * total_mult * (0.7 + 0.6 * np.random.random())
        
        likes = int(views * engagement_rate * 0.7)
        comments = int(views * engagement_rate * 0.2)
        shares = int(views * engagement_rate * 0.08)
        saves = int(views * engagement_rate * 0.02)
        clicks = int(views * 0.02 * total_mult)
        
        metrics = PostMetrics(
            likes=likes,
            comments=comments,
            shares=shares,
            views=views,
            clicks=clicks,
            saves=saves,
            timestamp=post.scheduled_time
        )
        metrics.calculate_engagement_rate()
        
        return metrics
    
    def _calculate_reward(self, metrics: PostMetrics) -> float:
        """Calculate reward from metrics."""
        # Multi-objective reward function
        engagement_reward = metrics.engagement_rate * 100
        
        # Normalize individual metrics
        likes_reward = np.log1p(metrics.likes) / 10
        comments_reward = np.log1p(metrics.comments) / 5  # Comments are more valuable
        shares_reward = np.log1p(metrics.shares) / 3
        saves_reward = np.log1p(metrics.saves) / 2
        
        # Combine with weights
        total_reward = (
            0.3 * engagement_reward +
            0.2 * likes_reward +
            0.2 * comments_reward +
            0.15 * shares_reward +
            0.1 * saves_reward +
            0.05 * (metrics.clicks / max(1, metrics.views))
        )
        
        return total_reward
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_idx = 0
        self.time_step = 0
        self.episode_rewards = []
        # Don't clear posted_history to maintain learning
        return self._get_state()
    
    def render(self, mode='human'):
        """Render current state (optional)."""
        if mode == 'human' and self.posted_history:
            recent_post = self.posted_history[-1]
            if recent_post.metrics:
                print(f"Last Post: {recent_post.sun_sign} | {recent_post.moon_phase}")
                print(f"Engagement: {recent_post.metrics.engagement_rate:.2%}")
                print(f"Likes: {recent_post.metrics.likes}, Views: {recent_post.metrics.views}")


# ============== Neural Networks ==============
class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dims: List[int], hidden_size: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Actor heads (one for each action dimension)
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_size, dim) for dim in action_dims
        ])
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass returning action logits and value."""
        shared_features = self.shared(state)
        
        # Get action logits for each dimension
        action_logits = [head(shared_features) for head in self.actor_heads]
        
        # Get value estimate
        value = self.critic(shared_features)
        
        return action_logits, value
    
    def get_action_and_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        action_logits, value = self.forward(state)
        
        # Sample actions from each dimension
        actions = []
        log_probs = []
        entropies = []
        
        for logits in action_logits:
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        # Stack actions
        actions = torch.stack(actions, dim=-1)
        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropies = torch.stack(entropies, dim=-1).sum(dim=-1)
        
        return actions, log_probs, entropies, value.squeeze()


class DQNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces."""
    
    def __init__(self, state_dim: int, action_dims: List[int], hidden_size: int = 256):
        super().__init__()
        
        # Calculate total action combinations
        self.action_dims = action_dims
        self.total_actions = np.prod(action_dims)
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.total_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions."""
        return self.network(state)
    
    def action_to_index(self, action: np.ndarray) -> int:
        """Convert multi-dimensional action to single index."""
        index = 0
        multiplier = 1
        for i in range(len(self.action_dims) - 1, -1, -1):
            index += action[i] * multiplier
            multiplier *= self.action_dims[i]
        return index
    
    def index_to_action(self, index: int) -> np.ndarray:
        """Convert single index to multi-dimensional action."""
        action = []
        for dim in reversed(self.action_dims):
            action.append(index % dim)
            index //= dim
        return np.array(list(reversed(action)))


# ============== RL Agents ==============
class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim: int, action_dims: List[int], 
                 lr: float = 3e-4, gamma: float = 0.99, 
                 eps_clip: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy = ActorCriticNetwork(state_dim, action_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, log_prob: float, value: float, done: bool):
        """Store transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns).to(self.device)
    
    def update(self, epochs: int = 4, batch_size: int = 64):
        """Update policy using PPO."""
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = self.compute_returns()
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Normalize advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Mini-batch training
            for idx in range(0, len(states), batch_size):
                batch_idx = slice(idx, min(idx + batch_size, len(states)))
                
                # Get current policy values
                _, log_probs, entropies, values = self.policy.get_action_and_value(states[batch_idx])
                
                # Calculate ratios
                ratios = torch.exp(log_probs - old_log_probs[batch_idx])
                
                # Calculate surrogate losses
                surr1 = ratios * advantages[batch_idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[batch_idx]
                
                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns[batch_idx])
                entropy_loss = -entropies.mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class DQNAgent:
    """Deep Q-Network agent with experience replay."""
    
    def __init__(self, state_dim: int, action_dims: List[int],
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dims = action_dims
        
        # Networks
        self.q_network = DQNetwork(state_dim, action_dims).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Random action
            return np.array([np.random.randint(dim) for dim in self.action_dims])
        
        # Greedy action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        return self.q_network.index_to_action(action_idx)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size: int = 32):
        """Update Q-network using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([self.q_network.action_to_index(a) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


# ============== Meta-Agent Orchestrator ==============
class AstrologyMetaAgent:
    """Meta-agent that manages the entire posting pipeline."""
    
    def __init__(self, algorithm: str = "PPO", 
                 calendar_file: str = "astro_calendar.csv",
                 platform: str = "instagram"):
        
        self.algorithm = algorithm
        self.platform = platform
        self.calendar_file = calendar_file
        
        # Load posts from calendar
        self.posts_queue = self._load_posts_from_calendar()
        
        # Initialize environment
        self.env = AstrologyPostingEnv(self.posts_queue, platform)
        
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        action_dims = list(self.env.action_space.nvec)
        
        if algorithm == "PPO":
            self.agent = PPOAgent(state_dim, action_dims)
        elif algorithm == "DQN":
            self.agent = DQNAgent(state_dim, action_dims)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Analytics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'average_engagement': [],
            'best_posts': []
        }
        
        # Policy tracker
        self.policy_history = []
        
    def _load_posts_from_calendar(self) -> List[AstrologyPost]:
        """Load posts from the generated calendar CSV."""
        posts = []
        
        try:
            df = pd.read_csv(self.calendar_file)
            
            for _, row in df.iterrows():
                post = AstrologyPost(
                    date=dt.datetime.strptime(row['scheduled_post_date'], '%Y-%m-%d').date(),
                    sun_sign=row['sun_sign'],
                    moon_phase=row['moon_phase'],
                    caption=row['caption'],
                    video_file=row.get('video_file', None) if pd.notna(row.get('video_file')) else None
                )
                posts.append(post)
                
        except Exception as e:
            print(f"Error loading calendar: {e}")
            # Generate sample posts for demo
            posts = self._generate_sample_posts()
        
        return posts
    
    def _generate_sample_posts(self) -> List[AstrologyPost]:
        """Generate sample posts for testing."""
        posts = []
        start_date = dt.date.today()
        
        signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo"]
        phases = ["New Moon", "Waxing Crescent", "Full Moon", "Waning Gibbous"]
        
        for i in range(30):
            date = start_date + dt.timedelta(days=i)
            post = AstrologyPost(
                date=date,
                sun_sign=signs[i % len(signs)],
                moon_phase=phases[i % len(phases)],
                caption=f"✨ {signs[i % len(signs)]} energy with {phases[i % len(phases)]} vibes! #astrology"
            )
            posts.append(post)
        
        return posts
    
    def train(self, num_episodes: int = 100, 
              save_interval: int = 10,
              update_interval: int = 2048):
        """Train the RL agent."""
        
        print(f"🚀 Starting {self.algorithm} training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Collect trajectory
            while not done:
                # Select action
                if self.algorithm == "PPO":
                    action = self.agent.select_action(state)
                    
                    # Get log prob and value for PPO
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    with torch.no_grad():
                        _, log_prob, _, value = self.agent.policy.get_action_and_value(state_tensor)
                    
                    # Step environment
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store transition
                    self.agent.store_transition(
                        state, action, reward, 
                        log_prob.item(), value.item(), done
                    )
                    
                elif self.algorithm == "DQN":
                    action = self.agent.select_action(state)
                    
                    # Step environment
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store transition
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    # Update Q-network
                    self.agent.update()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Render occasionally
                if episode % 10 == 0 and episode_length % 10 == 0:
                    self.env.render()
            
            # Update PPO policy at episode end
            if self.algorithm == "PPO":
                self.agent.update()
            
            # Update DQN target network
            if self.algorithm == "DQN" and episode % 10 == 0:
                self.agent.update_target_network()
            
            # Track metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            
            # Calculate average engagement
            if self.env.metrics_history:
                avg_engagement = np.mean([m.engagement_rate for m in self.env.metrics_history[-episode_length:]])
                self.training_history['average_engagement'].append(avg_engagement)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_engagement = np.mean(self.training_history['average_engagement'][-10:]) if self.training_history['average_engagement'] else 0
                
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Engagement: {avg_engagement:.2%}")
                if self.algorithm == "DQN":
                    print(f"  Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model
            if episode % save_interval == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pt")
            
            # Extract and save policy insights
            if episode % 20 == 0:
                self._extract_policy_insights()
        
        print("✅ Training complete!")
        self._generate_training_report()
    
    def _extract_policy_insights(self):
        """Extract insights from current policy."""
        insights = {
            'timestamp': dt.datetime.now(),
            'preferred_hours': self._get_preferred_posting_hours(),
            'sign_strategies': self._get_sign_specific_strategies(),
            'phase_strategies': self._get_phase_specific_strategies(),
            'content_preferences': self._get_content_preferences()
        }
        
        self.policy_history.append(insights)
        
    def _get_preferred_posting_hours(self) -> Dict[str, float]:
        """Analyze preferred posting hours from policy."""
        # Sample states and get action preferences
        hour_preferences = np.zeros(24)
        
        for _ in range(100):
            # Create random state
            state = self.env.observation_space.sample()
            
            if self.algorithm == "PPO":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    action_logits, _ = self.agent.policy(state_tensor)
                    hour_probs = torch.softmax(action_logits[0], dim=-1).cpu().numpy()[0]
                    hour_preferences += hour_probs
            
        hour_preferences /= hour_preferences.sum()
        
        return {f"{h:02d}:00": float(p) for h, p in enumerate(hour_preferences)}
    
    def _get_sign_specific_strategies(self) -> Dict[str, Dict]:
        """Extract sign-specific posting strategies."""
        strategies = {}
        
        signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                 "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
        
        for sign in signs:
            # Create state with this sign
            state = np.zeros(self.env._get_state_dim())
            # Set sign encoding
            sign_idx = signs.index(sign)
            state[7 + 24 + sign_idx] = 1  # After day_of_week and hour encodings
            
            # Get preferred actions
            if self.algorithm == "PPO":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    action_logits, _ = self.agent.policy(state_tensor)
                    
                    # Get most likely hour
                    hour = torch.argmax(action_logits[0]).item()
                    # Get video preference
                    use_video = torch.argmax(action_logits[1]).item()
                    # Get caption style
                    caption_style = torch.argmax(action_logits[2]).item()
                    
                    strategies[sign] = {
                        'preferred_hour': hour,
                        'use_video': bool(use_video),
                        'caption_style': caption_style
                    }
        
        return strategies
    
    def _get_phase_specific_strategies(self) -> Dict[str, Dict]:
        """Extract moon phase specific strategies."""
        strategies = {}
        
        phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                  "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
        
        for phase in phases:
            # Similar to sign-specific, but for moon phases
            state = np.zeros(self.env._get_state_dim())
            phase_idx = phases.index(phase)
            state[7 + 24 + 12 + phase_idx] = 1  # After day, hour, and sign encodings
            
            if self.algorithm == "PPO":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    action_logits, _ = self.agent.policy(state_tensor)
                    
                    strategies[phase] = {
                        'preferred_hour': torch.argmax(action_logits[0]).item(),
                        'use_video': bool(torch.argmax(action_logits[1]).item()),
                        'caption_style': torch.argmax(action_logits[2]).item()
                    }
        
        return strategies
    
    def _get_content_preferences(self) -> Dict[str, float]:
        """Analyze content preferences from policy."""
        video_preference = 0
        style_preferences = np.zeros(5)
        
        # Sample many states
        for _ in range(200):
            state = self.env.observation_space.sample()
            
            if self.algorithm == "PPO":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    action_logits, _ = self.agent.policy(state_tensor)
                    
                    video_probs = torch.softmax(action_logits[1], dim=-1).cpu().numpy()[0]
                    video_preference += video_probs[1]  # Probability of using video
                    
                    style_probs = torch.softmax(action_logits[2], dim=-1).cpu().numpy()[0]
                    style_preferences += style_probs
        
        return {
            'video_usage_rate': float(video_preference / 200),
            'caption_styles': {
                'style_0': float(style_preferences[0] / 200),
                'style_1': float(style_preferences[1] / 200),
                'style_2': float(style_preferences[2] / 200),
                'style_3': float(style_preferences[3] / 200),
                'style_4': float(style_preferences[4] / 200)
            }
        }
    
    def _generate_training_report(self):
        """Generate comprehensive training report."""
        report = {
            'training_summary': {
                'algorithm': self.algorithm,
                'total_episodes': len(self.training_history['episode_rewards']),
                'total_posts': sum(self.training_history['episode_lengths']),
                'average_reward': np.mean(self.training_history['episode_rewards']),
                'best_reward': max(self.training_history['episode_rewards']),
                'final_avg_engagement': np.mean(self.training_history['average_engagement'][-10:]) if self.training_history['average_engagement'] else 0
            },
            'optimal_strategies': {
                'posting_hours': self._get_preferred_posting_hours(),
                'sign_strategies': self._get_sign_specific_strategies(),
                'phase_strategies': self._get_phase_specific_strategies(),
                'content_preferences': self._get_content_preferences()
            },
            'performance_metrics': {
                'engagement_improvement': self._calculate_engagement_improvement(),
                'best_performing_combinations': self._get_best_performing_combinations()
            }
        }
        
        # Save report
        with open(f'training_report_{self.algorithm}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._plot_training_curves()
        self._plot_policy_heatmaps()
        
        print("\n📊 Training Report Generated!")
        print(f"  Algorithm: {report['training_summary']['algorithm']}")
        print(f"  Final Avg Engagement: {report['training_summary']['final_avg_engagement']:.2%}")
        print(f"  Engagement Improvement: {report['performance_metrics']['engagement_improvement']:.2%}")
    
    def _calculate_engagement_improvement(self) -> float:
        """Calculate improvement in engagement over training."""
        if len(self.training_history['average_engagement']) < 20:
            return 0.0
        
        initial_avg = np.mean(self.training_history['average_engagement'][:10])
        final_avg = np.mean(self.training_history['average_engagement'][-10:])
        
        if initial_avg > 0:
            return (final_avg - initial_avg) / initial_avg
        return 0.0
    
    def _get_best_performing_combinations(self) -> List[Dict]:
        """Get best performing content combinations."""
        if not self.env.posted_history:
            return []
        
        # Sort posts by engagement
        sorted_posts = sorted(
            [p for p in self.env.posted_history if p.metrics],
            key=lambda x: x.metrics.engagement_rate,
            reverse=True
        )
        
        best_combos = []
        for post in sorted_posts[:10]:
            best_combos.append({
                'sun_sign': post.sun_sign,
                'moon_phase': post.moon_phase,
                'hour': post.scheduled_time.hour if post.scheduled_time else None,
                'engagement_rate': post.metrics.engagement_rate,
                'likes': post.metrics.likes,
                'views': post.metrics.views
            })
        
        return best_combos
    
    def _plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Average engagement
        if self.training_history['average_engagement']:
            axes[0, 1].plot(self.training_history['average_engagement'])
            axes[0, 1].set_title('Average Engagement Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Engagement Rate')
        
        # Moving average rewards
        window = 10
        if len(self.training_history['episode_rewards']) >= window:
            moving_avg = pd.Series(self.training_history['episode_rewards']).rolling(window).mean()
            axes[1, 0].plot(moving_avg)
            axes[1, 0].set_title(f'{window}-Episode Moving Average Reward')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Average Reward')
        
        # Episode lengths
        axes[1, 1].plot(self.training_history['episode_lengths'])
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Posts')
        
        plt.tight_layout()
        plt.savefig(f'training_curves_{self.algorithm}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def _plot_policy_heatmaps(self):
        """Plot policy heatmaps for insights."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Posting hour preferences
        hour_prefs = self._get_preferred_posting_hours()
        hours = list(range(24))
        prefs = [hour_prefs.get(f"{h:02d}:00", 0) for h in hours]
        
        axes[0, 0].bar(hours, prefs)
        axes[0, 0].set_title('Preferred Posting Hours')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Preference Score')
        
        # Sign-specific strategies heatmap
        sign_strategies = self._get_sign_specific_strategies()
        signs = list(sign_strategies.keys())
        hours_by_sign = [sign_strategies[sign]['preferred_hour'] for sign in signs]
        
        # Create scatter plot
        axes[0, 1].scatter(hours_by_sign, range(len(signs)), s=100)
        axes[0, 1].set_yticks(range(len(signs)))
        axes[0, 1].set_yticklabels(signs)
        axes[0, 1].set_xlabel('Preferred Hour')
        axes[0, 1].set_title('Optimal Posting Hours by Zodiac Sign')
        axes[0, 1].set_xlim(-1, 24)
        
        # Phase strategies
        phase_strategies = self._get_phase_specific_strategies()
        phases = list(phase_strategies.keys())
        phase_hours = [phase_strategies[phase]['preferred_hour'] for phase in phases]
        
        axes[1, 0].scatter(phase_hours, range(len(phases)), s=100)
        axes[1, 0].set_yticks(range(len(phases)))
        axes[1, 0].set_yticklabels(phases)
        axes[1, 0].set_xlabel('Preferred Hour')
        axes[1, 0].set_title('Optimal Posting Hours by Moon Phase')
        axes[1, 0].set_xlim(-1, 24)
        
        # Content preferences
        content_prefs = self._get_content_preferences()
        video_rate = content_prefs['video_usage_rate']
        style_rates = list(content_prefs['caption_styles'].values())
        
        axes[1, 1].bar(['No Video', 'Use Video'], [1-video_rate, video_rate])
        axes[1, 1].set_title('Video Usage Preference')
        axes[1, 1].set_ylabel('Preference Score')
        
        plt.tight_layout()
        plt.savefig(f'policy_insights_{self.algorithm}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def deploy_policy(self, real_posting: bool = False):
        """Deploy the learned policy for actual posting."""
        print("\n🚀 Deploying learned policy...")
        
        # Load best checkpoint
        best_checkpoint = self._find_best_checkpoint()
        if best_checkpoint:
            self.agent.load(best_checkpoint)
        
        # Get upcoming posts
        upcoming_posts = [p for p in self.posts_queue if not p.posted]
        
        print(f"📅 Scheduling {len(upcoming_posts)} upcoming posts...")
        
        scheduled_posts = []
        for post in upcoming_posts:
            # Get optimal action for this post
            state = self._create_state_for_post(post)
            action = self.agent.select_action(state)
            
            # Parse action
            hour = int(action[0])
            use_video = bool(action[1])
            caption_style = int(action[2])
            
            # Schedule post
            scheduled_time = dt.datetime.combine(post.date, dt.time(hour, 0))
            
            scheduled_post = {
                'date': post.date.isoformat(),
                'time': scheduled_time.strftime("%H:%M"),
                'sun_sign': post.sun_sign,
                'moon_phase': post.moon_phase,
                'caption': self._enhance_caption(post.caption, caption_style),
                'use_video': use_video,
                'video_file': post.video_file if use_video else None,
                'tags': post.tags
            }
            
            scheduled_posts.append(scheduled_post)
            
            # Actually post if enabled
            if real_posting and dt.datetime.now() >= scheduled_time:
                self._post_to_platform(scheduled_post)
        
        # Save schedule
        schedule_file = f'posting_schedule_{dt.datetime.now().strftime("%Y%m%d")}.json'
        with open(schedule_file, 'w') as f:
            json.dump(scheduled_posts, f, indent=2)
        
        print(f"✅ Schedule saved to {schedule_file}")
        
        return scheduled_posts
    
    def _create_state_for_post(self, post: AstrologyPost) -> np.ndarray:
        """Create state representation for a specific post."""
        # This would match the state creation in the environment
        # Simplified version here
        return self.env._get_state()
    
    def _enhance_caption(self, base_caption: str, style: int) -> str:
        """Enhance caption based on learned style preference."""
        style_enhancements = {
            0: lambda c: c,  # Original
            1: lambda c: f"🌟 {c} ✨",  # Sparkly
            2: lambda c: f"{c}\n\nWhat cosmic energy are you feeling today? 💫",  # Engaging
            3: lambda c: f"COSMIC ALERT 🚨\n\n{c}",  # Urgent
            4: lambda c: f"{c}\n\n🔮 Follow for daily cosmic guidance 🌙"  # CTA
        }
        
        enhancer = style_enhancements.get(style, lambda c: c)
        return enhancer(base_caption)
    
    def _find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint based on performance."""
        import glob
        checkpoints = glob.glob("checkpoint_ep*.pt")
        
        if checkpoints:
            # In practice, you'd track which checkpoint had best validation performance
            # For now, return the latest
            return sorted(checkpoints)[-1]
        return None
    
    def _post_to_platform(self, post: Dict):
        """Actually post to social media platform (mock implementation)."""
        print(f"📱 Posting to {self.platform}: {post['sun_sign']} at {post['time']}")
        # Here you would integrate with actual social media APIs
        # For now, just log it
        pass
    
    def continuous_learning(self, feedback_window: int = 7):
        """Continuously learn from real posting feedback."""
        print("\n🔄 Starting continuous learning mode...")
        
        while True:
            # Wait for feedback window
            print(f"⏳ Waiting {feedback_window} days for feedback...")
            time.sleep(feedback_window * 24 * 3600)  # In production
            
            # Collect real metrics
            real_metrics = self._collect_real_metrics()
            
            # Update training data
            if real_metrics:
                self._update_with_real_metrics(real_metrics)
                
                # Retrain with new data
                print("🎯 Retraining with new data...")
                self.train(num_episodes=20)  # Shorter training for updates
                
                # Deploy updated policy
                self.deploy_policy()
            
            print("✅ Continuous learning cycle complete")
    
    def _collect_real_metrics(self) -> List[PostMetrics]:
        """Collect real metrics from social media platforms."""
        # Mock implementation - would connect to real APIs
        print("📊 Collecting real metrics from platforms...")
        return []
    
    def _update_with_real_metrics(self, metrics: List[PostMetrics]):
        """Update training data with real-world metrics."""
        # Incorporate real metrics into the environment's reward function
        pass
    
    def save_checkpoint(self, filename: str):
        """Save complete checkpoint including agent and meta-data."""
        checkpoint = {
            'algorithm': self.algorithm,
            'training_history': self.training_history,
            'policy_history': self.policy_history,
            'timestamp': dt.datetime.now().isoformat()
        }
        
        # Save agent state
        self.agent.save(filename)
        
        # Save meta-data
        meta_filename = filename.replace('.pt', '_meta.json')
        with open(meta_filename, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        print(f"💾 Checkpoint saved: {filename}")


# ============== Main Execution ==============
def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Astrology Social Media RL Meta-Agent")
    parser.add_argument("--algorithm", choices=["PPO", "DQN"], default="PPO",
                        help="RL algorithm to use")
    parser.add_argument("--calendar", default="astro_calendar.csv",
                        help="Path to astrology calendar CSV")
    parser.add_argument("--platform", default="instagram",
                        choices=["instagram", "facebook", "twitter"],
                        help="Social media platform")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy learned policy after training")
    parser.add_argument("--continuous", action="store_true",
                        help="Enable continuous learning mode")
    
    args = parser.parse_args()
    
    print("🌙 Astrology Social Media RL Meta-Agent")
    print("=" * 50)
    print(f"Algorithm: {args.algorithm}")
    print(f"Platform: {args.platform}")
    print(f"Calendar: {args.calendar}")
    print("=" * 50)
    
    # Create meta-agent
    agent = AstrologyMetaAgent(
        algorithm=args.algorithm,
        calendar_file=args.calendar,
        platform=args.platform
    )
    
    # Train agent
    agent.train(num_episodes=args.episodes)
    
    # Deploy if requested
    if args.deploy:
        agent.deploy_policy(real_posting=False)  # Set to True for actual posting
    
    # Start continuous learning if requested
    if args.continuous:
        agent.continuous_learning()


if __name__ == "__main__":
    main()
