"""
Trading environment for cryptocurrency market making
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

class CryptoTradingEnv(gym.Env):
    """A trading environment for cryptocurrency market making"""
    
    def __init__(self, market_data: pd.DataFrame):
        """
        Initialize the trading environment
        
        Args:
            market_data: DataFrame with columns [price, volume, liquidity, timestamp]
        """
        super().__init__()
        
        # Initialize market data
        self.market_data = market_data.copy()
        self.current_step = 0
        self.max_steps = len(market_data)
        
        # Trading parameters
        self.initial_balance = 10000.0  # Initial cash balance
        self.transaction_cost = 0.001   # 0.1% transaction cost
        
        # State variables
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.last_action = 0.0
        
        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Fill NaN values
        self.market_data['sma_7'] = self.market_data['price'].rolling(window=7, min_periods=1).mean()
        self.market_data['sma_30'] = self.market_data['price'].rolling(window=30, min_periods=1).mean()
        
    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        current_price = self.market_data.iloc[self.current_step]['price']
        sma_7 = self.market_data.iloc[self.current_step]['sma_7']
        sma_30 = self.market_data.iloc[self.current_step]['sma_30']
        
        # Normalize price indicators
        price_sma7_ratio = current_price / sma_7 - 1.0
        price_sma30_ratio = current_price / sma_30 - 1.0
        
        # Position and balance ratios
        position_ratio = self.current_position * current_price / self.initial_balance
        balance_ratio = self.current_balance / self.initial_balance - 1.0
        
        # RSI scaled to [0, 1]
        rsi = self.market_data.iloc[self.current_step]['rsi'] / 100.0
        
        return np.array([
            price_sma7_ratio,
            price_sma30_ratio,
            position_ratio,
            balance_ratio,
            rsi,
            self.last_action
        ], dtype=np.float32)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value including cash and position"""
        current_price = self.market_data.iloc[self.current_step]['price']
        return self.current_balance + (self.current_position * current_price)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Trading action in range [-1, 1]
            
        Returns:
            observation: Current observation
            reward: Reward for the action
            done: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get current price and calculate trade size
        current_price = self.market_data.iloc[self.current_step]['price']
        trade_size = action[0] * self.initial_balance / current_price
        
        # Calculate transaction cost
        transaction_cost = abs(trade_size * current_price * self.transaction_cost)
        
        # Update position and balance
        old_portfolio_value = self._calculate_portfolio_value()
        
        if trade_size > 0:  # Buy
            max_affordable = (self.current_balance - transaction_cost) / current_price
            trade_size = min(trade_size, max_affordable)
            if trade_size > 0:
                self.current_position += trade_size
                self.current_balance -= (trade_size * current_price + transaction_cost)
        else:  # Sell
            trade_size = max(trade_size, -self.current_position)
            if trade_size < 0:
                self.current_position += trade_size
                self.current_balance += (-trade_size * current_price - transaction_cost)
        
        # Calculate reward
        new_portfolio_value = self._calculate_portfolio_value()
        reward = (new_portfolio_value - old_portfolio_value) / self.initial_balance
        
        # Update state
        self.last_action = float(action[0])
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'portfolio_value': new_portfolio_value,
            'position': self.current_position,
            'balance': self.current_balance,
            'current_price': current_price
        }
        
        return self._get_observation(), reward, done, False, info
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.last_action = 0.0
        
        info = {
            'initial_portfolio_value': self.initial_balance,
            'current_price': self.market_data.iloc[0]['price']
        }
        
        return self._get_observation(), info
    
    def render(self):
        """Render the environment"""
        pass
        
    def close(self):
        """Close the environment"""
        pass 