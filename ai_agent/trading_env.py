import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from gymnasium import spaces

class CryptoTradingEnv(gym.Env):
    """Custom Environment for cryptocurrency trading"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000, transaction_cost: float = 0.001):
        super(CryptoTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.current_balance = initial_balance
        self.current_position = 0
        self.initial_portfolio_value = initial_balance

        # Action space: continuous values between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        # Observation space: price, technical indicators, position, balance
        num_features = len(df.columns)  # All features from DataFrame
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features + 2,),  # +2 for position and balance
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.initial_portfolio_value = self.initial_balance

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        self._take_action(action[0])
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        obs = self._get_observation()
        reward = self._calculate_reward()
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'initial_portfolio_value': self.initial_portfolio_value,
            'current_balance': self.current_balance,
            'current_position': self.current_position
        }
        
        return obs, reward, done, False, info

    def _take_action(self, action: float) -> None:
        """Execute the trade action"""
        current_price = self.df.iloc[self.current_step]['price']
        
        if action > 0:  # Buy
            max_buy_amount = self.current_balance / current_price
            buy_amount = max_buy_amount * abs(action)
            cost = buy_amount * current_price * (1 + self.transaction_cost)
            
            if cost <= self.current_balance:
                self.current_position += buy_amount
                self.current_balance -= cost
                
        elif action < 0:  # Sell
            sell_amount = self.current_position * abs(action)
            revenue = sell_amount * current_price * (1 - self.transaction_cost)
            
            self.current_position -= sell_amount
            self.current_balance += revenue

    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        features = self.df.iloc[self.current_step].values
        position = np.array([self.current_position])
        balance = np.array([self.current_balance])
        
        return np.concatenate([features, position, balance]).astype(np.float32)

    def _calculate_reward(self) -> float:
        """Calculate the reward for the current step"""
        current_portfolio_value = self._calculate_portfolio_value()
        prev_portfolio_value = self._calculate_portfolio_value(step_offset=1)
        
        return (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value

    def _calculate_portfolio_value(self, step_offset: int = 0) -> float:
        """Calculate total portfolio value"""
        step = max(0, self.current_step - step_offset)
        price = self.df.iloc[step]['price']
        return self.current_balance + self.current_position * price

    def render(self, mode='human'):
        """Render the environment"""
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.current_balance:.2f}')
        print(f'Position: {self.current_position:.6f}')
        print(f'Portfolio Value: {self._calculate_portfolio_value():.2f}')
        print(f'Current Price: {self.df.iloc[self.current_step]["price"]:.2f}')

    def close(self):
        """Clean up resources"""
        pass 