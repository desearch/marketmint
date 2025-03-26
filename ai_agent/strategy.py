import numpy as np
import pandas as pd
import logging
from web3 import Web3
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Any, List, Optional, Tuple

from .trading_env import CryptoTradingEnv

class AITrader:
    """
    AI trading agent using PPO for market making
    """
    
    def __init__(self):
        """
        Initialize AI trading agent
        """
        self.model = None
        self.env = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
    
    def preprocess_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for training
        
        Args:
            market_data: Raw market data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = market_data.copy()
        
        # Add technical indicators if not present
        if 'sma_7' not in df.columns:
            df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        if 'sma_30' not in df.columns:
            df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
        if 'rsi' not in df.columns:
            # Calculate RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)  # Fill NaN with neutral value
        
        # Add volume and liquidity if not present
        if 'volume_24h' not in df.columns:
            df['volume_24h'] = np.random.normal(500000, 50000, len(df))
        if 'liquidity' not in df.columns:
            df['liquidity'] = np.random.normal(1000000, 100000, len(df))
            
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def create_env(self, market_data: pd.DataFrame) -> CryptoTradingEnv:
        """
        Create trading environment
        
        Args:
            market_data: Preprocessed market data
            
        Returns:
            Trading environment instance
        """
        processed_data = self.preprocess_data(market_data)
        return CryptoTradingEnv(processed_data)
    
    def train(self, market_data: pd.DataFrame, total_timesteps: int = 10000,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the trading agent
        
        Args:
            market_data: Market data for training
            total_timesteps: Number of training timesteps
            save_path: Path to save trained model
            
        Returns:
            Training results
        """
        # Create environment
        self.env = self.create_env(market_data)
        env = DummyVecEnv([lambda: self.env])
        
        # Initialize and train model
        self.model = PPO('MlpPolicy', env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        
        if save_path:
            self.save(save_path)
            
        return {
            'total_timesteps': total_timesteps,
            'final_reward': float(self.env.current_balance - self.env.initial_balance)
        }
    
    def predict(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Make trading predictions
        
        Args:
            market_data: Market data for prediction
            
        Returns:
            Array of trading actions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Create environment
        env = self.create_env(market_data)
        obs, _ = env.reset()
        
        actions = []
        for _ in range(len(market_data)):
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(action[0])
            obs, _, done, _, _ = env.step(action)
            if done:
                break
                
        return np.array(actions)
    
    def execute_trade(self, amount: float) -> Dict[str, Any]:
        """
        Execute a trade with the given amount
        
        Args:
            amount: Amount to trade (positive for buy, negative for sell)
            
        Returns:
            Dictionary containing trade results
        """
        if not self.env:
            raise ValueError("Environment not initialized. Call train() first.")
            
        # Execute trade
        action = np.array([amount])
        observation, reward, done, truncated, info = self.env.step(action)
        
        return {
            'action': float(action[0]),
            'reward': float(reward),
            'portfolio_value': float(info['portfolio_value']),
            'current_price': float(info['current_price'])
        }
    
    def save(self, path: str) -> None:
        """
        Save trained model
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """
        Load trained model
        
        Args:
            path: Path to load model from
        """
        self.model = PPO.load(path)
        
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value
        
        Returns:
            Current portfolio value
        """
        if not self.env:
            raise ValueError("Environment not initialized. Call train() first.")
        return float(self.env.current_balance + self.env.current_position * self.env.market_data.iloc[self.env.current_step]['price']) 