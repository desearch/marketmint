import numpy as np
import pandas as pd
import logging
from web3 import Web3
from dotenv import load_dotenv
from stable_baselines3 import PPO
from typing import Dict, Any, List, Optional

from .trading_env import CryptoTradingEnv

class AITrader:
    """AI Trading Agent using PPO for cryptocurrency trading"""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        model_path: Optional[str] = None
    ):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.model_path = model_path
        self.model = None
        self.env = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data with technical indicators"""
        df = df.copy()
        
        # Calculate technical indicators
        df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
        
        # Calculate RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # Fill NaN values with neutral RSI
        
        return df
    
    def create_env(self, market_data: pd.DataFrame) -> CryptoTradingEnv:
        """Create and return the trading environment"""
        processed_data = self.preprocess_data(market_data)
        return CryptoTradingEnv(
            df=processed_data,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
    
    def train(
        self,
        market_data: pd.DataFrame,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the AI agent on historical market data"""
        self.env = self.create_env(market_data)
        
        # Initialize the PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save the trained model if path is provided
        if save_path:
            self.model.save(save_path)
            self.model_path = save_path
        
        return {
            "training_steps": total_timesteps,
            "final_portfolio_value": self.env._calculate_portfolio_value()
        }
    
    def predict(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Make trading predictions using the trained model"""
        if self.model is None:
            if self.model_path:
                self.model = PPO.load(self.model_path)
            else:
                raise ValueError("No trained model available. Please train or load a model first.")
        
        self.env = self.create_env(market_data)
        obs, _ = self.env.reset()
        done = False
        predictions = []
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)
            
            predictions.append({
                "step": self.env.current_step,
                "action": float(action[0]),
                "portfolio_value": info["portfolio_value"],
                "position": info["current_position"],
                "balance": info["current_balance"],
                "price": market_data.iloc[self.env.current_step]["price"]
            })
        
        return predictions
    
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model"""
        self.model = PPO.load(model_path)
        self.model_path = model_path
        self.logger.info(f"Loaded model from {model_path}")
        
    def save_model(self, save_path: str) -> None:
        """Save the current model"""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        self.model.save(save_path)
        self.model_path = save_path
        self.logger.info(f"Saved model to {save_path}") 