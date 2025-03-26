"""
Tests for AI trading agent
"""

import unittest
import numpy as np
import pandas as pd
from ai_agent.strategy import AITrader

class TestAITrader(unittest.TestCase):
    """Test cases for AITrader class"""
    
    def setUp(self):
        """Set up test environment"""
        self.trader = AITrader()
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        data = {
            'price': np.random.normal(1000, 50, 100),
            'volume_24h': np.random.normal(500000, 50000, 100),
            'liquidity': np.random.normal(1000000, 100000, 100),
            'timestamp': [d.timestamp() for d in dates],
            'sma_7': None,
            'sma_30': None,
            'rsi': np.random.normal(50, 10, 100)
        }
        self.market_data = pd.DataFrame(data)
        self.market_data['sma_7'] = self.market_data['price'].rolling(window=7, min_periods=1).mean()
        self.market_data['sma_30'] = self.market_data['price'].rolling(window=30, min_periods=1).mean()
        
    def test_preprocess_data(self):
        """Test data preprocessing"""
        processed_data = self.trader.preprocess_data(self.market_data)
        self.assertIn('volume_24h', processed_data.columns)
        self.assertIn('liquidity', processed_data.columns)
        self.assertIn('rsi', processed_data.columns)
        self.assertFalse(processed_data.isnull().any().any())
        
    def test_create_env(self):
        """Test trading environment creation"""
        env = self.trader.create_env(self.market_data)
        observation, info = env.reset()
        self.assertEqual(len(observation), 6)
        self.assertEqual(env.current_balance, 10000.0)
        self.assertEqual(env.current_position, 0.0)
        
    def test_train_and_predict(self):
        """Test model training and prediction"""
        # Train the model
        result = self.trader.train(
            self.market_data,
            total_timesteps=1000,  # Small number for testing
            save_path='test_model.zip'
        )
        self.assertIn('total_timesteps', result)
        self.assertIn('final_reward', result)
        
        # Make predictions
        predictions = self.trader.predict(self.market_data)
        self.assertEqual(len(predictions), len(self.market_data))
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in predictions))
        
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train and save model
        self.trader.train(
            self.market_data,
            total_timesteps=1000,
            save_path='test_model.zip'
        )
        
        # Create new trader and load model
        new_trader = AITrader()
        new_trader.load('test_model.zip')
        
        # Compare predictions
        pred1 = self.trader.predict(self.market_data)
        pred2 = new_trader.predict(self.market_data)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
        
    def test_execute_trade(self):
        """Test trade execution"""
        # Train model first
        self.trader.train(self.market_data, total_timesteps=1000)
        
        # Execute trade
        result = self.trader.execute_trade(1.0)
        self.assertIn('action', result)
        self.assertIn('reward', result)
        self.assertIn('portfolio_value', result)
        self.assertIn('current_price', result)
        
    def test_error_handling(self):
        """Test error handling"""
        # Test executing trade without training
        with self.assertRaises(ValueError):
            self.trader.execute_trade(1.0)
            
        # Test saving model without training
        with self.assertRaises(ValueError):
            self.trader.save('test_model.zip')
            
        # Test getting portfolio value without training
        with self.assertRaises(ValueError):
            self.trader.get_portfolio_value()