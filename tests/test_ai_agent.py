import unittest
import pandas as pd
import numpy as np
from ai_agent.strategy import AITrader
from ai_agent.trading_env import CryptoTradingEnv

def create_sample_market_data(n_steps: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)
    
    # Generate random walk prices
    price = 100 * np.ones(n_steps)
    returns = np.random.normal(0, 0.02, n_steps)
    for i in range(1, n_steps):
        price[i] = price[i-1] * (1 + returns[i])
    
    # Create DataFrame with price and volume
    df = pd.DataFrame({
        'price': price,
        'volume': np.random.uniform(1000, 10000, n_steps)
    })
    
    return df

class TestAITrader(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.market_data = create_sample_market_data()
        self.trader = AITrader(initial_balance=10000, transaction_cost=0.001)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        processed_data = self.trader.preprocess_data(self.market_data)
        
        # Check if technical indicators are calculated
        self.assertIn('sma_7', processed_data.columns)
        self.assertIn('sma_30', processed_data.columns)
        self.assertIn('rsi', processed_data.columns)
        
        # Check if indicators have correct values
        self.assertEqual(len(processed_data), len(self.market_data))
        self.assertTrue(all(processed_data['rsi'].between(0, 100)))
    
    def test_create_env(self):
        """Test trading environment creation"""
        env = self.trader.create_env(self.market_data)
        
        # Check if environment is created correctly
        self.assertIsInstance(env, CryptoTradingEnv)
        self.assertEqual(env.initial_balance, self.trader.initial_balance)
        self.assertEqual(env.transaction_cost, self.trader.transaction_cost)
    
    def test_train_and_predict(self):
        """Test model training and prediction"""
        # Train the model
        result = self.trader.train(
            self.market_data,
            total_timesteps=1000,  # Small number for testing
            save_path='test_model.zip'
        )
        
        # Check training results
        self.assertIn('training_steps', result)
        self.assertIn('final_portfolio_value', result)
        
        # Make predictions
        predictions = self.trader.predict(self.market_data)
        
        # Check predictions format
        self.assertTrue(len(predictions) > 0)
        self.assertIn('step', predictions[0])
        self.assertIn('action', predictions[0])
        self.assertIn('portfolio_value', predictions[0])
        self.assertIn('position', predictions[0])
        self.assertIn('balance', predictions[0])
        self.assertIn('price', predictions[0])
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train and save model
        self.trader.train(
            self.market_data,
            total_timesteps=1000,
            save_path='test_model.zip'
        )
        
        # Create new trader instance and load model
        new_trader = AITrader(initial_balance=10000, transaction_cost=0.001)
        new_trader.load_model('test_model.zip')
        
        # Make predictions with both models
        pred1 = self.trader.predict(self.market_data)
        pred2 = new_trader.predict(self.market_data)
        
        # Check if predictions are the same
        self.assertEqual(len(pred1), len(pred2))
        self.assertEqual(pred1[0]['action'], pred2[0]['action'])
    
    def tearDown(self):
        """Clean up after tests"""
        import os
        if os.path.exists('test_model.zip'):
            os.remove('test_model.zip')