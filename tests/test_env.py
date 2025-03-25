import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_agent.trading_env import CryptoTradingEnv

def create_sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = {
        'price': np.random.normal(1000, 50, 100),
        'liquidity': np.random.normal(1000000, 100000, 100),
        'volume_24h': np.random.normal(500000, 50000, 100),
        'timestamp': [d.timestamp() for d in dates]
    }
    df = pd.DataFrame(data)
    df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
    df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
    df['rsi'] = np.random.normal(50, 10, 100)
    return df

class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.market_data = create_sample_market_data()
        self.env = CryptoTradingEnv(self.market_data)

    def test_trading_env_initialization(self):
        """Test trading environment initialization"""
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_balance, 10000)
        self.assertEqual(self.env.current_position, 0)
        self.assertEqual(self.env.entry_price, 0)
        self.assertEqual(self.env.transaction_cost, 0.001)

    def test_trading_env_step_buy(self):
        """Test trading environment buy action"""
        observation, reward, done, info = self.env.step(np.array([1.0]))
        self.assertGreater(self.env.current_position, 0)
        self.assertEqual(self.env.current_balance, 0)
        self.assertEqual(self.env.entry_price, self.market_data.iloc[0]['price'])
        self.assertEqual(reward, 0)

    def test_trading_env_step_sell(self):
        """Test trading environment sell action"""
        # First buy
        self.env.step(np.array([1.0]))
        # Then sell
        observation, reward, done, info = self.env.step(np.array([-1.0]))
        self.assertEqual(self.env.current_position, 0)
        self.assertGreater(self.env.current_balance, 0)
        self.assertNotEqual(reward, 0)  # Should have a profit/loss reward

    def test_trading_env_reset(self):
        """Test trading environment reset"""
        # Execute some steps
        self.env.step(np.array([1.0]))
        self.env.step(np.array([-1.0]))
        # Reset
        observation = self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_balance, 10000)
        self.assertEqual(self.env.current_position, 0)
        self.assertEqual(self.env.entry_price, 0)
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (6,))

    def test_trading_env_observation_space(self):
        """Test trading environment observation space"""
        observation = self.env._get_observation()
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (6,))
        self.assertFalse(np.any(np.isnan(observation)))

    def test_trading_env_action_space(self):
        """Test trading environment action space"""
        self.assertEqual(self.env.action_space.shape, (1,))
        self.assertEqual(self.env.action_space.low[0], -1)
        self.assertEqual(self.env.action_space.high[0], 1)

    def test_trading_env_portfolio_value(self):
        """Test portfolio value calculation"""
        # Buy action
        _, _, _, info = self.env.step(np.array([1.0]))
        # Portfolio value should be initial value minus transaction costs
        expected_value = info['initial_portfolio_value'] * (1 - self.env.transaction_cost)
        self.assertLess(abs(info['portfolio_value'] - expected_value), 1e-6)
        
        # Sell action
        _, _, _, info = self.env.step(np.array([-1.0]))
        self.assertGreater(info['portfolio_value'], 0)

    def test_trading_env_transaction_costs(self):
        """Test transaction costs are applied correctly"""
        initial_balance = self.env.current_balance
        
        # Buy action
        _, _, _, info = self.env.step(np.array([1.0]))
        expected_position = (initial_balance * (1 - self.env.transaction_cost)) / self.market_data.iloc[0]['price']
        self.assertLess(abs(self.env.current_position - expected_position), 1e-6)

if __name__ == '__main__':
    unittest.main() 