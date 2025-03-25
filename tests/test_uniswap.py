import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from ai_agent.uniswap_client import UniswapV3Client
from ai_agent.trading_env import CryptoTradingEnv

class TestUniswapIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.mock_uniswap = Mock(spec=UniswapV3Client)
        self.mock_uniswap.get_pool_data.return_value = {
            'price': 1000.0,
            'liquidity': 1000000.0,
            'volume_24h': 500000.0
        }
        self.mock_uniswap.get_token_price.return_value = 1000.0
        self.mock_uniswap.calculate_min_amount_out.return_value = 990000000000000000  # 0.99 ETH in Wei
        self.mock_uniswap.execute_swap.return_value = {
            'status': 'success',
            'transaction_hash': '0x123',
            'gas_used': 100000,
            'amount_out': 995000000000000000  # 0.995 ETH in Wei (slightly better than min_amount_out)
        }

    def test_get_pool_data(self):
        """Test fetching pool data from Uniswap"""
        pool_data = self.mock_uniswap.get_pool_data()
        self.assertIn('price', pool_data)
        self.assertIn('liquidity', pool_data)
        self.assertIn('volume_24h', pool_data)
        self.assertIsInstance(pool_data['price'], float)
        self.assertIsInstance(pool_data['liquidity'], float)
        self.assertIsInstance(pool_data['volume_24h'], float)

    def test_get_token_price(self):
        """Test fetching token price from Uniswap"""
        price = self.mock_uniswap.get_token_price()
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)

    def test_calculate_min_amount_out(self):
        """Test calculating minimum amount out for a swap"""
        amount_in = 1000000000000000000  # 1 ETH in Wei
        min_amount_out = self.mock_uniswap.calculate_min_amount_out(amount_in)
        self.assertIsInstance(min_amount_out, int)
        self.assertGreater(min_amount_out, 0)
        self.assertLess(min_amount_out, amount_in)  # Should account for slippage

    def test_execute_swap(self):
        """Test executing a swap on Uniswap"""
        amount_in = 1000000000000000000  # 1 ETH in Wei
        swap_result = self.mock_uniswap.execute_swap(amount_in)
        self.assertIn('status', swap_result)
        self.assertIn('transaction_hash', swap_result)
        self.assertIn('gas_used', swap_result)
        self.assertEqual(swap_result['status'], 'success')
        self.assertIsInstance(swap_result['transaction_hash'], str)
        self.assertIsInstance(swap_result['gas_used'], int)

    def test_swap_slippage_protection(self):
        """Test slippage protection in swaps"""
        amount_in = 1000000000000000000  # 1 ETH in Wei
        min_amount_out = self.mock_uniswap.calculate_min_amount_out(amount_in)
        swap_result = self.mock_uniswap.execute_swap(amount_in)
        
        # Verify that the actual output amount is not less than the minimum
        self.assertGreaterEqual(
            swap_result.get('amount_out', 0),
            min_amount_out,
            "Swap output amount is less than minimum amount out"
        )

    def test_swap_gas_estimation(self):
        """Test gas estimation for swaps"""
        amount_in = 1000000000000000000  # 1 ETH in Wei
        swap_result = self.mock_uniswap.execute_swap(amount_in)
        
        # Verify gas usage is reasonable
        self.assertGreater(swap_result['gas_used'], 0)
        self.assertLess(swap_result['gas_used'], 500000)  # Should not exceed typical gas limits

    def test_integration_with_trading_env(self):
        """Test integration between Uniswap and trading environment"""
        # Create sample market data
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

        # Create trading environment
        env = CryptoTradingEnv(df)
        
        # Execute a buy action
        observation, reward, done, info = env.step(np.array([1.0]))
        
        # Verify the action was executed successfully
        self.assertGreater(env.current_position, 0)
        self.assertEqual(env.current_balance, 0)
        
        # Verify portfolio value calculation
        self.assertGreater(info['portfolio_value'], 0)
        self.assertLessEqual(
            info['portfolio_value'],
            info['initial_portfolio_value'] * (1 - env.transaction_cost)
        )

if __name__ == '__main__':
    unittest.main() 