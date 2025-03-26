import unittest
import numpy as np
from ai_agent.market_maker import MarketMaker
from ai_agent.strategies.pure_market_maker import PureMarketMaker

class TestMarketMaker(unittest.TestCase):
    """Test cases for MarketMaker class"""
    
    def setUp(self):
        """Set up test environment"""
        self.strategy = PureMarketMaker()
        self.market_maker = MarketMaker(self.strategy)
        
        # Initialize market state
        self.market_state = {
            'price': 1000.0,
            'portfolio_value': 10000.0,
            'position': 0.0,
            'timestamp': 1000
        }
        self.market_maker.update_market_state(self.market_state)
        
    def test_initialization(self):
        """Test market maker initialization"""
        self.assertIsInstance(self.market_maker.strategy, PureMarketMaker)
        
    def test_get_action(self):
        """Test getting trading action"""
        action = self.market_maker.get_action()
        self.assertIsInstance(action, float)
        self.assertTrue(-1.0 <= action <= 1.0)
        
    def test_get_orders(self):
        """Test getting orders"""
        orders = self.market_maker.get_orders()
        self.assertIsInstance(orders, dict)
        self.assertIn('bid_price', orders)
        self.assertIn('ask_price', orders)
        self.assertIn('bid_size', orders)
        self.assertIn('ask_size', orders)
        
    def test_get_spreads(self):
        """Test getting bid-ask spreads"""
        spreads = self.market_maker.get_spreads()
        self.assertIsInstance(spreads, dict)
        self.assertIn('bid_spread', spreads)
        self.assertIn('ask_spread', spreads)
        self.assertTrue(spreads['bid_spread'] >= 0)
        self.assertTrue(spreads['ask_spread'] >= 0)
        
    def test_get_inventory_metrics(self):
        """Test getting inventory metrics"""
        metrics = self.market_maker.get_inventory_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('target', metrics)
        self.assertIn('current', metrics)
        self.assertIn('deviation', metrics)
        
    def test_should_refresh_orders(self):
        """Test order refresh logic"""
        result = self.market_maker.should_refresh_orders()
        self.assertIsInstance(result, bool)
        
    def test_update_market_state(self):
        """Test market state update"""
        new_state = {
            'price': 1100.0,
            'portfolio_value': 11000.0,
            'position': 1.0,
            'timestamp': 2000
        }
        self.market_maker.update_market_state(new_state)
        action = self.market_maker.get_action()
        self.assertIsInstance(action, float)
        self.assertTrue(-1.0 <= action <= 1.0)
        
    def test_uninitialized_state(self):
        """Test error handling for uninitialized state"""
        market_maker = MarketMaker(PureMarketMaker())
        with self.assertRaises(ValueError):
            market_maker.get_action()
            
    def test_invalid_state(self):
        """Test error handling for invalid state"""
        with self.assertRaises(ValueError):
            self.market_maker.update_market_state({})
        with self.assertRaises(ValueError):
            self.market_maker.update_market_state({'price': 'invalid'})

if __name__ == '__main__':
    unittest.main() 