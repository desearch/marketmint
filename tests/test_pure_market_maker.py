import unittest
import numpy as np
import logging
from ai_agent.strategies.pure_market_maker import PureMarketMaker

class TestPureMarketMaker(unittest.TestCase):
    def setUp(self):
        # Enable debug logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.strategy = PureMarketMaker(
            target_spread=0.002,  # 0.2% target spread
            min_spread=0.001,     # 0.1% minimum spread
            max_position=100.0,   # Maximum position size
            position_limit=0.5,   # 50% of capital as position limit
            risk_aversion=1.0     # Risk aversion parameter
        )
        
        # Sample market state
        self.market_state = {
            'price': 100.0,
            'portfolio_value': 10000.0,
            'position': 0.0,
            'timestamp': 1000
        }
    
    def test_update_market_state(self):
        """Test market state update"""
        self.strategy.update_market_state(self.market_state)
        self.assertEqual(self.strategy.current_price, 100.0)
        self.assertEqual(self.strategy.current_position, 0.0)
        self.assertEqual(self.strategy.portfolio_value, 10000.0)
    
    def test_calculate_spread(self):
        """Test spread calculation"""
        self.strategy.update_market_state(self.market_state)
        
        # Test base spread
        spread = self.strategy.calculate_spread()
        self.assertEqual(spread, self.strategy.target_spread)
        
        # Test spread with position
        self.strategy.current_position = 50.0  # Half of max position
        spread = self.strategy.calculate_spread()
        self.assertTrue(spread > self.strategy.target_spread)
        self.assertTrue(spread >= self.strategy.min_spread)
    
    def test_calculate_skew(self):
        """Test skew calculation"""
        self.strategy.update_market_state(self.market_state)
        
        # Test no skew at zero position
        skew = self.strategy.calculate_skew()
        self.assertAlmostEqual(skew, 0.0, places=6)
        
        # Test positive skew with long position
        self.strategy.current_position = 50.0  # Half of max position
        skew = self.strategy.calculate_skew()
        self.assertTrue(skew > 0)
        self.assertLess(skew, 1)
        
        # Test negative skew with short position
        self.strategy.current_position = -50.0  # Half of max position short
        skew = self.strategy.calculate_skew()
        self.assertTrue(skew < 0)
        self.assertGreater(skew, -1)
    
    def test_get_orders(self):
        """Test order generation"""
        self.strategy.update_market_state(self.market_state)
        
        # Test with no position
        orders = self.strategy.get_orders()
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]['side'], 'buy')
        self.assertEqual(orders[1]['side'], 'sell')
        
        # Verify order prices
        self.assertTrue(orders[0]['price'] < self.market_state['price'])  # Buy below current price
        self.assertTrue(orders[1]['price'] > self.market_state['price'])  # Sell above current price
        
        # Test with long position
        self.strategy.current_position = 50.0
        orders = self.strategy.get_orders()
        self.assertTrue(orders[1]['amount'] > orders[0]['amount'])  # Should sell more than buy
        
        # Test with short position
        self.strategy.current_position = -50.0
        orders = self.strategy.get_orders()
        self.assertTrue(orders[0]['amount'] > orders[1]['amount'])  # Should buy more than sell
    
    def test_get_action(self):
        """Test action generation"""
        self.strategy.update_market_state(self.market_state)
        
        # Test with no position
        action = self.strategy.get_action()
        self.assertIsInstance(action, float)
        self.assertGreaterEqual(action, -1.0)
        self.assertLessEqual(action, 1.0)
        self.assertAlmostEqual(action, 0.0, places=6)  # Should be close to zero with no position
        
        # Test with long position
        self.strategy.current_position = 50.0  # Half of max position
        action = self.strategy.get_action()
        self.assertTrue(action < 0)  # Should want to reduce position
        
        # Test with short position
        self.strategy.current_position = -50.0  # Half of max position short
        action = self.strategy.get_action()
        self.assertTrue(action > 0)  # Should want to increase position
        
        # Test with no price
        self.strategy.current_price = None
        action = self.strategy.get_action()
        self.assertEqual(action, 0.0)  # Should return 0 when price is None

if __name__ == '__main__':
    unittest.main() 