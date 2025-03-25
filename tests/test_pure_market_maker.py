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
            bid_spread=0.01,
            ask_spread=0.01,
            min_spread=0.002,
            order_refresh_time=60,
            inventory_target_base_pct=0.5,
            inventory_range_multiplier=1.0,
            risk_factor=0.5
        )
        
        # Sample market state
        self.env_state = {
            'price': 100.0,
            'portfolio_value': 10000.0,
            'position': 50.0,
            'timestamp': 1000
        }
    
    def test_calculate_target_inventory(self):
        """Test target inventory calculation"""
        portfolio_value = 10000.0
        current_price = 100.0
        target = self.strategy.calculate_target_inventory(portfolio_value, current_price)
        expected = (portfolio_value * 0.5) / current_price  # 50% of portfolio in base currency
        self.assertEqual(target, expected)
    
    def test_calculate_inventory_bias(self):
        """Test inventory bias calculation"""
        # Test with current position equal to target
        bias = self.strategy.calculate_inventory_bias(50.0, 50.0)
        self.assertEqual(bias, 0.0)
        
        # Test with excess position
        bias = self.strategy.calculate_inventory_bias(75.0, 50.0)
        self.assertTrue(bias > 0)
        self.assertLessEqual(bias, 1.0)
        
        # Test with deficit position
        bias = self.strategy.calculate_inventory_bias(25.0, 50.0)
        self.assertTrue(bias < 0)
        self.assertGreaterEqual(bias, -1.0)
    
    def test_adjust_spreads(self):
        """Test spread adjustment based on inventory bias"""
        # Test with no bias
        bid_spread, ask_spread = self.strategy.adjust_spreads(0.01, 0.01, 0.0)
        self.assertEqual(bid_spread, 0.01)
        self.assertEqual(ask_spread, 0.01)
        
        # Test with positive bias (excess inventory)
        bid_spread, ask_spread = self.strategy.adjust_spreads(0.01, 0.01, 0.5)
        self.assertTrue(bid_spread > 0.01)  # Wider bid spread
        self.assertTrue(ask_spread < 0.01)  # Tighter ask spread
        
        # Test minimum spread enforcement
        bid_spread, ask_spread = self.strategy.adjust_spreads(0.001, 0.001, 0.0)
        self.assertEqual(bid_spread, self.strategy.min_spread)
        self.assertEqual(ask_spread, self.strategy.min_spread)
    
    def test_calculate_order_prices(self):
        """Test order price calculation"""
        mid_price = 100.0
        
        # Test with no inventory bias
        bid_price, ask_price = self.strategy.calculate_order_prices(mid_price, 0.0)
        self.assertEqual(bid_price, mid_price * 0.99)  # 1% below mid price
        self.assertEqual(ask_price, mid_price * 1.01)  # 1% above mid price
        
        # Test with price limits
        self.strategy.price_floor = 98.0
        self.strategy.price_ceiling = 102.0
        bid_price, ask_price = self.strategy.calculate_order_prices(mid_price, 0.0)
        self.assertEqual(bid_price, 98.0)
        self.assertEqual(ask_price, 102.0)
    
    def test_calculate_order_amounts(self):
        """Test order amount calculation"""
        # Test with fixed order amount
        self.strategy.order_amount = 1.0
        bid_amount, ask_amount = self.strategy.calculate_order_amounts(10000.0, 100.0, 50.0)
        self.assertEqual(bid_amount, 1.0)
        self.assertEqual(ask_amount, 1.0)
        
        # Test with dynamic order amounts
        self.strategy.order_amount = None
        bid_amount, ask_amount = self.strategy.calculate_order_amounts(10000.0, 100.0, 25.0)
        self.assertTrue(bid_amount > ask_amount)  # Should buy more when below target
        
        bid_amount, ask_amount = self.strategy.calculate_order_amounts(10000.0, 100.0, 75.0)
        self.assertTrue(ask_amount > bid_amount)  # Should sell more when above target
    
    def test_should_refresh_orders(self):
        """Test order refresh logic"""
        self.strategy.last_order_refresh = 0
        self.assertTrue(self.strategy.should_refresh_orders(61))
        self.assertFalse(self.strategy.should_refresh_orders(59))
    
    def test_generate_orders(self):
        """Test order generation"""
        orders = self.strategy.generate_orders(100.0, 10000.0, 50.0, 1000)
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]['side'], 'buy')
        self.assertEqual(orders[1]['side'], 'sell')
        
        # Test order caching
        same_orders = self.strategy.generate_orders(100.0, 10000.0, 50.0, 1030)
        self.assertEqual(orders, same_orders)  # Should return cached orders
        
        new_orders = self.strategy.generate_orders(100.0, 10000.0, 50.0, 1061)
        self.assertNotEqual(orders, new_orders)  # Should generate new orders
    
    def test_get_action(self):
        """Test action generation"""
        # Test initial state
        action = self.strategy.get_action(self.env_state)
        self.logger.debug(f"Initial action: {action}")
        self.assertIsInstance(action, float)
        self.assertGreaterEqual(action, -1.0)
        self.assertLessEqual(action, 1.0)
        
        # Test with excess position (50% above target)
        target = self.strategy.calculate_target_inventory(10000.0, 100.0)
        self.logger.debug(f"Target position: {target}")
        
        self.env_state['position'] = 75.0  # Above target
        self.logger.debug(f"Testing with position {self.env_state['position']} (target: {target})")
        action = self.strategy.get_action(self.env_state)
        self.logger.debug(f"Action for excess position: {action}")
        self.assertTrue(action < 0, f"Expected negative action for excess position, got {action}")
        
        # Test with deficit position (50% below target)
        self.env_state['position'] = 25.0  # Below target
        self.logger.debug(f"Testing with position {self.env_state['position']} (target: {target})")
        action = self.strategy.get_action(self.env_state)
        self.logger.debug(f"Action for deficit position: {action}")
        self.assertTrue(action > 0, f"Expected positive action for deficit position, got {action}")

if __name__ == '__main__':
    unittest.main() 