import unittest
import logging
from ai_agent.market_maker import MarketMaker
from ai_agent.strategies import PureMarketMaker

class TestMarketMaker(unittest.TestCase):
    def setUp(self):
        # Enable debug logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create market maker with default strategy
        self.market_maker = MarketMaker(
            bid_spread=0.01,
            ask_spread=0.01,
            min_spread=0.002,
            order_refresh_time=60,
            inventory_target_base_pct=0.5,
            inventory_range_multiplier=1.0,
            risk_factor=0.5
        )
        
        # Sample market state
        self.market_state = {
            'price': 100.0,
            'portfolio_value': 10000.0,
            'position': 50.0,
            'timestamp': 1000
        }
    
    def test_initialization(self):
        """Test market maker initialization"""
        self.assertIsInstance(self.market_maker.strategy, PureMarketMaker)
        
        # Test with pre-configured strategy
        strategy = PureMarketMaker(bid_spread=0.02)
        market_maker = MarketMaker(strategy=strategy)
        self.assertEqual(market_maker.strategy.bid_spread, 0.02)
    
    def test_update_market_state(self):
        """Test market state updates"""
        self.market_maker.update_market_state(self.market_state)
        self.assertEqual(self.market_maker.current_state, self.market_state)
    
    def test_get_orders(self):
        """Test order generation"""
        self.market_maker.update_market_state(self.market_state)
        orders = self.market_maker.get_orders()
        
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]['side'], 'buy')
        self.assertEqual(orders[1]['side'], 'sell')
        self.assertIn('price', orders[0])
        self.assertIn('amount', orders[0])
    
    def test_get_action(self):
        """Test action generation"""
        self.market_maker.update_market_state(self.market_state)
        action = self.market_maker.get_action()
        
        self.assertIsInstance(action, float)
        self.assertGreaterEqual(action, -1.0)
        self.assertLessEqual(action, 1.0)
    
    def test_should_refresh_orders(self):
        """Test order refresh logic"""
        self.market_maker.update_market_state(self.market_state)
        should_refresh = self.market_maker.should_refresh_orders()
        
        self.assertIsInstance(should_refresh, bool)
    
    def test_get_spreads(self):
        """Test spread calculation"""
        self.market_maker.update_market_state(self.market_state)
        spreads = self.market_maker.get_spreads()
        
        self.assertIn('bid_spread', spreads)
        self.assertIn('ask_spread', spreads)
        self.assertGreaterEqual(spreads['bid_spread'], self.market_maker.strategy.min_spread)
        self.assertGreaterEqual(spreads['ask_spread'], self.market_maker.strategy.min_spread)
    
    def test_get_inventory_metrics(self):
        """Test inventory metrics calculation"""
        self.market_maker.update_market_state(self.market_state)
        metrics = self.market_maker.get_inventory_metrics()
        
        self.assertIn('target_position', metrics)
        self.assertIn('current_position', metrics)
        self.assertIn('inventory_bias', metrics)
        self.assertEqual(metrics['current_position'], self.market_state['position'])
    
    def test_uninitialized_state(self):
        """Test error handling for uninitialized state"""
        with self.assertRaises(ValueError):
            self.market_maker.get_orders()
        
        with self.assertRaises(ValueError):
            self.market_maker.get_action()
        
        with self.assertRaises(ValueError):
            self.market_maker.should_refresh_orders()
        
        with self.assertRaises(ValueError):
            self.market_maker.get_spreads()
        
        with self.assertRaises(ValueError):
            self.market_maker.get_inventory_metrics()

if __name__ == '__main__':
    unittest.main() 