import unittest
import logging
import time
import concurrent.futures
from typing import Dict, Any
from ai_agent.market_maker import MarketMaker
from ai_agent.strategies import PureMarketMaker

class TestMarketMakerAdvanced(unittest.TestCase):
    def setUp(self):
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
    
    def test_edge_cases(self):
        """Test edge cases and extreme market conditions"""
        # Test valid edge cases
        edge_cases = [
            # Zero values
            {'price': 0.0, 'portfolio_value': 10000.0, 'position': 0.0, 'timestamp': 1000},
            {'price': 100.0, 'portfolio_value': 0.0, 'position': 50.0, 'timestamp': 1000},
            
            # Negative values
            {'price': -100.0, 'portfolio_value': 10000.0, 'position': -50.0, 'timestamp': 1000},
            {'price': 100.0, 'portfolio_value': -10000.0, 'position': 50.0, 'timestamp': 1000},
            
            # Extreme values
            {'price': 1e6, 'portfolio_value': 1e9, 'position': 1e6, 'timestamp': 1000},
            {'price': 1e-6, 'portfolio_value': 1e-3, 'position': 1e6, 'timestamp': 1000},
        ]
        
        for state in edge_cases:
            try:
                self.market_maker.update_market_state(state)
                # Test if we can still generate orders
                orders = self.market_maker.get_orders()
                self.assertIsInstance(orders, list)
                
                # Test if we can still get action
                action = self.market_maker.get_action()
                self.assertIsInstance(action, float)
                
                # Test if we can still get spreads
                spreads = self.market_maker.get_spreads()
                self.assertIsInstance(spreads, dict)
                
            except Exception as e:
                self.logger.error(f"Edge case failed: {state}, Error: {str(e)}")
                raise
        
        # Test invalid states separately
        invalid_states = [
            # Missing data
            {'price': 100.0, 'portfolio_value': 10000.0, 'timestamp': 1000},  # Missing position
            {'price': 100.0, 'position': 50.0, 'timestamp': 1000},  # Missing portfolio_value
        ]
        
        for state in invalid_states:
            with self.assertRaises(ValueError):
                self.market_maker.update_market_state(state)
    
    def test_performance(self):
        """Test performance under load"""
        # Test response time
        start_time = time.time()
        for _ in range(1000):
            self.market_maker.update_market_state(self.market_state)
            self.market_maker.get_orders()
            self.market_maker.get_action()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 1000
        self.logger.info(f"Average operation time: {avg_time:.6f} seconds")
        self.assertLess(avg_time, 0.01)  # Should be under 10ms per operation
        
        # Test memory usage
        import sys
        initial_size = sys.getsizeof(self.market_maker)
        self.logger.info(f"Initial object size: {initial_size} bytes")
        
        # Test with large number of orders
        for _ in range(100):
            self.market_maker.update_market_state(self.market_state)
            orders = self.market_maker.get_orders()
            self.assertLess(len(orders), 100)  # Should not generate too many orders
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        def operation():
            self.market_maker.update_market_state(self.market_state)
            orders = self.market_maker.get_orders()
            action = self.market_maker.get_action()
            return orders, action
        
        # Run 10 concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(operation) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), 10)
        for orders, action in results:
            self.assertIsInstance(orders, list)
            self.assertIsInstance(action, float)
    
    def test_market_conditions(self):
        """Test different market conditions"""
        market_conditions = [
            # Normal market
            {'price': 100.0, 'portfolio_value': 10000.0, 'position': 50.0, 'timestamp': 1000},
            
            # High volatility
            {'price': 150.0, 'portfolio_value': 15000.0, 'position': 75.0, 'timestamp': 1001},
            {'price': 50.0, 'portfolio_value': 5000.0, 'position': 25.0, 'timestamp': 1002},
            
            # Low liquidity
            {'price': 100.0, 'portfolio_value': 1000.0, 'position': 5.0, 'timestamp': 1003},
            
            # High inventory
            {'price': 100.0, 'portfolio_value': 10000.0, 'position': 95.0, 'timestamp': 1004},
            
            # Low inventory
            {'price': 100.0, 'portfolio_value': 10000.0, 'position': 5.0, 'timestamp': 1005},
        ]
        
        for state in market_conditions:
            self.market_maker.update_market_state(state)
            
            # Test order generation
            orders = self.market_maker.get_orders()
            self.assertIsInstance(orders, list)
            self.assertEqual(len(orders), 2)  # Should always have buy and sell orders
            
            # Test action generation
            action = self.market_maker.get_action()
            self.assertIsInstance(action, float)
            self.assertGreaterEqual(action, -1.0)
            self.assertLessEqual(action, 1.0)
            
            # Test spread adjustment
            spreads = self.market_maker.get_spreads()
            self.assertGreaterEqual(spreads['bid_spread'], self.market_maker.strategy.min_spread)
            self.assertGreaterEqual(spreads['ask_spread'], self.market_maker.strategy.min_spread)
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test invalid market state
        invalid_states = [
            {},  # Empty state
            {'price': 'invalid', 'portfolio_value': 10000.0, 'position': 50.0, 'timestamp': 1000},
            {'price': 100.0, 'portfolio_value': 'invalid', 'position': 50.0, 'timestamp': 1000},
            {'price': 100.0, 'portfolio_value': 10000.0, 'position': 'invalid', 'timestamp': 1000},
            {'price': 100.0, 'portfolio_value': 10000.0, 'position': 50.0, 'timestamp': 'invalid'},
        ]
        
        for state in invalid_states:
            with self.assertRaises(Exception):
                self.market_maker.update_market_state(state)
        
        # Test recovery after error
        self.market_maker.update_market_state(self.market_state)
        orders = self.market_maker.get_orders()
        self.assertIsInstance(orders, list)
        self.assertEqual(len(orders), 2)

if __name__ == '__main__':
    unittest.main() 