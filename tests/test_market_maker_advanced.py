import unittest
import logging
import numpy as np
from ai_agent.market_maker import MarketMaker
from ai_agent.strategies.pure_market_maker import PureMarketMaker

class TestMarketMakerAdvanced(unittest.TestCase):
    """Advanced test cases for MarketMaker class"""
    
    def setUp(self):
        """Set up test environment"""
        self.strategy = PureMarketMaker()
        self.market_maker = MarketMaker(strategy=self.strategy)
        
        # Initialize market state
        self.market_state = {
            'price': 1000.0,
            'portfolio_value': 10000.0,
            'position': 0.0,
            'timestamp': 1000
        }
        self.market_maker.update_market_state(self.market_state)
        
    def test_market_conditions(self):
        """Test market maker behavior under different market conditions"""
        # Test normal market conditions
        normal_state = {
            'price': 1000.0,
            'portfolio_value': 10000.0,
            'position': 0.0,
            'timestamp': 1000
        }
        self.market_maker.update_market_state(normal_state)
        action = self.market_maker.get_action()
        self.assertTrue(-1.0 <= action <= 1.0)
        
        # Test high volatility
        volatile_state = {
            'price': 1500.0,  # 50% price increase
            'portfolio_value': 15000.0,
            'position': 0.0,
            'timestamp': 2000
        }
        self.market_maker.update_market_state(volatile_state)
        action = self.market_maker.get_action()
        self.assertTrue(-1.0 <= action <= 1.0)
        
        # Test low liquidity
        low_liquidity_state = {
            'price': 1000.0,
            'portfolio_value': 10000.0,
            'position': 5.0,  # Large position relative to portfolio
            'timestamp': 3000
        }
        self.market_maker.update_market_state(low_liquidity_state)
        action = self.market_maker.get_action()
        self.assertTrue(-1.0 <= action <= 1.0)
        
    def test_edge_cases(self):
        """Test edge cases and extreme market conditions"""
        # Test valid edge cases
        edge_cases = [
            # Normal values
            {'price': 1000.0, 'portfolio_value': 10000.0, 'position': 0.0, 'timestamp': 1000},
            
            # Small values
            {'price': 0.01, 'portfolio_value': 100.0, 'position': 0.001, 'timestamp': 1000},
            
            # Large values
            {'price': 1e6, 'portfolio_value': 1e9, 'position': 1e3, 'timestamp': 1000},
            
            # Extreme position
            {'price': 1000.0, 'portfolio_value': 10000.0, 'position': 100.0, 'timestamp': 1000},
            {'price': 1000.0, 'portfolio_value': 10000.0, 'position': -100.0, 'timestamp': 1000}
        ]
        
        for state in edge_cases:
            try:
                self.market_maker.update_market_state(state)
                action = self.market_maker.get_action()
                self.assertTrue(-1.0 <= action <= 1.0)
            except Exception as e:
                self.fail(f"Edge case failed: {state}, Error: {str(e)}")
                
    def test_performance(self):
        """Test market maker performance metrics"""
        # Simulate market making over time
        timestamps = range(1000, 11000, 1000)
        total_pnl = 0.0
        position_changes = []
        
        for t in timestamps:
            state = {
                'price': 1000.0 + np.random.normal(0, 50),
                'portfolio_value': 10000.0 + total_pnl,
                'position': self.market_maker.strategy.last_action,
                'timestamp': t
            }
            self.market_maker.update_market_state(state)
            action = self.market_maker.get_action()
            
            # Track position changes
            position_changes.append(action)
            
            # Simulate PnL (simplified)
            price_change = np.random.normal(0, 10)
            total_pnl += action * price_change
            
        # Check performance metrics
        self.assertGreaterEqual(len(position_changes), len(timestamps))
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in position_changes))
        
    def test_concurrent_operations(self):
        """Test market maker behavior with concurrent updates"""
        # Simulate rapid market updates
        for _ in range(100):
            state = {
                'price': 1000.0 + np.random.normal(0, 50),
                'portfolio_value': 10000.0,
                'position': np.random.normal(0, 1),
                'timestamp': np.random.randint(1000, 10000)
            }
            self.market_maker.update_market_state(state)
            action = self.market_maker.get_action()
            self.assertTrue(-1.0 <= action <= 1.0)
            
    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test invalid market state
        invalid_states = [
            # Empty state
            {},
            
            # Missing required fields
            {'portfolio_value': 10000.0, 'position': 0.0, 'timestamp': 1000},
            {'price': 1000.0, 'position': 0.0, 'timestamp': 1000},
            {'price': 1000.0, 'portfolio_value': 10000.0, 'timestamp': 1000},
            
            # Invalid field types
            {'price': '1000.0', 'portfolio_value': 10000.0, 'position': 0.0, 'timestamp': 1000},
            {'price': 1000.0, 'portfolio_value': '10000.0', 'position': 0.0, 'timestamp': 1000},
            {'price': 1000.0, 'portfolio_value': 10000.0, 'position': '0.0', 'timestamp': 1000}
        ]
        
        for state in invalid_states:
            with self.assertRaises(ValueError):
                self.market_maker.update_market_state(state)
                
        # Test recovery after invalid state
        valid_state = {
            'price': 1000.0,
            'portfolio_value': 10000.0,
            'position': 0.0,
            'timestamp': 1000
        }
        self.market_maker.update_market_state(valid_state)
        action = self.market_maker.get_action()
        self.assertTrue(-1.0 <= action <= 1.0)

if __name__ == '__main__':
    unittest.main() 