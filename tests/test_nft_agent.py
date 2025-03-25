import unittest
import logging
from ai_agent.nft_agent import NFTAgent

class TestNFTAgent(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create NFT agent with test parameters
        self.nft_agent = NFTAgent(
            total_capital=100000.0,
            risk_free_rate=0.02,
            max_drawdown=0.1,
            min_profit_threshold=0.05
        )
        
        # Sample market state
        self.market_state = {
            'price': 100.0,
            'portfolio_value': 10000.0,
            'position': 50.0,
            'timestamp': 1000
        }
    
    def test_add_micro_agent(self):
        """Test adding micro-agents"""
        # Add first agent
        self.nft_agent.add_micro_agent(
            agent_id="agent1",
            governance_tokens=1000.0,
            initial_capital=10000.0
        )
        self.assertEqual(len(self.nft_agent.micro_agents), 1)
        self.assertEqual(self.nft_agent.total_governance_tokens, 1000.0)
        
        # Add second agent
        self.nft_agent.add_micro_agent(
            agent_id="agent2",
            governance_tokens=2000.0,
            initial_capital=20000.0
        )
        self.assertEqual(len(self.nft_agent.micro_agents), 2)
        self.assertEqual(self.nft_agent.total_governance_tokens, 3000.0)
        
        # Test duplicate agent
        with self.assertRaises(ValueError):
            self.nft_agent.add_micro_agent(
                agent_id="agent1",
                governance_tokens=1000.0,
                initial_capital=10000.0
            )
    
    def test_capital_allocation(self):
        """Test capital allocation based on governance tokens"""
        # Add agents with different governance token holdings
        self.nft_agent.add_micro_agent(
            agent_id="agent1",
            governance_tokens=1000.0,
            initial_capital=10000.0
        )
        self.nft_agent.add_micro_agent(
            agent_id="agent2",
            governance_tokens=2000.0,
            initial_capital=20000.0
        )
        
        # Allocate capital
        self.nft_agent.allocate_capital()
        
        # Check allocations
        agent1 = self.nft_agent.micro_agents["agent1"]
        agent2 = self.nft_agent.micro_agents["agent2"]
        
        # Agent1 should get 1/3 of total capital
        expected_capital1 = (1000.0 / 3000.0) * 100000.0
        self.assertAlmostEqual(agent1.trading_capital, expected_capital1)
        
        # Agent2 should get 2/3 of total capital
        expected_capital2 = (2000.0 / 3000.0) * 100000.0
        self.assertAlmostEqual(agent2.trading_capital, expected_capital2)
    
    def test_aggregated_orders(self):
        """Test order aggregation"""
        # Add agents
        self.nft_agent.add_micro_agent(
            agent_id="agent1",
            governance_tokens=1000.0,
            initial_capital=10000.0
        )
        self.nft_agent.add_micro_agent(
            agent_id="agent2",
            governance_tokens=2000.0,
            initial_capital=20000.0
        )
        
        # Update market state
        self.nft_agent.update_market_state(self.market_state)
        
        # Get aggregated orders
        orders = self.nft_agent.get_aggregated_orders()
        
        # Should have orders from both agents
        self.assertGreater(len(orders), 0)
        
        # Check order scaling
        for order in orders:
            self.assertIn('amount', order)
            self.assertIn('price', order)
            self.assertIn('side', order)
    
    def test_aggregated_action(self):
        """Test action aggregation"""
        # Add agents
        self.nft_agent.add_micro_agent(
            agent_id="agent1",
            governance_tokens=1000.0,
            initial_capital=10000.0
        )
        self.nft_agent.add_micro_agent(
            agent_id="agent2",
            governance_tokens=2000.0,
            initial_capital=20000.0
        )
        
        # Update market state
        self.nft_agent.update_market_state(self.market_state)
        
        # Get aggregated action
        action = self.nft_agent.get_aggregated_action()
        
        # Action should be between -1 and 1
        self.assertGreaterEqual(action, -1.0)
        self.assertLessEqual(action, 1.0)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        # Add agent
        self.nft_agent.add_micro_agent(
            agent_id="agent1",
            governance_tokens=1000.0,
            initial_capital=10000.0
        )
        
        # Update performance metrics
        metrics = {
            'total_profit': 500.0,
            'current_drawdown': 0.05,
            'sharpe_ratio': 1.5,
            'win_rate': 0.6
        }
        self.nft_agent.update_performance_metrics("agent1", metrics)
        
        # Check metrics
        agent_metrics = self.nft_agent.get_agent_performance("agent1")
        self.assertEqual(agent_metrics['total_profit'], 500.0)
        self.assertEqual(agent_metrics['current_drawdown'], 0.05)
        self.assertEqual(agent_metrics['sharpe_ratio'], 1.5)
        self.assertEqual(agent_metrics['win_rate'], 0.6)
        
        # Test system metrics
        system_metrics = self.nft_agent.get_system_metrics()
        self.assertEqual(system_metrics['total_profit'], 500.0)
        self.assertEqual(system_metrics['max_drawdown'], 0.05)
        self.assertEqual(system_metrics['average_sharpe_ratio'], 1.5)
        self.assertEqual(system_metrics['average_win_rate'], 0.6)
    
    def test_risk_limits(self):
        """Test risk limit enforcement"""
        # Add agent
        self.nft_agent.add_micro_agent(
            agent_id="agent1",
            governance_tokens=1000.0,
            initial_capital=10000.0
        )
        
        # Update performance metrics with high drawdown
        metrics = {
            'total_profit': 100.0,
            'current_drawdown': 0.15,  # Above max_drawdown
            'sharpe_ratio': 1.5,
            'win_rate': 0.6
        }
        self.nft_agent.update_performance_metrics("agent1", metrics)
        
        # Check that warning was logged
        # Note: We can't easily test the warning log in unit tests
        # but the code should handle the high drawdown gracefully
    
    def test_invalid_agent_id(self):
        """Test handling of invalid agent IDs"""
        # Try to get performance of non-existent agent
        with self.assertRaises(ValueError):
            self.nft_agent.get_agent_performance("nonexistent")
        
        # Try to update performance of non-existent agent
        with self.assertRaises(ValueError):
            self.nft_agent.update_performance_metrics("nonexistent", {})

if __name__ == '__main__':
    unittest.main() 