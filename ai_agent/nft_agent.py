from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from .market_maker import MarketMaker
from .strategies import PureMarketMaker

@dataclass
class MicroAgent:
    """Represents a micro-agent with its governance token holdings and trading parameters"""
    id: str
    governance_tokens: float
    trading_capital: float
    strategy: MarketMaker
    performance_metrics: Dict[str, float]
    risk_limits: Dict[str, float]

class NFTAgent:
    """
    NFT Agent that manages multiple micro-agents for trading.
    Uses governance tokens to allocate trading capital and manage risk.
    """
    
    def __init__(
        self,
        total_capital: float,
        risk_free_rate: float = 0.02,  # 2% risk-free rate
        max_drawdown: float = 0.1,      # 10% maximum drawdown
        min_profit_threshold: float = 0.05,  # 5% minimum profit threshold
    ):
        self.logger = logging.getLogger(__name__)
        self.total_capital = total_capital
        self.risk_free_rate = risk_free_rate
        self.max_drawdown = max_drawdown
        self.min_profit_threshold = min_profit_threshold
        
        # Initialize micro-agents
        self.micro_agents: Dict[str, MicroAgent] = {}
        self.total_governance_tokens = 0.0
        
        self.logger.info("NFT Agent initialized with total capital: %f", total_capital)
    
    def add_micro_agent(
        self,
        agent_id: str,
        governance_tokens: float,
        initial_capital: float,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new micro-agent to the system.
        
        Args:
            agent_id: Unique identifier for the micro-agent
            governance_tokens: Number of governance tokens held
            initial_capital: Initial trading capital (used for risk limits only)
            strategy_params: Optional parameters for the trading strategy
        """
        if agent_id in self.micro_agents:
            raise ValueError(f"Micro-agent {agent_id} already exists")
        
        # Create strategy with default or custom parameters
        strategy = MarketMaker(**(strategy_params or {}))
        
        # Calculate proportional capital based on governance tokens
        total_tokens = self.total_governance_tokens + governance_tokens
        proportional_capital = (governance_tokens / total_tokens) * self.total_capital if total_tokens > 0 else 0.0
        
        # Create micro-agent with zero initial trading capital
        micro_agent = MicroAgent(
            id=agent_id,
            governance_tokens=governance_tokens,
            trading_capital=0.0,  # Start with zero capital
            strategy=strategy,
            performance_metrics={
                'total_profit': 0.0,
                'current_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0
            },
            risk_limits={
                'max_position_size': proportional_capital,  # Risk limit based on proportional capital
                'max_daily_loss': proportional_capital * 0.02,   # 2% of proportional capital
                'min_profit_target': proportional_capital * self.min_profit_threshold
            }
        )
        
        self.micro_agents[agent_id] = micro_agent
        self.total_governance_tokens += governance_tokens
        
        self.logger.info(
            "Added micro-agent %s with %f governance tokens (proportional capital: %f)",
            agent_id, governance_tokens, proportional_capital
        )
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """
        Update market state for all micro-agents.
        
        Args:
            state: Current market state
        """
        for agent in self.micro_agents.values():
            agent.strategy.update_market_state(state)
    
    def allocate_capital(self) -> None:
        """Reallocate trading capital based on governance tokens"""
        total_tokens = sum(agent.governance_tokens for agent in self.micro_agents.values())
        if total_tokens == 0:
            return

        # Calculate proportional capital for each agent
        for agent in self.micro_agents.values():
            proportional_capital = (agent.governance_tokens / total_tokens) * self.total_capital
            agent.trading_capital = proportional_capital
            # Update risk limits based on new proportional capital
            agent.risk_limits.update({
                'max_position_size': proportional_capital,
                'max_daily_loss': proportional_capital * 0.02,
                'min_profit_target': proportional_capital * self.min_profit_threshold
            })
        
        self.logger.debug(
            "Capital allocation complete. Total allocated: %f",
            sum(agent.trading_capital for agent in self.micro_agents.values())
        )
    
    def get_aggregated_orders(self) -> List[Dict[str, Any]]:
        """
        Get aggregated orders from all micro-agents.
        Orders are scaled by each agent's trading capital.
        """
        aggregated_orders = []
        
        for agent in self.micro_agents.values():
            # Get orders from agent's strategy
            orders = agent.strategy.get_orders()
            
            # Scale order amounts by agent's capital allocation
            for order in orders:
                scaled_order = order.copy()
                scaled_order['amount'] *= (agent.trading_capital / self.total_capital)
                aggregated_orders.append(scaled_order)
        
        return aggregated_orders
    
    def get_aggregated_action(self) -> float:
        """
        Get aggregated trading action from all micro-agents.
        Actions are weighted by each agent's trading capital.
        """
        weighted_action = 0.0
        total_weight = 0.0
        
        for agent in self.micro_agents.values():
            action = agent.strategy.get_action()
            weight = agent.trading_capital / self.total_capital
            
            weighted_action += action * weight
            total_weight += weight
        
        return weighted_action / total_weight if total_weight > 0 else 0.0
    
    def update_performance_metrics(self, agent_id: str, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a micro-agent.
        
        Args:
            agent_id: ID of the micro-agent
            metrics: New performance metrics
        """
        if agent_id not in self.micro_agents:
            raise ValueError(f"Micro-agent {agent_id} not found")
            
        agent = self.micro_agents[agent_id]
        agent.performance_metrics.update(metrics)
        
        # Check if agent should be deactivated due to poor performance
        if (agent.performance_metrics['current_drawdown'] > self.max_drawdown or
            agent.performance_metrics['total_profit'] < agent.risk_limits['min_profit_target']):
            self.logger.warning(
                "Micro-agent %s performance below threshold. Current drawdown: %f, Total profit: %f",
                agent_id,
                agent.performance_metrics['current_drawdown'],
                agent.performance_metrics['total_profit']
            )
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific micro-agent.
        
        Args:
            agent_id: ID of the micro-agent
            
        Returns:
            Dictionary of performance metrics
        """
        if agent_id not in self.micro_agents:
            raise ValueError(f"Micro-agent {agent_id} not found")
            
        return self.micro_agents[agent_id].performance_metrics
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get aggregated performance metrics for all agents"""
        if not self.micro_agents:
            return {
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'average_sharpe_ratio': 0.0,
                'average_win_rate': 0.0
            }

        total_profit = sum(agent.performance_metrics['total_profit'] for agent in self.micro_agents.values())
        max_drawdown = max(agent.performance_metrics['current_drawdown'] for agent in self.micro_agents.values())
        avg_sharpe = sum(agent.performance_metrics['sharpe_ratio'] for agent in self.micro_agents.values()) / len(self.micro_agents)
        avg_win_rate = sum(agent.performance_metrics['win_rate'] for agent in self.micro_agents.values()) / len(self.micro_agents)

        return {
            'total_profit': round(total_profit, 2),
            'max_drawdown': round(max_drawdown, 2),
            'average_sharpe_ratio': round(avg_sharpe, 2),
            'average_win_rate': round(avg_win_rate, 2)
        } 