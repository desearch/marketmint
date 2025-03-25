from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from .market_maker import MarketMaker
from .strategies import PureMarketMaker

@dataclass
class Agent:
    """Represents a trading agent with its strategy and parameters"""
    id: str
    strategy: MarketMaker
    performance_metrics: Dict[str, float]
    risk_limits: Dict[str, float]

class StrategyRunner:
    """
    Strategy Runner that manages trading strategies and their agents.
    Each strategy can have multiple agents executing it.
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
        
        # Initialize strategies and their agents
        self.strategies: Dict[str, List[Agent]] = {}
        self.total_governance_tokens = 0.0
        
        self.logger.info("Strategy Runner initialized with total capital: %f", total_capital)
    
    def add_strategy(
        self,
        strategy_id: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new trading strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_params: Optional parameters for the trading strategy
        """
        if strategy_id in self.strategies:
            raise ValueError(f"Strategy {strategy_id} already exists")
        
        # Initialize empty list of agents for this strategy
        self.strategies[strategy_id] = []
        
        self.logger.info("Added strategy %s", strategy_id)
    
    def add_agent(
        self,
        strategy_id: str,
        agent_id: str,
        governance_tokens: float,
        risk_limits: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add a new agent to a strategy.
        
        Args:
            strategy_id: ID of the strategy
            agent_id: ID of the agent
            governance_tokens: Number of governance tokens held
            risk_limits: Optional risk limits for the agent
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Create agent with strategy's market maker
        if not self.strategies[strategy_id]:
            # Initialize strategy and market maker if none exist
            strategy = PureMarketMaker()
            market_maker = MarketMaker()
            market_maker.set_strategy(strategy)  # This will also set market_maker in strategy
            
            agent = Agent(
                id=agent_id,
                strategy=market_maker,  # Store the market maker instead of strategy
                performance_metrics={
                    'total_profit': 0.0,
                    'current_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'governance_tokens': governance_tokens
                },
                risk_limits=risk_limits or {
                    'max_position_size': self.total_capital * 0.1,  # 10% of total capital
                    'max_daily_loss': self.total_capital * 0.02,   # 2% of total capital
                    'min_profit_target': self.total_capital * self.min_profit_threshold
                }
            )
            self.strategies[strategy_id].append(agent)
        else:
            # Use existing strategy from first agent
            market_maker = self.strategies[strategy_id][0].strategy  # Get existing market maker
            agent = Agent(
                id=agent_id,
                strategy=market_maker,  # Store the market maker
                performance_metrics={
                    'total_profit': 0.0,
                    'current_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'governance_tokens': governance_tokens
                },
                risk_limits=risk_limits or {
                    'max_position_size': self.total_capital * 0.1,  # 10% of total capital
                    'max_daily_loss': self.total_capital * 0.02,   # 2% of total capital
                    'min_profit_target': self.total_capital * self.min_profit_threshold
                }
            )
            self.strategies[strategy_id].append(agent)
        
        self.total_governance_tokens += governance_tokens
        
        self.logger.info(
            "Added agent %s to strategy %s with %f governance tokens",
            agent_id, strategy_id, governance_tokens
        )
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """
        Update market state for all strategies and their agents.
        
        Args:
            state: Current market state
        """
        for agents in self.strategies.values():
            for agent in agents:
                agent.strategy.update_market_state(state)  # agent.strategy is now the market maker
    
    def get_aggregated_orders(self) -> List[Dict[str, Any]]:
        """
        Get aggregated orders from all strategies and their agents.
        Orders are scaled by each agent's governance token holdings.
        """
        aggregated_orders = []
        
        for strategy_id, agents in self.strategies.items():
            for agent in agents:
                # Get orders from agent's market maker
                orders = agent.strategy.get_orders()  # agent.strategy is now the market maker
                
                # Scale order amounts by agent's governance token holdings
                for order in orders:
                    scaled_order = order.copy()
                    scaled_order['amount'] *= (agent.performance_metrics['governance_tokens'] / self.total_governance_tokens)
                    aggregated_orders.append(scaled_order)
        
        return aggregated_orders
    
    def get_aggregated_action(self) -> float:
        """
        Get aggregated trading action from all strategies and their agents.
        Actions are weighted by each agent's governance token holdings.
        """
        weighted_action = 0.0
        total_weight = 0.0
        
        for strategy_id, agents in self.strategies.items():
            for agent in agents:
                action = agent.strategy.get_action()  # agent.strategy is now the market maker
                weight = agent.performance_metrics['governance_tokens'] / self.total_governance_tokens
                
                self.logger.debug(
                    "Strategy %s, Agent %s: action=%.4f, weight=%.4f, governance_tokens=%.2f",
                    strategy_id, agent.id, action, weight, agent.performance_metrics['governance_tokens']
                )
                
                weighted_action += action * weight
                total_weight += weight
        
        if total_weight == 0:
            self.logger.warning("No agents have governance tokens")
            return 0.0
            
        final_action = weighted_action / total_weight
        self.logger.debug(
            "Aggregated action: %.4f (weighted=%.4f, total_weight=%.4f)",
            final_action, weighted_action, total_weight
        )
        
        return final_action
    
    def distribute_profits(self, total_profit: float) -> Dict[str, float]:
        """
        Distribute profits to agents based on their governance token holdings.
        
        Args:
            total_profit: Total profit to distribute
            
        Returns:
            Dictionary mapping agent IDs to their profit share
        """
        if self.total_governance_tokens == 0:
            return {}
            
        profit_distribution = {}
        for strategy_id, agents in self.strategies.items():
            for agent in agents:
                profit_share = (agent.performance_metrics['governance_tokens'] / self.total_governance_tokens) * total_profit
                profit_distribution[agent.id] = profit_share
                agent.performance_metrics['total_profit'] += profit_share
                
        return profit_distribution
    
    def get_agent_performance(self, strategy_id: str, agent_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific agent.
        
        Args:
            strategy_id: ID of the strategy
            agent_id: ID of the agent
            
        Returns:
            Dictionary of performance metrics
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
            
        agent = next((a for a in self.strategies[strategy_id] if a.id == agent_id), None)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found in strategy {strategy_id}")
            
        return agent.performance_metrics
    
    def get_strategy_metrics(self, strategy_id: str) -> Dict[str, float]:
        """
        Get aggregated performance metrics for all agents in a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of aggregated metrics
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
            
        agents = self.strategies[strategy_id]
        if not agents:
            return {
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'average_sharpe_ratio': 0.0,
                'average_win_rate': 0.0
            }

        total_profit = sum(agent.performance_metrics['total_profit'] for agent in agents)
        max_drawdown = max(agent.performance_metrics['current_drawdown'] for agent in agents)
        avg_sharpe = sum(agent.performance_metrics['sharpe_ratio'] for agent in agents) / len(agents)
        avg_win_rate = sum(agent.performance_metrics['win_rate'] for agent in agents) / len(agents)

        return {
            'total_profit': round(total_profit, 2),
            'max_drawdown': round(max_drawdown, 2),
            'average_sharpe_ratio': round(avg_sharpe, 2),
            'average_win_rate': round(avg_win_rate, 2)
        } 