from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from .strategy_runner import StrategyRunner

@dataclass
class NFT:
    """Represents an NFT with its strategy and governance token holdings"""
    id: str
    strategy_id: str
    governance_tokens: float
    owner: str
    performance_metrics: Dict[str, float]

class NFTAgent:
    """
    NFT Agent that manages NFTs and their links to strategies.
    Each NFT is linked to a specific strategy in the StrategyRunner.
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
        
        # Initialize Strategy Runner
        self.strategy_runner = StrategyRunner(
            total_capital=total_capital,
            risk_free_rate=risk_free_rate,
            max_drawdown=max_drawdown,
            min_profit_threshold=min_profit_threshold
        )
        
        # Initialize NFTs
        self.nfts: Dict[str, NFT] = {}
        self.total_governance_tokens = 0.0
        
        self.logger.info("NFT Agent initialized with total capital: %f", total_capital)
    
    def add_strategy(
        self,
        strategy_id: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new trading strategy to the StrategyRunner.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_params: Optional parameters for the trading strategy
        """
        self.strategy_runner.add_strategy(strategy_id, strategy_params)
    
    def mint_nft(
        self,
        nft_id: str,
        owner: str,
        strategy_id: str,
        governance_tokens: float,
        risk_limits: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Mint a new NFT and link it to a strategy.
        
        Args:
            nft_id: Unique identifier for the NFT
            owner: Address of the NFT owner
            strategy_id: ID of the strategy to link to
            governance_tokens: Number of governance tokens held
            risk_limits: Optional risk limits for the agent
        """
        if nft_id in self.nfts:
            raise ValueError(f"NFT {nft_id} already exists")
        
        # Add agent to strategy in StrategyRunner
        self.strategy_runner.add_agent(
            strategy_id=strategy_id,
            agent_id=nft_id,
            governance_tokens=governance_tokens,
            risk_limits=risk_limits
        )
        
        # Create NFT
        nft = NFT(
            id=nft_id,
            strategy_id=strategy_id,
            governance_tokens=governance_tokens,
            owner=owner,
            performance_metrics={
                'total_profit': 0.0,
                'current_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'governance_tokens': governance_tokens
            }
        )
        
        self.nfts[nft_id] = nft
        self.total_governance_tokens += governance_tokens
        
        self.logger.info(
            "Minted NFT %s for owner %s linked to strategy %s with %f governance tokens",
            nft_id, owner, strategy_id, governance_tokens
        )
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """
        Update market state for all strategies.
        
        Args:
            state: Current market state
        """
        self.strategy_runner.update_market_state(state)
    
    def get_aggregated_orders(self) -> List[Dict[str, Any]]:
        """
        Get aggregated orders from all strategies.
        Orders are scaled by each NFT's governance token holdings.
        """
        return self.strategy_runner.get_aggregated_orders()
    
    def get_aggregated_action(self) -> float:
        """
        Get aggregated trading action from all strategies.
        Actions are weighted by each NFT's governance token holdings.
        """
        return self.strategy_runner.get_aggregated_action()
    
    def distribute_profits(self, total_profit: float) -> Dict[str, float]:
        """
        Distribute profits to NFT owners based on their governance token holdings.
        
        Args:
            total_profit: Total profit to distribute
            
        Returns:
            Dictionary mapping owner addresses to their profit share
        """
        # Get profit distribution from StrategyRunner
        agent_profits = self.strategy_runner.distribute_profits(total_profit)
        
        # Map agent profits to NFT owners
        owner_profits = {}
        for nft_id, profit in agent_profits.items():
            nft = self.nfts[nft_id]
            owner_profits[nft.owner] = owner_profits.get(nft.owner, 0.0) + profit
            nft.performance_metrics['total_profit'] += profit
            
        return owner_profits
    
    def get_nft_performance(self, nft_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific NFT.
        
        Args:
            nft_id: ID of the NFT
            
        Returns:
            Dictionary of performance metrics
        """
        if nft_id not in self.nfts:
            raise ValueError(f"NFT {nft_id} not found")
            
        nft = self.nfts[nft_id]
        return self.strategy_runner.get_agent_performance(nft.strategy_id, nft_id)
    
    def get_strategy_metrics(self, strategy_id: str) -> Dict[str, float]:
        """
        Get aggregated performance metrics for all NFTs in a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of aggregated metrics
        """
        return self.strategy_runner.get_strategy_metrics(strategy_id) 