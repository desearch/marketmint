"""
Core functions to support API endpoints for NFT trading system
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from .nft_agent import NFTAgent
from .profit_distributor import ProfitDistributor

logger = logging.getLogger(__name__)

class CoreFunctions:
    def __init__(self, nft_agent: NFTAgent, profit_distributor: ProfitDistributor):
        self.nft_agent = nft_agent
        self.profit_distributor = profit_distributor
        self.trade_history: List[Dict[str, Any]] = []
        self.current_market_state: Dict[str, Any] = {}
        self.proposals: List[Dict[str, Any]] = []  # Initialize proposals list
        
    def run_trade(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the AI agent once and execute a trade if signal allows
        
        Args:
            market_state: Current market state including price and volume
            
        Returns:
            Dict containing trade results
        """
        try:
            # Update market state
            self.current_market_state = market_state
            self.nft_agent.update_market_state(market_state)
            
            # Get aggregated action from all strategies
            action = self.nft_agent.get_aggregated_action()
            
            # Get orders based on action
            orders = self.nft_agent.get_aggregated_orders()
            
            # Record trade in history
            trade_record = {
                'timestamp': datetime.now(),
                'action': action,
                'price': market_state['price'],
                'volume': market_state.get('volume', 0),
                'orders': orders
            }
            self.trade_history.append(trade_record)
            
            return {
                'status': 'success',
                'trade': trade_record
            }
            
        except Exception as e:
            logger.error("Trade execution failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return a list of previous trades
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade records
        """
        return self.trade_history[-limit:]
    
    def get_market_price(self, token: str) -> Dict[str, Any]:
        """
        Return current price and volume for a given token
        
        Args:
            token: Token identifier
            
        Returns:
            Dict containing price and volume
        """
        if not self.current_market_state:
            return {
                'status': 'error',
                'message': 'No market data available'
            }
            
        return {
            'status': 'success',
            'token': token,
            'price': self.current_market_state.get('price', 0),
            'volume': self.current_market_state.get('volume', 0),
            'timestamp': datetime.now()
        }
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """
        Return current AI model status and metrics
        
        Returns:
            Dict containing strategy status and performance metrics
        """
        status = {
            'timestamp': datetime.now(),
            'strategies': {}
        }
        
        # Get metrics for each strategy
        for strategy_id in self.nft_agent.strategy_runner.strategies:
            metrics = self.nft_agent.get_strategy_metrics(strategy_id)
            status['strategies'][strategy_id] = {
                'metrics': metrics,
                'active_nfts': len(self.nft_agent.strategy_runner.strategies[strategy_id])
            }
            
        return status
    
    def distribute_profits(self, amount: float) -> Dict[str, Any]:
        """
        Trigger profit distribution to NFT holders
        
        Args:
            amount: Amount of profits to distribute
            
        Returns:
            Dict containing distribution results
        """
        # First distribute profits in the NFT agent
        owner_profits = self.nft_agent.distribute_profits(amount)
        
        # Then trigger on-chain distribution
        distribution_result = self.profit_distributor.distribute_profits(amount)
        
        return {
            'status': distribution_result['status'],
            'owner_profits': owner_profits,
            'transaction_hash': distribution_result.get('transaction_hash'),
            'timestamp': datetime.now()
        }
    
    def get_holder_earnings(self) -> Dict[str, Any]:
        """
        Show all current NFT holders and their cumulative earnings
        
        Returns:
            Dict containing holder earnings information
        """
        # Get on-chain NFT holders
        holders = self.profit_distributor.get_nft_holders()
        
        # Get total distributed profits
        total_distributed = self.profit_distributor.get_total_distributed()
        
        # Get earnings for each NFT from the agent
        nft_earnings = {}
        for nft_id, nft in self.nft_agent.nfts.items():
            metrics = self.nft_agent.get_nft_performance(nft_id)
            nft_earnings[nft_id] = {
                'owner': nft.owner,
                'strategy_id': nft.strategy_id,
                'governance_tokens': nft.governance_tokens,
                'total_profit': metrics['total_profit'],
                'performance_metrics': metrics
            }
            
        return {
            'status': 'success',
            'total_distributed': total_distributed,
            'holders': holders,
            'nft_earnings': nft_earnings,
            'timestamp': datetime.now()
        }
    
    def create_governance_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a proposal to change strategy or rebalance volume
        
        Args:
            proposal: Proposal details including type and parameters
            
        Returns:
            Dict containing proposal creation results
        """
        required_fields = ['type', 'description', 'parameters']
        if not all(field in proposal for field in required_fields):
            return {
                'status': 'error',
                'message': 'Missing required proposal fields'
            }
            
        # Store proposal details (in a real implementation, this would interact with governance contract)
        proposal_record = {
            'id': len(self.proposals) + 1,
            'timestamp': datetime.now(),
            'status': 'active',
            **proposal
        }
        self.proposals.append(proposal_record)
        
        return {
            'status': 'success',
            'proposal': proposal_record
        }
    
    def get_governance_proposals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return list of governance proposals
        
        Args:
            status: Optional filter for proposal status
            
        Returns:
            List of proposal records
        """
        if status:
            return [p for p in self.proposals if p['status'] == status]
        return self.proposals 