"""
Market Maker implementation that wraps trading strategies
"""

from typing import Dict, List, Any
import logging
from .strategies import PureMarketMaker

class MarketMaker:
    """
    Market Maker that wraps a trading strategy and provides a consistent interface
    """
    
    def __init__(self, **strategy_params):
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default strategy if none provided
        if 'strategy' in strategy_params:
            self.strategy = strategy_params.pop('strategy')
        else:
            self.strategy = PureMarketMaker(**strategy_params)
            self.strategy.market_maker = self
            
        self.current_state = None
        
        self.logger.info("Market maker initialized with strategy: %s", 
                        self.strategy.__class__.__name__)
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """
        Update market state
        
        Args:
            state: Current market state including price, position, etc.
        """
        required_fields = ['price', 'position', 'timestamp', 'portfolio_value']
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate field types
        if not isinstance(state['timestamp'], (int, float)):
            raise ValueError("Timestamp must be a number")
        if not isinstance(state['price'], (int, float)):
            raise ValueError("Price must be a number")
        if not isinstance(state['position'], (int, float)):
            raise ValueError("Position must be a number")
        if not isinstance(state['portfolio_value'], (int, float)):
            raise ValueError("Portfolio value must be a number")
        
        self.current_state = state
        if self.strategy:
            self.strategy.update_market_state(state)
            self.logger.debug(
                "Updated market state: price=%.2f, position=%.2f",
                state['price'], state['position']
            )
    
    def get_orders(self, market_state: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get current orders from the strategy
        
        Args:
            market_state: Optional market state. If not provided, uses current state.
            
        Returns:
            List of order dictionaries
        """
        if not self.strategy:
            return []
            
        state_to_use = market_state if market_state is not None else self.current_state
        if not state_to_use:
            return []
            
        orders = self.strategy.get_orders(state_to_use)
        self.logger.debug("Generated orders: %s", orders)
        return orders
    
    def get_action(self, state: Dict[str, Any] = None) -> float:
        """
        Get trading action from the strategy
        
        Args:
            state: Optional market state. If not provided, uses current state.
        
        Returns:
            Float representing the trading action
        """
        if not self.strategy:
            return 0.0
            
        # Use provided state or current state
        state_to_use = state if state is not None else self.current_state
        if not state_to_use:
            return 0.0
            
        action = self.strategy.get_action(state_to_use)
        self.logger.debug("Generated action: %.4f", action)
        return action
    
    def set_strategy(self, strategy) -> None:
        """Set the trading strategy"""
        self.strategy = strategy
        if strategy:
            strategy.market_maker = self
        self.logger.info("Set strategy: %s", strategy.__class__.__name__)
        if self.current_state and self.strategy:
            self.strategy.update_market_state(self.current_state)
    
    def should_refresh_orders(self) -> bool:
        """
        Check if orders should be refreshed.
        
        Returns:
            Boolean indicating if orders should be refreshed
        """
        # Refresh orders every 5 minutes by default
        return True
    
    def get_spreads(self, market_state: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Get current bid and ask spreads.
        
        Args:
            market_state: Optional market state. If not provided, uses current state.
            
        Returns:
            Dictionary with bid_spread and ask_spread
        """
        state_to_use = market_state if market_state is not None else self.current_state
        if not state_to_use:
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        target_position = self.strategy.calculate_target_inventory(
            state_to_use['portfolio_value'],
            state_to_use['price']
        )
        inventory_bias = self.strategy.calculate_inventory_bias(
            state_to_use['position'],
            target_position
        )
        
        bid_spread, ask_spread = self.strategy.adjust_spreads(
            self.strategy.bid_spread,
            self.strategy.ask_spread,
            inventory_bias
        )
        
        return {
            'bid_spread': bid_spread,
            'ask_spread': ask_spread
        }
    
    def get_inventory_metrics(self, market_state: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Get current inventory metrics.
        
        Args:
            market_state: Optional market state. If not provided, uses current state.
            
        Returns:
            Dictionary with target_position, current_position, and inventory_bias
        """
        state_to_use = market_state if market_state is not None else self.current_state
        if not state_to_use:
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        target_position = self.strategy.calculate_target_inventory(
            state_to_use['portfolio_value'],
            state_to_use['price']
        )
        inventory_bias = self.strategy.calculate_inventory_bias(
            state_to_use['position'],
            target_position
        )
        
        return {
            'target_position': target_position,
            'current_position': state_to_use['position'],
            'inventory_bias': inventory_bias
        } 