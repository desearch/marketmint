from typing import Dict, Any, List, Optional
import logging
from .strategies import PureMarketMaker

class MarketMaker:
    """
    High-level API for market making services.
    This class provides a clean interface for market making operations.
    """
    
    def __init__(
        self,
        strategy: Optional[PureMarketMaker] = None,
        **strategy_params
    ):
        """
        Initialize the market maker with a strategy.
        
        Args:
            strategy: Optional pre-configured strategy instance
            **strategy_params: Parameters to create a new strategy if none provided
        """
        self.logger = logging.getLogger(__name__)
        
        if strategy is None:
            self.strategy = PureMarketMaker(**strategy_params)
        else:
            self.strategy = strategy
            
        self.logger.info("Market maker initialized with strategy: %s", self.strategy.__class__.__name__)
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """
        Update the market maker with current market state.
        
        Args:
            state: Dictionary containing current market state
                  (price, portfolio_value, position, timestamp)
        """
        # Validate required fields
        required_fields = ['price', 'portfolio_value', 'position', 'timestamp']
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate field types
        if not isinstance(state['price'], (int, float)):
            raise ValueError("Price must be a number")
        if not isinstance(state['portfolio_value'], (int, float)):
            raise ValueError("Portfolio value must be a number")
        if not isinstance(state['position'], (int, float)):
            raise ValueError("Position must be a number")
        if not isinstance(state['timestamp'], (int, float)):
            raise ValueError("Timestamp must be a number")
        
        self.current_state = state
        self.logger.debug("Market state updated: %s", state)
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current market making orders.
        
        Returns:
            List of order dictionaries with side, price, and amount
        """
        if not hasattr(self, 'current_state'):
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        orders = self.strategy.generate_orders(
            self.current_state['price'],
            self.current_state['portfolio_value'],
            self.current_state['position'],
            self.current_state['timestamp']
        )
        
        self.logger.debug("Generated orders: %s", orders)
        return orders
    
    def get_action(self) -> float:
        """
        Get the current trading action.
        
        Returns:
            Float between -1 (full sell) and 1 (full buy)
        """
        if not hasattr(self, 'current_state'):
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        action = self.strategy.get_action(self.current_state)
        self.logger.debug("Generated action: %s", action)
        return action
    
    def should_refresh_orders(self) -> bool:
        """
        Check if orders should be refreshed.
        
        Returns:
            Boolean indicating if orders should be refreshed
        """
        if not hasattr(self, 'current_state'):
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        return self.strategy.should_refresh_orders(self.current_state['timestamp'])
    
    def get_spreads(self) -> Dict[str, float]:
        """
        Get current bid and ask spreads.
        
        Returns:
            Dictionary with bid_spread and ask_spread
        """
        if not hasattr(self, 'current_state'):
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        target_position = self.strategy.calculate_target_inventory(
            self.current_state['portfolio_value'],
            self.current_state['price']
        )
        inventory_bias = self.strategy.calculate_inventory_bias(
            self.current_state['position'],
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
    
    def get_inventory_metrics(self) -> Dict[str, float]:
        """
        Get current inventory metrics.
        
        Returns:
            Dictionary with target_position, current_position, and inventory_bias
        """
        if not hasattr(self, 'current_state'):
            raise ValueError("Market state not initialized. Call update_market_state first.")
            
        target_position = self.strategy.calculate_target_inventory(
            self.current_state['portfolio_value'],
            self.current_state['price']
        )
        inventory_bias = self.strategy.calculate_inventory_bias(
            self.current_state['position'],
            target_position
        )
        
        return {
            'target_position': target_position,
            'current_position': self.current_state['position'],
            'inventory_bias': inventory_bias
        } 