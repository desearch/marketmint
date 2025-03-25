"""
Pure Market Making strategy implementation
"""

from typing import Dict, List, Any
import numpy as np
import logging

class PureMarketMaker:
    """
    Pure market making strategy that provides liquidity around the current price
    """
    
    def __init__(
        self,
        target_spread: float = 0.002,  # 0.2% target spread
        min_spread: float = 0.001,     # 0.1% minimum spread
        max_position: float = 100.0,   # Maximum position size
        position_limit: float = 0.5,   # 50% of capital as position limit
        risk_aversion: float = 1.0     # Risk aversion parameter
    ):
        self.logger = logging.getLogger(__name__)
        
        self.target_spread = target_spread
        self.min_spread = min_spread
        self.max_position = max_position
        self.position_limit = position_limit
        self.risk_aversion = risk_aversion
        
        # Market state
        self.current_price = None
        self.current_position = 0.0
        self.portfolio_value = 0.0
        
        self.logger.info(
            "Initialized PureMarketMaker with target_spread=%.4f, min_spread=%.4f",
            target_spread, min_spread
        )
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """Update internal state with current market conditions"""
        self.current_price = state.get('price')
        self.current_position = state.get('position', 0.0)
        self.portfolio_value = state.get('portfolio_value', 0.0)
        
        self.logger.debug(
            "Updated market state: price=%.2f, position=%.2f",
            self.current_price, self.current_position
        )
    
    def calculate_spread(self) -> float:
        """Calculate the current spread based on position and market conditions"""
        # Base spread
        spread = self.target_spread
        
        # Adjust spread based on position
        position_utilization = abs(self.current_position) / self.max_position
        spread = max(self.min_spread, spread * (1 + position_utilization))
        
        return spread
    
    def calculate_skew(self) -> float:
        """Calculate price skew based on current position"""
        # Normalize position between -1 and 1
        normalized_position = self.current_position / self.max_position
        # Apply sigmoid to get smooth skew
        skew = 2 / (1 + np.exp(-2 * normalized_position * self.risk_aversion)) - 1
        return skew
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Generate limit orders around the current price"""
        if self.current_price is None:
            return []
        
        spread = self.calculate_spread()
        skew = self.calculate_skew()
        
        # Adjust spreads based on skew
        buy_spread = spread * (1 - skew)
        sell_spread = spread * (1 + skew)
        
        # Calculate order prices
        buy_price = self.current_price * (1 - buy_spread)
        sell_price = self.current_price * (1 + sell_spread)
        
        # Calculate base order size
        base_size = self.portfolio_value * self.position_limit
        
        # Adjust order sizes based on position
        position_ratio = self.current_position / self.max_position
        size_skew = np.tanh(position_ratio * self.risk_aversion)
        
        # When long, increase sell size and decrease buy size
        # When short, increase buy size and decrease sell size
        buy_size = base_size * (1 - size_skew)
        sell_size = base_size * (1 + size_skew)
        
        orders = [
            {
                'side': 'buy',
                'price': buy_price,
                'amount': buy_size
            },
            {
                'side': 'sell',
                'price': sell_price,
                'amount': sell_size
            }
        ]
        
        return orders
    
    def get_action(self) -> float:
        """
        Get trading action in range [-1, 1]
        -1: full sell, +1: full buy, 0: no action
        """
        if self.current_price is None:
            return 0.0
        
        # Calculate spread and skew
        spread = self.calculate_spread()
        skew = self.calculate_skew()
        
        # Calculate position-based action
        normalized_position = self.current_position / self.max_position
        position_action = -np.tanh(normalized_position * self.risk_aversion)
        
        # Calculate spread-based action
        spread_action = np.tanh((self.target_spread - spread) * 10)
        
        # Combine actions with weights
        action = 0.7 * position_action + 0.3 * spread_action
        
        # Apply skew to make actions more aggressive when needed
        action *= (1 + abs(skew))
        
        # Ensure action stays in [-1, 1] range
        action = np.clip(action, -1.0, 1.0)
        
        self.logger.debug(
            "Generated action %.2f (position=%.2f, max_position=%.2f, spread=%.4f, skew=%.2f)",
            action, self.current_position, self.max_position, spread, skew
        )
        
        return action 