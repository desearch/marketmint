import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from ..trading_env import CryptoTradingEnv

class PureMarketMaker:
    """
    Pure Market Making strategy inspired by Hummingbot.
    This strategy places and maintains orders on both sides of the order book
    to capture the bid-ask spread while managing inventory risk.
    """
    
    def __init__(
        self,
        bid_spread: float = 0.01,  # 1% spread below mid price
        ask_spread: float = 0.01,  # 1% spread above mid price
        min_spread: float = 0.002,  # Minimum spread required to place orders
        order_refresh_time: int = 60,  # Refresh orders every 60 seconds
        inventory_target_base_pct: float = 0.5,  # Target 50% inventory in base asset
        inventory_range_multiplier: float = 1.0,  # Multiplier for inventory target range
        risk_factor: float = 0.5,  # Factor for adjusting spreads based on inventory
        max_order_age: int = 1800,  # Maximum age of orders (30 minutes)
        order_amount: Optional[float] = None,  # Fixed order amount if specified
        price_ceiling: Optional[float] = None,  # Maximum price to place ask orders
        price_floor: Optional[float] = None,  # Minimum price to place bid orders
    ):
        self.bid_spread = bid_spread
        self.ask_spread = ask_spread
        self.min_spread = min_spread
        self.order_refresh_time = order_refresh_time
        self.inventory_target_base_pct = inventory_target_base_pct
        self.inventory_range_multiplier = inventory_range_multiplier
        self.risk_factor = risk_factor
        self.max_order_age = max_order_age
        self.order_amount = order_amount
        self.price_ceiling = price_ceiling
        self.price_floor = price_floor
        
        # Internal state
        self.last_order_refresh = 0
        self.current_orders: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def calculate_target_inventory(self, portfolio_value: float, current_price: float) -> float:
        """Calculate target inventory in base currency"""
        target = (portfolio_value * self.inventory_target_base_pct) / current_price
        self.logger.debug(f"Target inventory: {target:.2f}")
        return target
    
    def calculate_inventory_bias(self, current_position: float, target_position: float) -> float:
        """Calculate inventory bias for spread adjustment"""
        if target_position == 0:
            return 0
        
        # Calculate relative deviation from target
        inventory_deviation = (current_position - target_position) / target_position
        
        # Apply risk factor and clip to [-1, 1]
        bias = np.clip(inventory_deviation * self.risk_factor, -1, 1)
        
        # Increase bias when far from target
        if abs(inventory_deviation) > 1.0:
            bias = np.sign(bias)  # Full bias when deviation is large
        
        self.logger.debug(f"Inventory bias: {bias:.2f} (current: {current_position:.2f}, target: {target_position:.2f}, deviation: {inventory_deviation:.2f})")
        return bias
    
    def adjust_spreads(self, base_bid_spread: float, base_ask_spread: float, inventory_bias: float) -> tuple:
        """Adjust spreads based on inventory bias"""
        bid_spread = base_bid_spread * (1 + inventory_bias)
        ask_spread = base_ask_spread * (1 - inventory_bias)
        
        return (
            max(bid_spread, self.min_spread),
            max(ask_spread, self.min_spread)
        )
    
    def calculate_order_prices(self, mid_price: float, inventory_bias: float) -> tuple:
        """Calculate bid and ask prices"""
        adjusted_bid_spread, adjusted_ask_spread = self.adjust_spreads(
            self.bid_spread, self.ask_spread, inventory_bias
        )
        
        bid_price = mid_price * (1 - adjusted_bid_spread)
        ask_price = mid_price * (1 + adjusted_ask_spread)
        
        # Apply price limits if set
        if self.price_floor is not None:
            bid_price = max(min(bid_price, self.price_floor), self.price_floor)
        if self.price_ceiling is not None:
            ask_price = min(max(ask_price, self.price_ceiling), self.price_ceiling)
            
        return bid_price, ask_price
    
    def calculate_order_amounts(
        self,
        portfolio_value: float,
        current_price: float,
        current_position: float
    ) -> tuple:
        """Calculate bid and ask order amounts"""
        if self.order_amount is not None:
            return self.order_amount, self.order_amount
        
        # Calculate based on current portfolio value and position
        target_position = self.calculate_target_inventory(portfolio_value, current_price)
        position_diff = target_position - current_position
        
        # Base amount on the absolute difference from target
        base_amount = abs(position_diff)
        
        # Adjust order sizes based on the difference from target position
        if position_diff > 0:
            # Need to buy more
            bid_amount = base_amount * self.inventory_range_multiplier
            ask_amount = base_amount * (1 - self.inventory_range_multiplier)
        else:
            # Need to sell more
            bid_amount = base_amount * (1 - self.inventory_range_multiplier)
            ask_amount = base_amount * self.inventory_range_multiplier
            
        return bid_amount, ask_amount
    
    def should_refresh_orders(self, current_timestamp: int) -> bool:
        """Check if orders should be refreshed"""
        time_since_refresh = current_timestamp - self.last_order_refresh
        return time_since_refresh >= self.order_refresh_time
    
    def generate_orders(
        self,
        mid_price: float,
        portfolio_value: float,
        current_position: float,
        timestamp: int
    ) -> List[Dict[str, Any]]:
        """Generate new orders based on current market conditions"""
        if not self.should_refresh_orders(timestamp):
            return self.current_orders
            
        target_position = self.calculate_target_inventory(portfolio_value, mid_price)
        inventory_bias = self.calculate_inventory_bias(current_position, target_position)
        bid_price, ask_price = self.calculate_order_prices(mid_price, inventory_bias)
        bid_amount, ask_amount = self.calculate_order_amounts(
            portfolio_value, mid_price, current_position
        )
        
        self.current_orders = [
            {
                'side': 'buy',
                'price': bid_price,
                'amount': bid_amount,
                'timestamp': timestamp
            },
            {
                'side': 'sell',
                'price': ask_price,
                'amount': ask_amount,
                'timestamp': timestamp
            }
        ]
        
        self.last_order_refresh = timestamp
        return self.current_orders
    
    def get_action(self, env_state: Dict[str, Any]) -> float:
        """
        Convert market making orders into an action for the trading environment
        Returns a value between -1 (full sell) and 1 (full buy)
        """
        current_price = env_state['price']
        portfolio_value = env_state['portfolio_value']
        current_position = env_state['position']
        timestamp = env_state['timestamp']
        
        self.logger.debug(f"State: price={current_price:.2f}, portfolio={portfolio_value:.2f}, position={current_position:.2f}")
        
        # Calculate target position and bias
        target_position = self.calculate_target_inventory(portfolio_value, current_price)
        inventory_bias = self.calculate_inventory_bias(current_position, target_position)
        
        # Calculate relative position deviation
        position_deviation = (current_position - target_position) / target_position
        self.logger.debug(f"Position deviation: {position_deviation:.2f}")
        
        # When far from target position, take more aggressive action
        if abs(position_deviation) > 0.2:  # More than 20% deviation
            # Return a strong sell signal when we have too much inventory
            # and a strong buy signal when we have too little
            action = -np.sign(position_deviation)  # Negative because we want to reduce the deviation
            self.logger.debug(f"Taking aggressive action: {action} (deviation: {position_deviation:.2f})")
            return float(action)
        
        # For smaller deviations, use the order-based approach
        orders = self.generate_orders(
            current_price,  # Use current price as mid price
            portfolio_value,
            current_position,
            timestamp
        )
        
        # Convert orders to a single action
        net_position_change = 0
        for order in orders:
            if order['side'] == 'buy':
                net_position_change += order['amount']
            else:
                net_position_change -= order['amount']
        
        self.logger.debug(f"Net position change from orders: {net_position_change:.2f}")
                
        # Scale the action based on position deviation
        action = net_position_change * (1 - abs(position_deviation))
        self.logger.debug(f"Scaled action: {action:.2f}")
                
        # Normalize the position change to [-1, 1]
        max_position_change = max(
            abs(action),
            portfolio_value / current_price
        )
        action = np.clip(action / max_position_change, -1, 1)
        
        self.logger.debug(f"Final action: {action:.2f}")
        return float(action) 