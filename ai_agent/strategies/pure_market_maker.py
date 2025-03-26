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
    
    def __init__(self, bid_spread=0.01, ask_spread=0.01, min_spread=0.005, 
                 order_refresh_time=60, inventory_target_base_pct=0.5,
                 inventory_range_multiplier=1.0, risk_factor=0.5, **kwargs):
        # Strategy parameters
        self._bid_spread = bid_spread
        self._ask_spread = ask_spread
        self.min_spread = min_spread
        self.order_refresh_time = order_refresh_time
        self.inventory_target_base_pct = inventory_target_base_pct
        self.inventory_range_multiplier = inventory_range_multiplier
        self.risk_factor = risk_factor
        
        self.target_spread = max(bid_spread, ask_spread)
        self.max_position = 100000  # Maximum position size
        self.position_limit = 0.5   # Start reducing position at 50% of max
        self.risk_aversion = 1.0    # Higher risk aversion
        self.max_trade_size = 0.1   # 10% of max position per trade
        self.min_vol_threshold = 0.00001  # 0.001% minimum volatility threshold
        self.momentum_threshold = 0.00001  # 0.001% price move threshold
        self.mean_reversion_strength = 0.5  # Reduced mean reversion impact
        self.min_trade_size = 0.000005  # Minimum trade size of 0.0005% of max position
        
        # Internal state
        self.price_history = []
        self.vol_window = 20
        self.current_vol = 0
        self.vol_scale = 1.0
        self.consecutive_actions = 0
        self.last_action = 0.0
        self.market_maker = None  # Will be set by the MarketMaker
        
    @property
    def bid_spread(self) -> float:
        """Get current bid spread"""
        return self._bid_spread
        
    @property
    def ask_spread(self) -> float:
        """Get current ask spread"""
        return self._ask_spread
        
    def calculate_target_inventory(self, portfolio_value: float, price: float) -> float:
        """
        Calculate target inventory based on portfolio value and current price
        
        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            
        Returns:
            Target inventory in base currency units
        """
        # Target inventory is a percentage of portfolio value
        target_value = portfolio_value * self.position_limit
        if price <= 0:
            return 0.0
        return target_value / price
        
    def calculate_inventory_bias(self, current_position: float, target_position: float) -> float:
        """
        Calculate inventory bias based on current and target positions
        
        Args:
            current_position: Current position size
            target_position: Target position size
            
        Returns:
            Inventory bias in range [-1, 1]
        """
        # Calculate bias as normalized difference from target
        max_deviation = self.max_position * self.position_limit
        if max_deviation == 0:
            return 0.0
        bias = (current_position - target_position) / max_deviation
        return np.clip(bias, -1.0, 1.0)
        
    def adjust_spreads(self, base_bid_spread: float, base_ask_spread: float, inventory_bias: float) -> tuple[float, float]:
        """
        Adjust spreads based on inventory bias
        
        Args:
            base_bid_spread: Base bid spread
            base_ask_spread: Base ask spread
            inventory_bias: Current inventory bias
            
        Returns:
            Tuple of (adjusted_bid_spread, adjusted_ask_spread)
        """
        # Adjust spreads based on inventory bias
        # When long (positive bias), increase ask spread and decrease bid spread
        # When short (negative bias), increase bid spread and decrease ask spread
        bid_adjustment = 1.0 - inventory_bias
        ask_adjustment = 1.0 + inventory_bias
        
        adjusted_bid = max(self.min_spread, base_bid_spread * bid_adjustment)
        adjusted_ask = max(self.min_spread, base_ask_spread * ask_adjustment)
        
        return adjusted_bid, adjusted_ask
        
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """
        Update market state from dictionary input
        
        Args:
            state: Current market state
        """
        if not state or 'price' not in state:
            raise ValueError("Missing required field: price")
            
        price = state.get('price')
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("Price must be a positive number")
            
        self.price_history.append(price)
        if len(self.price_history) > self.vol_window:
            self.price_history.pop(0)
            
        if len(self.price_history) >= 2:
            returns = [np.log(self.price_history[i] / self.price_history[i-1]) 
                      for i in range(1, len(self.price_history))]
            self.current_vol = np.std(returns) * np.sqrt(252)
            # More conservative volatility scaling
            self.vol_scale = np.clip(1.0 / (1.0 + self.current_vol), 0.5, 1.0)
            
        logging.debug(f"Current vol: {self.current_vol:.4f}, Vol scale: {self.vol_scale:.4f}")
        
    def calculate_momentum(self) -> float:
        """Calculate momentum signal"""
        if len(self.price_history) < 2:
            return 0.0
            
        price_change = (self.price_history[-1] / self.price_history[-2]) - 1
        # Reduced momentum impact with lower cap
        return np.sign(price_change) * min(abs(price_change), 0.01) * 5.0
        
    def get_action(self, state: Dict[str, Any] = None) -> float:
        """
        Get trading action from state dictionary
        
        Args:
            state: Optional market state
            
        Returns:
            Float representing the trading action
        """
        if len(self.price_history) < 2:  # Only need 2 prices for momentum
            return 0.0
            
        if not state:
            return 0.0
            
        position = state.get('position', 0.0)
        normalized_position = position / self.max_position
        
        # Calculate components
        momentum = self.calculate_momentum()
        mean_reversion = -normalized_position * self.mean_reversion_strength
        
        # Combine signals with position-based weighting
        position_weight = min(1.0, abs(normalized_position) / self.position_limit)
        action = (1 - position_weight) * momentum + position_weight * mean_reversion
        
        # Apply stricter position limits starting at 50%
        position_factor = max(0.0, 1 - abs(normalized_position) / self.position_limit)
        action *= position_factor
        
        # Scale by volatility with lower maximum
        action *= self.vol_scale
        
        # Apply fixed minimum trade size
        if abs(action) < self.min_trade_size:
            action = 0.0
        else:
            # Limit maximum trade size and ensure within [-0.1, 0.1]
            action = np.clip(action, -self.max_trade_size, self.max_trade_size)
            action = np.clip(action, -0.1, 0.1)
            
            # Check for direction changes
            if np.sign(action) != np.sign(self.last_action):
                self.consecutive_actions = 0
            else:
                self.consecutive_actions += 1
                
            # More conservative consecutive trade scaling
            action *= min(1.5, 1 + self.consecutive_actions * 0.1)
        
        self.last_action = float(action)
        logging.debug(f"Action: {action:.6f}, Position: {normalized_position:.3f}, Vol: {self.current_vol:.4f}")
        return float(action)
        
    def get_orders(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get order size and prices from state dictionary
        
        Args:
            state: Current market state
            
        Returns:
            List of order dictionaries with side, price, and amount
        """
        if not state or 'price' not in state:
            return []
            
        price = state.get('price')
        if price <= 0:
            return []
            
        position = state.get('position', 0.0)
        portfolio_value = state.get('portfolio_value', 0.0)
        
        # Calculate target position and inventory bias
        target_position = self.calculate_target_inventory(portfolio_value, price)
        inventory_bias = self.calculate_inventory_bias(position, target_position)
        
        # Adjust spreads based on inventory
        bid_spread, ask_spread = self.adjust_spreads(self._bid_spread, self._ask_spread, inventory_bias)
        
        # Calculate order prices
        bid_price = price * (1 - bid_spread)
        ask_price = price * (1 + ask_spread)
        
        # Calculate base order size as percentage of max position
        base_size = self.max_position * self.max_trade_size
        
        # Adjust order sizes based on inventory bias
        bid_size = base_size * (1 - inventory_bias)  # Reduce buys when long
        ask_size = base_size * (1 + inventory_bias)  # Reduce sells when short
        
        # Apply minimum size filter
        if bid_size < self.min_trade_size * self.max_position:
            bid_size = 0
        if ask_size < self.min_trade_size * self.max_position:
            ask_size = 0
            
        orders = []
        if bid_size > 0:
            orders.append({
                'side': 'buy',
                'price': bid_price,
                'amount': bid_size
            })
        if ask_size > 0:
            orders.append({
                'side': 'sell',
                'price': ask_price,
                'amount': ask_size
            })
            
        return orders 