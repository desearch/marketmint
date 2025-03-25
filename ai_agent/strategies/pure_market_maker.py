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
    
    def __init__(self):
        # Strategy parameters
        self.target_spread = 0.01  # 1% target spread
        self.min_spread = 0.005    # 0.5% minimum spread
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
        self.last_action = 0
        self.market_maker = None  # Will be set by the MarketMaker
        
    def update_market_state(self, state):
        """Update market state from dictionary input"""
        price = state.get('price')
        position = state.get('position', 0.0)
        
        if price is None or price <= 0:
            return
            
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
        
    def calculate_momentum(self):
        if len(self.price_history) < 2:
            return 0
            
        price_change = (self.price_history[-1] / self.price_history[-2]) - 1
        # Reduced momentum impact with lower cap
        return np.sign(price_change) * min(abs(price_change), 0.01) * 5.0
        
    def get_action(self, state):
        """Get trading action from state dictionary"""
        price = state.get('price')
        position = state.get('position', 0.0)
        
        if len(self.price_history) < 2:  # Only need 2 prices for momentum
            return 0
            
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
            action = 0
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
        
        self.last_action = action
        logging.debug(f"Action: {action:.6f}, Position: {normalized_position:.3f}, Vol: {self.current_vol:.4f}")
        return action
        
    def get_orders(self, state):
        """Get order size from state dictionary"""
        action = self.get_action(state)
        size = action * self.max_position
        return size 