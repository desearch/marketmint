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
        max_position: float = 1000.0,   # Maximum position size
        position_limit: float = None,   # Position limit
        risk_aversion: float = 1.0,    # Risk aversion parameter
        max_trade_size: float = 0.05,   # Maximum trade size as fraction of max position
        min_vol_threshold: float = 0.01  # Minimum volatility threshold for trading (1% annualized)
    ):
        self.logger = logging.getLogger(__name__)
        
        self.target_spread = target_spread
        self.min_spread = min_spread
        self.max_position = max_position
        self.position_limit = position_limit or max_position
        self.risk_aversion = risk_aversion
        self.max_trade_size = max_trade_size
        self.min_vol_threshold = min_vol_threshold
        
        # Market state
        self.current_price = None
        self.position = 0.0
        self.portfolio_value = 0.0
        self.last_price = None
        self.price_history = []
        self.vol_window = 20
        self.vol_scale = 1.0
        self.current_vol = 0.0
        self.last_action = 0.0
        self.action_count = 0
        
        self.logger.info(
            "Initialized PureMarketMaker with target_spread=%.4f, min_spread=%.4f",
            target_spread, min_spread
        )
    
    def update_market_state(self, state: Dict[str, Any]) -> None:
        """Update internal state with current market conditions"""
        self.last_price = self.current_price
        self.current_price = state.get('price')
        self.position = state.get('position', 0.0)
        self.portfolio_value = state.get('portfolio_value', 0.0)
        
        # Update price history and maintain window
        self.price_history.append(self.current_price)
        if len(self.price_history) > self.vol_window:
            self.price_history.pop(0)
        
        # Calculate volatility scaling factor
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            self.current_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
            self.vol_scale = np.clip(1.0 / (1.0 + self.current_vol), 0.2, 1.0)  # Reduce position sizes in high vol
        
        self.logger.debug(
            "Updated market state: price=%.2f, position=%.2f",
            self.current_price, self.position
        )
    
    def calculate_spread(self) -> float:
        """Calculate the current spread based on position and market conditions"""
        # Base spread
        spread = self.target_spread
        
        # Adjust spread based on position
        position_utilization = abs(self.position) / self.max_position
        spread = max(self.min_spread, spread * (1 + position_utilization))
        
        # Adjust spread based on volatility
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            volatility = np.std(returns)
            spread = max(spread, self.min_spread * (1 + 5 * volatility))
        
        return spread
    
    def calculate_skew(self) -> float:
        """Calculate price skew based on current position"""
        # Normalize position between -1 and 1
        normalized_position = self.position / self.max_position
        # Apply sigmoid to get smooth skew
        skew = 2 / (1 + np.exp(-normalized_position * self.risk_aversion)) - 1
        return skew
    
    def calculate_momentum(self) -> float:
        """Calculate short-term price momentum with improved signal quality"""
        if len(self.price_history) < 10:
            return 0.0
        
        # Calculate multiple timeframe momentum
        short_ma = np.mean(self.price_history[-5:])
        med_ma = np.mean(self.price_history[-10:])
        long_ma = np.mean(self.price_history)
        
        # Require agreement between timeframes
        short_trend = np.sign(short_ma - med_ma)
        long_trend = np.sign(med_ma - long_ma)
        
        # More aggressive momentum threshold
        if short_trend == long_trend and abs(short_ma - long_ma) > self.current_price * 0.0005:
            return 0.4 * short_trend
        return 0.0
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Generate limit orders around the current price"""
        if self.current_price is None:
            return []
        
        spread = self.calculate_spread()
        skew = self.calculate_skew()
        
        # Adjust spreads based on skew - more aggressive
        buy_spread = spread * (1 - 2.0 * skew)
        sell_spread = spread * (1 + 2.0 * skew)
        
        # Calculate order prices
        buy_price = self.current_price * (1 - buy_spread)
        sell_price = self.current_price * (1 + sell_spread)
        
        # Calculate base order size - more aggressive
        # Use up to 100% of portfolio value
        base_size = self.portfolio_value
        
        # Adjust order sizes based on position - more aggressive
        position_ratio = self.position / self.max_position
        size_skew = 2.0 * np.tanh(position_ratio * self.risk_aversion)
        
        # When long, increase sell size and decrease buy size
        # When short, increase buy size and decrease sell size
        buy_size = base_size * (1 - size_skew)
        sell_size = base_size * (1 + size_skew)
        
        # Ensure minimum order size
        min_order_size = self.portfolio_value * 0.01  # 1% minimum order size
        buy_size = max(buy_size, min_order_size)
        sell_size = max(sell_size, min_order_size)
        
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
        
        self.logger.info(
            "Generated orders: buy(price=%.2f, size=%.2f), sell(price=%.2f, size=%.2f), spread=%.4f, skew=%.2f",
            buy_price, buy_size, sell_price, sell_size, spread, skew
        )
        
        return orders
    
    def get_action(self) -> float:
        """
        Get trading action in range [-1, 1]
        -1: full sell, +1: full buy, 0: no action
        """
        if self.current_price is None or len(self.price_history) < self.vol_window:
            return 0.0
        
        # Only trade if volatility is above minimum threshold
        if self.current_vol < self.min_vol_threshold:
            return 0.0
        
        # Calculate normalized position (-1 to 1)
        normalized_position = self.position / (self.max_position + 1e-10)
        
        # Mean reversion strength scales with position size and volatility
        mean_reversion = -0.3 * normalized_position * self.vol_scale
        
        # Add momentum component when position is small and trends align
        momentum = 0.0
        if abs(normalized_position) < 0.2:  # Increased threshold
            momentum = self.calculate_momentum() * self.vol_scale
        
        # Combine components
        action = mean_reversion + momentum
        
        # Scale by max trade size
        action = action * self.max_trade_size
        
        # Add tiny random component to avoid getting stuck
        if abs(action) > 0:
            action += np.random.normal(0, 0.002) * self.vol_scale
        
        # Reduce trading frequency by requiring minimum action size
        min_action = 0.005 * self.max_trade_size  # Reduced minimum action threshold
        if abs(action) < min_action:
            action = 0.0
        
        # Apply position limits
        if abs(self.position) > self.position_limit:
            action = -np.sign(self.position) * self.max_trade_size
        
        # Avoid frequent direction changes
        if self.last_action * action < 0:  # Direction change
            self.action_count += 1
            if self.action_count < 2:  # Reduced required consecutive signals
                action = 0.0
        else:
            self.action_count = 0
        
        self.last_action = action
        
        # Ensure action is within bounds
        return np.clip(action, -self.max_trade_size, self.max_trade_size)
        
        self.logger.debug(
            "Generated action %.4f (position=%.2f, max_position=%.2f, spread=%.4f, skew=%.2f, momentum=%.4f)",
            action, self.position, self.max_position, spread, skew, momentum
        )
        
        return action 