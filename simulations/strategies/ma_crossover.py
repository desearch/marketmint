import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseSimulation

class MACrossoverStrategy(BaseSimulation):
    """Moving Average Crossover Strategy"""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        max_position: float = None,
        short_window: int = 7,
        long_window: int = 30,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30
    ):
        super().__init__(initial_balance, transaction_cost, max_position)
        
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover and RSI"""
        df = data.copy()
        
        # Calculate moving averages
        df['sma_short'] = df['price'].rolling(window=self.short_window, min_periods=1).mean()
        df['sma_long'] = df['price'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Calculate RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['signal'] = 0
        
        # Buy signals:
        # 1. Short MA crosses above long MA
        # 2. RSI is oversold
        buy_condition = (
            (df['sma_short'] > df['sma_long']) &
            (df['sma_short'].shift(1) <= df['sma_long'].shift(1)) &
            (df['rsi'] < self.rsi_oversold)
        )
        
        # Sell signals:
        # 1. Short MA crosses below long MA
        # 2. RSI is overbought
        sell_condition = (
            (df['sma_short'] < df['sma_long']) &
            (df['sma_short'].shift(1) >= df['sma_long'].shift(1)) &
            (df['rsi'] > self.rsi_overbought)
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def execute_trade(self, signal: float, price: float) -> Dict[str, Any]:
        """Execute a trade based on the signal"""
        trade_result = {
            'timestamp': pd.Timestamp.now(),
            'price': price,
            'signal': signal,
            'position_before': self.current_position,
            'balance_before': self.current_balance
        }
        
        if signal > 0:  # Buy signal
            if self.current_position <= 0:  # Only buy if we're not already long
                # Calculate position size (accounting for transaction costs)
                max_position = self.current_balance / (price * (1 + self.transaction_cost))
                if self.max_position:
                    max_position = min(max_position, self.max_position)
                
                # Execute buy
                cost = max_position * price * (1 + self.transaction_cost)
                if cost <= self.current_balance:
                    self.current_position = max_position
                    self.current_balance -= cost
                    self.entry_price = price
                    self.total_trades += 1
                    
                    trade_result.update({
                        'action': 'buy',
                        'amount': max_position,
                        'cost': cost,
                        'position_after': self.current_position,
                        'balance_after': self.current_balance
                    })
        
        elif signal < 0:  # Sell signal
            if self.current_position > 0:  # Only sell if we have a position
                # Execute sell
                revenue = self.current_position * price * (1 - self.transaction_cost)
                profit = revenue - (self.current_position * self.entry_price)
                
                self.current_balance += revenue
                self.current_position = 0
                self.total_trades += 1
                
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.total_profit += profit
                
                trade_result.update({
                    'action': 'sell',
                    'amount': self.current_position,
                    'revenue': revenue,
                    'profit': profit,
                    'position_after': self.current_position,
                    'balance_after': self.current_balance
                })
        
        return trade_result 