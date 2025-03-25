from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseSimulation(ABC):
    """Base class for all trading simulations"""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        max_position: Optional[float] = None
    ):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Trading state
        self.current_balance = initial_balance
        self.current_position = 0
        self.entry_price = 0
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the strategy"""
        pass
    
    @abstractmethod
    def execute_trade(self, signal: float, price: float) -> Dict[str, Any]:
        """Execute a trade based on the signal"""
        pass
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the simulation on historical data"""
        # Reset state
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0
        self.trades = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        
        # Generate signals
        data_with_signals = self.generate_signals(data)
        
        # Run simulation
        for i in range(len(data)):
            current_price = data.iloc[i]['price']
            signal = data_with_signals.iloc[i]['signal']
            
            if signal != 0:  # If there's a trading signal
                trade_result = self.execute_trade(signal, current_price)
                self.trades.append(trade_result)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            
            # Update max drawdown
            self._update_max_drawdown(portfolio_value)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        return performance
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.current_balance + self.current_position * current_price
    
    def _update_max_drawdown(self, current_value: float) -> None:
        """Update maximum drawdown"""
        if len(self.portfolio_values) > 0:
            peak = max(self.portfolio_values)
            drawdown = (peak - current_value) / peak
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.portfolio_values:
            return {}
        
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Get risk metrics using the safer calculation
        risk_metrics = self.calculate_risk_metrics()
        
        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'max_drawdown': risk_metrics['max_drawdown'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'volatility': risk_metrics['volatility'],
            'trades': self.trades,
            'portfolio_values': self.portfolio_values
        }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        returns = self.calculate_returns()
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }
        
        std_dev = np.std(returns)
        if std_dev == 0:
            sharpe_ratio = 0.0  # No volatility means no risk-adjusted return
        else:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / std_dev
        
        max_drawdown = self.calculate_max_drawdown()
        volatility = std_dev * np.sqrt(252)  # Annualized volatility
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility)
        }

    def calculate_returns(self) -> np.ndarray:
        """Calculate portfolio returns"""
        if len(self.portfolio_values) < 2:
            return np.array([])
        return np.diff(self.portfolio_values) / self.portfolio_values[:-1]

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_drawdown = 0.0
        
        for value in self.portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return float(max_drawdown)