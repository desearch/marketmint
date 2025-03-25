import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    """Utility class for analyzing trading strategy performance"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        
        # Convert portfolio values to Series with datetime index if needed
        if isinstance(results['portfolio_values'], pd.Series):
            self.portfolio_values = results['portfolio_values']
        else:
            # Create a default datetime index if none provided
            dates = pd.date_range(start='2024-01-01', periods=len(results['portfolio_values']), freq='D')
            self.portfolio_values = pd.Series(results['portfolio_values'], index=dates)
        
        # Convert trades to DataFrame with datetime index if needed
        if isinstance(results['trades'], pd.DataFrame):
            self.trades = results['trades']
            if 'timestamp' not in self.trades.columns:
                self.trades['timestamp'] = pd.date_range(start='2024-01-01', periods=len(self.trades), freq='D')
                self.trades.set_index('timestamp', inplace=True)
        else:
            self.trades = pd.DataFrame()
    
    def plot_portfolio_value(self, title: str = "Portfolio Value Over Time") -> None:
        """Plot portfolio value over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_values.index, self.portfolio_values.values)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.show()
    
    def plot_drawdown(self, title: str = "Drawdown Over Time") -> None:
        """Plot drawdown over time"""
        # Calculate drawdown
        rolling_max = self.portfolio_values.expanding().max()
        drawdown = (rolling_max - self.portfolio_values) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown.values)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.show()
    
    def plot_trade_distribution(self, title: str = "Trade Profit Distribution") -> None:
        """Plot distribution of trade profits"""
        if 'profit' not in self.trades.columns:
            return
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.trades, x='profit', bins=50)
        plt.title(title)
        plt.xlabel("Profit")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        report = {
            'summary': {
                'initial_balance': self.results['initial_balance'],
                'final_value': self.results['final_value'],
                'total_return': self.results['total_return'],
                'total_trades': self.results['total_trades'],
                'win_rate': self.results['win_rate'],
                'max_drawdown': self.results['max_drawdown'],
                'sharpe_ratio': self.results['sharpe_ratio']
            },
            'trade_analysis': self._analyze_trades(),
            'risk_metrics': self._calculate_risk_metrics(),
            'monthly_returns': self._calculate_monthly_returns()
        }
        
        return report
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade statistics"""
        if self.trades.empty:
            return {}
        
        return {
            'average_profit': self.trades['profit'].mean() if 'profit' in self.trades.columns else 0,
            'profit_factor': self._calculate_profit_factor(),
            'average_win': self.trades[self.trades['profit'] > 0]['profit'].mean() if 'profit' in self.trades.columns else 0,
            'average_loss': self.trades[self.trades['profit'] < 0]['profit'].mean() if 'profit' in self.trades.columns else 0,
            'largest_win': self.trades['profit'].max() if 'profit' in self.trades.columns else 0,
            'largest_loss': self.trades['profit'].min() if 'profit' in self.trades.columns else 0
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics"""
        returns = self.portfolio_values.pct_change().dropna()
        
        return {
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(),
            'var_95': np.percentile(returns, 5),  # 95% Value at Risk
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()  # Conditional VaR
        }
    
    def _calculate_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns"""
        # Convert portfolio values to returns
        returns = self.portfolio_values.pct_change()
        
        # Resample to monthly frequency and calculate returns
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        
        return monthly_returns
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if 'profit' not in self.trades.columns:
            return 0
        
        gross_profit = self.trades[self.trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        
        if downside_std == 0:
            return float('inf')
        
        return (returns.mean() * 252) / downside_std
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (average annual return / max drawdown)"""
        if self.results['max_drawdown'] == 0:
            return float('inf')
        
        total_years = len(self.portfolio_values) / 252
        annual_return = (self.results['final_value'] / self.results['initial_balance']) ** (1 / total_years) - 1
        
        return annual_return / self.results['max_drawdown'] 