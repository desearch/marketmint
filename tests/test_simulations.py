import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from simulations.base import BaseSimulation
from simulations.strategies.ma_crossover import MACrossoverStrategy
from simulations.performance import PerformanceAnalyzer

def create_sample_data(n_steps: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=n_steps, freq='D')
    
    # Generate random walk prices
    price = 100 * np.ones(n_steps)
    returns = np.random.normal(0, 0.02, n_steps)
    for i in range(1, n_steps):
        price[i] = price[i-1] * (1 + returns[i])
    
    # Create DataFrame with price and volume
    df = pd.DataFrame({
        'price': price,
        'volume': np.random.uniform(1000, 10000, n_steps)
    }, index=dates)
    
    return df

class TestSimulation(BaseSimulation):
    """Concrete implementation of BaseSimulation for testing"""
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['signal'] = np.random.choice([-1, 0, 1], size=len(df))
        return df
    
    def execute_trade(self, signal: float, price: float) -> Dict[str, Any]:
        trade_result = {
            'timestamp': pd.Timestamp.now(),
            'price': price,
            'signal': signal,
            'position_before': self.current_position,
            'balance_before': self.current_balance
        }
        
        if signal > 0:  # Buy
            self.current_position = 1
            self.current_balance -= price
        elif signal < 0:  # Sell
            self.current_position = 0
            self.current_balance += price
        
        trade_result.update({
            'position_after': self.current_position,
            'balance_after': self.current_balance
        })
        
        return trade_result

class TestBaseSimulation(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.market_data = create_sample_data()
        self.simulation = TestSimulation(
            initial_balance=10000,
            transaction_cost=0.001
        )
    
    def test_initialization(self):
        """Test simulation initialization"""
        self.assertEqual(self.simulation.initial_balance, 10000)
        self.assertEqual(self.simulation.transaction_cost, 0.001)
        self.assertEqual(self.simulation.current_balance, 10000)
        self.assertEqual(self.simulation.current_position, 0)
        self.assertEqual(self.simulation.total_trades, 0)
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        # Test with no position
        self.assertEqual(
            self.simulation._calculate_portfolio_value(100),
            10000
        )
        
        # Test with position
        self.simulation.current_position = 1
        self.simulation.current_balance = 9000
        self.assertEqual(
            self.simulation._calculate_portfolio_value(100),
            9100  # 9000 + (1 * 100)
        )
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Simulate some portfolio values
        self.simulation.portfolio_values = [10000, 11000, 9000, 9500]
        
        # Calculate max drawdown
        self.simulation._update_max_drawdown(9000)
        self.assertAlmostEqual(self.simulation.max_drawdown, 0.1818, places=4)  # (11000 - 9000) / 11000

class TestMACrossoverStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.market_data = create_sample_data()
        self.strategy = MACrossoverStrategy(
            initial_balance=10000,
            transaction_cost=0.001,
            short_window=7,
            long_window=30
        )
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        data_with_signals = self.strategy.generate_signals(self.market_data)
        
        # Check if signals are generated
        self.assertIn('signal', data_with_signals.columns)
        self.assertIn('sma_short', data_with_signals.columns)
        self.assertIn('sma_long', data_with_signals.columns)
        self.assertIn('rsi', data_with_signals.columns)
        
        # Check signal values
        self.assertTrue(all(data_with_signals['signal'].isin([-1, 0, 1])))
    
    def test_trade_execution(self):
        """Test trade execution"""
        # Test buy signal
        buy_result = self.strategy.execute_trade(1, 100)
        self.assertAlmostEqual(self.strategy.current_position, 99.9, places=1)  # (10000 / 100) * (1 - 0.001)
        self.assertAlmostEqual(self.strategy.current_balance, 0.0, places=1)  # 10000 - (99.9 * 100 * 1.001)
        
        # Test sell signal
        sell_result = self.strategy.execute_trade(-1, 110)
        self.assertEqual(self.strategy.current_position, 0)
        self.assertGreater(self.strategy.current_balance, 10000)  # Should have made a profit
    
    def test_strategy_performance(self):
        """Test overall strategy performance"""
        # Run strategy
        results = self.strategy.run(self.market_data)
        
        # Check results
        self.assertIn('total_return', results)
        self.assertIn('total_trades', results)
        self.assertIn('win_rate', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('sharpe_ratio', results)
        
        # Check portfolio values
        self.assertEqual(len(results['portfolio_values']), len(self.market_data))
        self.assertTrue(all(isinstance(v, float) for v in results['portfolio_values']))

class TestPerformanceAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create sample results with datetime index
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_values = [10000 + i * 100 for i in range(100)]
        
        self.results = {
            'initial_balance': 10000,
            'final_value': 11000,
            'total_return': 0.1,
            'total_trades': 10,
            'winning_trades': 6,
            'losing_trades': 4,
            'win_rate': 0.6,
            'max_drawdown': 0.1,
            'sharpe_ratio': 1.5,
            'portfolio_values': pd.Series(portfolio_values, index=dates),
            'trades': pd.DataFrame({
                'timestamp': dates[:10],
                'profit': np.random.normal(100, 50, 10)
            })
        }
        
        self.analyzer = PerformanceAnalyzer(self.results)
    
    def test_report_generation(self):
        """Test performance report generation"""
        report = self.analyzer.generate_report()
        
        # Check report structure
        self.assertIn('summary', report)
        self.assertIn('trade_analysis', report)
        self.assertIn('risk_metrics', report)
        self.assertIn('monthly_returns', report)
        
        # Check summary metrics
        self.assertEqual(report['summary']['initial_balance'], 10000)
        self.assertEqual(report['summary']['final_value'], 11000)
        self.assertEqual(report['summary']['total_return'], 0.1)
    
    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        report = self.analyzer.generate_report()
        risk_metrics = report['risk_metrics']
        
        # Check risk metrics
        self.assertIn('volatility', risk_metrics)
        self.assertIn('sortino_ratio', risk_metrics)
        self.assertIn('calmar_ratio', risk_metrics)
        self.assertIn('var_95', risk_metrics)
        self.assertIn('cvar_95', risk_metrics)
    
    def test_trade_analysis(self):
        """Test trade analysis"""
        report = self.analyzer.generate_report()
        trade_analysis = report['trade_analysis']
        
        # Check trade analysis
        self.assertIn('average_profit', trade_analysis)
        self.assertIn('profit_factor', trade_analysis)
        self.assertIn('average_win', trade_analysis)
        self.assertIn('average_loss', trade_analysis)
        self.assertIn('largest_win', trade_analysis)
        self.assertIn('largest_loss', trade_analysis) 