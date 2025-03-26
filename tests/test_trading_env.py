"""
Tests for trading environment
"""

import pytest
import numpy as np
import pandas as pd
from ai_agent.trading_env import CryptoTradingEnv

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = {
        'price': np.random.normal(1000, 50, 100),
        'liquidity': np.random.normal(1000000, 100000, 100),
        'volume_24h': np.random.normal(500000, 50000, 100),
        'timestamp': [d.timestamp() for d in dates],
        'sma_7': None,
        'sma_30': None,
        'rsi': np.random.normal(50, 10, 100)
    }
    df = pd.DataFrame(data)
    df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
    df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
    return df

def test_trading_env_initialization(sample_market_data):
    """Test trading environment initialization"""
    env = CryptoTradingEnv(sample_market_data)
    assert env.current_step == 0
    assert env.current_balance == 10000.0
    assert env.current_position == 0.0
    assert env.last_action == 0.0

def test_trading_env_step_buy(sample_market_data):
    """Test trading environment buy action"""
    env = CryptoTradingEnv(sample_market_data)
    initial_balance = env.current_balance
    current_price = sample_market_data.iloc[0]['price']
    
    # Execute buy action
    observation, reward, done, truncated, info = env.step(np.array([1.0]))
    
    # Calculate expected position
    trade_size = initial_balance / current_price
    transaction_cost = trade_size * current_price * env.transaction_cost
    max_affordable = (initial_balance - transaction_cost) / current_price
    expected_position = max_affordable
    
    assert abs(env.current_position - expected_position) < 1e-6
    assert env.current_balance >= 0
    assert env.current_balance < initial_balance

def test_trading_env_step_sell(sample_market_data):
    """Test trading environment sell action"""
    env = CryptoTradingEnv(sample_market_data)
    initial_balance = env.current_balance
    current_price = sample_market_data.iloc[0]['price']
    
    # First buy
    env.step(np.array([1.0]))
    initial_position = env.current_position
    
    # Then sell
    observation, reward, done, truncated, info = env.step(np.array([-1.0]))
    
    # Calculate expected values
    transaction_cost = initial_position * current_price * env.transaction_cost
    expected_balance = initial_balance - (2 * transaction_cost)  # Cost for both buy and sell
    
    assert abs(env.current_position) < 1e-6  # Position should be close to 0
    assert abs(env.current_balance - expected_balance) < 1e-6
    assert reward != 0  # Should have a non-zero reward due to transaction costs

def test_trading_env_reset(sample_market_data):
    """Test trading environment reset"""
    env = CryptoTradingEnv(sample_market_data)
    
    # Execute some actions
    env.step(np.array([1.0]))
    env.step(np.array([-0.5]))
    
    # Reset environment
    observation, info = env.reset()
    
    assert env.current_step == 0
    assert env.current_position == 0.0
    assert env.current_balance == env.initial_balance
    assert env.last_action == 0.0

def test_trading_env_observation_space(sample_market_data):
    """Test observation space"""
    env = CryptoTradingEnv(sample_market_data)
    observation, _ = env.reset()
    
    assert observation.shape == (6,)
    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.float32

def test_trading_env_action_space(sample_market_data):
    """Test action space"""
    env = CryptoTradingEnv(sample_market_data)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
    assert env.action_space.shape == (1,)

def test_trading_env_portfolio_value(sample_market_data):
    """Test portfolio value calculation"""
    env = CryptoTradingEnv(sample_market_data)
    initial_balance = env.current_balance
    current_price = sample_market_data.iloc[0]['price']
    
    # Execute buy action
    observation, reward, done, truncated, info = env.step(np.array([1.0]))
    
    # Calculate expected portfolio value
    trade_size = initial_balance / current_price
    transaction_cost = trade_size * current_price * env.transaction_cost
    max_affordable = (initial_balance - transaction_cost) / current_price
    expected_position_value = max_affordable * current_price
    expected_portfolio_value = env.current_balance + expected_position_value
    
    assert abs(info['portfolio_value'] - expected_portfolio_value) < 1e-6

def test_trading_env_transaction_costs(sample_market_data):
    """Test transaction costs are applied correctly"""
    env = CryptoTradingEnv(sample_market_data)
    initial_balance = env.current_balance
    current_price = sample_market_data.iloc[0]['price']
    
    # Execute buy action
    observation, reward, done, truncated, info = env.step(np.array([1.0]))
    
    # Calculate expected position after transaction costs
    trade_size = initial_balance / current_price
    transaction_cost = trade_size * current_price * env.transaction_cost
    max_affordable = (initial_balance - transaction_cost) / current_price
    
    assert abs(env.current_position - max_affordable) < 1e-6
    assert env.current_balance >= 0
    assert env.current_balance < initial_balance 