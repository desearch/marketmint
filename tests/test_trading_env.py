import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_agent.trading_env import CryptoTradingEnv

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = {
        'price': np.random.normal(1000, 50, 100),
        'liquidity': np.random.normal(1000000, 100000, 100),
        'volume_24h': np.random.normal(500000, 50000, 100),
        'timestamp': [d.timestamp() for d in dates]
    }
    df = pd.DataFrame(data)
    df['sma_7'] = df['price'].rolling(window=7).mean()
    df['sma_30'] = df['price'].rolling(window=30).mean()
    df['rsi'] = np.random.normal(50, 10, 100)
    return df

def test_trading_env_initialization(sample_market_data):
    """Test trading environment initialization"""
    env = CryptoTradingEnv(sample_market_data)
    assert env.current_step == 0
    assert env.current_balance == 10000
    assert env.current_position == 0
    assert env.entry_price == 0
    assert env.transaction_cost == 0.001

def test_trading_env_step_buy(sample_market_data):
    """Test trading environment buy action"""
    env = CryptoTradingEnv(sample_market_data)
    observation, reward, done, info = env.step(np.array([1.0]))
    assert env.current_position > 0
    assert env.current_balance == 0
    assert env.entry_price == sample_market_data.iloc[0]['price']
    assert reward == 0

def test_trading_env_step_sell(sample_market_data):
    """Test trading environment sell action"""
    env = CryptoTradingEnv(sample_market_data)
    # First buy
    env.step(np.array([1.0]))
    # Then sell
    observation, reward, done, info = env.step(np.array([-1.0]))
    assert env.current_position == 0
    assert env.current_balance > 0
    assert reward != 0  # Should have a profit/loss reward

def test_trading_env_reset(sample_market_data):
    """Test trading environment reset"""
    env = CryptoTradingEnv(sample_market_data)
    # Execute some steps
    env.step(np.array([1.0]))
    env.step(np.array([-1.0]))
    # Reset
    observation = env.reset()
    assert env.current_step == 0
    assert env.current_balance == 10000
    assert env.current_position == 0
    assert env.entry_price == 0
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (6,)

def test_trading_env_observation_space(sample_market_data):
    """Test trading environment observation space"""
    env = CryptoTradingEnv(sample_market_data)
    observation = env._get_observation()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (6,)
    assert not np.any(np.isnan(observation))

def test_trading_env_action_space():
    """Test trading environment action space"""
    env = CryptoTradingEnv(pd.DataFrame())
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1
    assert env.action_space.high[0] == 1

def test_trading_env_portfolio_value(sample_market_data):
    """Test portfolio value calculation"""
    env = CryptoTradingEnv(sample_market_data)
    initial_value = env.current_balance
    
    # Buy action
    _, _, _, info = env.step(np.array([1.0]))
    assert abs(info['portfolio_value'] - initial_value) < 1e-6  # Account for transaction costs
    
    # Sell action
    _, _, _, info = env.step(np.array([-1.0]))
    assert info['portfolio_value'] > 0

def test_trading_env_transaction_costs(sample_market_data):
    """Test transaction costs are applied correctly"""
    env = CryptoTradingEnv(sample_market_data)
    initial_balance = env.current_balance
    
    # Buy action
    _, _, _, info = env.step(np.array([1.0]))
    expected_position = (initial_balance * (1 - env.transaction_cost)) / sample_market_data.iloc[0]['price']
    assert abs(env.current_position - expected_position) < 1e-6 