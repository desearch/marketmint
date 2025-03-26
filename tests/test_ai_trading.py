import pytest
import numpy as np
import pandas as pd
from ai_agent.strategy import AITrader
from ai_agent.market_maker import MarketMaker
from ai_agent.strategies.pure_market_maker import PureMarketMaker

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100),
        'liquidity': np.random.normal(10000, 1000, 100)
    }, index=dates)
    
    # Add technical indicators
    data['sma_7'] = data['close'].rolling(window=7, min_periods=1).mean()
    data['sma_30'] = data['close'].rolling(window=30, min_periods=1).mean()
    data['rsi'] = np.random.uniform(0, 100, 100)
    
    return data

@pytest.fixture
def ai_trader(sample_market_data):
    """Create AI trader instance"""
    return AITrader(
        market_data=sample_market_data,
        initial_balance=10000.0,
        transaction_cost=0.001
    )

@pytest.fixture
def market_maker():
    """Create market maker instance"""
    return MarketMaker(
        strategy=PureMarketMaker(
            initial_spread=0.002,
            min_spread=0.001,
            max_spread=0.005,
            inventory_target=0.0,
            inventory_limit=1.0
        )
    )

def test_ai_trader_initialization(ai_trader):
    """Test AI trader initialization"""
    assert ai_trader.initial_balance == 10000.0
    assert ai_trader.transaction_cost == 0.001
    assert ai_trader.market_data is not None

def test_ai_trader_preprocessing(ai_trader):
    """Test market data preprocessing"""
    processed_data = ai_trader.preprocess_data(ai_trader.market_data)
    
    assert processed_data is not None
    assert len(processed_data) == len(ai_trader.market_data)
    assert 'volume' in processed_data.columns
    assert 'liquidity' in processed_data.columns
    assert 'rsi' in processed_data.columns
    assert not processed_data.isnull().any().any()

def test_ai_trader_training(ai_trader):
    """Test AI trader training"""
    result = ai_trader.train(
        total_timesteps=1000,
        learning_rate=0.0003,
        batch_size=64
    )
    
    assert result['status'] == 'success'
    assert 'total_timesteps' in result
    assert 'final_reward' in result

def test_ai_trader_prediction(ai_trader):
    """Test AI trader prediction"""
    # Train the model first
    ai_trader.train(total_timesteps=1000)
    
    # Get prediction
    prediction = ai_trader.predict(ai_trader.market_data.iloc[-1])
    
    assert isinstance(prediction, float)
    assert -1.0 <= prediction <= 1.0

def test_market_maker_initialization(market_maker):
    """Test market maker initialization"""
    assert market_maker.strategy is not None
    assert isinstance(market_maker.strategy, PureMarketMaker)

def test_market_maker_action(market_maker):
    """Test market maker action generation"""
    # Set market state
    market_state = {
        'price': 100.0,
        'volume': 1000.0,
        'liquidity': 10000.0,
        'inventory': 0.0,
        'spread': 0.002
    }
    
    # Get action
    action = market_maker.get_action(market_state)
    
    assert isinstance(action, float)
    assert -1.0 <= action <= 1.0

def test_market_maker_orders(market_maker):
    """Test market maker order generation"""
    # Set market state
    market_state = {
        'price': 100.0,
        'volume': 1000.0,
        'liquidity': 10000.0,
        'inventory': 0.0,
        'spread': 0.002
    }
    
    # Get orders
    orders = market_maker.get_orders(market_state)
    
    assert isinstance(orders, dict)
    assert 'bid_price' in orders
    assert 'ask_price' in orders
    assert 'bid_size' in orders
    assert 'ask_size' in orders
    assert orders['bid_price'] < orders['ask_price']

def test_market_maker_spreads(market_maker):
    """Test market maker spread calculations"""
    # Set market state
    market_state = {
        'price': 100.0,
        'volume': 1000.0,
        'liquidity': 10000.0,
        'inventory': 0.0,
        'spread': 0.002
    }
    
    # Get spreads
    spreads = market_maker.get_spreads(market_state)
    
    assert isinstance(spreads, dict)
    assert 'bid_spread' in spreads
    assert 'ask_spread' in spreads
    assert spreads['bid_spread'] >= 0
    assert spreads['ask_spread'] >= 0

def test_market_maker_inventory(market_maker):
    """Test market maker inventory management"""
    # Set market state
    market_state = {
        'price': 100.0,
        'volume': 1000.0,
        'liquidity': 10000.0,
        'inventory': 0.0,
        'spread': 0.002
    }
    
    # Get inventory metrics
    metrics = market_maker.get_inventory_metrics(market_state)
    
    assert isinstance(metrics, dict)
    assert 'current_inventory' in metrics
    assert 'inventory_target' in metrics
    assert 'inventory_limit' in metrics
    assert abs(metrics['current_inventory']) <= metrics['inventory_limit'] 