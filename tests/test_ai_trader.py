import pytest
import pandas as pd
import numpy as np
from ai_agent.strategy import AITrader
from data.database import Database

@pytest.fixture
def market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'price': np.random.normal(2000, 100, len(dates)),
        'volume': np.random.normal(500000, 50000, len(dates)),
        'liquidity': np.random.normal(1000000, 100000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def ai_trader(market_data):
    """Initialize AI trader with sample market data"""
    trader = AITrader()
    trader.create_env(market_data)
    return trader

@pytest.fixture
def database():
    return Database()

def test_ai_trader_initialization(ai_trader):
    """Test AI trader initialization"""
    assert ai_trader.env is not None
    assert ai_trader.model is None

def test_ai_trader_preprocessing(ai_trader, market_data):
    """Test data preprocessing"""
    processed_data = ai_trader.preprocess_data(market_data)
    assert 'sma_7' in processed_data.columns
    assert 'sma_30' in processed_data.columns
    assert 'rsi' in processed_data.columns
    assert not processed_data.isnull().any().any()

def test_ai_trader_training(ai_trader, market_data):
    """Test model training"""
    results = ai_trader.train(market_data, total_timesteps=1000)
    assert ai_trader.model is not None
    assert 'total_timesteps' in results
    assert 'final_reward' in results

def test_ai_trader_prediction(ai_trader, market_data):
    """Test trading predictions"""
    ai_trader.train(market_data, total_timesteps=1000)
    predictions = ai_trader.predict(market_data)
    assert len(predictions) > 0
    assert all(-1.0 <= p <= 1.0 for p in predictions)

def test_trade_execution(ai_trader):
    """Test basic trade execution"""
    amount = 1.0  # 1 ETH
    result = ai_trader.execute_trade(amount)
    
    assert isinstance(result, dict)
    assert 'action' in result
    assert 'reward' in result
    assert 'portfolio_value' in result
    assert 'current_price' in result

def test_database_operations(database):
    """Test database operations"""
    # Test adding a trade
    database.add_trade(
        action='buy',
        amount=1.0,
        price=2000.0,
        status='completed',
        tx_hash='0x123'
    )
    
    # Test retrieving trade history
    trades = database.get_trade_history(limit=1)
    assert len(trades) > 0
    assert trades[0][2] == 'buy'  # action
    assert trades[0][3] == 1.0   # amount

def test_nft_ownership(database):
    """Test NFT ownership operations"""
    # Test updating NFT ownership
    database.update_nft_ownership(
        token_id=1,
        owner_address='0x456',
        share_percentage=1000  # 10%
    )
    
    # Test retrieving NFT ownership
    ownership = database.get_nft_ownership(1)
    assert ownership is not None
    assert ownership[1] == '0x456'  # owner_address
    assert ownership[2] == 1000     # share_percentage

def test_profit_distribution(database):
    """Test profit distribution operations"""
    # Test adding profit distribution
    database.add_profit_distribution(
        amount=10.0,  # 10 ETH
        total_shares=10000,
        tx_hash='0x789'
    )
    
    # Test retrieving profit distributions
    distributions = database.get_profit_distributions(limit=1)
    assert len(distributions) > 0
    assert distributions[0][2] == 10.0  # amount
    assert distributions[0][3] == 10000 # total_shares

def test_governance_proposals(database):
    """Test governance proposal operations"""
    # Test creating a proposal
    proposal_id = database.create_proposal(
        description='Test proposal',
        creator_address='0xabc'
    )
    
    # Test retrieving proposal status
    proposal = database.get_proposal_status(proposal_id)
    assert proposal is not None
    assert proposal[2] == 'Test proposal'  # description
    assert proposal[3] == 'active'         # status
    assert proposal[6] == '0xabc'          # creator_address 