import pytest
import pandas as pd
import numpy as np
from ai_agent.strategy import AITrader
from data.database import Database

@pytest.fixture
def ai_trader():
    return AITrader()

@pytest.fixture
def database():
    return Database()

def test_trade_execution(ai_trader):
    """Test basic trade execution"""
    amount = 1.0  # 1 ETH
    result = ai_trader.execute_trade(amount)
    
    assert isinstance(result, dict)
    assert 'status' in result
    assert 'message' in result

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