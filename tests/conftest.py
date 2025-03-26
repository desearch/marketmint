"""
Pytest configuration file for NFT agent tests
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ['TESTING'] = 'true'
    os.environ['WEB3_PROVIDER'] = 'http://localhost:8545'
    os.environ['CONTRACT_ADDRESS'] = '0x1234567890123456789012345678901234567890'
    yield
    del os.environ['TESTING']
    del os.environ['WEB3_PROVIDER']
    del os.environ['CONTRACT_ADDRESS']

@pytest.fixture
def test_data():
    """Common test data"""
    return {
        'strategy_id': 'test_strategy',
        'nft_id': 'test_nft',
        'owner_address': '0x1234567890123456789012345678901234567890',
        'governance_tokens': 1000,
        'total_capital': 1000000,
        'risk_free_rate': 0.02,
        'max_drawdown': 0.1,
        'min_profit_threshold': 0.05
    }

@pytest.fixture
def market_state():
    """Sample market state data"""
    return {
        'price': 100.0,
        'volume': 1000.0,
        'timestamp': 1000,
        'high': 105.0,
        'low': 95.0,
        'open': 98.0,
        'close': 102.0
    }

@pytest.fixture
def sample_metrics():
    """Sample performance metrics"""
    return {
        'total_profit': 1000.0,
        'current_drawdown': 0.05,
        'sharpe_ratio': 1.5,
        'win_rate': 0.6,
        'max_drawdown': 0.1,
        'avg_trade_size': 100.0,
        'num_trades': 50
    } 