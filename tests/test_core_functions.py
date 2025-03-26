"""
Tests for core functions that support API endpoints
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from ai_agent.core_functions import CoreFunctions
from ai_agent.nft_agent import NFTAgent, NFT
from ai_agent.profit_distributor import ProfitDistributor

@pytest.fixture
def mock_nft_agent():
    """Create a mock NFT agent with predefined behavior"""
    agent = Mock(spec=NFTAgent)
    
    # Mock NFTs
    agent.nfts = {
        'nft1': NFT(id='nft1', strategy_id='strategy1', governance_tokens=1000.0, owner='owner1', 
                    performance_metrics={'total_profit': 100.0}),
        'nft2': NFT(id='nft2', strategy_id='strategy1', governance_tokens=1500.0, owner='owner2',
                    performance_metrics={'total_profit': 150.0})
    }
    
    # Create mock strategy runner
    strategy_runner = Mock()
    strategy_runner.strategies = {
        'strategy1': ['agent1', 'agent2'],
        'strategy2': ['agent3']
    }
    agent.strategy_runner = strategy_runner
    
    # Mock methods
    agent.update_market_state.return_value = None
    agent.get_aggregated_action.return_value = 0.5
    agent.get_aggregated_orders.return_value = [
        {'type': 'buy', 'amount': 1.0, 'price': 100.0}
    ]
    agent.distribute_profits.return_value = {
        'owner1': 60.0,
        'owner2': 40.0
    }
    agent.get_strategy_metrics.return_value = {
        'total_profit': 1000.0,
        'sharpe_ratio': 2.5,
        'win_rate': 0.65
    }
    agent.get_nft_performance.return_value = {
        'total_profit': 100.0,
        'current_drawdown': 0.05,
        'sharpe_ratio': 1.8,
        'win_rate': 0.6
    }
    
    return agent

@pytest.fixture
def mock_profit_distributor():
    """Create a mock profit distributor with predefined behavior"""
    distributor = Mock(spec=ProfitDistributor)
    
    # Mock methods
    distributor.distribute_profits.return_value = {
        'status': 'success',
        'transaction_hash': '0x123...abc',
        'gas_used': 100000
    }
    distributor.get_nft_holders.return_value = [
        {'address': 'owner1', 'token_id': 'nft1', 'share': 0.6},
        {'address': 'owner2', 'token_id': 'nft2', 'share': 0.4}
    ]
    distributor.get_total_distributed.return_value = 1000.0
    
    return distributor

@pytest.fixture
def core_functions(mock_nft_agent, mock_profit_distributor):
    """Create CoreFunctions instance with mock dependencies"""
    return CoreFunctions(mock_nft_agent, mock_profit_distributor)

def test_run_trade_success(core_functions):
    """Test successful trade execution"""
    market_state = {
        'price': 100.0,
        'volume': 1000.0,
        'timestamp': datetime.now()
    }
    
    result = core_functions.run_trade(market_state)
    
    assert result['status'] == 'success'
    assert 'trade' in result
    assert result['trade']['action'] == 0.5
    assert result['trade']['price'] == 100.0
    assert result['trade']['volume'] == 1000.0
    assert len(core_functions.trade_history) == 1

def test_run_trade_failure(core_functions, mock_nft_agent):
    """Test trade execution failure"""
    mock_nft_agent.update_market_state.side_effect = Exception("Market data invalid")
    
    result = core_functions.run_trade({'price': 100.0})
    
    assert result['status'] == 'error'
    assert 'message' in result
    assert len(core_functions.trade_history) == 0

def test_get_trade_history(core_functions):
    """Test retrieving trade history"""
    # Add some test trades
    trades = [
        {'timestamp': datetime.now(), 'action': 0.5, 'price': 100.0},
        {'timestamp': datetime.now(), 'action': -0.3, 'price': 101.0}
    ]
    core_functions.trade_history.extend(trades)
    
    history = core_functions.get_trade_history(limit=1)
    assert len(history) == 1
    assert history[0] == trades[1]

def test_get_market_price_available(core_functions):
    """Test getting market price when data is available"""
    core_functions.current_market_state = {
        'price': 100.0,
        'volume': 1000.0
    }
    
    result = core_functions.get_market_price('BTC')
    
    assert result['status'] == 'success'
    assert result['token'] == 'BTC'
    assert result['price'] == 100.0
    assert result['volume'] == 1000.0

def test_get_market_price_unavailable(core_functions):
    """Test getting market price when no data is available"""
    core_functions.current_market_state = {}
    
    result = core_functions.get_market_price('BTC')
    
    assert result['status'] == 'error'
    assert 'message' in result

def test_get_strategy_status(core_functions):
    """Test getting strategy status and metrics"""
    status = core_functions.get_strategy_status()
    
    assert 'timestamp' in status
    assert 'strategies' in status
    assert len(status['strategies']) == 2
    assert 'metrics' in status['strategies']['strategy1']
    assert status['strategies']['strategy1']['active_nfts'] == 2

def test_distribute_profits(core_functions):
    """Test profit distribution to NFT holders"""
    result = core_functions.distribute_profits(100.0)
    
    assert result['status'] == 'success'
    assert result['owner_profits'] == {'owner1': 60.0, 'owner2': 40.0}
    assert result['transaction_hash'] == '0x123...abc'
    assert 'timestamp' in result

def test_get_holder_earnings(core_functions):
    """Test getting NFT holder earnings"""
    result = core_functions.get_holder_earnings()
    
    assert result['status'] == 'success'
    assert result['total_distributed'] == 1000.0
    assert len(result['holders']) == 2
    assert len(result['nft_earnings']) == 2
    assert result['nft_earnings']['nft1']['owner'] == 'owner1'
    assert result['nft_earnings']['nft1']['total_profit'] == 100.0

def test_create_governance_proposal_success(core_functions):
    """Test creating a valid governance proposal"""
    proposal = {
        'type': 'strategy_change',
        'description': 'Update strategy parameters',
        'parameters': {'target_spread': 0.003}
    }
    
    result = core_functions.create_governance_proposal(proposal)
    
    assert result['status'] == 'success'
    assert result['proposal']['id'] == 1
    assert result['proposal']['type'] == 'strategy_change'
    assert len(core_functions.proposals) == 1

def test_create_governance_proposal_invalid(core_functions):
    """Test creating an invalid governance proposal"""
    proposal = {
        'type': 'strategy_change',
        # Missing required fields
    }
    
    result = core_functions.create_governance_proposal(proposal)
    
    assert result['status'] == 'error'
    assert 'message' in result
    assert len(core_functions.proposals) == 0

def test_get_governance_proposals(core_functions):
    """Test getting governance proposals with and without status filter"""
    proposals = [
        {
            'id': 1,
            'type': 'strategy_change',
            'status': 'active',
            'timestamp': datetime.now()
        },
        {
            'id': 2,
            'type': 'volume_rebalance',
            'status': 'completed',
            'timestamp': datetime.now()
        }
    ]
    core_functions.proposals.extend(proposals)
    
    # Test getting all proposals
    all_proposals = core_functions.get_governance_proposals()
    assert len(all_proposals) == 2
    
    # Test filtering by status
    active_proposals = core_functions.get_governance_proposals(status='active')
    assert len(active_proposals) == 1
    assert active_proposals[0]['id'] == 1 