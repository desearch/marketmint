"""
Tests for NFTAgent class
"""

import pytest
from unittest.mock import Mock, patch
from ai_agent.nft_agent import NFTAgent, NFT

@pytest.fixture
def mock_strategy_runner():
    """Create a mock StrategyRunner instance"""
    return Mock()

@pytest.fixture
def mock_blockchain_bridge():
    """Create a mock BlockchainBridge instance"""
    return Mock()

@pytest.fixture
def agent(mock_strategy_runner, mock_blockchain_bridge):
    """Create an NFTAgent instance with mocked dependencies"""
    with patch('ai_agent.nft_agent.StrategyRunner') as mock_runner_class:
        mock_runner_class.return_value = mock_strategy_runner
        return NFTAgent(
            total_capital=1000000,
            blockchain_bridge=mock_blockchain_bridge,
            risk_free_rate=0.02,
            max_drawdown=0.1,
            min_profit_threshold=0.05
        )

def test_add_strategy(agent, mock_strategy_runner, mock_blockchain_bridge):
    """Test adding a strategy"""
    # Setup mock
    strategy_id = 'test_strategy'
    strategy_params = {'param1': 'value1'}
    mock_blockchain_bridge.create_strategy.return_value = {
        'status': 'success',
        'transaction_hash': '0x1234'
    }
    
    # Call method
    result = agent.add_strategy(strategy_id, strategy_params)
    
    # Verify result
    assert result['status'] == 'success'
    assert result['strategy_id'] == strategy_id
    
    # Verify interactions
    mock_strategy_runner.add_strategy.assert_called_once_with(strategy_id, strategy_params)
    mock_blockchain_bridge.create_strategy.assert_called_once_with(strategy_id)

def test_mint_nft(agent, mock_strategy_runner, mock_blockchain_bridge):
    """Test minting an NFT"""
    # Setup mock
    nft_id = 'test_nft'
    owner = '0x1234'
    strategy_id = 'test_strategy'
    governance_tokens = 1000
    risk_limits = {'max_drawdown': 0.2}
    
    mock_blockchain_bridge.mint_nft.return_value = {
        'status': 'success',
        'token_id': 1
    }
    
    # Call method
    result = agent.mint_nft(nft_id, owner, strategy_id, governance_tokens, risk_limits)
    
    # Verify result
    assert result['status'] == 'success'
    assert result['nft_id'] == nft_id
    assert result['token_id'] == 1
    
    # Verify interactions
    mock_strategy_runner.add_agent.assert_called_once_with(
        strategy_id=strategy_id,
        agent_id=nft_id,
        governance_tokens=governance_tokens,
        risk_limits=risk_limits
    )
    
    mock_blockchain_bridge.mint_nft.assert_called_once_with(
        recipient=owner,
        strategy_id=strategy_id,
        governance_tokens=int(governance_tokens)
    )
    
    # Verify NFT was stored
    nft = agent.nfts[nft_id]
    assert nft.id == nft_id
    assert nft.strategy_id == strategy_id
    assert nft.governance_tokens == governance_tokens
    assert nft.owner == owner
    assert nft.token_id == 1

def test_update_market_state(agent, mock_strategy_runner):
    """Test updating market state"""
    state = {'price': 100, 'volume': 1000}
    
    # Call method
    agent.update_market_state(state)
    
    # Verify interaction
    mock_strategy_runner.update_market_state.assert_called_once_with(state)

def test_get_aggregated_orders(agent, mock_strategy_runner):
    """Test getting aggregated orders"""
    # Setup mock
    mock_orders = [
        {'action': 'buy', 'size': 100},
        {'action': 'sell', 'size': 50}
    ]
    mock_strategy_runner.get_aggregated_orders.return_value = mock_orders
    
    # Call method
    orders = agent.get_aggregated_orders()
    
    # Verify result
    assert orders == mock_orders
    mock_strategy_runner.get_aggregated_orders.assert_called_once()

def test_get_aggregated_action(agent, mock_strategy_runner):
    """Test getting aggregated action"""
    # Setup mock
    mock_strategy_runner.get_aggregated_action.return_value = 0.75
    
    # Call method
    action = agent.get_aggregated_action()
    
    # Verify result
    assert action == 0.75
    mock_strategy_runner.get_aggregated_action.assert_called_once()

def test_distribute_profits(agent, mock_strategy_runner, mock_blockchain_bridge):
    """Test profit distribution"""
    # Setup mocks
    total_profit = 1000
    mock_strategy_runner.distribute_profits.return_value = {
        'nft1': 600,  # 60% share
        'nft2': 400   # 40% share
    }
    
    # Setup NFTs
    agent.nfts = {
        'nft1': NFT(
            id='nft1',
            strategy_id='strategy1',
            governance_tokens=1500,
            owner='0x1234',
            performance_metrics={'total_profit': 0},
            token_id=1
        ),
        'nft2': NFT(
            id='nft2',
            strategy_id='strategy1',
            governance_tokens=1000,
            owner='0x5678',
            performance_metrics={'total_profit': 0},
            token_id=2
        )
    }
    
    mock_blockchain_bridge.distribute_profits.return_value = {
        'status': 'success',
        'amount_distributed': 1000
    }
    
    # Call method
    owner_profits = agent.distribute_profits(total_profit)
    
    # Verify results
    assert owner_profits['0x1234'] == 600
    assert owner_profits['0x5678'] == 400
    
    # Verify blockchain interaction
    mock_blockchain_bridge.distribute_profits.assert_called_once_with(
        'strategy1',
        1000
    )
    
    # Verify NFT metrics were updated
    assert agent.nfts['nft1'].performance_metrics['total_profit'] == 600
    assert agent.nfts['nft2'].performance_metrics['total_profit'] == 400

def test_get_nft_performance(agent, mock_strategy_runner, mock_blockchain_bridge):
    """Test getting NFT performance metrics"""
    # Setup mocks
    nft_id = 'test_nft'
    strategy_id = 'test_strategy'
    mock_metrics = {
        'total_profit': 1000,
        'current_drawdown': 0.05,
        'sharpe_ratio': 1.5
    }
    mock_strategy_runner.get_agent_performance.return_value = mock_metrics
    
    mock_blockchain_bridge.get_nft_data.return_value = {
        'status': 'success',
        'unclaimed_profits': 100
    }
    
    # Setup NFT
    agent.nfts[nft_id] = NFT(
        id=nft_id,
        strategy_id=strategy_id,
        governance_tokens=1000,
        owner='0x1234',
        performance_metrics={},
        token_id=1
    )
    
    # Call method
    metrics = agent.get_nft_performance(nft_id)
    
    # Verify results
    assert metrics == {**mock_metrics, 'unclaimed_profits': 100}
    
    # Verify interactions
    mock_strategy_runner.get_agent_performance.assert_called_once_with(strategy_id, nft_id)
    mock_blockchain_bridge.get_nft_data.assert_called_once_with(1)

def test_get_strategy_metrics(agent, mock_strategy_runner, mock_blockchain_bridge):
    """Test getting strategy metrics"""
    # Setup mocks
    strategy_id = 'test_strategy'
    mock_metrics = {
        'total_profit': 2000,
        'avg_drawdown': 0.03,
        'avg_sharpe': 1.8
    }
    mock_strategy_runner.get_strategy_metrics.return_value = mock_metrics
    
    mock_blockchain_bridge.get_strategy_data.return_value = {
        'status': 'success',
        'total_profits': 1500,
        'total_governance_tokens': 3000
    }
    
    # Call method
    metrics = agent.get_strategy_metrics(strategy_id)
    
    # Verify results
    expected_metrics = {
        **mock_metrics,
        'total_distributed_profits': 1500,
        'total_governance_tokens': 3000
    }
    assert metrics == expected_metrics
    
    # Verify interactions
    mock_strategy_runner.get_strategy_metrics.assert_called_once_with(strategy_id)
    mock_blockchain_bridge.get_strategy_data.assert_called_once_with(strategy_id)

def test_error_handling(agent, mock_blockchain_bridge):
    """Test error handling in NFTAgent"""
    # Test invalid NFT ID
    with pytest.raises(ValueError, match="NFT test_nft not found"):
        agent.get_nft_performance('test_nft')
    
    # Test blockchain errors
    mock_blockchain_bridge.create_strategy.return_value = {
        'status': 'error',
        'message': 'Transaction failed'
    }
    result = agent.add_strategy('test_strategy')
    assert result['status'] == 'error'
    assert 'Transaction failed' in result['message'] 