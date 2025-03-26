"""
Tests for BlockchainBridge class
"""

import pytest
from unittest.mock import Mock, patch
from web3 import Web3
from ai_agent.blockchain_bridge import BlockchainBridge

@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance"""
    mock = Mock()
    mock.eth.account.from_key.return_value = Mock(address='0x1234')
    mock.eth.gas_price = 20000000000
    mock.to_wei = Web3.to_wei
    mock.from_wei = Web3.from_wei
    mock.to_checksum_address = lambda x: x
    return mock

@pytest.fixture
def mock_contract():
    """Create a mock contract instance"""
    return Mock()

@pytest.fixture
def bridge(mock_web3, mock_contract):
    """Create a BlockchainBridge instance with mocked dependencies"""
    with patch('json.load') as mock_load:
        mock_load.return_value = {}
        mock_web3.eth.contract.return_value = mock_contract
        return BlockchainBridge(
            web3_provider='http://localhost:8545',
            contract_address='0x1234',
            private_key='0x5678',
            contract_abi_path='fake_path.json'
        )

def test_create_strategy(bridge, mock_web3, mock_contract):
    """Test strategy creation"""
    # Setup mock
    tx_hash = '0x9876'
    receipt = {
        'transactionHash': bytes.fromhex('9876'),
        'gasUsed': 100000
    }
    
    mock_web3.eth.get_transaction_count.return_value = 0
    mock_web3.eth.send_raw_transaction.return_value = tx_hash
    mock_web3.eth.wait_for_transaction_receipt.return_value = receipt
    
    # Call method
    result = bridge.create_strategy('test_strategy')
    
    # Verify result
    assert result['status'] == 'success'
    assert result['transaction_hash'] == '0x9876'
    assert result['gas_used'] == 100000
    
    # Verify contract interaction
    mock_contract.functions.createStrategy.assert_called_once_with('test_strategy')

def test_mint_nft(bridge, mock_web3, mock_contract):
    """Test NFT minting"""
    # Setup mock
    tx_hash = '0x9876'
    receipt = {
        'transactionHash': bytes.fromhex('9876'),
        'gasUsed': 100000
    }
    token_id = 1
    
    mock_web3.eth.get_transaction_count.return_value = 0
    mock_web3.eth.send_raw_transaction.return_value = tx_hash
    mock_web3.eth.wait_for_transaction_receipt.return_value = receipt
    
    # Setup event logs
    mock_contract.events.NFTMinted.return_value.process_receipt.return_value = [{
        'args': {'tokenId': token_id}
    }]
    
    # Call method
    result = bridge.mint_nft(
        recipient='0x5678',
        strategy_id='test_strategy',
        governance_tokens=1000
    )
    
    # Verify result
    assert result['status'] == 'success'
    assert result['transaction_hash'] == '0x9876'
    assert result['gas_used'] == 100000
    assert result['token_id'] == token_id
    
    # Verify contract interaction
    mock_contract.functions.mintNFT.assert_called_once_with(
        '0x5678',
        'test_strategy',
        1000
    )

def test_distribute_profits(bridge, mock_web3, mock_contract):
    """Test profit distribution"""
    # Setup mock
    tx_hash = '0x9876'
    receipt = {
        'transactionHash': bytes.fromhex('9876'),
        'gasUsed': 100000
    }
    
    mock_web3.eth.get_transaction_count.return_value = 0
    mock_web3.eth.send_raw_transaction.return_value = tx_hash
    mock_web3.eth.wait_for_transaction_receipt.return_value = receipt
    
    # Call method
    result = bridge.distribute_profits('test_strategy', 1.5)  # 1.5 ETH
    
    # Verify result
    assert result['status'] == 'success'
    assert result['transaction_hash'] == '0x9876'
    assert result['gas_used'] == 100000
    assert result['amount_distributed'] == 1.5
    
    # Verify contract interaction
    mock_contract.functions.distributeStrategyProfits.assert_called_once()
    call_args = mock_contract.functions.distributeStrategyProfits.call_args[0]
    assert call_args[0] == 'test_strategy'

def test_get_nft_data(bridge, mock_contract):
    """Test getting NFT data"""
    # Setup mock
    mock_contract.functions.getNFTData.return_value = (
        'test_strategy',
        1000,
        Web3.to_wei(0.5, 'ether')
    )
    
    # Call method
    result = bridge.get_nft_data(1)
    
    # Verify result
    assert result['status'] == 'success'
    assert result['token_id'] == 1
    assert result['strategy_id'] == 'test_strategy'
    assert result['governance_tokens'] == 1000
    assert result['unclaimed_profits'] == 0.5
    
    # Verify contract interaction
    mock_contract.functions.getNFTData.assert_called_once_with(1)

def test_get_strategy_data(bridge, mock_contract):
    """Test getting strategy data"""
    # Setup mock
    mock_contract.functions.getStrategyData.return_value = (
        3000,  # total governance tokens
        Web3.to_wei(2.5, 'ether'),  # total profits
        True  # is active
    )
    
    # Call method
    result = bridge.get_strategy_data('test_strategy')
    
    # Verify result
    assert result['status'] == 'success'
    assert result['strategy_id'] == 'test_strategy'
    assert result['total_governance_tokens'] == 3000
    assert result['total_profits'] == 2.5
    assert result['is_active'] == True
    
    # Verify contract interaction
    mock_contract.functions.getStrategyData.assert_called_once_with('test_strategy')

def test_error_handling(bridge, mock_contract):
    """Test error handling in bridge methods"""
    # Setup mock to raise exception
    mock_contract.functions.createStrategy.side_effect = Exception("Test error")
    
    # Test error handling in create_strategy
    result = bridge.create_strategy('test_strategy')
    assert result['status'] == 'error'
    assert 'Test error' in result['message']
    
    # Test error handling in other methods
    mock_contract.functions.mintNFT.side_effect = Exception("Test error")
    result = bridge.mint_nft('0x1234', 'test_strategy', 1000)
    assert result['status'] == 'error'
    assert 'Test error' in result['message']
    
    mock_contract.functions.distributeStrategyProfits.side_effect = Exception("Test error")
    result = bridge.distribute_profits('test_strategy', 1.0)
    assert result['status'] == 'error'
    assert 'Test error' in result['message'] 