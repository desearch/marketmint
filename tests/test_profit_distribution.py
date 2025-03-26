import pytest
from web3 import Web3
from ai_agent.profit_distributor import ProfitDistributor
from ai_agent.nft_agent import NFTAgent
from ai_agent.blockchain_bridge import BlockchainBridge

@pytest.fixture
def web3_provider():
    return "http://localhost:8545"  # Local testnet

@pytest.fixture
def contract_addresses():
    return {
        "nftContract": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "profitDistributor": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
    }

@pytest.fixture
def blockchain_bridge(web3_provider, contract_addresses):
    return BlockchainBridge(
        web3_provider=web3_provider,
        contract_address=contract_addresses["nftContract"],
        private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # Test account
        contract_abi_path="contracts/artifacts/contracts/AMMAgentNFT.sol/AMMAgentNFT.json"
    )

@pytest.fixture
def nft_agent(blockchain_bridge):
    return NFTAgent(
        total_capital=10000.0,
        blockchain_bridge=blockchain_bridge,
        risk_free_rate=0.02,
        max_drawdown=0.1,
        min_profit_threshold=0.05
    )

@pytest.fixture
def profit_distributor(blockchain_bridge, contract_addresses):
    return ProfitDistributor(
        web3_provider="http://localhost:8545",
        contract_address=contract_addresses["profitDistributor"],
        private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        contract_abi_path="contracts/artifacts/contracts/ProfitDistributor.sol/ProfitDistributor.json"
    )

def test_profit_distribution_initialization(profit_distributor):
    """Test profit distributor initialization"""
    assert profit_distributor.contract is not None
    assert profit_distributor.w3 is not None

def test_profit_distribution_to_holders(profit_distributor, nft_agent):
    """Test profit distribution to NFT holders"""
    # Test data
    total_profit = 1.0  # 1 ETH
    
    # Distribute profits
    result = profit_distributor.distribute_profits(total_profit)
    
    assert result['status'] == 'success'
    assert result['amount_distributed'] == total_profit
    
    # Verify distribution in contract
    distributed = profit_distributor.get_total_distributed()
    assert distributed >= total_profit

def test_profit_claiming_by_holders(profit_distributor, nft_agent):
    """Test profit claiming by NFT holders"""
    # Test data
    token_id = 1
    recipient = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    
    # Get claimable amount before claiming
    claimable_before = profit_distributor.get_claimable_profits(token_id)
    
    # Claim profits
    result = profit_distributor.claim_profits(token_id)
    
    assert result['status'] == 'success'
    assert result['amount_claimed'] > 0
    
    # Verify claimable amount decreased
    claimable_after = profit_distributor.get_claimable_profits(token_id)
    assert claimable_after < claimable_before

def test_nft_holder_shares(profit_distributor, nft_agent):
    """Test NFT holder share calculations"""
    # Get NFT holders
    holders = profit_distributor.get_nft_holders()
    
    assert len(holders) > 0
    for holder in holders:
        assert holder['address'] is not None
        assert holder['token_id'] is not None
        assert holder['share'] > 0

def test_distribution_history(profit_distributor):
    """Test profit distribution history tracking"""
    # Get distribution history
    history = profit_distributor.get_distribution_history(limit=10)
    
    assert isinstance(history, list)
    for entry in history:
        assert 'amount' in entry
        assert 'timestamp' in entry
        assert 'nft_holders' in entry
        assert entry['amount'] >= 0

def test_total_distributed_amount(profit_distributor):
    """Test total distributed amount tracking"""
    # Get total distributed
    total = profit_distributor.get_total_distributed()
    
    assert total >= 0
    assert isinstance(total, float)

def test_error_handling(profit_distributor):
    """Test error handling in profit distribution"""
    # Test invalid profit amount
    with pytest.raises(ValueError):
        profit_distributor.distribute_profits(-1.0)  # Negative profit
    
    # Test invalid token ID
    with pytest.raises(ValueError):
        profit_distributor.claim_profits(999999)  # Non-existent token ID

def test_profit_distribution_events(profit_distributor):
    """Test profit distribution events"""
    # Distribute profits
    result = profit_distributor.distribute_profits(1.0)
    
    # Get events
    events = profit_distributor.get_distribution_events()
    
    assert len(events) > 0
    latest_event = events[0]
    assert latest_event['amount'] == 1.0
    assert latest_event['timestamp'] is not None 