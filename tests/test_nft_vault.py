import pytest
from web3 import Web3
from ai_agent.blockchain_bridge import BlockchainBridge
from ai_agent.nft_agent import NFTAgent
from ai_agent.profit_distributor import ProfitDistributor

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
        contract_address=contract_addresses["profitDistributor"],
        private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        contract_abi_path="contracts/artifacts/contracts/ProfitDistributor.sol/ProfitDistributor.json",
        rpc_url="http://localhost:8545"
    )

def test_nft_minting(nft_agent):
    """Test NFT minting functionality"""
    # Test data
    recipient = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    strategy_id = "test_strategy_1"
    governance_tokens = 1000
    
    # Create strategy first
    result = nft_agent.add_strategy(strategy_id)
    assert result['status'] == 'success'
    
    # Mint NFT
    result = nft_agent.mint_nft(
        nft_id="test_nft_1",
        owner=recipient,
        strategy_id=strategy_id,
        governance_tokens=governance_tokens
    )
    
    assert result['status'] == 'success'
    assert result['nft_id'] == "test_nft_1"
    assert result['token_id'] is not None

def test_profit_distribution(profit_distributor, nft_agent):
    """Test profit distribution to NFT holders"""
    # Test data
    total_profit = 1.0  # 1 ETH
    
    # Create strategy and mint NFT first
    strategy_id = "test_strategy_2"
    recipient = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    governance_tokens = 1000
    
    result = nft_agent.add_strategy(strategy_id)
    assert result['status'] == 'success'
    
    result = nft_agent.mint_nft(
        nft_id="test_nft_2",
        owner=recipient,
        strategy_id=strategy_id,
        governance_tokens=governance_tokens
    )
    assert result['status'] == 'success'
    
    # Distribute profits
    result = profit_distributor.distribute_profits(total_profit)
    
    assert result['status'] == 'success'
    assert result['amount_distributed'] == total_profit

def test_profit_claiming(profit_distributor, nft_agent):
    """Test profit claiming by NFT holders"""
    # Test data
    token_id = 1
    recipient = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    
    # Create strategy and mint NFT first
    strategy_id = "test_strategy_3"
    governance_tokens = 1000
    
    result = nft_agent.add_strategy(strategy_id)
    assert result['status'] == 'success'
    
    result = nft_agent.mint_nft(
        nft_id="test_nft_3",
        owner=recipient,
        strategy_id=strategy_id,
        governance_tokens=governance_tokens
    )
    assert result['status'] == 'success'
    
    # Claim profits
    result = profit_distributor.claim_profits(token_id)
    
    assert result['status'] == 'success'
    assert result['amount_claimed'] > 0

def test_nft_holder_shares(nft_agent):
    """Test NFT holder share calculations"""
    # Test data
    strategy_id = "test_strategy_4"
    recipient = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    governance_tokens = 1000
    nft_id = "test_nft_4"
    
    # Create strategy and mint NFT first
    result = nft_agent.add_strategy(strategy_id)
    assert result['status'] == 'success'
    
    result = nft_agent.mint_nft(
        nft_id=nft_id,
        owner=recipient,
        strategy_id=strategy_id,
        governance_tokens=governance_tokens
    )
    assert result['status'] == 'success'
    
    # Get NFT data
    nft = nft_agent.nfts.get(nft_id)
    
    assert nft is not None
    assert nft.governance_tokens > 0
    assert nft.performance_metrics['total_profit'] >= 0

def test_vault_balance(profit_distributor):
    """Test vault balance tracking"""
    # Get vault balance
    balance = profit_distributor.get_vault_balance()
    
    assert balance >= 0  # Balance should be non-negative

def test_error_handling(nft_agent, profit_distributor):
    """Test error handling in NFT and profit distribution"""
    # Test invalid NFT ID
    with pytest.raises(ValueError):
        nft_agent.get_nft_performance("invalid_nft_id")
    
    # Test invalid profit distribution
    with pytest.raises(ValueError):
        profit_distributor.distribute_profits(-1.0)  # Negative profit 