"""
Tests for StrategyNFT smart contract using eth-tester
"""

import pytest
from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3, EthereumTesterProvider
from eth_account import Account
import json
import os
from eth_utils import to_wei, from_wei

@pytest.fixture
def eth_tester():
    """Create an EthereumTester instance"""
    return EthereumTester(PyEVMBackend())

@pytest.fixture
def web3(eth_tester):
    """Create a Web3 instance with EthereumTesterProvider"""
    provider = EthereumTesterProvider(eth_tester)
    w3 = Web3(provider)
    w3.eth.default_account = w3.eth.accounts[0]
    return w3

@pytest.fixture
def accounts(web3):
    """Get test accounts"""
    return web3.eth.accounts[:3]

@pytest.fixture
def contract_json():
    """Load contract JSON"""
    contract_path = os.path.join(os.path.dirname(__file__), '..', 'contracts', 'StrategyNFT.json')
    with open(contract_path) as f:
        return json.load(f)

@pytest.fixture
def strategy_nft(web3, contract_json, accounts):
    """Deploy the StrategyNFT contract"""
    contract = web3.eth.contract(
        abi=contract_json['abi'],
        bytecode=contract_json['bytecode']
    )
    
    # Deploy contract
    tx_hash = contract.constructor().transact({'from': accounts[0]})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    
    return web3.eth.contract(
        address=tx_receipt.contractAddress,
        abi=contract_json['abi']
    )

@pytest.fixture
def strategy_id():
    """Return a test strategy ID"""
    return "test_strategy"

def test_create_strategy(web3, strategy_nft, accounts, strategy_id):
    """Test strategy creation"""
    # Create strategy
    tx_hash = strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Verify event
    logs = strategy_nft.events.StrategyCreated().process_receipt(tx_receipt)
    assert len(logs) == 1
    assert logs[0]['args']['strategyId'] == strategy_id
    
    # Verify strategy data
    strategy = strategy_nft.functions.strategies(strategy_id).call()
    assert strategy[0] == strategy_id  # strategyId
    assert strategy[1] == 0  # totalGovernanceTokens
    assert strategy[2] == True  # isActive

def test_create_duplicate_strategy(web3, strategy_nft, accounts, strategy_id):
    """Test creating duplicate strategy fails"""
    # Create first strategy
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    # Try to create duplicate
    with pytest.raises(Exception, match="Strategy already exists"):
        strategy_nft.functions.createStrategy(strategy_id).transact({
            'from': accounts[0]
        })

def test_mint_nft(web3, strategy_nft, accounts, strategy_id):
    """Test NFT minting"""
    recipient = accounts[1]
    governance_tokens = 1000
    
    # Create strategy first
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    # Mint NFT
    tx_hash = strategy_nft.functions.mintNFT(
        recipient,
        strategy_id,
        governance_tokens
    ).transact({'from': accounts[0]})
    
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Verify event
    logs = strategy_nft.events.NFTMinted().process_receipt(tx_receipt)
    assert len(logs) == 1
    event = logs[0]['args']
    token_id = event['tokenId']
    assert event['owner'] == recipient
    assert event['strategyId'] == strategy_id
    assert event['governanceTokens'] == governance_tokens
    
    # Verify NFT data
    nft = strategy_nft.functions.nftData(token_id).call()
    assert nft[0] == strategy_id  # strategyId
    assert nft[1] == governance_tokens  # governanceTokens
    assert nft[2] == 0  # unclaimedProfits

def test_mint_nft_invalid_strategy(web3, strategy_nft, accounts):
    """Test minting NFT with invalid strategy fails"""
    with pytest.raises(Exception, match="Strategy does not exist"):
        strategy_nft.functions.mintNFT(
            accounts[1],
            "invalid_strategy",
            1000
        ).transact({'from': accounts[0]})

def test_distribute_profits(web3, strategy_nft, accounts, strategy_id):
    """Test profit distribution"""
    amount = to_wei(1, 'ether')
    
    # Create strategy and mint NFTs
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    strategy_nft.functions.mintNFT(
        accounts[1],
        strategy_id,
        1000
    ).transact({'from': accounts[0]})
    
    strategy_nft.functions.mintNFT(
        accounts[2],
        strategy_id,
        2000
    ).transact({'from': accounts[0]})
    
    # Distribute profits
    tx_hash = strategy_nft.functions.distributeStrategyProfits(
        strategy_id
    ).transact({
        'from': accounts[0],
        'value': amount
    })
    
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Verify event
    logs = strategy_nft.events.ProfitsDistributed().process_receipt(tx_receipt)
    assert len(logs) == 1
    assert logs[0]['args']['strategyId'] == strategy_id
    assert logs[0]['args']['amount'] == amount
    
    # Verify strategy profits
    assert strategy_nft.functions.strategyProfits(strategy_id).call() == amount

def test_claim_profits(web3, strategy_nft, accounts, strategy_id):
    """Test profit claiming"""
    amount = to_wei(3, 'ether')
    
    # Setup: Create strategy and mint NFTs
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    # Mint NFTs with 1:2 ratio
    tx1 = strategy_nft.functions.mintNFT(
        accounts[1],
        strategy_id,
        1000
    ).transact({'from': accounts[0]})
    receipt1 = web3.eth.wait_for_transaction_receipt(tx1)
    token_id_1 = strategy_nft.events.NFTMinted().process_receipt(receipt1)[0]['args']['tokenId']
    
    tx2 = strategy_nft.functions.mintNFT(
        accounts[2],
        strategy_id,
        2000
    ).transact({'from': accounts[0]})
    receipt2 = web3.eth.wait_for_transaction_receipt(tx2)
    token_id_2 = strategy_nft.events.NFTMinted().process_receipt(receipt2)[0]['args']['tokenId']
    
    # Distribute profits
    strategy_nft.functions.distributeStrategyProfits(
        strategy_id
    ).transact({
        'from': accounts[0],
        'value': amount
    })
    
    # Get initial balances
    balance1_before = web3.eth.get_balance(accounts[1])
    balance2_before = web3.eth.get_balance(accounts[2])
    
    # Claim profits
    tx1 = strategy_nft.functions.claimProfits(token_id_1).transact({
        'from': accounts[1]
    })
    tx2 = strategy_nft.functions.claimProfits(token_id_2).transact({
        'from': accounts[2]
    })
    
    # Get final balances
    balance1_after = web3.eth.get_balance(accounts[1])
    balance2_after = web3.eth.get_balance(accounts[2])
    
    # Verify profit distribution (1:2 ratio)
    expected_profit1 = to_wei(1, 'ether')
    expected_profit2 = to_wei(2, 'ether')
    
    # Account for gas costs in balance checks
    assert balance1_after > balance1_before
    assert balance2_after > balance2_before
    
    # Verify events
    receipt1 = web3.eth.wait_for_transaction_receipt(tx1)
    receipt2 = web3.eth.wait_for_transaction_receipt(tx2)
    
    logs1 = strategy_nft.events.ProfitsClaimed().process_receipt(receipt1)
    logs2 = strategy_nft.events.ProfitsClaimed().process_receipt(receipt2)
    
    assert logs1[0]['args']['amount'] == expected_profit1
    assert logs2[0]['args']['amount'] == expected_profit2

def test_claim_profits_unauthorized(web3, strategy_nft, accounts, strategy_id):
    """Test claiming profits by non-owner fails"""
    # Create strategy and mint NFT
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    tx = strategy_nft.functions.mintNFT(
        accounts[1],
        strategy_id,
        1000
    ).transact({'from': accounts[0]})
    receipt = web3.eth.wait_for_transaction_receipt(tx)
    token_id = strategy_nft.events.NFTMinted().process_receipt(receipt)[0]['args']['tokenId']
    
    # Distribute profits
    strategy_nft.functions.distributeStrategyProfits(
        strategy_id
    ).transact({
        'from': accounts[0],
        'value': to_wei(1, 'ether')
    })
    
    # Try to claim profits from non-owner account
    with pytest.raises(Exception, match="Not token owner"):
        strategy_nft.functions.claimProfits(token_id).transact({
            'from': accounts[2]
        })

def test_get_nft_data(web3, strategy_nft, accounts, strategy_id):
    """Test getting NFT data"""
    # Create strategy and mint NFT
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    governance_tokens = 1000
    tx = strategy_nft.functions.mintNFT(
        accounts[1],
        strategy_id,
        governance_tokens
    ).transact({'from': accounts[0]})
    receipt = web3.eth.wait_for_transaction_receipt(tx)
    token_id = strategy_nft.events.NFTMinted().process_receipt(receipt)[0]['args']['tokenId']
    
    # Get NFT data
    nft_data = strategy_nft.functions.getNFTData(token_id).call()
    
    assert nft_data[0] == strategy_id  # strategyId
    assert nft_data[1] == governance_tokens  # governanceTokens
    assert nft_data[2] == 0  # unclaimedProfits

def test_get_strategy_data(web3, strategy_nft, accounts, strategy_id):
    """Test getting strategy data"""
    # Create strategy and mint NFTs
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    strategy_nft.functions.mintNFT(
        accounts[1],
        strategy_id,
        1000
    ).transact({'from': accounts[0]})
    
    strategy_nft.functions.mintNFT(
        accounts[2],
        strategy_id,
        2000
    ).transact({'from': accounts[0]})
    
    # Distribute profits
    amount = to_wei(1, 'ether')
    strategy_nft.functions.distributeStrategyProfits(
        strategy_id
    ).transact({
        'from': accounts[0],
        'value': amount
    })
    
    # Get strategy data
    strategy_data = strategy_nft.functions.getStrategyData(strategy_id).call()
    
    assert strategy_data[0] == 3000  # totalGovernanceTokens
    assert strategy_data[1] == amount  # totalProfits
    assert strategy_data[2] == True  # isActive

def test_error_handling(web3, strategy_nft, accounts):
    """Test error handling in contract"""
    strategy_id = "test_strategy"
    
    # Test creating duplicate strategy
    strategy_nft.functions.createStrategy(strategy_id).transact({
        'from': accounts[0]
    })
    
    with pytest.raises(Exception):
        strategy_nft.functions.createStrategy(strategy_id).transact({
            'from': accounts[0]
        })
    
    # Test minting NFT for non-existent strategy
    with pytest.raises(Exception):
        strategy_nft.functions.mintNFT(
            accounts[1],
            "nonexistent_strategy",
            1000
        ).transact({'from': accounts[0]})
    
    # Test claiming profits from non-existent NFT
    with pytest.raises(Exception):
        strategy_nft.functions.claimProfits(999).transact({
            'from': accounts[1]
        }) 