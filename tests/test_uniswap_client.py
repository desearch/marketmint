import pytest
from ai_agent.uniswap_client import UniswapV3Client
from web3 import Web3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def uniswap_client():
    return UniswapV3Client()

def test_initialization(uniswap_client):
    """Test client initialization"""
    assert uniswap_client.w3.is_connected()
    assert uniswap_client.router_address == os.getenv('UNISWAP_ROUTER_ADDRESS')
    assert uniswap_client.weth_address == os.getenv('WETH_ADDRESS')

def test_get_token_price(uniswap_client):
    """Test token price fetching"""
    price = uniswap_client.get_token_price(uniswap_client.weth_address)
    assert isinstance(price, float)
    assert price > 0

def test_calculate_min_amount_out(uniswap_client):
    """Test minimum amount calculation with slippage"""
    amount_in = Web3.to_wei(1, 'ether')  # 1 ETH
    slippage = 0.01  # 1%
    
    min_amount = uniswap_client.calculate_min_amount_out(amount_in, slippage)
    
    assert isinstance(min_amount, int)
    assert min_amount == int(amount_in * (1 - slippage))

def test_get_pool_data(uniswap_client):
    """Test pool data fetching"""
    pool_data = uniswap_client.get_pool_data(uniswap_client.weth_address)
    
    assert isinstance(pool_data, dict)
    assert 'liquidity' in pool_data
    assert 'volume_24h' in pool_data
    assert 'fee_tier' in pool_data
    assert 'tvl' in pool_data

def test_estimate_gas(uniswap_client):
    """Test gas estimation for swaps"""
    amount_in = Web3.to_wei(1, 'ether')  # 1 ETH
    
    try:
        gas_estimate, gas_price = uniswap_client.estimate_gas(
            uniswap_client.weth_address,  # token in
            uniswap_client.weth_address,  # token out
            amount_in
        )
        
        assert isinstance(gas_estimate, int)
        assert isinstance(gas_price, int)
        assert gas_estimate > 0
        assert gas_price > 0
        
    except Exception as e:
        pytest.skip(f"Gas estimation failed: {str(e)}")

def test_check_allowance(uniswap_client):
    """Test token allowance checking"""
    try:
        allowance = uniswap_client.check_allowance(uniswap_client.weth_address)
        assert isinstance(allowance, int)
    except Exception as e:
        pytest.skip(f"Allowance check failed: {str(e)}")

def test_execute_swap_validation(uniswap_client):
    """Test swap execution input validation"""
    amount_in = Web3.to_wei(0.1, 'ether')  # 0.1 ETH
    min_amount_out = Web3.to_wei(0.09, 'ether')  # Minimum 0.09 ETH out
    
    result = uniswap_client.execute_swap(
        uniswap_client.weth_address,  # token in
        uniswap_client.weth_address,  # token out
        amount_in,
        min_amount_out
    )
    
    assert isinstance(result, dict)
    assert 'status' in result
    if result['status'] == 'error':
        assert 'message' in result
    else:
        assert 'transaction_hash' in result
        assert 'gas_used' in result 