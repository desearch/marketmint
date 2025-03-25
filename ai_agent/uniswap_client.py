from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv
import json
import requests
from typing import Dict, Tuple

# Load environment variables
load_dotenv()

# Uniswap V3 ABIs
UNISWAP_V3_ROUTER_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"}
        ],
        "name": "exactInput",
        "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function"
    }
]

class UniswapV3Client:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('RPC_URL')))
        self.account = Account.from_key(os.getenv('PRIVATE_KEY'))
        self.router_address = os.getenv('UNISWAP_ROUTER_ADDRESS')
        self.weth_address = os.getenv('WETH_ADDRESS')
        
        # Initialize router contract
        self.router = self.w3.eth.contract(
            address=self.router_address,
            abi=UNISWAP_V3_ROUTER_ABI
        )

    def get_token_price(self, token_address: str) -> float:
        """Get token price in ETH from Uniswap V3 pool"""
        # TODO: Implement price fetching using The Graph or direct pool contract calls
        # For now, return a mock price
        return 1000.0  # Mock ETH price

    def calculate_min_amount_out(self, amount_in: float, slippage: float = 0.01) -> int:
        """Calculate minimum amount out based on slippage tolerance"""
        return int(amount_in * (1 - slippage))

    def execute_swap(self, 
                    token_in: str, 
                    token_out: str, 
                    amount_in: int,
                    min_amount_out: int) -> Dict:
        """Execute a swap on Uniswap V3"""
        try:
            # Prepare the swap parameters
            path = [token_in, token_out]
            deadline = self.w3.eth.get_block('latest')['timestamp'] + 300  # 5 minutes

            # Build the transaction
            swap_tx = self.router.functions.exactInput(
                amount_in,
                min_amount_out,
                path,
                self.account.address
            ).build_transaction({
                'from': self.account.address,
                'gas': int(os.getenv('GAS_LIMIT', 300000)),
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'value': amount_in if token_in == self.weth_address else 0
            })

            # Sign and send the transaction
            signed_tx = self.account.sign_transaction(swap_tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'effective_gas_price': receipt['effectiveGasPrice']
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_pool_data(self, token_address: str) -> Dict:
        """Get pool data for a token pair (token/ETH)"""
        try:
            # TODO: Implement pool data fetching using The Graph
            # For now, return mock data
            return {
                'liquidity': 1000000,
                'volume_24h': 500000,
                'fee_tier': 3000,  # 0.3%
                'tvl': 2000000
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def estimate_gas(self, 
                    token_in: str, 
                    token_out: str, 
                    amount_in: int) -> Tuple[int, int]:
        """Estimate gas cost for a swap"""
        try:
            gas_price = self.w3.eth.gas_price
            gas_estimate = self.router.functions.exactInput(
                amount_in,
                0,  # min amount out (not relevant for estimation)
                [token_in, token_out],
                self.account.address
            ).estimate_gas({
                'from': self.account.address,
                'value': amount_in if token_in == self.weth_address else 0
            })
            
            return gas_estimate, gas_price
        except Exception as e:
            raise Exception(f"Gas estimation failed: {str(e)}")

    def check_allowance(self, token_address: str) -> int:
        """Check token allowance for router contract"""
        try:
            token_contract = self.w3.eth.contract(
                address=token_address,
                abi=[{
                    "constant": True,
                    "inputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"}
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                }]
            )
            
            return token_contract.functions.allowance(
                self.account.address,
                self.router_address
            ).call()
        except Exception as e:
            raise Exception(f"Failed to check allowance: {str(e)}") 