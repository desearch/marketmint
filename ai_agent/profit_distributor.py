from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv
import json
import logging
from typing import Dict, List
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.getenv('LOG_FILE', 'amm.log')
)
logger = logging.getLogger(__name__)

class ProfitDistributor:
    def __init__(
        self,
        contract_address: str,
        private_key: str,
        contract_abi_path: str,
        rpc_url: str = "http://localhost:8545",
        gas_limit: int = 300000
    ):
        """
        Initialize profit distributor
        
        Args:
            contract_address: Address of the ProfitDistributor contract
            private_key: Private key for signing transactions
            contract_abi_path: Path to contract ABI JSON file
            rpc_url: RPC URL for Web3 provider
            gas_limit: Gas limit for transactions
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.account = Account.from_key(private_key)
        self.profit_distributor_address = contract_address
        self.gas_limit = gas_limit
        
        # Load contract ABI
        with open(contract_abi_path, 'r') as f:
            contract_json = json.load(f)
            self.contract_abi = contract_json['abi']
        
        # Initialize contract
        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.profit_distributor_address),
            abi=self.contract_abi
        )

    def distribute_profits(self, amount: float) -> Dict:
        """
        Distribute profits to NFT holders
        
        Args:
            amount: Amount of profits to distribute (in ETH)
            
        Returns:
            Transaction status and details
        """
        try:
            if amount <= 0:
                raise ValueError("Profit amount must be positive")
                
            # Convert amount to Wei
            amount_in_wei = Web3.to_wei(amount, 'ether')
            
            # Build transaction
            tx = self.contract.functions.distributeProfits().build_transaction({
                'from': self.account.address,
                'value': amount_in_wei,
                'gas': self.gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Get distribution details
            distribution = self.contract.functions.getLastDistribution().call()
            
            return {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'distribution': {
                    'total_amount': Web3.from_wei(distribution[0], 'ether'),
                    'timestamp': datetime.fromtimestamp(distribution[1]),
                    'nft_holders': distribution[2]
                }
            }
            
        except ValueError as e:
            logger.error("Invalid profit amount: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }
        except Exception as e:
            logger.error("Profit distribution failed: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

    def claim_profits(self, token_id: int) -> Dict:
        """
        Claim profits for a specific NFT token
        
        Args:
            token_id: Token ID to claim profits for
            
        Returns:
            Transaction status and claimed amount
        """
        try:
            # Build transaction
            tx = self.contract.functions.claimProfits(token_id).build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Get claim details from event logs
            logs = self.contract.events.ProfitsClaimed().process_receipt(receipt)
            claimed_amount = Web3.from_wei(logs[0]['args']['amount'], 'ether')
            
            return {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'amount_claimed': claimed_amount
            }
            
        except Exception as e:
            logger.error("Failed to claim profits: %s", str(e))
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_nft_holders(self) -> List[Dict]:
        """Get list of NFT holders and their shares"""
        try:
            holders = self.contract.functions.getNFTHolders().call()
            return [
                {
                    'address': holder[0],
                    'token_id': holder[1],
                    'share': holder[2]
                }
                for holder in holders
            ]
        except Exception as e:
            logger.error("Failed to get NFT holders: %s", str(e))
            return []

    def get_distribution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent profit distribution history"""
        try:
            history = self.contract.functions.getDistributionHistory(limit).call()
            return [
                {
                    'amount': Web3.from_wei(dist[0], 'ether'),
                    'timestamp': datetime.fromtimestamp(dist[1]),
                    'nft_holders': dist[2]
                }
                for dist in history
            ]
        except Exception as e:
            logger.error("Failed to get distribution history: %s", str(e))
            return []

    def get_total_distributed(self) -> float:
        """Get total amount of profits distributed"""
        try:
            total = self.contract.functions.totalDistributed().call()
            return Web3.from_wei(total, 'ether')
        except Exception as e:
            logger.error("Failed to get total distributed: %s", str(e))
            return 0.0

    def get_vault_balance(self) -> float:
        """Get current balance of the profit distributor vault"""
        try:
            balance = self.w3.eth.get_balance(self.profit_distributor_address)
            return Web3.from_wei(balance, 'ether')
        except Exception as e:
            logger.error("Failed to get vault balance: %s", str(e))
            return 0.0 