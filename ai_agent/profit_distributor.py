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
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('RPC_URL')))
        self.account = Account.from_key(os.getenv('PRIVATE_KEY'))
        self.profit_distributor_address = os.getenv('PROFIT_DISTRIBUTOR_ADDRESS')
        
        # Load contract ABI
        with open('contracts/artifacts/contracts/ProfitDistributor.sol/ProfitDistributor.json', 'r') as f:
            contract_json = json.load(f)
            self.contract_abi = contract_json['abi']
        
        # Initialize contract
        self.contract = self.w3.eth.contract(
            address=self.profit_distributor_address,
            abi=self.contract_abi
        )

    def distribute_profits(self, amount: float) -> Dict:
        """Distribute profits to NFT holders"""
        try:
            # Convert amount to Wei
            amount_in_wei = Web3.to_wei(amount, 'ether')
            
            # Build transaction
            tx = self.contract.functions.distributeProfits().build_transaction({
                'from': self.account.address,
                'value': amount_in_wei,
                'gas': int(os.getenv('GAS_LIMIT', 300000)),
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
            
        except Exception as e:
            logger.error("Profit distribution failed: %s", str(e))
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