"""
Bridge between NFT agent and blockchain smart contracts
"""

from typing import Dict, List, Any, Optional
from web3 import Web3
from web3.contract import Contract
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BlockchainBridge:
    def __init__(
        self,
        web3_provider: str,
        contract_address: str,
        private_key: str,
        contract_abi_path: str
    ):
        """
        Initialize blockchain connection and contract interface
        
        Args:
            web3_provider: Web3 provider URL (e.g. Infura endpoint)
            contract_address: Deployed StrategyNFT contract address
            private_key: Private key for the owner account
            contract_abi_path: Path to contract ABI JSON file
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        
        # Load contract ABI from artifact
        with open(contract_abi_path) as f:
            contract_artifact = json.load(f)
            contract_abi = contract_artifact['abi']  # Extract just the ABI array
        
        # Initialize contract
        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(contract_address),
            abi=contract_abi
        )
        
        # Setup account
        self.account = self.w3.eth.account.from_key(private_key)
        self.w3.eth.default_account = self.account.address
        
        logger.info(f"Initialized blockchain bridge with contract at {contract_address}")
    
    def create_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Create a new strategy in the smart contract
        
        Args:
            strategy_id: Unique identifier for the strategy
            
        Returns:
            Transaction receipt
        """
        try:
            # Build transaction
            tx = self.contract.functions.createStrategy(strategy_id).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed']
            }
            
        except Exception as e:
            logger.error(f"Failed to create strategy: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def mint_nft(
        self,
        recipient: str,
        strategy_id: str,
        governance_tokens: int
    ) -> Dict[str, Any]:
        """
        Mint a new NFT for a strategy
        
        Args:
            recipient: Address to receive the NFT
            strategy_id: Strategy identifier
            governance_tokens: Number of governance tokens
            
        Returns:
            Transaction receipt and token ID
        """
        try:
            # Convert governance tokens to share percentage (basis points)
            share_percentage = min(governance_tokens * 100, 10000)  # Cap at 100%
            
            # Build transaction
            tx = self.contract.functions.mint(
                self.w3.to_checksum_address(recipient),
                share_percentage
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Get token ID from event logs
            logs = self.contract.events.Transfer().process_receipt(receipt)
            token_id = logs[0]['args']['tokenId']
            
            return {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'token_id': token_id
            }
            
        except Exception as e:
            logger.error(f"Failed to mint NFT: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def distribute_profits(
        self,
        strategy_id: str,
        amount: float
    ) -> Dict[str, Any]:
        """
        Distribute profits to NFT holders of a strategy
        
        Args:
            strategy_id: Strategy identifier
            amount: Amount of profits to distribute (in ETH)
            
        Returns:
            Transaction receipt
        """
        try:
            # Convert amount to Wei
            amount_in_wei = self.w3.to_wei(amount, 'ether')
            
            # Build transaction
            tx = self.contract.functions.distributeStrategyProfits(strategy_id).build_transaction({
                'from': self.account.address,
                'value': amount_in_wei,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'status': 'success',
                'transaction_hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'amount_distributed': amount
            }
            
        except Exception as e:
            logger.error(f"Failed to distribute profits: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_nft_data(self, token_id: int) -> Dict[str, Any]:
        """
        Get data for a specific NFT
        
        Args:
            token_id: Token ID to query
            
        Returns:
            NFT data including share percentage and unclaimed profits
        """
        try:
            share_percentage = self.contract.functions.getSharePercentage(token_id).call()
            total_shares = self.contract.functions.getTotalShares().call()
            
            return {
                'status': 'success',
                'token_id': token_id,
                'share_percentage': share_percentage,
                'total_shares': total_shares
            }
            
        except Exception as e:
            logger.error(f"Failed to get NFT data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_strategy_data(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get data for a specific strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Strategy data including total governance tokens and profits
        """
        try:
            total_tokens, total_profits, is_active = self.contract.functions.getStrategyData(strategy_id).call()
            
            return {
                'status': 'success',
                'strategy_id': strategy_id,
                'total_governance_tokens': total_tokens,
                'total_profits': self.w3.from_wei(total_profits, 'ether'),
                'is_active': is_active
            }
            
        except Exception as e:
            logger.error(f"Failed to get strategy data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_total_shares(self) -> Dict[str, Any]:
        """
        Get total shares issued
        
        Returns:
            Total shares in basis points
        """
        try:
            total_shares = self.contract.functions.getTotalShares().call()
            
            return {
                'status': 'success',
                'total_shares': total_shares
            }
            
        except Exception as e:
            logger.error(f"Failed to get total shares: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 