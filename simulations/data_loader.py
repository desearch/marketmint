import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from web3 import Web3
from dotenv import load_dotenv
import os

class BlockchainDataLoader:
    """Utility class for loading and preprocessing blockchain data"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Web3
        self.rpc_url = rpc_url or os.getenv('RPC_URL')
        if not self.rpc_url:
            raise ValueError("RPC URL not provided and not found in environment variables")
        
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
    
    def load_historical_data(
        self,
        token_address: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """Load historical price and volume data for a token"""
        try:
            # Convert dates to timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Calculate number of blocks based on interval
            block_interval = self._get_block_interval(interval)
            
            # Get block numbers for start and end dates
            start_block = self.w3.eth.get_block_by_timestamp(start_ts)['number']
            end_block = self.w3.eth.get_block_by_timestamp(end_ts)['number']
            
            # Fetch data in chunks to avoid timeout
            data = []
            current_block = start_block
            
            while current_block <= end_block:
                chunk_end = min(current_block + block_interval, end_block)
                chunk_data = self._fetch_block_data(token_address, current_block, chunk_end)
                data.extend(chunk_data)
                current_block = chunk_end + 1
                
                self.logger.info(f"Fetched data from block {current_block} to {chunk_end}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            raise
    
    def _get_block_interval(self, interval: str) -> int:
        """Convert time interval to number of blocks"""
        # Average block time is ~12 seconds on Ethereum
        BLOCKS_PER_HOUR = 300
        BLOCKS_PER_DAY = BLOCKS_PER_HOUR * 24
        
        if interval == '1h':
            return BLOCKS_PER_HOUR
        elif interval == '1d':
            return BLOCKS_PER_DAY
        elif interval == '1w':
            return BLOCKS_PER_DAY * 7
        else:
            raise ValueError(f"Unsupported interval: {interval}")
    
    def _fetch_block_data(
        self,
        token_address: str,
        start_block: int,
        end_block: int
    ) -> List[Dict]:
        """Fetch price and volume data for a range of blocks"""
        data = []
        
        for block_number in range(start_block, end_block + 1):
            try:
                # Get block timestamp
                block = self.w3.eth.get_block(block_number)
                timestamp = block['timestamp']
                
                # Get token price and volume from Uniswap pool
                # This is a placeholder - implement actual Uniswap V3 pool interaction
                price = self._get_token_price(token_address, block_number)
                volume = self._get_token_volume(token_address, block_number)
                
                data.append({
                    'timestamp': timestamp,
                    'block_number': block_number,
                    'price': price,
                    'volume': volume
                })
                
            except Exception as e:
                self.logger.warning(f"Error fetching data for block {block_number}: {str(e)}")
                continue
        
        return data
    
    def _get_token_price(self, token_address: str, block_number: int) -> float:
        """Get token price at a specific block"""
        # Placeholder - implement actual Uniswap V3 pool interaction
        # This should query the Uniswap V3 pool contract to get the current price
        return 0.0
    
    def _get_token_volume(self, token_address: str, block_number: int) -> float:
        """Get token volume at a specific block"""
        # Placeholder - implement actual Uniswap V3 pool interaction
        # This should query the Uniswap V3 pool contract to get the volume
        return 0.0
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame"""
        # Calculate price changes
        df['returns'] = df['price'].pct_change()
        
        # Moving averages
        df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
        df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        bb_std = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df 