import sqlite3
import json
from datetime import datetime
from pathlib import Path

class Database:
    def __init__(self):
        self.db_path = Path(__file__).parent / "amm_data.db"
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create trade history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                action TEXT,
                amount REAL,
                price REAL,
                status TEXT,
                tx_hash TEXT
            )
        ''')

        # Create NFT ownership table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nft_ownership (
                token_id INTEGER PRIMARY KEY,
                owner_address TEXT,
                share_percentage INTEGER,
                last_updated DATETIME
            )
        ''')

        # Create profit distribution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profit_distributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                amount REAL,
                total_shares INTEGER,
                tx_hash TEXT
            )
        ''')

        # Create governance proposals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS governance_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                description TEXT,
                status TEXT,
                votes_for INTEGER,
                votes_against INTEGER,
                creator_address TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_trade(self, action, amount, price, status, tx_hash=None):
        """Add a new trade to the history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trade_history 
            (timestamp, action, amount, price, status, tx_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), action, amount, price, status, tx_hash))
        
        conn.commit()
        conn.close()

    def update_nft_ownership(self, token_id, owner_address, share_percentage):
        """Update NFT ownership information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO nft_ownership 
            (token_id, owner_address, share_percentage, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (token_id, owner_address, share_percentage, datetime.now()))
        
        conn.commit()
        conn.close()

    def add_profit_distribution(self, amount, total_shares, tx_hash=None):
        """Record a profit distribution"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO profit_distributions 
            (timestamp, amount, total_shares, tx_hash)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now(), amount, total_shares, tx_hash))
        
        conn.commit()
        conn.close()

    def create_proposal(self, description, creator_address):
        """Create a new governance proposal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO governance_proposals 
            (timestamp, description, status, votes_for, votes_against, creator_address)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), description, 'active', 0, 0, creator_address))
        
        proposal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return proposal_id

    def get_trade_history(self, limit=100):
        """Get recent trade history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trade_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        trades = cursor.fetchall()
        conn.close()
        return trades

    def get_nft_ownership(self, token_id):
        """Get NFT ownership information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM nft_ownership 
            WHERE token_id = ?
        ''', (token_id,))
        
        ownership = cursor.fetchone()
        conn.close()
        return ownership

    def get_profit_distributions(self, limit=100):
        """Get recent profit distributions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM profit_distributions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        distributions = cursor.fetchall()
        conn.close()
        return distributions

    def get_proposal_status(self, proposal_id):
        """Get governance proposal status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM governance_proposals 
            WHERE id = ?
        ''', (proposal_id,))
        
        proposal = cursor.fetchone()
        conn.close()
        return proposal 