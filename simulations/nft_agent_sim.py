import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from ai_agent.nft_agent import NFTAgent
from ai_agent.strategies import PureMarketMaker

class NFTAgentSimulator:
    """Simulator for testing NFT Agent performance with synthetic or real data"""
    
    def __init__(
        self,
        total_capital: float = 1000000.0,
        n_strategies: int = 2,
        n_nfts_per_strategy: int = 2,
        simulation_days: int = 30,
        data_interval_minutes: int = 5
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NFT Agent
        self.nft_agent = NFTAgent(
            total_capital=total_capital,
            risk_free_rate=0.02,
            max_drawdown=0.15,
            min_profit_threshold=0.03
        )
        
        # Simulation parameters
        self.n_strategies = n_strategies
        self.n_nfts_per_strategy = n_nfts_per_strategy
        self.simulation_days = simulation_days
        self.data_interval_minutes = data_interval_minutes
        
        # Performance tracking
        self.portfolio_values = []
        self.strategy_profits = {f"strategy{i+1}": [] for i in range(n_strategies)}
        self.nft_profits = {}
        self.timestamps = []
        
    def initialize_strategies_and_nfts(self):
        """Initialize strategies and NFTs"""
        # Create strategies with optimized parameters
        for i in range(self.n_strategies):
            strategy_id = f"strategy{i+1}"
            
            # Configure strategy parameters with optimized values
            base_params = {
                'target_spread': 0.002 * (1.0 + 0.1 * i),  # 10% increase per strategy
                'min_spread': 0.001 * (1.0 + 0.1 * i),
                'max_position': self.nft_agent.total_capital * 0.1,  # 10% of capital
                'position_limit': 0.5,  # Start reducing at 50% utilization
                'risk_aversion': 1.0,
                'min_trade_size': 0.00001,  # 0.001% of max position
                'max_trade_size': 0.1,  # 10% of max position per trade
                'momentum_threshold': 0.00001,  # 0.001% price change
                'min_vol_threshold': 0.00001,  # 0.001% volatility
                'mean_reversion_strength': 0.5
            }
            
            self.nft_agent.add_strategy(
                strategy_id=strategy_id,
                strategy_params=base_params
            )
            
            # Create NFTs for each strategy with exponential governance token distribution
            for j in range(self.n_nfts_per_strategy):
                nft_id = f"nft{i+1}_{j+1}"
                owner = f"owner{i+1}_{j+1}"
                # Exponentially increasing governance tokens with smaller steps
                governance_tokens = 1000 * (1.5 ** j)  # 1000, 1500, 2250, etc.
                
                self.nft_agent.mint_nft(
                    nft_id=nft_id,
                    owner=owner,
                    strategy_id=strategy_id,
                    governance_tokens=governance_tokens,
                    risk_limits={
                        'max_position_size': base_params['max_position'] * 0.5,  # 50% of strategy max position
                        'max_daily_loss': self.nft_agent.total_capital * 0.01,   # 1% of total capital
                        'min_profit_target': self.nft_agent.total_capital * self.nft_agent.min_profit_threshold
                    }
                )
                
                # Initialize profit tracking for this NFT
                self.nft_profits[nft_id] = []
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for simulation"""
        # Calculate number of intervals
        intervals = int((self.simulation_days * 24 * 60) / self.data_interval_minutes)
        
        # Generate time series
        base_time = datetime.now()
        times = [base_time + timedelta(minutes=i*self.data_interval_minutes) for i in range(intervals)]
        
        # Generate price series with random walk and some volatility clusters
        price = 100.0
        prices = [price]
        volatility = 0.0002  # Base volatility
        
        for _ in range(intervals-1):
            volatility = max(0.0001, min(0.001, volatility + np.random.normal(0, 0.0001)))
            price *= np.exp(np.random.normal(0, volatility))
            prices.append(price)
        
        # Generate volume with some patterns
        volumes = np.random.lognormal(mean=np.log(1000), sigma=0.5, size=intervals)
        # Add time-of-day effect
        hour_of_day = np.array([t.hour for t in times])
        volumes *= 1 + np.sin(hour_of_day * np.pi / 12)
        
        return pd.DataFrame({
            'timestamp': times,
            'price': prices,
            'volume': volumes
        })
    
    def load_real_data(self, file_path: str) -> pd.DataFrame:
        """Load real market data from CSV file"""
        df = pd.read_csv(file_path)
        required_columns = ['timestamp', 'price', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def run_simulation(self):
        """Run the simulation and return the results."""
        # Initialize strategies and NFTs if not already done
        if not self.nft_agent.strategy_runner.strategies:
            self.initialize_strategies_and_nfts()
        
        # Generate synthetic data if not already loaded
        if not hasattr(self, 'market_data'):
            self.market_data = self.generate_synthetic_data()
        
        results = []
        self.logger.info(f"Starting simulation with {len(self.market_data)} data points")
        
        # Initialize tracking variables
        portfolio_value = self.nft_agent.total_capital
        strategy_positions = {strategy_id: 0.0 for strategy_id in self.nft_agent.strategy_runner.strategies.keys()}
        strategy_values = {strategy_id: 0.0 for strategy_id in self.nft_agent.strategy_runner.strategies.keys()}
        
        # Initialize profit tracking for each NFT and owner
        self.nft_profits = {nft_id: [0.0] for nft_id in self.nft_agent.nfts.keys()}
        self.owner_profits = {nft.owner: [0.0] for nft in self.nft_agent.nfts.values()}
        self.portfolio_values = [portfolio_value]
        self.strategy_profits = {strategy_id: [0.0] for strategy_id in self.nft_agent.strategy_runner.strategies.keys()}
        self.timestamps = [self.market_data['timestamp'].iloc[0]]
        
        # Create mapping from owner to NFTs
        owner_to_nfts = {}
        for nft_id, nft in self.nft_agent.nfts.items():
            if nft.owner not in owner_to_nfts:
                owner_to_nfts[nft.owner] = []
            owner_to_nfts[nft.owner].append(nft_id)
        
        for i in range(len(self.market_data)):
            timestamp = self.market_data['timestamp'].iloc[i]
            price = self.market_data['price'].iloc[i]
            volume = self.market_data['volume'].iloc[i]
            
            # Validate price and volume data
            if pd.isna(price) or pd.isna(volume) or price <= 0 or volume <= 0:
                self.logger.warning(f"Invalid price or volume at timestamp {timestamp}, skipping iteration")
                continue
            
            # Track timestamp
            self.timestamps.append(timestamp)
            
            state = {
                'price': float(price),  # Ensure price is float
                'volume': float(volume),  # Ensure volume is float
                'timestamp': int(timestamp.timestamp()),
                'portfolio_value': float(portfolio_value)  # Ensure portfolio value is float
            }
            
            # Track portfolio value
            self.portfolio_values.append(portfolio_value)
            
            # Process each strategy
            timestep_pnl = 0.0  # Track total PnL for this timestep
            
            for strategy_id, agents in self.nft_agent.strategy_runner.strategies.items():
                strategy_pnl = 0.0
                
                for agent in agents:
                    # Update state with current position
                    state['position'] = float(strategy_positions[strategy_id])  # Ensure position is float
                    agent.strategy.update_market_state(state)
                    
                    # Get action from strategy
                    action = agent.strategy.get_action(state)
                    
                    # Get strategy parameters from the underlying strategy
                    underlying_strategy = agent.strategy.strategy
                    max_position = float(underlying_strategy.max_position)
                    position_limit = float(underlying_strategy.position_limit)
                    max_trade_size = float(underlying_strategy.max_trade_size)
                    
                    # Scale trade size based on action and limits
                    trade_size = float(action * max_trade_size * max_position)
                    
                    # Apply position limits
                    current_position = strategy_positions[strategy_id]
                    max_allowed_position = max_position * position_limit
                    
                    if abs(current_position + trade_size) > max_allowed_position:
                        # Calculate remaining position capacity
                        remaining_capacity = max_allowed_position - abs(current_position)
                        # Scale trade size to fit within limits
                        trade_size = np.sign(trade_size) * min(abs(trade_size), remaining_capacity)
                    
                    # Calculate slippage based on trade size relative to volume
                    slippage = min(abs(trade_size / volume), 0.01)  # Cap slippage at 1%
                    
                    # Apply slippage to price
                    executed_price = price * (1 + slippage * np.sign(trade_size))
                    
                    # Calculate PnL
                    trade_value = trade_size * executed_price
                    strategy_positions[strategy_id] = float(current_position + trade_size)
                    position_value = strategy_positions[strategy_id] * price
                    
                    # Update strategy value and calculate PnL
                    old_value = strategy_values[strategy_id]
                    strategy_values[strategy_id] = float(position_value - trade_value)
                    trade_pnl = float(strategy_values[strategy_id] - old_value)
                    
                    # Validate PnL calculation
                    if pd.isna(trade_pnl):
                        self.logger.warning(f"Invalid PnL calculated for strategy {strategy_id}, setting to 0")
                        trade_pnl = 0.0
                    
                    strategy_pnl += trade_pnl
                    timestep_pnl += trade_pnl
                    
                    # Log trade details
                    self.logger.info(f"Strategy {strategy_id} - Action: {action:.4f}, Size: {trade_size:.2f}, "
                                   f"Price: {executed_price:.2f}, Slippage: {slippage:.4f}, PnL: {trade_pnl:.2f}")
                    
                    results.append({
                        'timestamp': timestamp,
                        'strategy_id': strategy_id,
                        'action': float(action),
                        'trade_size': float(trade_size),
                        'price': float(executed_price),
                        'slippage': float(slippage),
                        'pnl': float(trade_pnl)
                    })
                
                # Update strategy profits
                self.strategy_profits[strategy_id].append(float(strategy_pnl))
            
            # Update portfolio value
            portfolio_value += timestep_pnl
            
            # Distribute profits to NFTs for this timestep
            if timestep_pnl != 0:  # Only distribute if there are profits/losses
                profit_distribution = self.nft_agent.distribute_profits(timestep_pnl)
                
                # Update owner profits
                for owner_id, profit in profit_distribution.items():
                    current_profit = float(profit)
                    self.owner_profits[owner_id].append(current_profit)
                    
                    # Distribute owner's profit equally among their NFTs
                    if owner_id in owner_to_nfts:
                        nft_profit = current_profit / len(owner_to_nfts[owner_id])
                        for nft_id in owner_to_nfts[owner_id]:
                            self.nft_profits[nft_id].append(float(nft_profit))
            else:
                # If no PnL, append zeros to maintain array lengths
                for owner_id in self.owner_profits:
                    self.owner_profits[owner_id].append(0.0)
                for nft_id in self.nft_profits:
                    self.nft_profits[nft_id].append(0.0)
        
        return results
    
    def plot_results(self):
        """Plot simulation results"""
        plt.figure(figsize=(20, 15))
        
        # Plot portfolio value
        plt.subplot(4, 1, 1)
        plt.plot(self.timestamps[1:], self.portfolio_values[1:])  # Skip first point which is initial value
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Plot strategy profits
        plt.subplot(4, 1, 2)
        for strategy_id, profits in self.strategy_profits.items():
            plt.plot(self.timestamps[1:], profits[1:], label=strategy_id)  # Skip first point which is 0
        plt.title('Strategy Profits Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot owner profits
        plt.subplot(4, 1, 3)
        for owner_id, profits in self.owner_profits.items():
            plt.plot(self.timestamps[1:], profits[1:], label=f'Owner {owner_id}')  # Skip first point which is 0
        plt.title('Owner Profits Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot NFT profits
        plt.subplot(4, 1, 4)
        for nft_id, profits in self.nft_profits.items():
            plt.plot(self.timestamps[1:], profits[1:], label=f'NFT {nft_id}')  # Skip first point which is 0
        plt.title('NFT Profits Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run simulator
    simulator = NFTAgentSimulator(
        total_capital=1000000.0,
        n_strategies=2,
        n_nfts_per_strategy=2,
        simulation_days=30,
        data_interval_minutes=5
    )
    
    results = simulator.run_simulation()
    simulator.plot_results() 