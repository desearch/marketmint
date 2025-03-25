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
        # Create strategies with different parameters
        for i in range(self.n_strategies):
            strategy_id = f"strategy{i+1}"
            self.nft_agent.add_strategy(
                strategy_id=strategy_id,
                strategy_params={
                    'target_spread': 0.002 * (i + 1),  # Increasing spreads
                    'min_spread': 0.001 * (i + 1),
                    'max_position': 100 * (2 ** i)
                }
            )
            
            # Create NFTs for each strategy
            for j in range(self.n_nfts_per_strategy):
                nft_id = f"nft{i+1}_{j+1}"
                owner = f"owner{i+1}_{j+1}"
                # Exponentially increasing governance tokens
                governance_tokens = 1000 * (2 ** j)  # 1000, 2000, 4000, etc.
                
                self.nft_agent.mint_nft(
                    nft_id=nft_id,
                    owner=owner,
                    strategy_id=strategy_id,
                    governance_tokens=governance_tokens,
                    risk_limits={
                        'max_position_size': self.nft_agent.total_capital * 0.1,  # 10% of total capital
                        'max_daily_loss': self.nft_agent.total_capital * 0.02,   # 2% of total capital
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
    
    def run_simulation(self, data: Optional[pd.DataFrame] = None) -> Dict[str, List[float]]:
        """Run simulation with either synthetic or real data"""
        # Initialize strategies and NFTs if not already done
        if not self.nft_agent.strategy_runner.strategies:
            self.initialize_strategies_and_nfts()
        
        # Generate synthetic data if none provided
        if data is None:
            data = self.generate_synthetic_data()
        
        portfolio_value = self.nft_agent.total_capital
        current_position = 0.0  # Start with no position
        self.portfolio_values = [portfolio_value]
        self.timestamps = [data['timestamp'].iloc[0]]
        
        # Reset profit tracking
        for strategy_id in self.strategy_profits:
            self.strategy_profits[strategy_id] = [0.0]
        for nft_id in self.nft_profits:
            self.nft_profits[nft_id] = [0.0]
        
        self.logger.info("Starting simulation with %d data points", len(data))
        
        # Run simulation
        for i in range(1, len(data)):
            # Update market state
            current_state = {
                'price': data['price'].iloc[i],
                'volume': data['volume'].iloc[i],
                'timestamp': int(data['timestamp'].iloc[i].timestamp()),  # Convert to Unix timestamp
                'portfolio_value': portfolio_value,
                'position': current_position
            }
            
            self.logger.debug(
                "Step %d: Price=%.2f, Position=%.2f, Portfolio=%.2f",
                i, current_state['price'], current_position, portfolio_value
            )
            
            # Update market state for all strategies
            self.nft_agent.update_market_state(current_state)
            
            # Get aggregated action
            action = self.nft_agent.get_aggregated_action()
            self.logger.debug("Aggregated action: %.2f", action)
            
            # Calculate price change and trade size
            price_change = (data['price'].iloc[i] - data['price'].iloc[i-1]) / data['price'].iloc[i-1]
            
            # Use more aggressive position sizing based on action magnitude
            # Allow up to 100% of portfolio to be used
            position_size = abs(action) * portfolio_value
            trade_size = position_size * np.sign(action)  # Apply direction from action
            
            # Update position with some slippage
            slippage = np.random.normal(0, 0.001)  # 0.1% slippage
            executed_trade_size = trade_size * (1 + slippage)
            current_position += executed_trade_size
            
            # Calculate P&L including transaction costs
            transaction_cost = abs(executed_trade_size) * 0.001  # 0.1% transaction cost
            position_pnl = current_position * price_change - transaction_cost
            portfolio_value += position_pnl
            
            self.logger.info(
                "Trade executed: action=%.4f, size=%.2f, executed_size=%.2f, price_change=%.4f, pnl=%.2f, cost=%.2f, portfolio=%.2f",
                action, trade_size, executed_trade_size, price_change, position_pnl, transaction_cost, portfolio_value
            )
            
            # Update metrics
            self.portfolio_values.append(portfolio_value)
            self.timestamps.append(data['timestamp'].iloc[i])
            
            # Update strategy and NFT profits
            for strategy_id in self.strategy_profits:
                strategy_metrics = self.nft_agent.get_strategy_metrics(strategy_id)
                current_profit = self.strategy_profits[strategy_id][-1] + strategy_metrics['total_profit']
                self.strategy_profits[strategy_id].append(current_profit)
                
                # Update individual NFT profits
                for nft_id, nft in self.nft_agent.nfts.items():
                    if nft.strategy_id == strategy_id:
                        nft_metrics = self.nft_agent.get_nft_performance(nft_id)
                        current_profit = self.nft_profits[nft_id][-1] + nft_metrics['total_profit']
                        self.nft_profits[nft_id].append(current_profit)
                        
                        self.logger.debug(
                            "Strategy %s, NFT %s: total_profit=%.2f",
                            strategy_id, nft_id, current_profit
                        )
        
        # Distribute final profits to NFT owners
        final_profits = self.nft_agent.distribute_profits(portfolio_value - self.nft_agent.total_capital)
        self.logger.info("Final profit distribution: %s", final_profits)
        
        self.logger.info("Simulation completed")
        return {
            'portfolio_values': self.portfolio_values,
            'timestamps': self.timestamps,
            'strategy_profits': self.strategy_profits,
            'nft_profits': self.nft_profits
        }
    
    def plot_results(self):
        """Plot simulation results"""
        plt.figure(figsize=(15, 15))
        
        # Plot portfolio value
        plt.subplot(3, 1, 1)
        plt.plot(self.timestamps, self.portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Plot strategy profits
        plt.subplot(3, 1, 2)
        for strategy_id, profits in self.strategy_profits.items():
            plt.plot(self.timestamps, profits, label=strategy_id)
        plt.title('Strategy Profits Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot individual NFT profits
        plt.subplot(3, 1, 3)
        for nft_id, profits in self.nft_profits.items():
            plt.plot(self.timestamps, profits, label=nft_id)
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