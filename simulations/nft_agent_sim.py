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
        n_agents: int = 3,
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
        self.n_agents = n_agents
        self.simulation_days = simulation_days
        self.data_interval_minutes = data_interval_minutes
        
        # Performance tracking
        self.portfolio_values = []
        self.agent_profits = {f"agent{i+1}": [] for i in range(n_agents)}
        self.timestamps = []
        
    def initialize_agents(self):
        """Initialize micro-agents with different governance token holdings"""
        # Create agents with varying governance token amounts
        for i in range(self.n_agents):
            agent_id = f"agent{i+1}"
            # Exponentially increasing governance tokens
            governance_tokens = 1000 * (2 ** i)  # 1000, 2000, 4000, etc.
            initial_capital = 10000 * (2 ** i)   # 10000, 20000, 40000, etc.
            
            self.nft_agent.add_micro_agent(
                agent_id=agent_id,
                governance_tokens=governance_tokens,
                initial_capital=initial_capital,
                strategy_params={
                    'target_spread': 0.002 * (i + 1),  # Increasing spreads
                    'min_spread': 0.001 * (i + 1),
                    'max_position': 100 * (2 ** i)
                }
            )
        
        # Initial capital allocation
        self.nft_agent.allocate_capital()
        
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
        # Initialize agents if not already done
        if not self.nft_agent.micro_agents:
            self.initialize_agents()
        
        # Generate synthetic data if none provided
        if data is None:
            data = self.generate_synthetic_data()
        
        portfolio_value = self.nft_agent.total_capital
        current_position = 0.0  # Start with no position
        self.portfolio_values = [portfolio_value]
        self.timestamps = [data['timestamp'].iloc[0]]
        
        # Reset agent profits tracking
        for agent_id in self.agent_profits:
            self.agent_profits[agent_id] = [0.0]
        
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
            
            # Update market state for all agents
            self.nft_agent.update_market_state(current_state)
            
            # Get aggregated action
            action = self.nft_agent.get_aggregated_action()
            self.logger.debug("Aggregated action: %.2f", action)
            
            # Calculate price change and trade size
            price_change = (data['price'].iloc[i] - data['price'].iloc[i-1]) / data['price'].iloc[i-1]
            
            # Use more aggressive position sizing based on action magnitude
            position_size = abs(action) * portfolio_value * 0.5  # Use up to 50% of portfolio
            trade_size = position_size * np.sign(action)  # Apply direction from action
            
            # Update position with some slippage
            slippage = np.random.normal(0, 0.001)  # 0.1% slippage
            current_position += trade_size * (1 + slippage)
            
            # Calculate P&L including transaction costs
            transaction_cost = abs(trade_size) * 0.001  # 0.1% transaction cost
            position_pnl = current_position * price_change - transaction_cost
            portfolio_value += position_pnl
            
            self.logger.debug(
                "Trade: size=%.2f, price_change=%.4f, pnl=%.2f, transaction_cost=%.2f",
                trade_size, price_change, position_pnl, transaction_cost
            )
            
            # Update metrics
            self.portfolio_values.append(portfolio_value)
            self.timestamps.append(data['timestamp'].iloc[i])
            
            # Update individual agent profits
            for agent_id, agent in self.nft_agent.micro_agents.items():
                agent_action = agent.strategy.get_action()
                # Calculate agent's share of the trade based on their capital
                agent_share = agent.trading_capital / self.nft_agent.total_capital
                agent_trade_size = trade_size * agent_share
                agent_pnl = agent_trade_size * price_change - (transaction_cost * agent_share)
                current_profit = self.agent_profits[agent_id][-1] + agent_pnl
                self.agent_profits[agent_id].append(current_profit)
                
                self.logger.debug(
                    "Agent %s: action=%.2f, trade_size=%.2f, pnl=%.2f, total_profit=%.2f",
                    agent_id, agent_action, agent_trade_size, agent_pnl, current_profit
                )
        
        self.logger.info("Simulation completed")
        return {
            'portfolio_values': self.portfolio_values,
            'timestamps': self.timestamps,
            'agent_profits': self.agent_profits
        }
    
    def plot_results(self):
        """Plot simulation results"""
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.portfolio_values, label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot agent profits
        plt.subplot(2, 1, 2)
        for agent_id, profits in self.agent_profits.items():
            plt.plot(self.timestamps, profits, label=agent_id)
        plt.title('Agent Profits Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    print("\nStarting simulation with synthetic data...")
    
    # Create and run simulator
    simulator = NFTAgentSimulator()
    results = simulator.run_simulation()
    
    # Plot results
    simulator.plot_results()
    
    # Print final metrics
    initial_value = simulator.portfolio_values[0]
    final_value = simulator.portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    print(f"\nSimulation Results:")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    for agent_id, profits in simulator.agent_profits.items():
        print(f"{agent_id} Total Profit: ${profits[-1]:,.2f}")

if __name__ == "__main__":
    main() 