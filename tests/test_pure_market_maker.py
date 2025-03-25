import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from ai_agent.strategies import PureMarketMaker
from ai_agent.market_maker import MarketMaker

def generate_test_data(n_points=100):
    """Generate synthetic market data for testing"""
    base_time = datetime.now()
    times = [base_time + timedelta(minutes=i*5) for i in range(n_points)]
    
    # Generate price series with random walk
    price = 100.0
    prices = [price]
    volatility = 0.002  # Higher volatility for testing
    
    for _ in range(n_points-1):
        price *= np.exp(np.random.normal(0, volatility))
        prices.append(price)
    
    # Generate volume with patterns
    volumes = np.random.lognormal(mean=np.log(1000), sigma=0.5, size=n_points)
    
    return pd.DataFrame({
        'timestamp': times,
        'price': prices,
        'volume': volumes
    })

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize strategy with test parameters
    strategy = PureMarketMaker(
        target_spread=0.002,  # 0.2% target spread
        min_spread=0.001,     # 0.1% minimum spread
        max_position=1000.0,  # Larger position size for testing
        position_limit=0.5,   # 50% of capital as position limit
        risk_aversion=1.0     # Standard risk aversion
    )
    
    # Wrap strategy in MarketMaker
    market_maker = MarketMaker()
    market_maker.set_strategy(strategy)
    
    # Generate test data
    data = generate_test_data(n_points=100)
    logger.info("Generated %d test data points", len(data))
    
    # Initialize tracking variables
    position = 0.0
    portfolio_value = 1000000.0  # $1M starting capital
    trades = []
    positions = [position]
    portfolio_values = [portfolio_value]
    actions = []
    
    # Run simulation
    for i in range(1, len(data)):
        # Update market state
        current_state = {
            'price': data['price'].iloc[i],
            'volume': data['volume'].iloc[i],
            'timestamp': int(data['timestamp'].iloc[i].timestamp()),
            'portfolio_value': portfolio_value,
            'position': position
        }
        
        market_maker.update_market_state(current_state)
        
        # Get action from strategy
        action = market_maker.get_action()
        actions.append(action)
        
        # Calculate trade size and execute
        position_size = abs(action) * portfolio_value
        trade_size = position_size * np.sign(action)
        
        # Apply slippage
        slippage = np.random.normal(0, 0.001)
        executed_size = trade_size * (1 + slippage)
        
        # Update position
        old_position = position
        position += executed_size
        
        # Calculate P&L
        price_change = (data['price'].iloc[i] - data['price'].iloc[i-1]) / data['price'].iloc[i-1]
        transaction_cost = abs(executed_size) * 0.001  # 0.1% transaction cost
        trade_pnl = old_position * price_change - transaction_cost
        
        portfolio_value += trade_pnl
        
        # Record trade
        trades.append({
            'timestamp': data['timestamp'].iloc[i],
            'price': data['price'].iloc[i],
            'action': action,
            'trade_size': executed_size,
            'position': position,
            'pnl': trade_pnl,
            'portfolio_value': portfolio_value
        })
        
        positions.append(position)
        portfolio_values.append(portfolio_value)
        
        logger.info(
            "Step %d: Price=%.2f, Action=%.4f, Size=%.2f, Position=%.2f, PnL=%.2f, Portfolio=%.2f",
            i, current_state['price'], action, executed_size, position, trade_pnl, portfolio_value
        )
    
    # Calculate performance metrics
    trades_df = pd.DataFrame(trades)
    total_return = (portfolio_value - 1000000.0) / 1000000.0 * 100
    sharpe_ratio = np.mean(trades_df['pnl']) / np.std(trades_df['pnl']) if len(trades_df) > 0 else 0
    max_drawdown = np.min(np.minimum.accumulate(portfolio_values) / np.maximum.accumulate(portfolio_values) - 1)
    
    logger.info("\nPerformance Summary:")
    logger.info("Total Return: %.2f%%", total_return)
    logger.info("Sharpe Ratio: %.2f", sharpe_ratio)
    logger.info("Max Drawdown: %.2f%%", max_drawdown * 100)
    logger.info("Number of Trades: %d", len(trades))
    logger.info("Average Position Size: %.2f", np.mean(np.abs(positions)))
    logger.info("Average Action Size: %.4f", np.mean(np.abs(actions)))
    
    # Print some statistics about the actions
    logger.info("\nAction Statistics:")
    logger.info("Mean Action: %.4f", np.mean(actions))
    logger.info("Std Action: %.4f", np.std(actions))
    logger.info("Min Action: %.4f", np.min(actions))
    logger.info("Max Action: %.4f", np.max(actions))
    logger.info("Zero Actions: %.2f%%", np.mean(np.array(actions) == 0) * 100)

if __name__ == "__main__":
    main() 