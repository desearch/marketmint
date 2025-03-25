import click
from .profit_distribution import profit
from ai_agent.strategy import AITrader
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.getenv('LOG_FILE', 'amm.log')
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """AMM Agent CLI - Manage your AI-powered trading system"""
    pass

@cli.group()
def trading():
    """Trading strategy management commands"""
    pass

@trading.command()
@click.argument('amount', type=float)
def execute(amount: float):
    """Execute a trade based on AI predictions"""
    trader = AITrader()
    result = trader.execute_trade(amount)
    
    if result['status'] == 'success':
        click.echo(f"Trade executed successfully")
        click.echo(f"Transaction hash: {result['transaction_hash']}")
        click.echo(f"Gas used: {result['gas_used']}")
    elif result['status'] == 'hold':
        click.echo("Model suggests holding position")
    else:
        click.echo(f"Error: {result['message']}", err=True)

@trading.command()
def train():
    """Train the AI model with historical data"""
    trader = AITrader()
    result = trader.train_model()
    
    if result['status'] == 'success':
        click.echo("Model trained successfully")
    else:
        click.echo(f"Error: {result['message']}", err=True)

# Add profit distribution commands
cli.add_command(profit)

if __name__ == '__main__':
    cli() 