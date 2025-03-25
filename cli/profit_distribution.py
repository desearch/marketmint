import click
from ai_agent.profit_distributor import ProfitDistributor
import json
from datetime import datetime
from typing import Dict, List

@click.group()
def profit():
    """Profit distribution management commands"""
    pass

@profit.command()
@click.argument('amount', type=float)
def distribute(amount: float):
    """Distribute profits to NFT holders"""
    distributor = ProfitDistributor()
    result = distributor.distribute_profits(amount)
    
    if result['status'] == 'success':
        click.echo(f"Successfully distributed {amount} ETH")
        click.echo(f"Transaction hash: {result['transaction_hash']}")
        click.echo(f"Gas used: {result['gas_used']}")
        click.echo(f"Distribution details:")
        click.echo(f"  Total amount: {result['distribution']['total_amount']} ETH")
        click.echo(f"  Timestamp: {result['distribution']['timestamp']}")
        click.echo(f"  NFT holders: {result['distribution']['nft_holders']}")
    else:
        click.echo(f"Error: {result['message']}", err=True)

@profit.command()
def holders():
    """List all NFT holders and their shares"""
    distributor = ProfitDistributor()
    holders = distributor.get_nft_holders()
    
    if not holders:
        click.echo("No NFT holders found")
        return
    
    click.echo("NFT Holders:")
    for holder in holders:
        click.echo(f"  Address: {holder['address']}")
        click.echo(f"  Token ID: {holder['token_id']}")
        click.echo(f"  Share: {holder['share']}%")
        click.echo("---")

@profit.command()
@click.option('--limit', default=10, help='Number of distributions to show')
def history(limit: int):
    """Show profit distribution history"""
    distributor = ProfitDistributor()
    history = distributor.get_distribution_history(limit)
    
    if not history:
        click.echo("No distribution history found")
        return
    
    click.echo("Distribution History:")
    for dist in history:
        click.echo(f"  Amount: {dist['amount']} ETH")
        click.echo(f"  Timestamp: {dist['timestamp']}")
        click.echo(f"  NFT holders: {dist['nft_holders']}")
        click.echo("---")

@profit.command()
def total():
    """Show total amount of profits distributed"""
    distributor = ProfitDistributor()
    total = distributor.get_total_distributed()
    click.echo(f"Total profits distributed: {total} ETH")

if __name__ == '__main__':
    profit() 