# AI-Powered NFT-Governed AMM

An innovative Automatic Market Maker (AMM) system that combines AI trading strategies with NFT-based governance.

## 🚀 Features

- NFT-based governance system using ERC-721 tokens
- AI-powered trading using FinRL
- Automated profit distribution to NFT holders
- CLI-based interface for all operations
- Integration with Uniswap v3

## 🛠️ Project Structure

```
amm-nft-ai-agent/
│
├── contracts/          # Smart contracts
├── cli/               # CLI tools and commands
├── ai_agent/          # AI trading strategies
├── data/              # Local data storage
└── tests/             # Test files
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- Ethereum wallet with testnet ETH
- Access to Ethereum testnet (Sepolia/Goerli)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amm-nft-ai-agent.git
cd amm-nft-ai-agent
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Node.js dependencies:
```bash
cd contracts
npm install
```

4. Create a `.env` file with your configuration:
```
PRIVATE_KEY=your_private_key
RPC_URL=your_ethereum_node_url
```

## 🚀 Usage

### CLI Commands

- Run AI trading:
```bash
python cli.py trade
```

- Distribute profits:
```bash
python cli.py distribute
```

- View earnings:
```bash
python cli.py earnings
```

- Create governance proposal:
```bash
python cli.py vote --proposal="upgrade AI"
```

- Check proposal status:
```bash
python cli.py proposal-status
```

## 🔒 Security

- Never share your private keys
- Always use environment variables for sensitive data
- Test thoroughly on testnet before mainnet deployment

## 🧪 Testing

1. Start local testnet:
```bash
ganache-cli
```

2. Deploy contracts:
```bash
cd contracts
npx hardhat run scripts/deploy.js --network localhost
```

3. Run tests:
```bash
python -m pytest tests/
```

## 📝 License

MIT License - See LICENSE file for details 