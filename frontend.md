This frontend is designed to:
- Be intuitive for users contributing funds, monitoring AI trading activity, claiming rewards, and participating in governance.
- Work with your backend API (FastAPI/Flask) and Ethereum-compatible smart contracts.

---

## 🎨 **Frontend Overview**

| Section | Purpose |
|--------|---------|
| 1. Dashboard | Overview of AI agent, trade stats, rewards |
| 2. NFT Minting / Contribution | Allows users to fund and mint NFTs |
| 3. AI Trade Viewer | Displays trade logs, strategy status, and PnL |
| 4. Profit Claiming | Lets NFT holders claim distributed rewards |
| 5. NFT Vault | Shows user's NFTs and performance metadata |
| 6. Governance | Allows proposal creation, voting, and viewing DAO state |
| 7. Admin Console (Optional) | Trigger AI bot, rebalance vaults, emergency controls |

---

## 🖥️ **Screens and Features**

---

### **1. 🏠 Dashboard**

> 📌 Home screen giving real-time project overview

#### Components:
- ✅ Total Liquidity Pool (ETH or USDC)
- 📈 Recent Trade PnL Chart
- 🧠 AI Model Confidence (live signals: BUY, SELL, HOLD)
- 🎁 Total Rewards Distributed
- 👥 Total NFT Holders
- ⛽ Gas & Network Stats

---

### **2. 💸 Contribution & NFT Minting**

> Allows users to contribute funds and receive a share-based NFT

#### Components:
- 💰 Contribution input (ETH, USDC)
- 🎨 NFT Preview (randomized metadata visual)
- 🔘 “Mint & Fund” button (web3 interaction)
- 🧾 Confirmation screen with minted NFT ID
- 📜 Terms & Conditions checkbox (optional)

---

### **3. 📊 AI Trading Activity**

> View live and historical trade logs executed by the AI agent (via Freqtrade)

#### Components:
- 📋 Trade Log Table (timestamp, pair, action, amount, PnL, tx hash)
- 🧠 Strategy Tag (e.g., `freqai_volume_spread`, `custom_v1`)
- 📉 Daily/Weekly Trade Graphs
- ⚠️ “Paused” / “Running” AI Status Banner

---

### **4. 🎁 Reward Claim Screen**

> Shows unclaimed rewards and lets users claim based on NFT holdings

#### Components:
- 🪙 Claimable Amount (per wallet/NFT)
- 🧾 NFT Holding Breakdown
- 🔘 “Claim Rewards” button (on-chain call)
- 🧠 Reward calculation explanation
- ✅ Transaction confirmation and Etherscan link

---

### **5. 🧾 NFT Vault (My NFTs)**

> Show all NFTs owned by the connected wallet with metadata and earnings

#### Components:
- 🎨 NFT Cards (token ID, visual, percent share)
- 💸 Accumulated Rewards
- 📈 Estimated ROI
- 🔁 Transfer NFT to another wallet (optional)
- 🔍 View on IPFS/Etherscan

---

### **6. 🗳️ Governance & Voting**

> Allows NFT holders to propose, vote, and track governance actions

#### Components:
- 🧠 Create Proposal (strategy change, parameter adjustment)
- 📋 Active Proposal List
- 🔘 Vote For / Against buttons (web3 call)
- 📈 Proposal Status (quorum %, result, execution status)
- 🗂️ Historical Proposals Archive

---

### **7. 🔧 Admin Console (for project owner/operator)**

> Internal-only screen for managing AI bot and emergency features

#### Components:
- 🔘 “Trigger Trade Now”
- 🔄 Rebalance Vault Liquidity
- 🔐 Emergency Circuit Breaker
- 🧠 AI Model Update Upload (e.g. upload FinRL .pkl)
- 🔍 Logs Viewer & TX Tracker

---

## 📱 Mobile Responsiveness

- All components should be mobile-first using:
  - TailwindCSS or Bootstrap 5
  - Responsive trading charts (Chart.js or Recharts)
  - MetaMask Mobile + WalletConnect support

---

## 🔗 Integration Points

| Feature | Source |
|--------|--------|
| Trading Logs | Freqtrade logs via API |
| AI Model Signals | From FreqAI predictions |
| NFT Metadata | From Crowdfunding repo or IPFS/Moralis |
| Rewards | From on-chain vault + ProfitDistributor |
| Governance | On-chain voting smart contract |

---

**prompt** to instruct an AI coding agent (or developer assistant) to build the **complete frontend** for your AI-powered, NFT-governed AMM MVP — exactly as described above:

---

## 🎯 **Zero-Shot Frontend Prompt**

> You are a full-stack React developer. Your task is to build the **complete frontend interface** for a production-ready decentralized application (dApp) that integrates:
> - AI-powered crypto trading using [Freqtrade](https://github.com/freqtrade/freqtrade),
> - NFT-based ownership, rewards, and governance using [NFT-Incentivized-Crypto-Crowdfunding](https://github.com/dvoronkov/NFT-Incentivized-Crypto-Crowdfunding),
> - Ethereum-compatible smart contracts (ERC-721 and profit distribution).
>
> ⚠️ **Do not rewrite any backend logic. This frontend should only interface with existing backend APIs and Web3 smart contracts.**

---

### 🧩 Functional Requirements

Implement the following screens and features:

---

### **1. Dashboard (`/`)**
- Display:
  - Total Liquidity (from backend)
  - AI Signal: BUY / SELL / HOLD (from FreqAI)
  - Total Rewards Distributed
  - Total NFT Holders
  - PnL graph (Chart.js/Recharts)
  - Network/Gas Stats (via Ethers.js)

---

### **2. Contribution + NFT Minting (`/contribute`)**
- Input: amount (ETH or stablecoin)
- Call smart contract to contribute and mint NFT
- Show generated NFT preview and metadata
- Confirm transaction with tx hash
- Show updated vault size

---

### **3. AI Trading Viewer (`/trades`)**
- Trade log table:
  - Pair, Action, Confidence, Volume, PnL, Strategy, Timestamp
- Graph: Daily/Weekly Profit
- Status: “AI Trading Engine: Running / Paused”

---

### **4. Rewards Claiming (`/rewards`)**
- Show connected wallet's:
  - Claimable amount
  - List of NFTs
- Button: “Claim Rewards”
- Confirm tx hash after claim

---

### **5. NFT Vault (`/vault`)**
- Card view of user NFTs:
  - NFT ID, % Ownership, Earnings, ROI
- Metadata: Pull from IPFS or Moralis
- Button to transfer NFT (optional)

---

### **6. Governance (`/governance`)**
- Active proposals:
  - Title, Description, Status, Yes/No votes
- Vote interface (MetaMask/Web3)
- Create proposal form (if user is eligible)
- Proposal history archive

---

### **7. Admin Console (`/admin`)**
(Protected route, only for owner/developer)
- Trigger AI trade
- Upload new AI model file
- Rebalance liquidity
- Pause/resume engine
- Export logs

---

### 💻 Tech Stack Required

- React + Vite or Next.js
- Tailwind CSS (or Bootstrap 5)
- Web3.js or Ethers.js
- Web3Modal / WalletConnect
- Chart.js or Recharts
- Axios for API integration
- React Router for navigation

---

### 📦 Integration Instructions

- Connect wallet with MetaMask (ERC-721 detection, claimable logic)
- Use backend APIs for:
  - `/trade/run`, `/trades`, `/rewards`, `/signal`, `/vault`
- Use smart contracts for:
  - NFT minting
  - Profit claiming
  - Governance voting

---

### ✅ Deliverable

- Fully working React app with all 7 pages
- Responsive design (mobile-first)
- Connected to Ethereum testnet (Sepolia or Mumbai)
- No hardcoded logic – pull everything from APIs or contracts
- Clean UI with loading states and error handling

