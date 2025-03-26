// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract StrategyNFT is ERC721, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    struct Strategy {
        string strategyId;
        uint256 totalGovernanceTokens;
        bool isActive;
    }
    
    struct NFTData {
        string strategyId;
        uint256 governanceTokens;
        uint256 unclaimedProfits;
    }
    
    // NFT ID counter
    Counters.Counter private _tokenIds;
    
    // Mapping from token ID to NFT data
    mapping(uint256 => NFTData) public nftData;
    
    // Mapping from strategy ID to Strategy
    mapping(string => Strategy) public strategies;
    
    // Mapping from strategy ID to accumulated profits
    mapping(string => uint256) public strategyProfits;
    
    // Events
    event StrategyCreated(string strategyId);
    event NFTMinted(uint256 tokenId, address owner, string strategyId, uint256 governanceTokens);
    event ProfitsDistributed(string strategyId, uint256 amount);
    event ProfitsClaimed(uint256 tokenId, address owner, uint256 amount);
    
    constructor() ERC721("Strategy NFT", "SNFT") {}
    
    function createStrategy(string memory strategyId) external onlyOwner {
        require(!strategies[strategyId].isActive, "Strategy already exists");
        
        strategies[strategyId] = Strategy({
            strategyId: strategyId,
            totalGovernanceTokens: 0,
            isActive: true
        });
        
        emit StrategyCreated(strategyId);
    }
    
    function mintNFT(
        address recipient,
        string memory strategyId,
        uint256 governanceTokens
    ) external onlyOwner returns (uint256) {
        require(strategies[strategyId].isActive, "Strategy does not exist");
        require(governanceTokens > 0, "Governance tokens must be greater than 0");
        
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _safeMint(recipient, newTokenId);
        
        nftData[newTokenId] = NFTData({
            strategyId: strategyId,
            governanceTokens: governanceTokens,
            unclaimedProfits: 0
        });
        
        strategies[strategyId].totalGovernanceTokens += governanceTokens;
        
        emit NFTMinted(newTokenId, recipient, strategyId, governanceTokens);
        return newTokenId;
    }
    
    function distributeStrategyProfits(string memory strategyId) external payable onlyOwner {
        require(strategies[strategyId].isActive, "Strategy does not exist");
        require(msg.value > 0, "No profits to distribute");
        
        Strategy storage strategy = strategies[strategyId];
        require(strategy.totalGovernanceTokens > 0, "No governance tokens issued");
        
        strategyProfits[strategyId] += msg.value;
        
        emit ProfitsDistributed(strategyId, msg.value);
    }
    
    function claimProfits(uint256 tokenId) external nonReentrant {
        require(_exists(tokenId), "NFT does not exist");
        require(ownerOf(tokenId) == msg.sender, "Not token owner");
        
        NFTData storage nft = nftData[tokenId];
        Strategy storage strategy = strategies[nft.strategyId];
        
        uint256 totalProfits = strategyProfits[nft.strategyId];
        uint256 share = (totalProfits * nft.governanceTokens) / strategy.totalGovernanceTokens;
        uint256 claimable = share - nft.unclaimedProfits;
        
        require(claimable > 0, "No profits to claim");
        
        nft.unclaimedProfits += claimable;
        
        (bool success, ) = payable(msg.sender).call{value: claimable}("");
        require(success, "Transfer failed");
        
        emit ProfitsClaimed(tokenId, msg.sender, claimable);
    }
    
    function getNFTData(uint256 tokenId) external view returns (
        string memory strategyId,
        uint256 governanceTokens,
        uint256 unclaimedProfits
    ) {
        require(_exists(tokenId), "NFT does not exist");
        NFTData memory nft = nftData[tokenId];
        return (nft.strategyId, nft.governanceTokens, nft.unclaimedProfits);
    }
    
    function getStrategyData(string memory strategyId) external view returns (
        uint256 totalGovernanceTokens,
        uint256 totalProfits,
        bool isActive
    ) {
        Strategy memory strategy = strategies[strategyId];
        return (
            strategy.totalGovernanceTokens,
            strategyProfits[strategyId],
            strategy.isActive
        );
    }
} 