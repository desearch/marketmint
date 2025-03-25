// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./AMMAgentNFT.sol";

contract ProfitDistributor is Ownable, ReentrancyGuard {
    AMMAgentNFT public nftContract;
    
    // Mapping to track unclaimed profits per token
    mapping(uint256 => uint256) public unclaimedProfits;
    
    // Total profits distributed
    uint256 public totalProfitsDistributed;

    event ProfitDistributed(uint256 amount);
    event ProfitClaimed(uint256 tokenId, uint256 amount);

    constructor(address _nftContract) {
        nftContract = AMMAgentNFT(_nftContract);
    }

    // Function to receive ETH
    receive() external payable {}

    // Distribute profits to NFT holders
    function distributeProfits() external payable onlyOwner nonReentrant {
        require(msg.value > 0, "No profits to distribute");
        
        uint256 totalShares = nftContract.getTotalShares();
        require(totalShares > 0, "No NFT shares issued");

        // Calculate profit per share (in wei)
        uint256 profitPerShare = msg.value / totalShares;
        
        // Update total profits distributed
        totalProfitsDistributed += msg.value;
        
        emit ProfitDistributed(msg.value);
    }

    // Claim profits for a specific NFT
    function claimProfits(uint256 tokenId) external nonReentrant {
        require(nftContract.ownerOf(tokenId) == msg.sender, "Not token owner");
        
        uint256 sharePercentage = nftContract.getSharePercentage(tokenId);
        uint256 totalShares = nftContract.getTotalShares();
        
        // Calculate claimable amount
        uint256 claimableAmount = (totalProfitsDistributed * sharePercentage / totalShares) - unclaimedProfits[tokenId];
        require(claimableAmount > 0, "No profits to claim");

        // Update unclaimed profits
        unclaimedProfits[tokenId] += claimableAmount;

        // Transfer profits
        (bool success, ) = msg.sender.call{value: claimableAmount}("");
        require(success, "Transfer failed");

        emit ProfitClaimed(tokenId, claimableAmount);
    }

    // View function to check claimable profits
    function getClaimableProfits(uint256 tokenId) external view returns (uint256) {
        uint256 sharePercentage = nftContract.getSharePercentage(tokenId);
        uint256 totalShares = nftContract.getTotalShares();
        
        return (totalProfitsDistributed * sharePercentage / totalShares) - unclaimedProfits[tokenId];
    }

    // Emergency withdrawal function
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        (bool success, ) = owner().call{value: balance}("");
        require(success, "Transfer failed");
    }
} 