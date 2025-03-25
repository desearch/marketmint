// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

contract AMMAgentNFT is ERC721, Ownable {
    using Counters for Counters.Counter;
    using Strings for uint256;
    Counters.Counter private _tokenIds;

    // Mapping from token ID to share percentage (in basis points, 1% = 100)
    mapping(uint256 => uint256) public sharePercentages;
    
    // Total shares issued
    uint256 public totalShares;

    // Base URI for token metadata
    string private _baseTokenURI;

    constructor() ERC721("AMM Agent NFT", "AMMNFT") {
        _baseTokenURI = "";
    }

    function mint(address recipient, uint256 sharePercentage) 
        public 
        onlyOwner 
        returns (uint256) 
    {
        require(sharePercentage > 0 && sharePercentage <= 10000, "Invalid share percentage");
        require(totalShares + sharePercentage <= 10000, "Total shares cannot exceed 100%");

        _tokenIds.increment();
        uint256 newItemId = _tokenIds.current();

        _mint(recipient, newItemId);
        sharePercentages[newItemId] = sharePercentage;
        totalShares += sharePercentage;

        return newItemId;
    }

    function _baseURI() internal view override returns (string memory) {
        return _baseTokenURI;
    }

    function setBaseURI(string memory baseURI) public onlyOwner {
        _baseTokenURI = baseURI;
    }

    function tokenURI(uint256 tokenId) 
        public 
        view 
        override 
        returns (string memory) 
    {
        require(_exists(tokenId), "Token does not exist");
        return string(abi.encodePacked(_baseURI(), tokenId.toString()));
    }

    function getSharePercentage(uint256 tokenId) 
        public 
        view 
        returns (uint256) 
    {
        require(_exists(tokenId), "Token does not exist");
        return sharePercentages[tokenId];
    }

    function getTotalShares() public view returns (uint256) {
        return totalShares;
    }
} 