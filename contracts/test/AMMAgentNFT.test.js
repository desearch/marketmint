const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("AMMAgentNFT", function () {
  let AMMAgentNFT;
  let nftContract;
  let owner;
  let addr1;
  let addr2;

  beforeEach(async function () {
    [owner, addr1, addr2] = await ethers.getSigners();
    AMMAgentNFT = await ethers.getContractFactory("AMMAgentNFT");
    nftContract = await AMMAgentNFT.deploy();
    await nftContract.deployed();
  });

  describe("Minting", function () {
    it("Should mint a new NFT with correct share percentage", async function () {
      const sharePercentage = 1000; // 10%
      await nftContract.mint(addr1.address, sharePercentage);
      
      expect(await nftContract.ownerOf(1)).to.equal(addr1.address);
      expect(await nftContract.getSharePercentage(1)).to.equal(sharePercentage);
      expect(await nftContract.getTotalShares()).to.equal(sharePercentage);
    });

    it("Should not allow minting more than 100% total shares", async function () {
      await nftContract.mint(addr1.address, 6000); // 60%
      await nftContract.mint(addr2.address, 4000); // 40%
      
      await expect(
        nftContract.mint(addr1.address, 100) // Try to mint 1% more
      ).to.be.revertedWith("Total shares cannot exceed 100%");
    });

    it("Should not allow invalid share percentages", async function () {
      await expect(
        nftContract.mint(addr1.address, 10001) // More than 100%
      ).to.be.revertedWith("Invalid share percentage");
      
      await expect(
        nftContract.mint(addr1.address, 0) // Zero shares
      ).to.be.revertedWith("Invalid share percentage");
    });
  });

  describe("Token URI", function () {
    it("Should set and get base URI", async function () {
      const baseURI = "https://example.com/token/";
      await nftContract.setBaseURI(baseURI);
      
      await nftContract.mint(addr1.address, 1000);
      expect(await nftContract.tokenURI(1)).to.equal(baseURI + "1");
    });
  });

  describe("Share Management", function () {
    it("Should track total shares correctly", async function () {
      await nftContract.mint(addr1.address, 3000); // 30%
      await nftContract.mint(addr2.address, 7000); // 70%
      
      expect(await nftContract.getTotalShares()).to.equal(10000);
    });

    it("Should return correct share percentage for token", async function () {
      await nftContract.mint(addr1.address, 2500); // 25%
      
      expect(await nftContract.getSharePercentage(1)).to.equal(2500);
    });
  });
}); 