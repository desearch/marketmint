const hre = require("hardhat");

async function main() {
  console.log("Deploying contracts...");

  // Deploy AMMAgentNFT
  const AMMAgentNFT = await hre.ethers.getContractFactory("AMMAgentNFT");
  const nftContract = await AMMAgentNFT.deploy();
  await nftContract.deployed();
  console.log("AMMAgentNFT deployed to:", nftContract.address);

  // Deploy ProfitDistributor
  const ProfitDistributor = await hre.ethers.getContractFactory("ProfitDistributor");
  const profitDistributor = await ProfitDistributor.deploy(nftContract.address);
  await profitDistributor.deployed();
  console.log("ProfitDistributor deployed to:", profitDistributor.address);

  // Save contract addresses to a file
  const addresses = {
    nftContract: nftContract.address,
    profitDistributor: profitDistributor.address
  };

  const fs = require('fs');
  fs.writeFileSync(
    'contract-addresses.json',
    JSON.stringify(addresses, null, 2)
  );

  console.log("Contract addresses saved to contract-addresses.json");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 