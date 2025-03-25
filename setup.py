from setuptools import setup, find_packages

setup(
    name="ai_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0"
    ],
    author="Market Mint",
    author_email="info@marketmint.com",
    description="NFT-based trading agent system",
    keywords="trading,nft,market-making",
    python_requires=">=3.8"
) 