from setuptools import setup, find_packages

setup(
    name="amm-agent",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.0.0",
        "web3>=5.0.0",
        "python-dotenv>=0.19.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "stable-baselines3>=1.5.0",
        "finrl>=0.3.0"
    ],
    entry_points={
        "console_scripts": [
            "amm-agent=cli.main:cli",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered AMM trading system with profit distribution to NFT holders",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/amm-agent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
) 