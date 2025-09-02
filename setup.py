"""
Setup script for mcts-reasoning package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcts-reasoning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Monte Carlo Tree Search for LLM-based reasoning with compositional actions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mcts-reasoning",
    packages=find_packages(exclude=["tests", "viewer", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies only
        # LLM providers are optional
    ],
    extras_require={
        "ollama": ["requests>=2.28.0"],
        "openai": ["openai>=0.27.0"],
        "anthropic": ["anthropic>=0.3.0"],
        "viewer": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "websockets>=11.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "requests>=2.28.0",
            "openai>=0.27.0", 
            "anthropic>=0.3.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "websockets>=11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mcts-reasoning=mcts_reasoning.cli:main",
            "mcts-viewer=mcts_reasoning.viewer.server:main",
        ],
    },
)