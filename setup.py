"""
Setup script for mcts-reasoning package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcts-reasoning",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Monte Carlo Tree Search for LLM-based reasoning with advanced compositional prompting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mcts-reasoning",
    packages=find_packages(exclude=["tests", "examples"]),
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
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",  # For UCB1 calculations
    ],
    extras_require={
        "ollama": ["requests>=2.28.0"],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "tui": [
            "rich>=13.0.0",  # For beautiful TUI
            "prompt_toolkit>=3.0.0",  # For advanced input with history and completion
        ],
        "mcp": [
            # Future: official MCP SDK when available
            # For now, MCP support is built-in
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocs-autorefs>=0.5.0",
            "pymdown-extensions>=10.0.0",
        ],
        "all": [
            "numpy>=1.20.0",
            "requests>=2.28.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "rich>=13.0.0",
            "prompt_toolkit>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            # Primary commands
            "mcts=mcts_reasoning.cli:main",              # Non-interactive CLI
            "mcts-shell=mcts_reasoning.tui:main",  # Interactive shell (TUI)

            # Backwards compatibility aliases
            "mcts-tui=mcts_reasoning.tui:main",     # Legacy TUI name
            "mcts-reasoning-tui=mcts_reasoning.tui:main",
        ],
    },
)