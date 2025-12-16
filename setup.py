"""
Setup script for mcts-reasoning package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcts-reasoning",
    version="0.4.0",
    author="Alex Towell",
    author_email="lex@metafunctor.com",
    description="Monte Carlo Tree Search for LLM-based reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/queelius/mcts-reasoning",
    packages=find_packages(exclude=["tests", "examples", "archive"]),
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
        "requests>=2.28.0",  # For Ollama provider
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "requests>=2.28.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mcts-reason=mcts_reasoning.cli:main",
        ],
    },
)