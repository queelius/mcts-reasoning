#!/usr/bin/env python3
"""
Test Benchmarking Suite

Tests the benchmarking framework with sample datasets and configurations.
"""

import os
from pathlib import Path
from mcts_reasoning.benchmarking import BenchmarkRunner, BenchmarkProblem
from mcts_reasoning.dataset_loader import DatasetLoader
from mcts_reasoning.compositional.providers import MockLLMProvider, get_llm
from mcts_reasoning.compositional.rag import get_math_compositional_rag


def test_benchmark_with_mock():
    """Test benchmarking with mock LLM."""
    print("=" * 80)
    print("TESTING BENCHMARKING SUITE WITH MOCK LLM")
    print("=" * 80)

    # Create mock LLM
    llm = MockLLMProvider(responses={
        "evaluate": "0.7",
        "quality": "0.75",
        # Mock answers
        "solve": "The answer is 6",
        "calculate": "The result is 12"
    })

    # Load sample dataset
    benchmarks_dir = Path(__file__).parent / "benchmarks"

    if not benchmarks_dir.exists():
        print(f"⚠️  Benchmarks directory not found at {benchmarks_dir}")
        print("   Creating sample problems...")

        # Create sample problems
        problems = [
            BenchmarkProblem(
                id="test_001",
                category="arithmetic",
                question="What is 2 + 3?",
                answer="5",
                difficulty="easy",
                metadata={"answer_type": "number"}
            ),
            BenchmarkProblem(
                id="test_002",
                category="arithmetic",
                question="What is 4 * 3?",
                answer="12",
                difficulty="easy",
                metadata={"answer_type": "number"}
            ),
            BenchmarkProblem(
                id="test_003",
                category="algebra",
                question="Solve for x: x + 2 = 7",
                answer="5",
                difficulty="medium",
                metadata={"answer_type": "number"}
            )
        ]
    else:
        # Load from file
        print(f"\n✓ Found benchmarks directory: {benchmarks_dir}")
        try:
            problems = DatasetLoader.load_json(str(benchmarks_dir / "gsm8k_sample.json"))
            print(f"✓ Loaded {len(problems)} problems from gsm8k_sample.json")
        except Exception as e:
            print(f"⚠️  Could not load dataset: {e}")
            print("   Using minimal sample...")
            problems = [
                BenchmarkProblem(
                    id="test_001",
                    category="arithmetic",
                    question="What is 2 + 3?",
                    answer="5",
                    difficulty="easy"
                )
            ]

    # Create benchmark runner
    runner = BenchmarkRunner(llm)

    # Test basic configuration
    print(f"\n[Test 1] Basic Benchmark Run")
    print("─" * 80)

    config = {
        'llm': llm,
        'compositional': True,
        'solution_detection': True,
        'exploration': 1.414,
        'max_depth': 3
    }

    results, summary = runner.run_benchmark(
        problems[:5],  # First 5 problems
        config,
        simulations=10,
        verbose=True
    )

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Total Problems: {summary.total_problems}")
    print(f"Correct: {summary.correct}")
    print(f"Incorrect: {summary.incorrect}")
    print(f"Errors: {summary.errors}")
    print(f"Accuracy: {summary.accuracy:.1%}")
    print(f"Avg Time: {summary.avg_time:.2f}s")
    print(f"Avg LLM Calls: {summary.avg_llm_calls:.0f}")

    print(f"\nCategory Breakdown:")
    for category, stats in summary.category_breakdown.items():
        print(f"  {category:15s} - {stats['correct']}/{stats['total']} "
              f"({stats['accuracy']:.1%})")

    # Save results
    output_path = "benchmark_results_mock.json"
    runner.save_results(results, summary, output_path)
    print(f"\n✓ Results saved to {output_path}")

    # Test configuration comparison
    print(f"\n[Test 2] Configuration Comparison")
    print("─" * 80)

    configurations = {
        "baseline": {
            'llm': llm,
            'compositional': True,
            'solution_detection': False,
            'meta_reasoning': False,
            'reflection': False
        },
        "full_features": {
            'llm': llm,
            'compositional': True,
            'solution_detection': True,
            'meta_reasoning': False,  # Keep disabled for mock (too many calls)
            'reflection': False
        }
    }

    comparison = runner.compare_configurations(
        problems[:3],  # Just 3 for comparison
        configurations,
        simulations=10,
        max_problems=3
    )

    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*80}")

    for config_name, (results, summary) in comparison.items():
        print(f"\n{config_name}:")
        print(f"  Accuracy: {summary.accuracy:.1%}")
        print(f"  Avg Time: {summary.avg_time:.2f}s")
        print(f"  Avg LLM Calls: {summary.avg_llm_calls:.0f}")

    print(f"\n{'='*80}")
    print("✅ Benchmarking suite test PASSED")
    print(f"{'='*80}")


def test_benchmark_with_real_llm():
    """Test benchmarking with real LLM."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("\n⚠️  No API keys found - skipping real LLM benchmark test")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test with real LLM")
        return

    print("\n" + "=" * 80)
    print("TESTING BENCHMARKING SUITE WITH REAL LLM")
    print("=" * 80)

    # Get real LLM
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
    llm = get_llm(provider)
    print(f"✓ Connected to {provider} LLM\n")

    # Load dataset
    benchmarks_dir = Path(__file__).parent / "benchmarks"

    datasets = {}

    if benchmarks_dir.exists():
        # Load all datasets
        datasets = DatasetLoader.load_directory(str(benchmarks_dir))
        print(f"✓ Loaded {len(datasets)} datasets:")
        for name, problems in datasets.items():
            print(f"  - {name}: {len(problems)} problems")

    if not datasets:
        print("⚠️  No datasets found, using sample problems")
        datasets = {
            "sample": [
                BenchmarkProblem(
                    id="real_001",
                    category="algebra",
                    question="Solve for x: 2x + 3 = 11",
                    answer="4",
                    metadata={"answer_type": "number"}
                ),
                BenchmarkProblem(
                    id="real_002",
                    category="number_theory",
                    question="What is the smallest prime number greater than 10?",
                    answer="11",
                    metadata={"answer_type": "number"}
                )
            ]
        }

    # Select dataset to test
    dataset_name = list(datasets.keys())[0]
    problems = datasets[dataset_name]

    print(f"\nTesting with dataset: {dataset_name}")
    print(f"Total problems: {len(problems)}")

    # Create runner
    runner = BenchmarkRunner(llm)

    # Get math RAG
    rag_store = get_math_compositional_rag()

    # Test with full features
    print(f"\n[Real LLM Benchmark]")
    print("─" * 80)

    config = {
        'llm': llm,
        'compositional': True,
        'rag_store': rag_store,
        'solution_detection': True,
        'meta_reasoning': False,  # Disabled to reduce LLM calls
        'reflection': False,      # Disabled to reduce LLM calls
        'learning': False,
        'exploration': 1.414,
        'max_depth': 4
    }

    results, summary = runner.run_benchmark(
        problems,
        config,
        simulations=20,
        max_problems=5,  # Test on first 5 problems
        verbose=True
    )

    print(f"\n{'='*80}")
    print("REAL LLM BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Problems: {summary.total_problems}")
    print(f"Accuracy: {summary.accuracy:.1%} ({summary.correct}/{summary.total_problems})")
    print(f"Avg Time: {summary.avg_time:.2f}s")
    print(f"Avg LLM Calls: {summary.avg_llm_calls:.0f}")

    print(f"\nCategory Breakdown:")
    for category, stats in summary.category_breakdown.items():
        print(f"  {category:15s} - Accuracy: {stats['accuracy']:.1%} "
              f"({stats['correct']}/{stats['total']})")

    # Show sample results
    print(f"\nSample Results:")
    for i, result in enumerate(results[:3], 1):
        status = "✓" if result.is_correct else "✗"
        print(f"\n  {i}. {status} {result.question[:60]}...")
        print(f"     Ground Truth: {result.ground_truth}")
        print(f"     Predicted: {result.predicted_answer}")
        print(f"     Time: {result.time_seconds:.1f}s, LLM Calls: {result.llm_calls}")

    # Save results
    output_path = f"benchmark_results_{dataset_name}_real.json"
    runner.save_results(results, summary, output_path)
    print(f"\n✓ Results saved to {output_path}")

    print(f"\n{'='*80}")


def test_dataset_loader():
    """Test dataset loading functionality."""
    print("\n" + "=" * 80)
    print("TESTING DATASET LOADER")
    print("=" * 80)

    benchmarks_dir = Path(__file__).parent / "benchmarks"

    if not benchmarks_dir.exists():
        print(f"⚠️  Benchmarks directory not found")
        return

    # Test loading JSON
    print("\n[Test 1] Loading JSON datasets")
    print("─" * 80)

    datasets = DatasetLoader.load_directory(str(benchmarks_dir))

    print(f"✓ Found {len(datasets)} datasets:")
    for name, problems in datasets.items():
        print(f"\n  Dataset: {name}")
        print(f"    Problems: {len(problems)}")

        # Show categories
        categories = {}
        for p in problems:
            categories[p.category] = categories.get(p.category, 0) + 1

        print(f"    Categories: {dict(categories)}")

        # Show sample
        if problems:
            sample = problems[0]
            print(f"    Sample: {sample.question[:60]}...")

    # Test filtering
    print("\n[Test 2] Filtering problems")
    print("─" * 80)

    if datasets:
        dataset_name = list(datasets.keys())[0]
        all_problems = datasets[dataset_name]

        # Filter by category
        categories = set(p.category for p in all_problems)
        if categories:
            test_category = list(categories)[0]
            filtered = DatasetLoader.filter_problems(
                all_problems,
                category=test_category
            )
            print(f"✓ Filtered by category '{test_category}': {len(filtered)} problems")

        # Filter by difficulty
        filtered_easy = DatasetLoader.filter_problems(
            all_problems,
            difficulty="easy"
        )
        print(f"✓ Filtered by difficulty 'easy': {len(filtered_easy)} problems")

        # Limit count
        filtered_limited = DatasetLoader.filter_problems(
            all_problems,
            max_count=3
        )
        print(f"✓ Limited to 3 problems: {len(filtered_limited)} problems")

    print(f"\n{'='*80}")
    print("✅ Dataset loader test PASSED")
    print(f"{'='*80}")


def main():
    """Run all benchmarking tests."""
    print("\n" + "=" * 80)
    print("BENCHMARKING SUITE TEST")
    print("=" * 80)

    # Test dataset loader
    test_dataset_loader()

    # Test with mock LLM
    test_benchmark_with_mock()

    # Test with real LLM if available
    test_benchmark_with_real_llm()

    print("\n" + "=" * 80)
    print("ALL BENCHMARKING TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
