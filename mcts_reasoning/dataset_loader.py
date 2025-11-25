"""
Dataset Loader for Benchmarking

Loads benchmark problems from various formats (JSON, CSV).
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .benchmarking import BenchmarkProblem


class DatasetLoader:
    """Loads benchmark datasets from files."""

    @staticmethod
    def load_json(file_path: str) -> List[BenchmarkProblem]:
        """
        Load problems from JSON file.

        Expected format:
        {
          "name": "Dataset Name",
          "description": "Description",
          "problems": [
            {
              "id": "prob_001",
              "category": "algebra",
              "question": "Solve x + 2 = 5",
              "answer": "3",
              "difficulty": "easy",
              "metadata": {...}
            },
            ...
          ]
        }

        Args:
            file_path: Path to JSON file

        Returns:
            List of BenchmarkProblem instances
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        problems = []
        for prob_data in data.get('problems', []):
            problem = BenchmarkProblem(
                id=prob_data['id'],
                category=prob_data['category'],
                question=prob_data['question'],
                answer=prob_data['answer'],
                difficulty=prob_data.get('difficulty'),
                metadata=prob_data.get('metadata')
            )
            problems.append(problem)

        return problems

    @staticmethod
    def load_csv(file_path: str, id_col: str = 'id',
                category_col: str = 'category',
                question_col: str = 'question',
                answer_col: str = 'answer') -> List[BenchmarkProblem]:
        """
        Load problems from CSV file.

        Args:
            file_path: Path to CSV file
            id_col: Column name for problem ID
            category_col: Column name for category
            question_col: Column name for question
            answer_col: Column name for answer

        Returns:
            List of BenchmarkProblem instances
        """
        import csv

        problems = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                problem = BenchmarkProblem(
                    id=row[id_col],
                    category=row.get(category_col, 'general'),
                    question=row[question_col],
                    answer=row[answer_col],
                    difficulty=row.get('difficulty'),
                    metadata={k: v for k, v in row.items()
                             if k not in [id_col, category_col, question_col, answer_col, 'difficulty']}
                )
                problems.append(problem)

        return problems

    @staticmethod
    def load_directory(directory: str, pattern: str = "*.json") -> Dict[str, List[BenchmarkProblem]]:
        """
        Load all datasets from a directory.

        Args:
            directory: Path to directory
            pattern: File pattern to match (default: *.json)

        Returns:
            Dict of {dataset_name: problems}
        """
        dir_path = Path(directory)
        datasets = {}

        for file_path in dir_path.glob(pattern):
            dataset_name = file_path.stem
            if pattern.endswith('.json'):
                problems = DatasetLoader.load_json(str(file_path))
            elif pattern.endswith('.csv'):
                problems = DatasetLoader.load_csv(str(file_path))
            else:
                continue

            datasets[dataset_name] = problems

        return datasets

    @staticmethod
    def filter_problems(problems: List[BenchmarkProblem],
                       category: str = None,
                       difficulty: str = None,
                       max_count: int = None) -> List[BenchmarkProblem]:
        """
        Filter problems by criteria.

        Args:
            problems: List of problems
            category: Filter by category (optional)
            difficulty: Filter by difficulty (optional)
            max_count: Maximum number to return (optional)

        Returns:
            Filtered list of problems
        """
        filtered = problems

        if category:
            filtered = [p for p in filtered if p.category == category]

        if difficulty:
            filtered = [p for p in filtered if p.difficulty == difficulty]

        if max_count:
            filtered = filtered[:max_count]

        return filtered


__all__ = ['DatasetLoader']
