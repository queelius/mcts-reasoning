"""
Meta-Reasoning: LLM Suggests Next Action

Instead of purely random or RAG-weighted action selection, the LLM can analyze
the current reasoning state and suggest which type of cognitive operation would
be most productive to try next.

This enables adaptive reasoning strategies based on current progress.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActionSuggestion:
    """LLM's suggestion for next reasoning action."""

    operation: str  # Suggested CognitiveOperation
    focus: Optional[str] = None  # Suggested FocusAspect
    style: Optional[str] = None  # Suggested ReasoningStyle
    reasoning: str = ""  # LLM's explanation
    confidence: float = 0.5  # 0-1


class MetaReasoner:
    """
    Uses LLM to suggest next reasoning actions based on current state.

    Analyzes the current reasoning state and suggests which cognitive
    operations would be most productive to explore next.
    """

    def __init__(self, llm, temperature: float = 0.2):
        """
        Initialize meta-reasoner.

        Args:
            llm: LLM provider for meta-reasoning
            temperature: Temperature for meta-reasoning (lower = more focused)
        """
        self.llm = llm
        self.temperature = temperature
        self._suggestion_count = 0
        self._suggestions_history = []

    def suggest_next_action(
        self,
        current_state: str,
        original_question: str,
        available_operations: Optional[List[str]] = None
    ) -> ActionSuggestion:
        """
        Ask LLM to suggest next reasoning action.

        Args:
            current_state: Current reasoning state/thoughts
            original_question: Original problem being solved
            available_operations: List of available operations (optional)

        Returns:
            ActionSuggestion with recommended operation and reasoning
        """
        # Default operations if not provided
        if available_operations is None:
            available_operations = [
                "decompose", "analyze", "synthesize", "verify",
                "abstract", "concretize", "compare", "evaluate",
                "generate", "refine"
            ]

        # Build prompt
        ops_list = ", ".join(available_operations)

        # Truncate state for token efficiency
        state_preview = current_state[-1500:] if len(current_state) > 1500 else current_state

        prompt = f"""You are a meta-reasoning system analyzing a reasoning process.

Original Problem:
{original_question}

Current Reasoning State (recent):
{state_preview}

Available Cognitive Operations:
{ops_list}

Task: Analyze the current reasoning state and suggest which cognitive operation would be MOST PRODUCTIVE to try next.

Consider:
1. What has been tried so far?
2. What key aspects are missing or need more work?
3. What would advance the reasoning most effectively?
4. Is the reasoning stuck in a pattern?

Operations Guide:
- decompose: Break problem into smaller parts
- analyze: Examine components/structure carefully
- synthesize: Combine insights into coherent whole
- verify: Check correctness/validity
- abstract: Identify general patterns/principles
- concretize: Make abstract ideas specific/concrete
- compare: Contrast different approaches/aspects
- evaluate: Assess quality/completeness
- generate: Create new solutions/ideas
- refine: Improve/polish existing reasoning

Respond in this EXACT format:
OPERATION: [one operation from the list above]
FOCUS: [what aspect to focus on - one word/phrase]
STYLE: [systematic/intuitive/critical/creative]
CONFIDENCE: [0.0 to 1.0]
REASONING: [One sentence explaining why this operation would be most productive]"""

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=self.temperature
            )

            # Parse response
            suggestion = self._parse_suggestion(response, available_operations)

            self._suggestion_count += 1
            self._suggestions_history.append({
                'operation': suggestion.operation,
                'confidence': suggestion.confidence,
                'reasoning': suggestion.reasoning
            })

            logger.info(f"Meta-reasoning suggestion #{self._suggestion_count}: "
                       f"{suggestion.operation} (confidence={suggestion.confidence:.2f})")
            logger.debug(f"  Reasoning: {suggestion.reasoning}")

            return suggestion

        except Exception as e:
            logger.error(f"Meta-reasoning failed: {e}")
            # Fallback to neutral suggestion
            return ActionSuggestion(
                operation=available_operations[0] if available_operations else "analyze",
                confidence=0.3,
                reasoning="Meta-reasoning failed, using fallback"
            )

    def _parse_suggestion(
        self,
        response: str,
        available_operations: List[str]
    ) -> ActionSuggestion:
        """Parse LLM response into ActionSuggestion."""
        lines = response.strip().split('\n')

        operation = None
        focus = None
        style = None
        confidence = 0.5
        reasoning = ""

        for line in lines:
            line = line.strip()

            if line.startswith("OPERATION:"):
                op_text = line.split(":", 1)[1].strip().lower()
                # Extract operation name
                for op in available_operations:
                    if op in op_text:
                        operation = op
                        break

            elif line.startswith("FOCUS:"):
                focus = line.split(":", 1)[1].strip()

            elif line.startswith("STYLE:"):
                style = line.split(":", 1)[1].strip().lower()

            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_text = line.split(":", 1)[1].strip()
                    confidence = float(conf_text)
                except:
                    confidence = 0.5

            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Fallback if operation not found
        if operation is None:
            operation = available_operations[0] if available_operations else "analyze"
            logger.warning(f"Could not parse operation from response, using '{operation}'")

        return ActionSuggestion(
            operation=operation,
            focus=focus,
            style=style,
            reasoning=reasoning,
            confidence=confidence
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-reasoning statistics."""
        if not self._suggestions_history:
            return {
                'suggestion_count': 0,
                'average_confidence': 0.0,
                'most_suggested': None
            }

        # Count operation frequencies
        from collections import Counter
        op_counts = Counter(s['operation'] for s in self._suggestions_history)
        most_common = op_counts.most_common(1)

        avg_confidence = sum(s['confidence'] for s in self._suggestions_history) / len(self._suggestions_history)

        return {
            'suggestion_count': self._suggestion_count,
            'average_confidence': avg_confidence,
            'most_suggested': most_common[0] if most_common else None,
            'operation_distribution': dict(op_counts),
            'recent_suggestions': self._suggestions_history[-5:]
        }


class MetaReasoningActionSelector:
    """
    Wrapper around ActionSelector that uses meta-reasoning to guide action selection.

    Instead of replacing action selection entirely, this biases the selection
    toward LLM-suggested operations while maintaining exploration.
    """

    def __init__(self, base_selector, meta_reasoner: MetaReasoner, bias_strength: float = 3.0):
        """
        Initialize meta-reasoning action selector.

        Args:
            base_selector: Base ActionSelector (compositional)
            meta_reasoner: MetaReasoner instance
            bias_strength: How much to bias toward suggested operation (multiplier)
        """
        self.base_selector = base_selector
        self.meta_reasoner = meta_reasoner
        self.bias_strength = bias_strength

    def get_valid_actions(self, current_state: Optional[str] = None,
                         previous_action = None,
                         n_samples: int = 15,
                         problem: Optional[str] = None):
        """Get actions biased toward meta-reasoning suggestion."""
        # Get base actions
        base_actions = self.base_selector.get_valid_actions(
            current_state=current_state,
            previous_action=previous_action,
            n_samples=n_samples,
            problem=problem
        )

        # If we have state and problem, get meta-reasoning suggestion
        if current_state and problem and self.meta_reasoner:
            # Get available operations from base selector
            from mcts_reasoning.compositional import CognitiveOperation
            available_ops = [op.value for op in CognitiveOperation]

            # Get suggestion
            suggestion = self.meta_reasoner.suggest_next_action(
                current_state, problem, available_ops
            )

            # Find actions matching the suggestion and boost them
            boosted_actions = []
            other_actions = []

            for action in base_actions:
                from mcts_reasoning.compositional.actions import CompositionalAction
                if isinstance(action, CompositionalAction):
                    if action.operation.value == suggestion.operation:
                        boosted_actions.append(action)
                    else:
                        other_actions.append(action)
                else:
                    other_actions.append(action)

            # Return biased list: repeat suggested actions to increase probability
            # Use confidence to modulate bias strength
            boost_factor = int(self.bias_strength * suggestion.confidence) + 1
            biased_actions = boosted_actions * boost_factor + other_actions

            logger.debug(f"Meta-reasoning bias: {len(boosted_actions)} actions boosted "
                        f"{boost_factor}x (suggested: {suggestion.operation})")

            return biased_actions if biased_actions else base_actions

        # Fallback to base actions
        return base_actions


__all__ = ['MetaReasoner', 'ActionSuggestion', 'MetaReasoningActionSelector']
