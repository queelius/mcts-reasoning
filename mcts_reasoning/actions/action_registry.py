from typing import Dict, List, Type, Optional
import random

from .base_action import BaseAction, ComposableAction
from .thinking_actions import (
    ThinkStepByStepAction,
    IdentifyPrincipleAction, 
    AnalyzeConstraintsAction,
    ConsiderAlternativesAction,
    BreakIntoSubproblemsAction,
    SolveSubproblemAction,
    ReflectAndRefineAction
)
from .context_actions import (
    SummarizeContextAction,
    DistillContextAction,
    ExtractMainIdeaAction,
    ExtractAssumptionsAction,
    ExtractEvidenceAction,
    ReframeContextAction,
    IsolateSubproblemContextAction,
    ProduceFinalAnswerAction
)


class ActionRegistry:
    """Registry for all available thinking actions."""
    
    def __init__(self):
        self._actions: Dict[str, BaseAction] = {}
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register all default thinking actions."""
        actions = [
            # Problem-solving actions
            ThinkStepByStepAction(),
            IdentifyPrincipleAction(),
            AnalyzeConstraintsAction(),
            ConsiderAlternativesAction(),
            BreakIntoSubproblemsAction(),
            SolveSubproblemAction(),
            ReflectAndRefineAction(),
            
            # Context manipulation actions
            SummarizeContextAction(),
            DistillContextAction(),
            ExtractMainIdeaAction(),
            ExtractAssumptionsAction(),
            ExtractEvidenceAction(),
            ReframeContextAction(),
            IsolateSubproblemContextAction(),
            ProduceFinalAnswerAction()
        ]
        
        for action in actions:
            self.register_action(action)
    
    def register_action(self, action: BaseAction):
        """Register a new action."""
        self._actions[action.action_type] = action
    
    def get_action(self, action_type: str) -> Optional[BaseAction]:
        """Get an action by type."""
        return self._actions.get(action_type)
    
    def list_actions(self) -> Dict[str, str]:
        """List all available actions with their descriptions."""
        return {
            action_type: action.description 
            for action_type, action in self._actions.items()
        }
    
    def get_applicable_actions(self, node_depth: int, is_subproblem: bool = False, 
                              context_length: int = 0) -> List[str]:
        """Get actions that are applicable given the context."""
        if is_subproblem:
            # For subproblems, focus on direct solving approaches
            return [
                "think_step_by_step",
                "identify_principle",
                "analyze_constraints",
                "consider_alternatives",
                "isolate_subproblem_context"  # Can further isolate context
            ]
        
        # Context actions become more relevant as context grows
        context_actions = []
        if context_length > 1000:  # Add context actions when context is substantial
            context_actions = [
                "summarize_context",
                "distill_context",
                "extract_main_idea"
            ]
        if context_length > 2000:  # Add more aggressive context management
            context_actions.extend([
                "extract_evidence",
                "extract_assumptions"
            ])
        
        if node_depth == 0:
            # Root level - start with problem-solving actions
            return [
                "think_step_by_step",
                "identify_principle", 
                "analyze_constraints",
                "consider_alternatives",
                "break_into_subproblems"
            ]
        elif node_depth < 3:
            # Early depth - most actions available plus context management
            return [
                "think_step_by_step",
                "identify_principle",
                "analyze_constraints", 
                "consider_alternatives",
                "break_into_subproblems",
                "reflect_and_refine",
                "reframe_context"  # Try different perspectives early
            ] + context_actions
        elif node_depth < 6:
            # Mid depth - balance refinement with context management
            return [
                "consider_alternatives",
                "reflect_and_refine",
                "think_step_by_step",
                "distill_context",  # Start distilling insights
                "extract_main_idea"
            ] + context_actions
        else:
            # Deep levels - focus on convergence and clarity
            return [
                "extract_main_idea",
                "distill_context",
                "summarize_context",
                "reflect_and_refine"
            ]
    
    def sample_action(self, applicable_actions: List[str], 
                     exclude: List[str] = None) -> Optional[str]:
        """Sample a random action from applicable actions."""
        if exclude is None:
            exclude = []
        
        available = [action for action in applicable_actions if action not in exclude]
        
        if not available:
            return None
        
        return random.choice(available)
    
    def get_weighted_action_selection(self, applicable_actions: List[str],
                                    prefer_decomposition: bool = False) -> str:
        """Get action with weighted selection based on preferences."""
        if not applicable_actions:
            return "think_step_by_step"  # Fallback
        
        # Define weights for different actions
        weights = {
            "break_into_subproblems": 3.0 if prefer_decomposition else 1.5,
            "think_step_by_step": 2.0,
            "identify_principle": 1.5,
            "analyze_constraints": 1.2,
            "consider_alternatives": 1.8,
            "reflect_and_refine": 1.0,
            "solve_subproblem": 2.0  # This is typically called automatically
        }
        
        # Calculate weighted probabilities
        action_weights = []
        for action in applicable_actions:
            weight = weights.get(action, 1.0)
            action_weights.append(weight)
        
        total_weight = sum(action_weights)
        if total_weight == 0:
            return random.choice(applicable_actions)
        
        # Weighted random selection
        r = random.random() * total_weight
        cumulative = 0
        
        for i, weight in enumerate(action_weights):
            cumulative += weight
            if r <= cumulative:
                return applicable_actions[i]
        
        return applicable_actions[-1]  # Fallback
    
    def is_composable_action(self, action_type: str) -> bool:
        """Check if an action is composable (can have children)."""
        action = self.get_action(action_type)
        return isinstance(action, ComposableAction)