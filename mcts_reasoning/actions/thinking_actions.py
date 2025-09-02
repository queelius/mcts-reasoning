from typing import List, Dict, Any
import re

from .base_action import BaseAction, ComposableAction, ActionResult
from ..llm.ollama_client import LLMManager


class ThinkStepByStepAction(BaseAction):
    """Action that prompts for step-by-step reasoning."""
    
    def __init__(self):
        super().__init__("think_step_by_step", "Think through this problem step by step")
    
    def get_prompt_template(self) -> str:
        return """
Let's think through this step by step.

Context: {context}

Original Question: {original_question}

Please break down your reasoning into clear, numbered steps. Think carefully about each step before moving to the next.
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        prompt = self.get_prompt_template().format(
            context=context,
            original_question=original_question
        )
        
        response = llm_manager.reason(prompt, temperature=0.7)
        
        return ActionResult(
            response=response.text,
            new_context=f"{context}\n\nStep-by-step reasoning:\n{response.text}"
        )


class IdentifyPrincipleAction(BaseAction):
    """Action that identifies the main principle or concept in the problem."""
    
    def __init__(self):
        super().__init__("identify_principle", "Identify the main principle or key concept")
    
    def get_prompt_template(self) -> str:
        return """
What is the main principle, key concept, or fundamental idea that applies to this problem?

Context: {context}

Original Question: {original_question}

Please identify and explain the core principle that would help solve this problem. Focus on the underlying concept rather than specific steps.
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        prompt = self.get_prompt_template().format(
            context=context,
            original_question=original_question
        )
        
        response = llm_manager.reason(prompt, temperature=0.6)
        
        return ActionResult(
            response=response.text,
            new_context=f"{context}\n\nKey principle identified:\n{response.text}"
        )


class AnalyzeConstraintsAction(BaseAction):
    """Action that analyzes constraints and limitations of the problem."""
    
    def __init__(self):
        super().__init__("analyze_constraints", "Analyze constraints and limitations")
    
    def get_prompt_template(self) -> str:
        return """
What are the constraints, limitations, or requirements that must be considered for this problem?

Context: {context}

Original Question: {original_question}

Please identify:
1. Any explicit constraints mentioned
2. Implicit limitations or assumptions
3. Requirements that must be satisfied
4. Boundaries or scope limitations
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        prompt = self.get_prompt_template().format(
            context=context,
            original_question=original_question
        )
        
        response = llm_manager.reason(prompt, temperature=0.5)
        
        return ActionResult(
            response=response.text,
            new_context=f"{context}\n\nConstraints analysis:\n{response.text}"
        )


class ConsiderAlternativesAction(BaseAction):
    """Action that considers alternative approaches or solutions."""
    
    def __init__(self):
        super().__init__("consider_alternatives", "Consider alternative approaches")
    
    def get_prompt_template(self) -> str:
        return """
What are some alternative approaches or different ways to think about this problem?

Context: {context}

Original Question: {original_question}

Please brainstorm different approaches, perspectives, or methods that could be used. Consider:
- Different methodologies
- Various angles of approach  
- Alternative interpretations
- Different solution strategies
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        prompt = self.get_prompt_template().format(
            context=context,
            original_question=original_question
        )
        
        response = llm_manager.reason(prompt, temperature=0.8)
        
        return ActionResult(
            response=response.text,
            new_context=f"{context}\n\nAlternative approaches:\n{response.text}"
        )


class BreakIntoSubproblemsAction(ComposableAction):
    """Action that breaks the problem into smaller subproblems."""
    
    def __init__(self):
        super().__init__("break_into_subproblems", "Break into smaller subproblems")
    
    def should_preserve_original_question(self) -> bool:
        return False  # Subproblems get their own questions
    
    def get_prompt_template(self) -> str:
        return """
Break this problem down into smaller, independent subproblems that can be solved separately.

Context: {context}

Original Question: {original_question}

Please:
1. Identify 2-4 smaller subproblems that together would solve the main problem
2. Make sure each subproblem can be solved independently
3. Format each subproblem as a clear, standalone question
4. Explain how solving these subproblems would contribute to the overall solution

Format your response as:
SUBPROBLEM 1: [question]
SUBPROBLEM 2: [question]
...
EXPLANATION: [how these contribute to the overall solution]
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        prompt = self.get_prompt_template().format(
            context=context,
            original_question=original_question
        )
        
        response = llm_manager.reason(prompt, temperature=0.6)
        
        # Parse subproblems from the response
        subproblems = self._extract_subproblems(response.text)
        
        return ActionResult(
            response=response.text,
            new_context=f"Breaking down: {original_question}\n\n{response.text}",
            children_actions=["solve_subproblem"] * len(subproblems),
            requires_composition=True,
            metadata={"subproblems": subproblems}
        )
    
    def _extract_subproblems(self, response: str) -> List[str]:
        """Extract subproblem questions from the response."""
        subproblems = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for lines starting with "SUBPROBLEM N:"
            match = re.match(r'SUBPROBLEM\s+\d+:\s*(.+)', line, re.IGNORECASE)
            if match:
                subproblems.append(match.group(1).strip())
        
        return subproblems
    
    def compose_results(self, children_responses: List[str], 
                       original_context: str, llm_manager: LLMManager) -> str:
        """Compose the subproblem solutions into a complete answer."""
        compose_prompt = f"""
I broke down a problem into subproblems and got solutions for each. Please compose these into a complete, coherent answer to the original question.

Original Context: {original_context}

Subproblem Solutions:
{chr(10).join(f"Solution {i+1}: {resp}" for i, resp in enumerate(children_responses))}

Please provide a complete, well-structured answer that integrates all the subproblem solutions.
"""
        
        response = llm_manager.reason(compose_prompt, temperature=0.5)
        # Ensure we return a string, not an LLMResponse object
        if isinstance(response, str):
            return response
        return response.text if hasattr(response, 'text') else str(response)


class SolveSubproblemAction(BaseAction):
    """Action for solving individual subproblems."""
    
    def __init__(self):
        super().__init__("solve_subproblem", "Solve a specific subproblem")
    
    def should_preserve_original_question(self) -> bool:
        return False  # The subproblem becomes the new question
    
    def get_prompt_template(self) -> str:
        return """
Solve this specific subproblem thoroughly:

Subproblem: {subproblem}

Context from parent: {context}

Please provide a complete solution to this subproblem. Be thorough and clear in your reasoning.
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, subproblem: str = None, **kwargs) -> ActionResult:
        
        if not subproblem:
            subproblem = original_question  # Fallback if no specific subproblem provided
        
        prompt = self.get_prompt_template().format(
            subproblem=subproblem,
            context=context
        )
        
        response = llm_manager.reason(prompt, temperature=0.7)
        
        return ActionResult(
            response=response.text,
            new_context=f"Subproblem: {subproblem}\n\nSolution: {response.text}"
        )


class ReflectAndRefineAction(BaseAction):
    """Action that reflects on and refines previous reasoning."""
    
    def __init__(self):
        super().__init__("reflect_and_refine", "Reflect on and refine the reasoning")
    
    def get_prompt_template(self) -> str:
        return """
Let's reflect on the reasoning so far and see if we can refine or improve it.

Context: {context}

Original Question: {original_question}

Please:
1. Review the reasoning provided so far
2. Identify any gaps, weaknesses, or areas for improvement
3. Refine or enhance the reasoning
4. Provide an improved version if possible
"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        prompt = self.get_prompt_template().format(
            context=context,
            original_question=original_question
        )
        
        response = llm_manager.reason(prompt, temperature=0.6)
        
        return ActionResult(
            response=response.text,
            new_context=f"{context}\n\nReflection and refinement:\n{response.text}"
        )